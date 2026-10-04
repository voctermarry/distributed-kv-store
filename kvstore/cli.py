"""Command line entry point: one canonical JSON document per invocation.

Exit codes are part of the contract: 0 success, 2 input/usage error, 3 a report was produced and its
verdict is negative (a miss, or a recovery that had to repair something). Nothing but JSON is ever
written to stdout; errors go to stderr in the same document shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from . import __version__
from .errors import KVError
from .store import Store

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NEGATIVE = 3


def canonical(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _emit(document: dict[str, Any]) -> None:
    sys.stdout.write(canonical(document) + "\n")


def _store(args: argparse.Namespace) -> Store:
    # Deliberately NOT opened here: `with _store(args) as store:` calls `Store.__enter__`, and opening
    # twice leaked the first WAL handle and replayed the log a second time (found by a test that saw a
    # ResourceWarning line ahead of its error document).
    return Store(directory=args.directory, memtable_limit_bytes=args.memtable_limit)


def _command_describe(_: argparse.Namespace) -> int:
    _emit(
        {
            "name": "distributed-kv-store",
            "version": __version__,
            "operations": ["compact", "delete", "describe", "flush", "get", "put", "scan", "stats", "verify"],
            "layout": {"sealedTable": "table-<6 digits>.sst", "log": "wal.log"},
            "logRecord": {"checksum": "crc32 of the canonical payload, 8 hex", "ops": ["put", "del"]},
            "exitCodes": {"ok": EXIT_OK, "error": EXIT_ERROR, "negativeVerdict": EXIT_NEGATIVE},
        }
    )
    return EXIT_OK


def _command_put(args: argparse.Namespace) -> int:
    with _store(args) as store:
        store.put(args.key, args.value)
        _emit({"key": args.key, "written": True, "bytes": len(args.value), "writes": store.writes})
    return EXIT_OK


def _command_delete(args: argparse.Namespace) -> int:
    with _store(args) as store:
        found, value = store.get(args.key)
        store.delete(args.key)
        _emit({"key": args.key, "deleted": True, "wasPresent": found, "wasValue": value, "writes": store.writes})
    return EXIT_OK


def _command_get(args: argparse.Namespace) -> int:
    with _store(args) as store:
        found, value = store.get(args.key)
    _emit({"key": args.key, "found": found, "value": value})
    return EXIT_OK if found else EXIT_NEGATIVE


def _command_scan(args: argparse.Namespace) -> int:
    with _store(args) as store:
        rows = store.scan(start=args.start, end=args.end, limit=args.limit)
    _emit({"rows": [{"key": key, "value": value} for key, value in rows], "count": len(rows)})
    return EXIT_OK if rows else EXIT_NEGATIVE


def _command_stats(args: argparse.Namespace) -> int:
    with _store(args) as store:
        _emit(store.stats().to_document())
    return EXIT_OK


def _command_flush(args: argparse.Namespace) -> int:
    with _store(args) as store:
        path = store.flush()
        _emit({"flushed": path is not None, "table": path, "stats": store.stats().to_document()})
    return EXIT_OK if path else EXIT_NEGATIVE


def _command_compact(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = store.compact()
    _emit(result)
    return EXIT_OK


def _command_verify(args: argparse.Namespace) -> int:
    report = Store(directory=args.directory).verify()
    _emit(report)
    repaired = bool(report["wal"]["truncatedBytes"])
    return EXIT_NEGATIVE if repaired else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distributed-kv-store", description="LSM key/value store")
    parser.add_argument("--version", action="version", version=f"distributed-kv-store {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe", help="print capabilities as JSON").set_defaults(handler=_command_describe)

    def with_store(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--dir", dest="directory", required=True)
        sub.add_argument("--memtable-limit", dest="memtable_limit", type=int, default=1 << 20)
        return sub

    put = with_store("put", "write a key")
    put.add_argument("--key", required=True)
    put.add_argument("--value", required=True)
    put.set_defaults(handler=_command_put)

    delete = with_store("delete", "write a tombstone")
    delete.add_argument("--key", required=True)
    delete.set_defaults(handler=_command_delete)

    get = with_store("get", "read a key")
    get.add_argument("--key", required=True)
    get.set_defaults(handler=_command_get)

    scan = with_store("scan", "list a key range")
    scan.add_argument("--start")
    scan.add_argument("--end")
    scan.add_argument("--limit", type=int)
    scan.set_defaults(handler=_command_scan)

    with_store("stats", "store counters").set_defaults(handler=_command_stats)
    with_store("flush", "seal the memtable into a table").set_defaults(handler=_command_flush)
    with_store("compact", "merge every table into one").set_defaults(handler=_command_compact)

    verify = subparsers.add_parser("verify", help="re-open every artifact and report repairs")
    verify.add_argument("--dir", dest="directory", required=True)
    verify.set_defaults(handler=_command_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KVError as error:
        sys.stderr.write(canonical(error.to_document()) + "\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
