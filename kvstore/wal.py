"""Write-ahead log: append-only, one checksummed record per line, crash-safe recovery.

Line format (a single space separates the checksum from the payload):

    <crc32 of the canonical payload, 8 lowercase hex><space><canonical JSON payload>

The payload is `{"op":"put"|"del","key":...,"seq":N,"value":...}` with `value` omitted for `del`.
Recovery reads forward, verifies every checksum, and stops at the first record that is missing,
truncated or corrupt -- that is exactly the shape a crash leaves behind. The store then truncates
the file to the last good offset so the next append starts from a consistent place.
"""

from __future__ import annotations

import json
import os
import tempfile
import zlib
from dataclasses import dataclass
from typing import Iterable, Iterator

from .errors import ParseError, ValidationError

PUT = "put"
DEL = "del"
OPS = (PUT, DEL)


def fsync_directory(path: str) -> None:
    """Fsync the directory that owns ``path``'s entries, so renames/unlinks survive a crash.

    A rename or unlink only reaches disk once the *directory* is fsynced; the file fsync alone
    does not order the directory entry. Failures (e.g. a platform without directory fsync) are
    tolerated -- the atomic replacement still holds, only the crash guarantee weakens.
    """
    try:
        handle = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def durable_replace(path: str, data: bytes = b"") -> None:
    """Atomically replace ``path`` with ``data`` (empty by default) and force it onto disk.

    Temp file write + fsync, atomic rename, directory fsync: on reopen the target is either the
    old file or the new one, never a torn mix, and the replacement itself survives a power loss.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".wal-", suffix=".tmp", dir=directory)
    handle = os.fdopen(fd, "wb")
    try:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    except BaseException:
        handle.close()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, path)
    fsync_directory(path)


def canonical(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encode_record(payload: dict[str, object]) -> str:
    body = canonical(payload)
    checksum = format(zlib.crc32(body.encode("utf-8")) & 0xFFFFFFFF, "08x")
    return f"{checksum} {body}\n"


def decode_record(line: str, *, line_number: int | None = None) -> dict[str, object]:
    stripped = line.rstrip("\n")
    if not stripped:
        raise ParseError("empty record", line=line_number)
    checksum, _, body = stripped.partition(" ")
    if len(checksum) != 8 or not body:
        raise ParseError("record must be '<8 hex><space><json>'", line=line_number)
    expected = format(zlib.crc32(body.encode("utf-8")) & 0xFFFFFFFF, "08x")
    if checksum != expected:
        raise ParseError("record checksum mismatch", line=line_number)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ParseError(f"invalid record JSON: {error.msg}", line=line_number) from error
    if not isinstance(payload, dict):
        raise ParseError("record payload must be an object", line=line_number)
    if payload.get("op") not in OPS:
        raise ParseError(f"op must be one of {', '.join(OPS)}", line=line_number)
    key = payload.get("key")
    if not isinstance(key, str) or not key:
        raise ParseError("key must be a non-empty string", line=line_number)
    sequence = payload.get("seq")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ParseError("seq must be a positive integer", line=line_number)
    if payload["op"] == PUT and not isinstance(payload.get("value"), str):
        raise ParseError("put requires a string value", line=line_number)
    return payload


@dataclass(slots=True)
class RecoveryReport:
    records: list[dict[str, object]]
    truncated_bytes: int
    last_sequence: int
    good_bytes: int


class WriteAheadLog:
    """Append records; recover them after a crash; truncate a torn tail."""

    def __init__(self, path: str, *, sync: bool = True) -> None:
        self.path = path
        self.sync = sync
        self._handle = None
        self._sequence = 0

    # -- lifecycle -----------------------------------------------------------
    def open(self) -> None:
        if self._handle is not None:  # reopening must not leak the previous handle
            self.close()
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        self._handle = open(self.path, "a+", encoding="utf-8", newline="\n")
        self._handle.seek(0, os.SEEK_END)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            if self.sync:
                os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "WriteAheadLog":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- writing -------------------------------------------------------------
    @property
    def last_sequence(self) -> int:
        return self._sequence

    def append(self, op: str, key: str, value: str | None = None) -> int:
        if op not in OPS:
            raise ValidationError(f"op must be one of {', '.join(OPS)}", value=op)
        if not key:
            raise ValidationError("key must be non-empty", value=key)
        if op == PUT and value is None:
            raise ValidationError("put requires a value", value=key)
        if self._handle is None:
            raise ValidationError("log is not open")
        self._sequence += 1
        payload: dict[str, object] = {"op": op, "key": key, "seq": self._sequence}
        if op == PUT:
            payload["value"] = value
        self._handle.write(encode_record(payload))
        self._handle.flush()
        if self.sync:
            os.fsync(self._handle.fileno())
        return self._sequence

    def reset(self) -> None:
        """Used by `Store.flush`/`Store.compact`: the log only ever describes what is not sealed.

        The truncation is itself a durable atomic replacement, so a crash between sealing a table
        and reopening the log cannot leave a stale, non-empty log behind that replay would fold
        back in on top of the sealed data.
        """
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        durable_replace(self.path)
        self._sequence = 0
        self.open()

    # -- reading -------------------------------------------------------------
    def recover(self) -> RecoveryReport:
        """Read what is intact; report how many bytes of torn tail were dropped."""
        records: list[dict[str, object]] = []
        good_bytes = 0
        last_sequence = 0
        if not os.path.exists(self.path):
            return RecoveryReport([], 0, 0, 0)
        with open(self.path, "rb") as handle:
            raw = handle.read()
        for number, line in enumerate(raw.splitlines(keepends=True), start=1):
            try:
                payload = decode_record(line.decode("utf-8"), line_number=number)
            except (ParseError, UnicodeDecodeError):
                break
            records.append(payload)
            good_bytes += len(line)
            last_sequence = int(payload["seq"])  # type: ignore[arg-type]
        truncated = len(raw) - good_bytes
        if truncated:
            with open(self.path, "r+b") as handle:
                handle.truncate(good_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        self._sequence = last_sequence
        return RecoveryReport(records, truncated, last_sequence, good_bytes)

    def records(self) -> Iterator[dict[str, object]]:
        yield from self.recover().records


def replay(records: Iterable[dict[str, object]]) -> dict[str, str | None]:
    """Fold WAL records into the state they describe (None marks a deleted key)."""
    state: dict[str, str | None] = {}
    for payload in records:
        if payload["op"] == PUT:
            state[str(payload["key"])] = str(payload["value"])
        else:
            state[str(payload["key"])] = None
    return state
