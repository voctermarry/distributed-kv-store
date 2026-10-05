"""Hard-exit crash injection driver for the compaction recovery tests.

Runs in a real subprocess (the tests spawn it with subprocess.run) so ``os._exit`` behaves like
kill -9: no ``finally`` blocks, no buffered flushes, no interpreter cleanup. It seeds one store,
arms a single exit at a named point inside ``Store.compact`` (or its helpers), and either dies or
prints the normal compact result JSON.

Usage: python _crash_driver.py <point> <dir> [seed-mode]

seed modes:
  flushed  two sealed tables (the newer one carries a tombstone over an older value) plus one
           unflushed memtable row -- compaction covers sealed layers and the WAL at once
  mem      one older sealed table plus a newer WAL tombstone/value and an unflushed row: the
           commit still has to retire the WAL for the tombstone to keep hiding the sealed value

points (flushed unless noted):
  during_write            killed after the first row reaches the entry spool
  during_finish           killed on entry to TableWriter.finish (spool complete, no final file)
  during_finalize         killed after the assembled .out is fsynced, before its rename
  pre_manifest            killed with the finished .new on disk, before the commit record
  post_manifest           killed right after the manifest rename+fsync, before the install
  after_rename            killed right after the new table took its live-table name
  after_old_1             killed right after table-000001.sst (the oldest) is unlinked
  after_old_2             killed right after table-000002.sst (the newer) is unlinked
  after_wal_clear         killed right after the WAL was durably emptied
  after_manifest_unlink   killed right after the commit record itself is removed
  exit_after_success      compact returns normally, then the process dies without close()
"""

from __future__ import annotations

import json
import os
import sys

import kvstore.sstable as sstable_mod
from kvstore.sstable import TableWriter
from kvstore.store import Store

EXIT_KILLED = 17


def seed(store: Store, mode: str) -> None:
    store.put("ghost", "old-value")
    store.put("keep1", "v1")
    store.flush()  # table-000001: {ghost: old-value, keep1: v1}
    store.delete("ghost")
    store.put("keep1", "v1-new")
    if mode == "flushed":
        store.flush()  # table-000002: {ghost: tombstone, keep1: v1-new}
        store.put("mem-only", "m")  # stays unflushed: the WAL record compact must absorb
    else:
        # The tombstone and the newer value stay unflushed: the commit must retire the WAL or
        # the sealed table's old value resurfaces on reopen.
        store.put("mem-only", "m")


def main(argv: list[str]) -> int:
    point, directory = argv[1], argv[2]
    mode = argv[3] if len(argv) > 3 else "flushed"

    store = Store(directory).open()
    seed(store, mode)

    def die(code: int = EXIT_KILLED) -> None:
        os._exit(code)

    # -- write-phase points (class patches; this process owns exactly one store) -------------
    if point == "during_write":
        original_add = TableWriter.add

        def add_then_die(self, key, value):  # type: ignore[no-untyped-def]
            original_add(self, key, value)
            die()

        TableWriter.add = add_then_die
    elif point == "during_finish":
        TableWriter.finish = lambda self: die()  # type: ignore[assignment]
    elif point == "during_finalize":
        original_os_replace = sstable_mod.os.replace

        def replace_then_die(source, destination):  # type: ignore[no-untyped-def]
            if str(source).endswith(".out"):
                die()
            return original_os_replace(source, destination)

        sstable_mod.os.replace = replace_then_die

    # -- commit/install points: wrap Store methods at class level (slots ban instance attrs) --
    original_write_manifest = Store._write_manifest

    def manifest_hook(self, path, plan):  # type: ignore[no-untyped-def]
        if point == "pre_manifest":
            die()
        original_write_manifest(self, path, plan)
        if point == "post_manifest":
            die()

    Store._write_manifest = manifest_hook  # type: ignore[assignment]

    original_replace = Store._durably_replace

    def replace_hook(self, source, destination):  # type: ignore[no-untyped-def]
        original_replace(self, source, destination)
        if point == "after_rename":
            die()

    Store._durably_replace = replace_hook  # type: ignore[assignment]

    original_unlink = Store._durably_unlink

    def unlink_hook(self, path):  # type: ignore[no-untyped-def]
        original_unlink(self, path)
        base = os.path.basename(str(path))
        if point == "after_old_1" and base == "table-000001.sst":
            die()
        if point == "after_old_2" and base == "table-000002.sst":
            die()
        if point == "after_manifest_unlink" and base.endswith(".manifest"):
            die()

    Store._durably_unlink = unlink_hook  # type: ignore[assignment]

    original_clear = Store._clear_wal_for_compact

    def clear_hook(self):  # type: ignore[no-untyped-def]
        original_clear(self)
        if point == "after_wal_clear":
            die()

    Store._clear_wal_for_compact = clear_hook  # type: ignore[assignment]

    result = store.compact()
    if point == "exit_after_success":
        # Success was already returned to the caller of compact(); publish the same result
        # document and then die without close()/flush(): a successful return alone must be
        # durable, and the test asserts the document shape survives the hard exit.
        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
        die(0)
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
