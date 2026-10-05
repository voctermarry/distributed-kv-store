"""Subprocess driver for compaction crash tests.

It builds one fixed dataset, starts a compaction and calls ``os._exit`` at a precise point -- a
hard process kill that skips every ``except``/cleanup handler, exactly like ``SIGKILL`` or power
loss. The caller reopens the directory from a fresh process and checks recovery.

Dataset (three sealed tables plus a live memtable):

    table 1:  a -> "1", b -> "2", z -> "zed"
    table 2:  a tombstone, b -> "3"
    table 3:  d -> "5"
    memtable: c -> "4", b tombstone          (WAL holds exactly these two records)

Logical content regardless of how an interrupted compaction is resolved:

    visible rows (scan/snapshot): c -> "4", d -> "5", z -> "zed"
    a and b resolve to no value: a newer tombstone shadows the older values.

Kill points:

    w0  "writing" manifest durable, new table not created yet
    wm  new table spool partially written (after the first entry row)
    wf  new table sealed ("final file" present), manifest still "writing"
    c0  manifest flipped to "committed", no old table removed yet
    c1  one of three old tables removed
    c2  two of three old tables removed
    cw  every old table removed + directory fsynced, WAL not yet truncated
    wr  WAL truncated, committed manifest not yet removed
    ok  compaction reported success -- die immediately afterwards

Store is a slotted dataclass, so the instrumentation patches class/module bindings rather than
instance attributes.
"""

from __future__ import annotations

import os
import sys

import kvstore.store as store_module
from kvstore.sstable import TableWriter
from kvstore.store import TABLE_PATTERN, Store
from kvstore.wal import WriteAheadLog

EXPECTED = [("c", "4"), ("d", "5"), ("z", "zed")]
DELETED = ("a", "b")
FORWARD_POINTS = {"c0", "c1", "c2", "cw", "wr"}
ROLLBACK_POINTS = {"w0", "wm", "wf"}
MANIFEST_BASENAME = ".compact.json"


def die() -> None:
    os._exit(0)


def build(store: Store) -> None:
    store.put("a", "1")
    store.put("b", "2")
    store.put("z", "zed")
    store.flush()
    store.delete("a")
    store.put("b", "3")
    store.flush()
    store.put("d", "5")
    store.flush()
    store.put("c", "4")
    store.delete("b")


def install(point: str, store: Store) -> None:
    original_write_manifest = Store._write_manifest
    original_add = TableWriter.add
    original_finish = TableWriter.finish
    original_unlink = store_module.os.unlink
    original_reset = WriteAheadLog.reset
    state = {"removed": 0}

    def write_manifest(self, document):
        original_write_manifest(self, document)
        if document["phase"] == "writing" and point == "w0":
            die()
        if document["phase"] == "committed" and point == "c0":
            die()

    def add(self, key, value):
        original_add(self, key, value)
        if point == "wm":
            die()

    def finish(self):
        table = original_finish(self)
        if point == "wf":
            die()
        return table

    def unlink(path):
        name = os.path.basename(str(path))
        if TABLE_PATTERN.match(name):
            original_unlink(path)
            state["removed"] += 1
            if point == "c1" and state["removed"] == 1:
                die()
            if point == "c2" and state["removed"] == 2:
                die()
        elif name == MANIFEST_BASENAME and point == "wr":
            die()  # WAL already truncated, committed manifest still on disk
        else:
            original_unlink(path)

    def reset(self):
        if point == "cw" and self is store.wal:
            die()
        original_reset(self)

    Store._write_manifest = write_manifest
    TableWriter.add = add
    TableWriter.finish = finish
    store_module.os.unlink = unlink
    WriteAheadLog.reset = reset


def main() -> None:
    directory, point = sys.argv[1], sys.argv[2]
    store = Store(directory).open()
    build(store)
    install(point, store)
    store.compact()
    if point == "ok":
        die()


if __name__ == "__main__":
    main()
