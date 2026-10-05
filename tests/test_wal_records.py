"""Live ``wal_records`` accounting.

``StoreStats.wal_records`` (and ``walRecords`` in ``to_document``) must always equal the number of
*complete, recoverable records* the WAL file currently holds -- not the number of distinct keys,
the memtable size, or a snapshot taken at open time. Every durable append counts as one record
(overwrites, repeated deletes and deletes of missing keys included); sealing the memtable, whether
through an explicit flush, an automatic memtable-limit flush or a compaction, retires the log and
zeros the counter. A failed append must not count ahead of durability; a failed *seal* after a
durable append must keep that record counted and recoverable.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from kvstore.errors import OutputError
from kvstore.sstable import SSTable
from kvstore.store import Store
from kvstore.wal import WriteAheadLog


def physical_wal_records(directory: str) -> int:
    """Ground truth from outside the Store: recover the WAL file as a fresh opener would."""
    report = WriteAheadLog(os.path.join(directory, "wal.log")).recover()
    return len(report.records)


class WalRecordCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name
        self.wal_path = os.path.join(self.path, "wal.log")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_every_append_counts_one_record_even_for_the_same_key(self) -> None:
        with Store(self.path).open() as store:
            self.assertEqual(store.stats().wal_records, 0)
            store.put("a", "1")
            store.put("a", "2")  # overwrite: a second record, not a replacement
            store.put("a", "3")
            store.delete("a")
            store.delete("missing")  # a key that never existed still appends a record
            store.delete("a")  # deleting an already-deleted key appends again
            stats = store.stats()
            self.assertEqual(stats.wal_records, 6)
            # The counter is a record count: it matches neither distinct keys nor memtable size.
            self.assertEqual(stats.memtable_entries, 2)
            self.assertEqual(store.writes, 6)
            self.assertEqual(physical_wal_records(self.path), 6)
            doc = stats.to_document()
            self.assertEqual(doc["walRecords"], 6)
            self.assertEqual(
                set(doc),
                {
                    "directory",
                    "tables",
                    "sealedEntries",
                    "memtableEntries",
                    "memtableBytes",
                    "tombstones",
                    "walRecords",
                    "walTruncatedBytes",
                    "nextTableNumber",
                },
            )
        with Store(self.path).open() as reopened:
            # Recovery accepts the same six complete records; replay only folds their state.
            self.assertEqual(reopened.stats().wal_records, 6)
            self.assertEqual(reopened.stats().memtable_entries, 2)
            self.assertEqual(reopened.get("a"), (True, None))
            self.assertEqual(reopened.get("missing"), (True, None))

    def test_explicit_flush_zeros_the_counter_and_an_empty_flush_leaves_it(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.delete("a")
            self.assertEqual(store.stats().wal_records, 2)
            path = store.flush()
            self.assertIsNotNone(path)
            self.assertEqual(store.stats().wal_records, 0)
            self.assertEqual(physical_wal_records(self.path), 0)
            # Nothing live: None comes back and the retired log must not gain a phantom record.
            self.assertIsNone(store.flush())
            self.assertEqual(store.stats().wal_records, 0)
            store.put("b", "2")
            self.assertEqual(store.stats().wal_records, 1)
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 1)
            self.assertEqual(reopened.get("b"), (True, "2"))

    def test_automatic_memtable_limit_flush_leaves_zero_on_return(self) -> None:
        with Store(self.path, memtable_limit_bytes=256).open() as store:
            store.put("k1", "v" * 200)  # 202 bytes: not yet at the limit
            self.assertEqual(store.stats().wal_records, 1)
            store.put("k2", "v" * 200)  # pushes past the limit: sealed automatically
            stats = store.stats()
            self.assertEqual(stats.wal_records, 0)
            self.assertEqual(stats.memtable_entries, 0)
            self.assertEqual(stats.tables, 1)
            self.assertEqual(physical_wal_records(self.path), 0)
            # The appended record was taken over by the new table; reads still see it.
            self.assertEqual(store.get("k1"), (True, "v" * 200))
            self.assertEqual(store.get("k2"), (True, "v" * 200))
            store.put("k3", "3")  # the post-seal log starts counting from one again
            self.assertEqual(store.stats().wal_records, 1)
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 1)
            self.assertEqual(reopened.get("k1"), (True, "v" * 200))
            self.assertEqual(reopened.get("k3"), (True, "3"))

    def test_compact_leaves_the_counter_at_zero(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.put("b", "2")
            store.flush()
            store.delete("a")
            store.put("b", "3")
            self.assertEqual(store.stats().wal_records, 2)
            result = store.compact()
            self.assertGreaterEqual(result["removedTables"], 1)
            self.assertEqual(store.stats().wal_records, 0)
            self.assertEqual(physical_wal_records(self.path), 0)
            # Compacting again, with an empty memtable, keeps it at zero.
            store.compact()
            self.assertEqual(store.stats().wal_records, 0)
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 0)
            self.assertEqual(reopened.scan(), [("b", "3")])

    def test_failed_append_does_not_increment_the_counter(self) -> None:
        failure = OutputError("simulated disk failure", path=self.wal_path)
        with Store(self.path).open() as store:
            with mock.patch.object(WriteAheadLog, "append", side_effect=failure):
                with self.assertRaises(OutputError):
                    store.put("a", "1")
                with self.assertRaises(OutputError):
                    store.delete("b")
            # Nothing was durable: no record, no write, no memtable entry.
            stats = store.stats()
            self.assertEqual(stats.wal_records, 0)
            self.assertEqual(store.writes, 0)
            self.assertEqual(stats.memtable_entries, 0)
            self.assertEqual(physical_wal_records(self.path), 0)
            store.put("a", "2")  # the log still works and starts counting from one
            self.assertEqual(store.stats().wal_records, 1)
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 1)
            self.assertEqual(reopened.get("a"), (True, "2"))

    def test_failed_seal_after_a_durable_append_keeps_the_record_counted(self) -> None:
        with Store(self.path, memtable_limit_bytes=256).open() as store:
            store.put("a", "1")
            self.assertIsNotNone(store.flush())  # one sealed table, log retired
            self.assertEqual(store.stats().wal_records, 0)
            disk_failure = OSError("simulated seal failure")
            with mock.patch.object(SSTable, "write", side_effect=disk_failure):
                # 502 value bytes force the automatic seal as part of this very put; the WAL append
                # has already landed durably before SSTable.write is attempted.
                with self.assertRaises(OutputError):
                    store.put("b", "v" * 500)
            stats = store.stats()
            self.assertEqual(stats.wal_records, 1)
            self.assertEqual(physical_wal_records(self.path), 1)
            # The unsealed memtable still serves the acknowledged write in this session.
            self.assertEqual(store.get("a"), (True, "1"))
            self.assertEqual(store.get("b"), (True, "v" * 500))
        # A fresh process gets the same answer straight from the retained WAL record.
        with Store(self.path, memtable_limit_bytes=256).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 1)
            self.assertEqual(reopened.get("b"), (True, "v" * 500))
            # The store is still usable: sealing now succeeds and retires the record.
            self.assertIsNotNone(reopened.flush())
            self.assertEqual(reopened.stats().wal_records, 0)
            self.assertEqual(physical_wal_records(self.path), 0)

    def test_reopen_matches_the_running_counter_and_excludes_a_torn_tail(self) -> None:
        # Single open() per with-block (the CLI idiom): an explicit Store(path).open() *and*
        # __enter__ would open twice, and the first recovery already truncates and reports the
        # torn bytes, leaving none for the second to see.
        with Store(self.path) as store:
            store.put("a", "1")
            store.put("b", "2")
            store.delete("a")
            self.assertEqual(store.stats().wal_records, 3)
        with Store(self.path) as store:
            stats = store.stats()
            self.assertEqual(stats.wal_records, 3)
            self.assertEqual(stats.wal_truncated_bytes, 0)
            self.assertEqual(physical_wal_records(self.path), 3)
        # Simulate a crash mid-append: an incomplete record follows the three good ones.
        with open(self.wal_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("deadbeef {\"op\":\"put\",\"key\":\"c\",\"seq\":4,\"va")
        with Store(self.path) as store:
            stats = store.stats()
            self.assertEqual(stats.wal_records, 3)  # only records before the truncation point
            self.assertGreater(stats.wal_truncated_bytes, 0)
            self.assertEqual(store.get("c"), (False, None))
        # The repair is durable: a further open reports no repairs and the same three records.
        with Store(self.path) as store:
            stats = store.stats()
            self.assertEqual(stats.wal_records, 3)
            self.assertEqual(stats.wal_truncated_bytes, 0)

    def test_stats_document_stays_in_lockstep_with_the_counter(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.delete("a")
            stats = store.stats()
            self.assertEqual(stats.tombstones, 1)
            self.assertEqual(
                stats.to_document(),
                {
                    "directory": os.path.abspath(self.path),
                    "tables": 0,
                    "sealedEntries": 0,
                    "memtableEntries": 1,
                    "memtableBytes": stats.memtable_bytes,
                    "tombstones": 1,
                    "walRecords": 2,
                    "walTruncatedBytes": 0,
                    "nextTableNumber": 1,
                },
            )
            store.flush()
            self.assertEqual(store.stats().to_document()["walRecords"], 0)


if __name__ == "__main__":
    unittest.main()
