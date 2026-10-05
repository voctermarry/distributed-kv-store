"""Live WAL record counting.

``wal_records`` / ``walRecords`` must always equal the number of *complete, recoverable* records
the current ``wal.log`` holds -- not the number of distinct keys, not the memtable size, and not a
snapshot taken at ``open()``. Every durable append adds exactly one (overwrites, repeated
deletes and deletes of never-existing keys each count); sealing the memtable retires the log and
zeroes the count, an empty flush leaves it untouched, and ``compact`` leaves it at zero. A failed
append counts nothing; an append that is already durable when a later sealing step fails still
counts and replays identically after a reopen.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from kvstore.errors import OutputError
from kvstore.sstable import SSTable
from kvstore.store import Store
from kvstore.wal import WriteAheadLog


def recover_records(directory: str) -> int:
    """Count intact WAL records straight from disk, independently of any open Store."""
    return len(WriteAheadLog(os.path.join(directory, "wal.log")).recover().records)


class WalRecordCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_put_and_delete_increment_the_live_counter(self) -> None:
        with Store(self.path).open() as store:
            self.assertEqual(store.stats().wal_records, 0)
            store.put("a", "1")
            store.put("b", "2")
            store.delete("a")
            stats = store.stats()
            # Three records even though only two keys exist: the counter counts records, not keys.
            self.assertEqual(stats.wal_records, 3)
            self.assertEqual(stats.to_document()["walRecords"], 3)
            self.assertEqual(stats.memtable_entries, 2)  # a (tombstone) and b

    def test_overwrites_repeated_deletes_and_missing_deletes_each_count(self) -> None:
        with Store(self.path).open() as store:
            store.put("k", "v1")
            store.put("k", "v2")
            store.put("k", "v3")  # overwrite: a third record
            store.delete("k")
            store.delete("k")  # deleting an already-deleted key is still a record
            store.delete("never-existed")  # so is deleting a key that never existed
            stats = store.stats()
            self.assertEqual(stats.wal_records, 6)
            self.assertEqual(stats.memtable_entries, 2)  # k tombstone + never-existed tombstone
        self.assertEqual(recover_records(self.path), 6)
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 6)

    def test_stats_document_keeps_its_field_set_and_counter(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.delete("a")
            stats = store.stats()
            document = stats.to_document()
            self.assertEqual(
                set(document),
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
            self.assertEqual(document["walRecords"], stats.wal_records)
            self.assertEqual(document["walRecords"], 2)
            self.assertEqual(document["walTruncatedBytes"], stats.wal_truncated_bytes)
            self.assertEqual(document["memtableEntries"], 1)
            self.assertEqual(document["tombstones"], 1)

    def test_explicit_flush_zeroes_the_counter_empty_flush_leaves_it(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.put("b", "2")
            self.assertEqual(store.stats().wal_records, 2)
            self.assertIsNotNone(store.flush())
            self.assertEqual(store.stats().wal_records, 0)
            self.assertEqual(store.stats().memtable_entries, 0)
            # An empty memtable seals nothing and must not touch the counter.
            self.assertIsNone(store.flush())
            self.assertEqual(store.stats().wal_records, 0)
            # The retired log starts fresh: the next write counts from zero.
            store.put("c", "3")
            self.assertEqual(store.stats().wal_records, 1)
            self.assertEqual(store.stats().to_document()["walRecords"], 1)

    def test_automatic_flush_zeroes_the_counter_for_the_triggering_write(self) -> None:
        # The reported bug: a write that sealed the memtable still showed the open-time (or
        # pre-seal) count until the process was restarted.
        with Store(self.path, memtable_limit_bytes=64).open() as store:
            store.put("a", "x" * 40)  # 41 bytes: not yet full
            self.assertEqual(store.stats().wal_records, 1)
            store.put("b", "y" * 40)  # crosses the limit: durable append, then auto-seal
            stats = store.stats()
            self.assertEqual(stats.wal_records, 0)  # the new table took the append over
            self.assertEqual(stats.to_document()["walRecords"], 0)
            self.assertEqual(stats.memtable_entries, 0)
            self.assertEqual(stats.tables, 1)
            self.assertEqual(store.get("a"), (True, "x" * 40))
            self.assertEqual(store.get("b"), (True, "y" * 40))
        # Reopening sees the sealed state directly: an empty log and both values.
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 0)
            self.assertEqual(reopened.get("a"), (True, "x" * 40))
            self.assertEqual(reopened.get("b"), (True, "y" * 40))
        self.assertEqual(recover_records(self.path), 0)

    def test_compact_leaves_the_counter_at_zero(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.flush()
            store.put("b", "2")
            store.delete("a")
            self.assertEqual(store.stats().wal_records, 2)
            store.compact()
            self.assertEqual(store.stats().wal_records, 0)
        self.assertEqual(recover_records(self.path), 0)
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 0)
            self.assertEqual(reopened.scan(), [("b", "2")])

    def test_reopen_count_equals_the_pre_close_record_count(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.put("a", "2")
            store.delete("a")
            store.put("b", "3")
            self.assertEqual(store.stats().wal_records, 4)
        with Store(self.path).open() as reopened:
            stats = reopened.stats()
            self.assertEqual(stats.wal_records, 4)  # same as before close
            self.assertEqual(stats.wal_truncated_bytes, 0)
            self.assertEqual(stats.memtable_entries, 2)  # four records fold to two keys
        self.assertEqual(recover_records(self.path), 4)

    def test_torn_tail_counts_only_records_before_the_truncation_point(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.put("b", "2")
        with open(os.path.join(self.path, "wal.log"), "a", encoding="utf-8", newline="\n") as handle:
            handle.write("deadbeef {\"op\":\"put\",\"key\":\"c\",\"seq\":3,\"va")  # torn tail
        # Open exactly once (the ``with Store().open()`` idiom re-enters and runs recovery a
        # second time) so the report observed is the one produced by the actual repair.
        reopened = Store(self.path).open()
        try:
            stats = reopened.stats()
            self.assertEqual(stats.wal_records, 2)  # the torn record is not recoverable
            self.assertGreater(stats.wal_truncated_bytes, 0)  # the repair is still reported
            self.assertEqual(stats.to_document()["walTruncatedBytes"], stats.wal_truncated_bytes)
            self.assertEqual(reopened.get("c"), (False, None))
        finally:
            reopened.close()
        # Recovery repaired the file in place: exactly the two good records remain, a later
        # recovery has nothing left to truncate.
        clean = WriteAheadLog(os.path.join(self.path, "wal.log")).recover()
        self.assertEqual(len(clean.records), 2)
        self.assertEqual(clean.truncated_bytes, 0)

    def test_failed_append_before_durability_does_not_count(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            self.assertEqual(store.stats().wal_records, 1)

            original_append = WriteAheadLog.append

            def failing_append(self, op, key, value=None):  # type: ignore[no-untyped-def]
                raise OutputError("wal append failed before the record landed", path=self.path)

            WriteAheadLog.append = failing_append
            try:
                with self.assertRaises(OutputError):
                    store.put("b", "2")
                with self.assertRaises(OutputError):
                    store.delete("c")
            finally:
                WriteAheadLog.append = original_append

            stats = store.stats()
            self.assertEqual(stats.wal_records, 1)  # nothing was counted ahead of durability
            self.assertEqual(store.writes, 1)  # writes keeps meaning only acknowledged operations
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 1)
            self.assertEqual(reopened.get("b"), (False, None))
            self.assertEqual(reopened.get("c"), (False, None))
        self.assertEqual(recover_records(self.path), 1)

    def test_successful_append_is_counted_when_the_auto_seal_then_fails(self) -> None:
        with Store(self.path, memtable_limit_bytes=64).open() as store:
            store.put("a", "x" * 40)  # 41 bytes: under the limit
            original_write = SSTable.write

            def failing_write(path, items):  # type: ignore[no-untyped-def]
                raise OSError("simulated disk failure during sealing")

            # The append for b is durable first; only the memtable-limit seal that follows fails.
            SSTable.write = failing_write
            try:
                with self.assertRaises(OSError):
                    store.put("b", "y" * 40)
            finally:
                SSTable.write = original_write

            stats = store.stats()
            self.assertEqual(stats.wal_records, 2)  # the WAL keeps both recoverable rows
            self.assertEqual(store.writes, 2)
            self.assertEqual(stats.memtable_entries, 2)  # the failed seal sealed nothing
        # No table was produced, yet neither acknowledged write is lost: a reopen replays both.
        self.assertFalse(any(name.endswith(".sst") for name in os.listdir(self.path)))
        self.assertEqual(recover_records(self.path), 2)
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 2)
            self.assertEqual(reopened.get("a"), (True, "x" * 40))
            self.assertEqual(reopened.get("b"), (True, "y" * 40))

    def test_store_recovers_after_a_failed_seal_and_a_retry_retires_the_log(self) -> None:
        with Store(self.path, memtable_limit_bytes=64).open() as store:
            store.put("a", "x" * 40)
            original_write = SSTable.write

            def failing_write(path, items):  # type: ignore[no-untyped-def]
                raise OSError("simulated disk failure during sealing")

            SSTable.write = failing_write
            try:
                with self.assertRaises(OSError):
                    store.put("b", "y" * 40)
            finally:
                SSTable.write = original_write
            self.assertEqual(store.stats().wal_records, 2)
            # Sealing now succeeds; the memtable (never cleared by the failed attempt) seals and
            # the same two records retire from the log.
            self.assertIsNotNone(store.flush())
            self.assertEqual(store.stats().wal_records, 0)
        with Store(self.path).open() as reopened:
            self.assertEqual(reopened.stats().wal_records, 0)
            self.assertEqual(reopened.scan(), [("a", "x" * 40), ("b", "y" * 40)])

    def test_writes_is_independent_of_the_live_record_count(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.flush()  # retires the log but is not itself a write operation
            store.put("b", "2")
            store.delete("a")
            self.assertEqual(store.writes, 3)  # every acknowledged put/delete, cumulatively
            self.assertEqual(store.stats().wal_records, 2)  # only what the current log still holds
            # And writes keeps advancing on overwrites while wal_records tracks each record.
            store.put("b", "again")
            self.assertEqual(store.writes, 4)
            self.assertEqual(store.stats().wal_records, 3)


if __name__ == "__main__":
    unittest.main()
