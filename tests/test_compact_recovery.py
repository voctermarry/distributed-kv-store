"""Crash consistency of compaction.

Each scenario seeds the same logical state, kills a *real* subprocess (os._exit, so no finally
blocks or buffered writes run) at one precise point inside compaction, reopens the directory in a
fresh process, and checks that recovery deterministically adopts one physical layout -- the old
tables plus the WAL, or the compacted table -- while get/scan/snapshot see identical data either
way. Reopening repeatedly must converge, and a later flush and a second compaction must work.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

from kvstore.errors import CorruptionError, OutputError
from kvstore.sstable import TableWriter
from kvstore.store import TABLE_PATTERN, Store

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_crash_driver.py")

# What both layouts must resolve to: the newer layer's tombstone hides the older value, the newer
# value wins, and the unflushed WAL row survives compaction.
EXPECTED_ROWS = [("keep1", "v1-new"), ("mem-only", "m")]

PRE_COMMIT_POINTS = ("during_write", "during_finish", "during_finalize", "pre_manifest")
POST_COMMIT_POINTS = (
    "post_manifest",
    "after_rename",
    "after_old_1",
    "after_old_2",
    "after_wal_clear",
    "after_manifest_unlink",
    "exit_after_success",
)


def run_crash(point: str, directory: str, mode: str = "flushed") -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    return subprocess.run(
        [sys.executable, DRIVER, point, directory, mode],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def run_cli(argv: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "kvstore.cli", *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def table_files(directory: str) -> list[str]:
    return sorted(name for name in os.listdir(directory) if TABLE_PATTERN.match(name))


def junk_files(directory: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(directory)
        if name.startswith(("compaction-", "compact-"))
    )


class CompactionCrashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    # -- assertions ----------------------------------------------------------
    def _assert_logical_state(self, store: Store) -> None:
        # scan and snapshot must never leak the tombstoned key or its older value.
        self.assertEqual(store.scan(), EXPECTED_ROWS)
        self.assertEqual(store.snapshot().items(), EXPECTED_ROWS)
        self.assertEqual(store.get("keep1"), (True, "v1-new"))
        self.assertEqual(store.get("mem-only"), (True, "m"))
        self.assertIsNone(store.get("ghost")[1])  # the old value must never resurface
        # A range scan covering the tombstoned key must not reveal it either.
        self.assertEqual(store.scan(start="g", end="h"), [])
        self.assertEqual([k for k, _ in store.scan(start="k")], ["keep1", "mem-only"])

    def _assert_layout_consistency(self) -> None:
        """stats/verify/next-table-number agree with the layout that was actually adopted."""
        with Store(self.path).open() as store:
            stats = store.stats()
            on_disk = table_files(self.path)
            self.assertEqual(stats.tables, len(on_disk))
            numbers = [int(TABLE_PATTERN.match(name).group(1)) for name in on_disk]
            self.assertEqual(stats.next_table_number, (max(numbers) + 1) if numbers else 1)
            self.assertEqual(stats.sealed_entries, sum(table.count for table in store.tables))
            self.assertEqual(stats.memtable_entries, stats.wal_records)
            report = store.verify()
            self.assertEqual(len(report["tables"]), len(on_disk))
            self.assertEqual(report["wal"]["truncatedBytes"], 0)
        # Nothing intermediate may survive a settled recovery.
        self.assertEqual(junk_files(self.path), [])

    def _assert_reopens_converge(self) -> None:
        seen = None
        for _ in range(3):
            with Store(self.path).open() as store:
                rows = store.scan()
                listing = tuple(sorted(os.listdir(self.path)))
            self.assertEqual(rows, EXPECTED_ROWS)
            if seen is None:
                seen = listing
            else:
                self.assertEqual(listing, seen)

    def _assert_later_writes_work(self) -> None:
        with Store(self.path).open() as store:
            old_table_count = len(table_files(self.path))
            next_number = store.stats().next_table_number
            store.put("later", "L")
            path = store.flush()
            self.assertIsNotNone(path)
            self.assertEqual(os.path.basename(path), f"table-{next_number:06d}.sst")
            result = store.compact()
            # Flush adds exactly one table, so compaction retires old_table_count + 1 tables.
            self.assertEqual(result["removedTables"], old_table_count + 1)
        with Store(self.path).open() as store:
            self.assertEqual(
                store.scan(),
                [("keep1", "v1-new"), ("later", "L"), ("mem-only", "m")],
            )
            self.assertIsNone(store.get("ghost")[1])
        # The store reached by the CLI compact entry point must behave identically.
        cli = run_cli(["compact", "--dir", self.path])
        self.assertEqual(cli.returncode, 0, cli.stderr)
        document = json.loads(cli.stdout)
        self.assertEqual(document["keys"], 3)
        self.assertEqual(document["removedTables"], 1)
        with Store(self.path).open() as store:
            self.assertEqual(
                store.scan(),
                [("keep1", "v1-new"), ("later", "L"), ("mem-only", "m")],
            )

    def _check_point(self, point: str, mode: str = "flushed") -> None:
        crashed = run_crash(point, self.path, mode)
        if point in PRE_COMMIT_POINTS:
            self.assertNotEqual(crashed.returncode, 0)
            # Intermediate artifacts are present but must not look like live tables.
            self.assertTrue(junk_files(self.path))
            self.assertEqual(len(table_files(self.path)), 2)
        with Store(self.path).open() as store:
            self._assert_logical_state(store)
            if point in PRE_COMMIT_POINTS:
                self.assertEqual(store.wal_records, 1)  # the unflushed row replays
                self.assertEqual(store.get("ghost"), (True, None))  # tombstone still live
            else:
                self.assertEqual(store.wal_records, 0)
        self._assert_reopens_converge()
        self._assert_layout_consistency()
        self._assert_later_writes_work()

    # -- the crash matrix ----------------------------------------------------
    def test_pre_commit_crashes_roll_back(self) -> None:
        for point in PRE_COMMIT_POINTS:
            with self.subTest(point=point):
                self._check_point(point)
            self.tearDown()
            self.setUp()

    def test_post_commit_crashes_roll_forward(self) -> None:
        for point in POST_COMMIT_POINTS:
            with self.subTest(point=point):
                self._check_point(point)
            self.tearDown()
            self.setUp()

    def test_memtable_covered_compaction_recovers(self) -> None:
        # One sealed old table plus newer tombstone/value only in the WAL: WAL retirement is what
        # keeps the tombstone in force after every crash point.
        for point in ("during_write", "pre_manifest", "post_manifest", "after_wal_clear", "exit_after_success"):
            with self.subTest(point=point):
                run_crash(point, self.path, "mem")
                with Store(self.path).open() as store:
                    self.assertEqual(store.scan(), EXPECTED_ROWS)
                    self.assertIsNone(store.get("ghost")[1])
                self._assert_reopens_converge()
                self.assertEqual(junk_files(self.path), [])
                self._assert_layout_consistency()
            self.tearDown()
            self.setUp()

    def test_verify_settles_an_unfinished_compaction(self) -> None:
        run_crash("post_manifest", self.path)
        cli = run_cli(["verify", "--dir", self.path])
        self.assertEqual(cli.returncode, 0, cli.stderr)
        document = json.loads(cli.stdout)
        self.assertEqual(len(document["tables"]), 1)
        self.assertEqual(document["tables"][0]["keys"], 2)
        self.assertEqual(document["wal"]["records"], 0)
        self.assertEqual(junk_files(self.path), [])

    def test_verify_after_rollback_does_not_report_corruption(self) -> None:
        run_crash("pre_manifest", self.path)
        cli = run_cli(["verify", "--dir", self.path])
        self.assertEqual(cli.returncode, 0, cli.stderr)
        document = json.loads(cli.stdout)
        self.assertEqual(len(document["tables"]), 2)
        self.assertEqual(document["wal"]["records"], 1)
        self.assertEqual(junk_files(self.path), [])

    def test_corrupt_sealed_table_still_raises_with_pending_debris(self) -> None:
        run_crash("pre_manifest", self.path)
        # Damage a genuinely committed old table: debris cleanup may run, but opening the sealed
        # table must still surface CorruptionError carrying its path.
        target = os.path.join(self.path, "table-000001.sst")
        with open(target, "r+b") as handle:
            handle.seek(40)
            handle.write(b"Z")
        with self.assertRaises(CorruptionError) as caught:
            Store(self.path).open()
        self.assertEqual(caught.exception.context.get("path"), target)

    def test_filesystem_failure_during_compaction_is_output_error(self) -> None:
        with Store(self.path).open() as store:
            store.put("ghost", "old")
            store.flush()
            store.delete("ghost")
            store.put("keep", "k")
            original_add = TableWriter.add

            def failing_add(self, key, value):  # type: ignore[no-untyped-def]
                raise OSError("simulated disk failure")

            TableWriter.add = failing_add
            try:
                with self.assertRaises(OutputError):
                    store.compact()
            finally:
                TableWriter.add = original_add
            # Nothing was committed: the live view keeps the old, acknowledged state.
            self.assertEqual(store.get("keep"), (True, "k"))
            self.assertEqual(store.get("ghost"), (True, None))
            self.assertEqual(len(table_files(self.path)), 1)
            self.assertEqual(junk_files(self.path), [])
        with Store(self.path).open() as store:
            self.assertEqual(store.scan(), [("keep", "k")])
            # The store is fully usable: a retry of compaction succeeds.
            self.assertEqual(store.compact()["keys"], 1)

    def test_successful_compact_is_durable_even_if_killed_immediately(self) -> None:
        crashed = run_crash("exit_after_success", self.path)
        self.assertEqual(crashed.returncode, 0)
        document = json.loads(crashed.stdout)
        self.assertEqual(set(document), {"table", "keys", "removedTables"})
        self.assertEqual(document["keys"], 2)
        self.assertEqual(document["removedTables"], 2)
        with Store(self.path).open() as store:
            self.assertEqual(store.scan(), EXPECTED_ROWS)
        self._assert_layout_consistency()

    def test_clean_directory_without_manifest_opens_without_repair(self) -> None:
        # A normally completed compaction leaves a clean, manifest-free directory; reopening it
        # must perform no repairs and serve the compacted data.
        run_crash("exit_after_success", self.path)
        listing = set(os.listdir(self.path))
        self.assertIn("wal.log", listing)
        self.assertFalse(any(name.endswith(".manifest") for name in listing))
        cli = run_cli(["verify", "--dir", self.path])
        self.assertEqual(cli.returncode, 0, cli.stderr)
        with Store(self.path).open() as store:
            self.assertEqual(store.scan(), EXPECTED_ROWS)


if __name__ == "__main__":
    unittest.main()
