"""Crash consistency of compaction.

A driver subprocess (``crash_driver.py``) builds one dataset, starts a compaction and hard-kills
itself with ``os._exit`` at every interesting point -- skipping every exception handler and
cleanup, the same shape as SIGKILL or power loss. These tests reopen the directory from another
process and assert that:

* the interrupted compaction is detected and deterministically finished or undone;
* the logical data (get / scan / snapshot) is identical under either physical layout, in
  particular when a newer-layer tombstone hides an older value;
* repeated close/reopen converges and stays stable;
* stats, verify and the next table number match the adopted layout, and flush/compact keep
  working;
* leftover temp artifacts are never mistaken for tables and never raise corruption_error;
* a genuinely corrupt *sealed* table still raises CorruptionError;
* a persistence failure raises OutputError and the pre-compaction state reopens intact.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

from kvstore.cli import EXIT_NEGATIVE, EXIT_OK, main
from kvstore.errors import CorruptionError, OutputError
from kvstore.sstable import SSTable
from kvstore.store import MANIFEST_NAME, Store

from tests import crash_driver

DRIVER = os.path.abspath(crash_driver.__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(DRIVER))
EXPECTED = crash_driver.EXPECTED
DELETED = crash_driver.DELETED
ALL_POINTS = sorted(crash_driver.ROLLBACK_POINTS | crash_driver.FORWARD_POINTS | {"ok"})


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def crash_compact(directory: str, point: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "tests.crash_driver", directory, point],
        check=True,
        cwd=REPO_ROOT,
    )


def logical_view(directory: str) -> None:
    """Assert get/scan/snapshot all present the post-compaction logical state.

    A deleted key presents *no value* under either adopted layout: pre-compaction the tombstone
    is still physically present and ``get`` reports a found delete marker (``(True, None)`` --
    the store's deliberate "deleted" vs "never existed" distinction), post-compaction the
    tombstone was dropped and the same key is a full miss (``(False, None)``). What must be
    identical is the logical content: the value is None in both, the old value never returns,
    and scan/snapshot are byte-identical.
    """
    with Store(directory).open() as store:
        for key, value in EXPECTED:
            assert store.get(key) == (True, value), key
        for key in DELETED:
            found, value = store.get(key)
            assert value is None, key
            assert all(k != key for k, _ in store.scan()), key
        assert store.scan() == EXPECTED
        assert sorted(store.snapshot().items()) == EXPECTED


class CompactionCrashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _table_files(self) -> list[str]:
        return sorted(name for name in os.listdir(self.path) if name.endswith(".sst"))

    # -- every kill window ---------------------------------------------------
    def test_every_kill_window_recovers_to_the_same_logical_data(self) -> None:
        for point in ALL_POINTS:
            with self.subTest(point=point):
                crash_compact(self.path, point)
                logical_view(self.path)
                # Recovery is idempotent and converges: several more close/reopen cycles must
                # keep the same answer and never rediscover an "unfinished" compaction.
                for _ in range(3):
                    logical_view(self.path)
                # The CLI reads through the same open() recovery path.
                for key, value in EXPECTED:
                    code, out, err = run_cli(["get", "--dir", self.path, "--key", key])
                    self.assertEqual((code, err), (EXIT_OK, ""))
                    self.assertEqual(json.loads(out)["value"], value)
                for key in DELETED:
                    _, out, _ = run_cli(["get", "--dir", self.path, "--key", key])
                    self.assertIsNone(json.loads(out)["value"])
                code, out, _ = run_cli(["scan", "--dir", self.path])
                self.assertEqual(code, EXIT_OK)
                self.assertEqual([(r["key"], r["value"]) for r in json.loads(out)["rows"]], EXPECTED)
                self._reset()

    def _reset(self) -> None:
        self.directory.cleanup()
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def test_pre_commit_kills_roll_back_to_the_source_layout(self) -> None:
        for point in sorted(crash_driver.ROLLBACK_POINTS):
            with self.subTest(point=point):
                crash_compact(self.path, point)
                with Store(self.path).open() as store:
                    self.assertEqual(self._table_files(), ["table-000001.sst", "table-000002.sst", "table-000003.sst"])
                    self.assertEqual(store.stats().next_table_number, 4)
                    self.assertEqual(store.stats().tables, 3)
                self.assertFalse(os.path.exists(os.path.join(self.path, MANIFEST_NAME)))
                self._reset()

    def test_post_commit_kills_roll_forward_to_the_new_layout(self) -> None:
        for point in sorted(crash_driver.FORWARD_POINTS | {"ok"}):
            with self.subTest(point=point):
                crash_compact(self.path, point)
                with Store(self.path).open() as store:
                    self.assertEqual(self._table_files(), ["table-000004.sst"])
                    stats = store.stats()
                    self.assertEqual(stats.tables, 1)
                    self.assertEqual(stats.next_table_number, 5)
                    self.assertEqual(stats.sealed_entries, len(EXPECTED))
                    self.assertEqual(stats.memtable_entries, 0)
                    self.assertEqual(stats.wal_records, 0)
                self.assertFalse(os.path.exists(os.path.join(self.path, MANIFEST_NAME)))
                self._reset()

    def test_older_value_never_resurrects_after_rollback(self) -> None:
        # "a" and "b" hold older values below newer tombstones; a rollback must not let the
        # interrupted output table or the WAL replay bring those values back.
        crash_compact(self.path, "wm")
        with Store(self.path).open() as store:
            for key in DELETED:
                found, value = store.get(key)
                self.assertIsNone(value, key)          # no value: deleted, never the old value
                self.assertNotIn(key, dict(store.scan()))
            self.assertEqual(store.get("c"), (True, "4"))  # memtable/WAL record still present
            self.assertEqual(store.get("z"), (True, "zed"))

    # -- verify / stats after recovery ---------------------------------------
    def test_verify_is_clean_after_every_recovery(self) -> None:
        for point in ALL_POINTS:
            with self.subTest(point=point):
                crash_compact(self.path, point)
                with Store(self.path).open() as store:
                    report = store.verify()
                self.assertEqual(report["wal"]["truncatedBytes"], 0, point)
                self.assertTrue(report["tables"], point)
                code, out, err = run_cli(["verify", "--dir", self.path])
                self.assertEqual((code, err), (EXIT_OK, ""), (point, out, err))
                self._reset()

    # -- continued service ---------------------------------------------------
    def test_flush_and_compact_keep_working_after_recovery(self) -> None:
        for point in ALL_POINTS:
            with self.subTest(point=point):
                crash_compact(self.path, point)
                with Store(self.path).open() as store:
                    store.put("n", "9")
                    path = store.flush()
                    expected_number = 5 if point in crash_driver.FORWARD_POINTS | {"ok"} else 4
                    self.assertTrue(path.endswith(f"table-{expected_number:06d}.sst"))
                    result = store.compact()
                    self.assertEqual(result["keys"], len(EXPECTED) + 1)
                    self.assertEqual(store.get("n"), (True, "9"))
                with Store(self.path).open() as store:
                    self.assertEqual(store.scan(), sorted(EXPECTED + [("n", "9")]))
                self._reset()

    def test_writes_after_rollback_do_not_duplicate_or_lose_keys(self) -> None:
        crash_compact(self.path, "w0")
        with Store(self.path).open() as store:
            store.put("c", "44")  # overwrite the still-replayed WAL record
            store.put("e", "6")
            store.flush()
            self.assertEqual(store.get("c"), (True, "44"))
            self.assertEqual(store.get("e"), (True, "6"))
        with Store(self.path).open() as store:
            self.assertEqual(store.get("c"), (True, "44"))
            self.assertEqual(store.get("e"), (True, "6"))

    # -- stray intermediate artifacts ----------------------------------------
    def test_stray_intermediates_are_ignored_and_cleaned(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.flush()
        for stray in (
            "table-000009.sst.tmp",
            "table-000009.sst.out",
            ".compact.json.tmp",
            ".wal-abcd.tmp",
        ):
            with open(os.path.join(self.path, stray), "wb") as handle:
                handle.write(b"garbage")
        # No manifest: nothing to recover, but leftovers must not be read as tables nor error.
        with Store(self.path).open() as store:
            self.assertEqual(len(store.tables), 1)
            self.assertEqual(store.get("a"), (True, "1"))
            store.verify()
        remaining = set(os.listdir(self.path))
        self.assertNotIn("table-000009.sst.tmp", remaining)
        self.assertNotIn("table-000009.sst.out", remaining)
        self.assertNotIn(".compact.json.tmp", remaining)
        self.assertNotIn(".wal-abcd.tmp", remaining)
        code, _, err = run_cli(["verify", "--dir", self.path])
        self.assertEqual((code, err), (EXIT_OK, ""))

    def test_manifest_without_any_table_is_a_harmless_rollback(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.flush()
        with open(os.path.join(self.path, MANIFEST_NAME), "w") as handle:
            json.dump({"phase": "writing", "result": "table-000002.sst", "sources": ["table-000001.sst"]}, handle)
        with Store(self.path).open() as store:
            self.assertEqual(store.get("a"), (True, "1"))
        self.assertFalse(os.path.exists(os.path.join(self.path, MANIFEST_NAME)))

    # -- corrupt committed output falls back, corrupt sealed table does not --
    def test_corrupt_committed_output_rolls_back_to_sources(self) -> None:
        crash_compact(self.path, "c0")
        # Damage the just-sealed output table while the committed manifest still points at it.
        result = os.path.join(self.path, "table-000004.sst")
        self.assertTrue(os.path.exists(result))
        with open(result, "r+b") as handle:
            handle.seek(40)
            handle.write(b"Z")
        with Store(self.path).open() as store:  # must roll back, not raise corruption_error
            self.assertEqual(store.get("z"), (True, "zed"))
            self.assertIsNone(store.get("a")[1])  # deleted, old value hidden
        with Store(self.path).open() as store:
            result2 = store.compact()
            self.assertEqual(result2["keys"], len(EXPECTED))

    def test_genuinely_corrupt_sealed_table_still_raises_with_path(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            path = store.flush()
        assert path is not None
        with open(path, "r+b") as handle:
            handle.seek(40)
            handle.write(b"Z")
        with self.assertRaises(CorruptionError) as caught:
            Store(self.path).open()
        self.assertEqual(caught.exception.context.get("path"), path)

    # -- persistence failure --------------------------------------------------
    def test_persistence_failure_is_output_error_and_state_reopens(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.flush()
            store.delete("a")
            real_replace = os.replace

            def fail_replace(source, target, *args):
                if str(target).endswith(MANIFEST_NAME):
                    raise OSError("simulated fs failure")
                return real_replace(source, target, *args)

            os.replace = fail_replace
            try:
                with self.assertRaises(OutputError):
                    store.compact()
            finally:
                os.replace = real_replace
            self.assertFalse(os.path.exists(os.path.join(self.path, MANIFEST_NAME)))
        # The pre-compaction logical state is intact on the next open.
        with Store(self.path).open() as store:
            self.assertEqual(store.get("a"), (True, None))
            self.assertEqual(store.scan(), [])
            result = store.compact()
            self.assertEqual(result["removedTables"], 1)
        with Store(self.path).open() as store:
            self.assertEqual(store.get("a"), (False, None))


if __name__ == "__main__":
    unittest.main()
