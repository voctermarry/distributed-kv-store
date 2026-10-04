"""Store semantics: visibility order, tombstones, flush, compaction, snapshots, CLI surface."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from kvstore.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main
from kvstore.errors import CorruptionError, ValidationError
from kvstore.sstable import SSTable
from kvstore.store import Store


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_put_get_round_trip(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            self.assertEqual(store.get("a"), (True, "1"))
            self.assertEqual(store.get("missing"), (False, None))

    def test_later_write_wins_across_a_flush(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.flush()
            store.put("a", "2")
            self.assertEqual(store.get("a"), (True, "2"))
        with Store(self.path).open() as store:
            self.assertEqual(store.get("a"), (True, "2"))

    def test_delete_hides_older_values_after_reopen(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.flush()
            store.delete("a")
        with Store(self.path).open() as store:
            self.assertEqual(store.get("a"), (True, None))
            self.assertEqual(store.scan(), [])

    def test_scan_is_ordered_and_bounded(self) -> None:
        with Store(self.path).open() as store:
            for key in ("c", "a", "b"):
                store.put(key, key.upper())
            self.assertEqual([key for key, _ in store.scan()], ["a", "b", "c"])
            self.assertEqual([key for key, _ in store.scan(start="b")], ["b", "c"])
            self.assertEqual([key for key, _ in store.scan(end="c")], ["a", "b"])
            self.assertEqual([key for key, _ in store.scan(limit=2)], ["a", "b"])

    def test_scan_rejects_bad_ranges_and_limits(self) -> None:
        with Store(self.path).open() as store:
            with self.assertRaises(ValidationError):
                store.scan(start="z", end="a")
            with self.assertRaises(ValidationError):
                store.scan(limit=0)

    def test_flush_seals_the_log_and_keeps_data(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            path = store.flush()
            self.assertIsNotNone(path)
            self.assertTrue(os.path.basename(path).startswith("table-"))
            self.assertEqual(store.stats().wal_records, 0)
            self.assertEqual(store.get("a"), (True, "1"))

    def test_flush_without_writes_reports_nothing(self) -> None:
        with Store(self.path).open() as store:
            self.assertIsNone(store.flush())

    def test_compact_merges_tables_and_drops_tombstones(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.put("b", "2")
            store.flush()
            store.delete("a")
            store.put("b", "3")
            store.flush()
            before = store.stats()
            self.assertEqual(before.tables, 2)
            result = store.compact()
            self.assertEqual(result["removedTables"], 2)
            self.assertEqual(result["keys"], 1)
            self.assertEqual(store.get("a"), (False, None))
            self.assertEqual(store.get("b"), (True, "3"))
        with Store(self.path).open() as store:
            self.assertEqual(store.scan(), [("b", "3")])

    def test_snapshot_is_a_stable_view(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            snapshot = store.snapshot()
            store.put("a", "2")
            store.put("b", "3")
            self.assertEqual(snapshot.get("a"), "1")
            self.assertIsNone(snapshot.get("b"))
            self.assertEqual(store.get("a"), (True, "2"))

    def test_verify_reports_clean_state(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            store.flush()
            report = store.verify()
            self.assertEqual(report["wal"]["truncatedBytes"], 0)
            self.assertEqual(len(report["tables"]), 1)

    def test_empty_keys_are_rejected(self) -> None:
        with Store(self.path).open() as store:
            with self.assertRaises(ValidationError):
                store.put("", "1")

    def test_corrupt_table_is_reported(self) -> None:
        with Store(self.path).open() as store:
            store.put("a", "1")
            path = store.flush()
        assert path is not None
        with open(path, "r+b") as handle:
            handle.seek(40)
            handle.write(b"Z")
        with self.assertRaises(CorruptionError):
            SSTable.open(path)


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_describe_lists_the_contract(self) -> None:
        code, out, err = run_cli(["describe"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(document["exitCodes"], {"ok": 0, "error": 2, "negativeVerdict": 3})
        self.assertEqual(document["layout"]["log"], "wal.log")

    def test_put_then_get_across_processes(self) -> None:
        self.assertEqual(run_cli(["put", "--dir", self.path, "--key", "a", "--value", "1"])[0], EXIT_OK)
        code, out, _ = run_cli(["get", "--dir", self.path, "--key", "a"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["value"], "1")

    def test_get_missing_key_is_a_negative_verdict(self) -> None:
        code, out, _ = run_cli(["get", "--dir", self.path, "--key", "nope"])
        self.assertEqual(code, EXIT_NEGATIVE)
        self.assertFalse(json.loads(out)["found"])

    def test_flush_without_writes_is_a_negative_verdict(self) -> None:
        code, out, _ = run_cli(["flush", "--dir", self.path])
        self.assertEqual(code, EXIT_NEGATIVE)
        self.assertFalse(json.loads(out)["flushed"])

    def test_scan_reports_rows_and_exit_code(self) -> None:
        run_cli(["put", "--dir", self.path, "--key", "a", "--value", "1"])
        code, out, _ = run_cli(["scan", "--dir", self.path])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["count"], 1)
        self.assertEqual(run_cli(["scan", "--dir", self.path, "--start", "b"])[0], EXIT_NEGATIVE)

    def test_invalid_key_is_an_error_document(self) -> None:
        code, out, err = run_cli(["put", "--dir", self.path, "--key", "", "--value", "1"])
        # stderr must be *only* the error document: a stray warning line ahead of it breaks every
        # caller that parses stderr as JSON (this assertion is what caught a leaked WAL handle).
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_verify_reports_a_repaired_log_as_a_negative_verdict(self) -> None:
        run_cli(["put", "--dir", self.path, "--key", "a", "--value", "1"])
        with open(os.path.join(self.path, "wal.log"), "a", encoding="utf-8", newline="\n") as handle:
            handle.write("deadbeef {\"op\":\"put\",\"key\":\"b\",\"seq\":9,\"va")
        code, out, _ = run_cli(["verify", "--dir", self.path])
        self.assertEqual(code, EXIT_NEGATIVE)
        self.assertGreater(json.loads(out)["wal"]["truncatedBytes"], 0)


if __name__ == "__main__":
    unittest.main()
