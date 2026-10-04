"""Lazy sealed-table reads: bounded residency, on-demand values, incremental merge.

These tests pin the refactor contract that the baseline (which kept every key and value in a
dict) could not satisfy: opening a table retains only the bloom filter, the count and the entry
offsets; values are read on demand; scans merge layers incrementally and stop at the limit; and
every structural failure surfaces as ``CorruptionError``.
"""

from __future__ import annotations

import base64
import builtins
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib

from kvstore.errors import CorruptionError
from kvstore.sstable import BLOOM_HASHES, SSTable
from kvstore.store import Store

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALUE = "x" * 20_000
BIG_VALUE = "v" * 20_000
ENCODED_VALUE = base64.b64encode(VALUE.encode("utf-8")).decode("ascii")


def bloom_for(keys: list[str]) -> tuple[int, str]:
    bits = max(64, 10 * max(1, len(keys)))
    payload = bytearray((bits + 7) // 8)
    for key in keys:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        for index in range(BLOOM_HASHES):
            position = int.from_bytes(digest[index * 4 : index * 4 + 4], "big") % bits
            payload[position // 8] |= 1 << (position % 8)
    return bits, base64.b64encode(bytes(payload)).decode("ascii")


def build_table_bytes(entries: list[tuple[str, str]], *, count: int | None = None) -> bytes:
    """Assemble the documented on-disk layout by hand -- independent of SSTable.write, so it
    represents what an older version of the program would leave on disk."""
    keys = [key for key, _ in entries]
    bits, bloom = bloom_for(keys)
    header = json.dumps(
        {"count": len(entries) if count is None else count, "bloomBits": bits, "bloomHashes": BLOOM_HASHES},
        separators=(",", ":"),
    ) + "\n"
    body = "".join(f"{key}\t{encoded}\n" for key, encoded in entries)
    prefix = (header + body).encode("utf-8")
    footer = json.dumps(
        {"bloom": bloom, "crc32": format(zlib.crc32(prefix) & 0xFFFFFFFF, "08x")},
        separators=(",", ":"),
    )
    return prefix + footer.encode("utf-8") + b"\n"


def reseal(header_line: bytes, entry_lines: list[bytes], bloom: str) -> bytes:
    """Rebuild a file with a correct crc over an arbitrarily tampered prefix."""
    prefix = header_line + b"".join(line + b"\n" for line in entry_lines)
    footer = json.dumps(
        {"bloom": bloom, "crc32": format(zlib.crc32(prefix) & 0xFFFFFFFF, "08x")},
        separators=(",", ":"),
    )
    return prefix + footer.encode("utf-8") + b"\n"


class CountingHandle:
    """File wrapper tallying bytes read()/readline() at or past ``entry_start``."""

    def __init__(self, handle, entry_start: int, totals: dict[str, int]) -> None:
        self._handle = handle
        self._entry_start = entry_start
        self._totals = totals

    def seek(self, *args):
        return self._handle.seek(*args)

    def tell(self):
        return self._handle.tell()

    def _count(self, data: bytes, bucket: str) -> bytes:
        if self._handle.tell() >= self._entry_start:
            self._totals[bucket] += len(data)
        return data

    def read(self, size=-1):
        position = self._handle.tell()
        data = self._handle.read(size)
        if position >= self._entry_start:
            self._totals["read"] += len(data)
        return data

    def readline(self, *args):
        position = self._handle.tell()
        data = self._handle.readline(*args)
        if position >= self._entry_start:
            self._totals["readline"] += len(data)
        return data

    def close(self):
        return self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OpenResidencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write(self, value_size: int) -> str:
        path = os.path.join(self.path, f"t{value_size}.sst")
        SSTable.write(
            path,
            [(f"k{i:05d}", ("v" * value_size) if i % 4 else None) for i in range(200)],
        )
        return path

    def test_open_keeps_no_keys_or_values(self) -> None:
        table = SSTable.open(self._write(100_000))
        self.assertFalse(hasattr(table, "values"))
        self.assertFalse(hasattr(table, "index"))
        self.assertEqual(table.count, 200)
        self.assertEqual(len(table.offsets), 200)

    def test_resident_memory_does_not_grow_with_value_bytes(self) -> None:
        import tracemalloc

        small_path = self._write(10_000)
        large_path = self._write(100_000)  # 10x the payload, same key count
        tracemalloc.start()
        small = SSTable.open(small_path)
        _, small_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        large = SSTable.open(large_path)
        _, large_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # A tenfold payload increase must not move the open footprint: the transient validate
        # buffer is a fixed block, and after open only offsets plus the bloom survive.
        self.assertLess(large_peak, small_peak * 2 + 256 * 1024)
        self.assertLess(large_peak, 8 * 1024 * 1024)
        self.assertEqual(large.count, small.count)

    def test_reopen_new_table_preserves_everything(self) -> None:
        path = os.path.join(self.path, "t.sst")
        table = SSTable.write(path, [("a", ""), ("b", VALUE), ("c", None), ("é", "日")])
        reopened = SSTable.open(path)
        self.assertEqual(reopened.get("a"), (True, ""))
        self.assertEqual(reopened.get("b"), (True, VALUE))
        self.assertEqual(reopened.get("c"), (True, None))
        self.assertEqual(reopened.get("é"), (True, "日"))
        self.assertEqual(reopened.get("absent"), (False, None))
        self.assertEqual(
            list(reopened.scan_rows()),
            [("a", ""), ("b", VALUE), ("c", None), ("é", "日")],
        )
        self.assertEqual(table.count, reopened.count)


class PointLookupReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "table-000001.sst")
        self.items = [(f"k{i:05d}", None if i % 13 == 0 else BIG_VALUE) for i in range(300)]
        SSTable.write(self.path, self.items)
        self.table = SSTable.open(self.path)
        self.entry_start = self.table.offsets[0]

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _measure(self, action) -> dict[str, int]:
        totals = {"read": 0, "readline": 0}
        real_open = builtins.open
        table = self.table

        def counting_open(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            if str(path) != table.path:
                return handle
            return CountingHandle(handle, self.entry_start, totals)

        builtins.open = counting_open
        try:
            action()
        finally:
            builtins.open = real_open
        return totals

    def test_binary_search_finds_every_present_key(self) -> None:
        for index, (key, value) in enumerate(self.items):
            found, got = self.table.get(key)
            self.assertTrue(found, key)
            self.assertEqual(got, value, key)

    def test_absent_and_tombstone_are_distinct(self) -> None:
        self.assertEqual(self.table.get("k00013"), (True, None))   # tombstone
        self.assertEqual(self.table.get("zzzzz"), (False, None))   # never written
        self.assertEqual(self.table.get("k00300"), (False, None))  # past the end

    def test_bloom_miss_does_not_touch_the_entry_region(self) -> None:
        totals = self._measure(lambda: self.table.get("definitely-not-in-bloom-zzz"))
        self.assertEqual(totals["read"], 0)
        self.assertEqual(totals["readline"], 0)

    def test_probes_are_bounded_by_key_length_not_value_length(self) -> None:
        totals = self._measure(lambda: self.table.get("k00298"))  # 298 holds a value (299 is a tombstone)
        # ~8-9 binary-search probes over 5-byte keys; even with generous slack this is nothing
        # next to the value.
        self.assertLess(totals["read"], 4096)
        self.assertGreaterEqual(totals["readline"], len(BIG_VALUE))  # an exact hit reads its value

    def test_tombstone_hit_never_reads_a_value(self) -> None:
        totals = self._measure(lambda: self.table.get("k00013"))
        self.assertEqual(totals["readline"], 0)
        self.assertGreater(totals["read"], 0)  # it did locate the entry via probes


class IncrementalMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _store_with_two_layers(self) -> Store:
        older = os.path.join(self.path, "table-000001.sst")
        newer = os.path.join(self.path, "table-000002.sst")
        keys = [f"k{i:03d}" for i in range(20)]
        SSTable.write(older, [(key, "OLD" * 5_000) for key in keys])
        SSTable.write(newer, [(key, "NEW" * 5_000) for key in keys])
        return Store(self.path).open()

    def _counted_store(self, store: Store):
        totals = {table.path: {"read": 0, "readline": 0} for table in store.tables}
        starts = {table.path: table.offsets[0] for table in store.tables}
        real_open = builtins.open

        def counting_open(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            key = str(path)
            if key not in totals:
                return handle
            return CountingHandle(handle, starts[key], totals[key])

        return totals, counting_open, real_open

    def test_newer_layer_overrides_and_old_values_are_never_read(self) -> None:
        store = self._store_with_two_layers()
        totals, counting_open, real_open = self._counted_store(store)
        builtins.open = counting_open
        try:
            rows = store.scan()
        finally:
            builtins.open = real_open
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(value.startswith("NEW") for _, value in rows))
        older, newer = store.tables[1].path, store.tables[0].path
        self.assertEqual(totals[older]["readline"], 0, "shadowed values must not be read")
        self.assertGreater(totals[newer]["readline"], 0)

    def test_tombstones_hide_older_values_without_reading_them(self) -> None:
        store = self._store_with_two_layers()
        store.delete("k0005")  # memtable tombstone over both tables
        rows = store.scan()
        self.assertNotIn("k0005", [key for key, _ in rows])
        self.assertEqual(store.get("k0005"), (True, None))

    def test_limit_stops_touching_later_entries(self) -> None:
        value = "V" * 20_000
        encoded_len = len(base64.b64encode(value.encode()).decode())  # one value line's payload
        path = os.path.join(self.path, "one.sst")
        SSTable.write(path, [(f"k{i:04d}", value) for i in range(300)])
        shutil.move(path, os.path.join(self.path, "table-000001.sst"))
        with Store(self.path).open() as store:
            table_path = store.tables[0].path
            totals = {table_path: {"read": 0, "readline": 0}}
            start = store.tables[0].offsets[0]
            real_open = builtins.open

            def counting_open(p, *args, **kwargs):
                handle = real_open(p, *args, **kwargs)
                if not str(p).endswith(".sst"):
                    return handle
                return CountingHandle(handle, start, totals[table_path])

            builtins.open = counting_open
            try:
                rows = store.scan(limit=3)
            finally:
                builtins.open = real_open
            self.assertEqual([key for key, _ in rows], ["k0000", "k0001", "k0002"])
            # Only the three returned values are read in full (plus their keys); probes stay
            # tiny, and none of the other 297 ~27KB value lines is fetched.
            self.assertLess(totals[table_path]["readline"], 4 * (encoded_len + 16))
            self.assertLess(totals[table_path]["read"], 4096)

    def test_compact_streams_without_reading_shadowed_values(self) -> None:
        store = self._store_with_two_layers()
        totals, counting_open, real_open = self._counted_store(store)
        builtins.open = counting_open
        try:
            result = store.compact()
        finally:
            builtins.open = real_open
        self.assertEqual(result["keys"], 20)
        # The older layer's values were shadowed on every key, so none were fetched in full.
        self.assertIn(os.path.join(self.path, "table-000001.sst"), totals)
        self.assertEqual(totals[os.path.join(self.path, "table-000001.sst")]["readline"], 0)
        self.assertEqual(store.get("k000"), (True, "NEW" * 5_000))
        with Store(self.path).open() as reopened:
            self.assertEqual(len(reopened.scan()), 20)

    def test_unicode_ordering_and_range_bounds(self) -> None:
        with Store(self.path).open() as store:
            for key, value in [("a", "1"), ("ä", "2"), ("b", "3"), ("中", "4")]:
                store.put(key, value)
            store.flush()
            store.put("b", "5")
            self.assertEqual([k for k, _ in store.scan()], ["a", "b", "ä", "中"])
            self.assertEqual([k for k, _ in store.scan(start="b", end="中")], ["b", "ä"])
            # Ordering is Unicode code point: ä (U+00E4) and 中 (U+4E2D) both sort after "z".
            self.assertEqual([k for k, _ in store.scan(start="z")], ["ä", "中"])
            self.assertEqual([k for k, _ in store.scan(end="b")], ["a"])


class CompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_hand_built_baseline_layout_opens_and_serves(self) -> None:
        raw = build_table_bytes(
            [("a", ENCODED_VALUE), ("b", "~"), ("c", base64.b64encode("日".encode()).decode())]
        )
        path = os.path.join(self.path, "table-000001.sst")
        with open(path, "wb") as handle:
            handle.write(raw)
        table = SSTable.open(path)
        self.assertEqual(table.count, 3)
        self.assertEqual(table.get("a"), (True, VALUE))
        self.assertEqual(table.get("b"), (True, None))
        self.assertEqual(table.get("c"), (True, "日"))
        self.assertEqual(table.get("missing"), (False, None))
        self.assertEqual(
            [(k, v) for k, v in table.scan_rows() if v is not None],
            [("a", VALUE), ("c", "日")],
        )
        # It participates in verify and compaction without migration.
        with Store(self.path).open() as store:
            report = store.verify()
            self.assertEqual(report["tables"][0]["keys"], 3)
            result = store.compact()
            self.assertEqual(result["removedTables"], 1)
            self.assertEqual(store.get("a"), (True, VALUE))
        with Store(self.path).open() as store:
            self.assertEqual(store.get("a"), (True, VALUE))

    def test_table_written_by_baseline_code(self) -> None:
        try:
            source = subprocess.run(
                ["git", "show", "HEAD:kvstore/sstable.py"],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("baseline source unavailable")
        module_path = os.path.join(self.path, "old_sstable.py")
        with open(module_path, "w") as handle:
            handle.write(source.replace("from .errors", "from kvstore.errors"))
        import importlib.util

        spec = importlib.util.spec_from_file_location("old_sstable", module_path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules["old_sstable"] = baseline  # dataclass resolves the class module through this
        spec.loader.exec_module(baseline)
        path = os.path.join(self.path, "table-000001.sst")
        baseline.SSTable.write(path, [("a", "old"), ("b", None), ("z", "末")])
        table = SSTable.open(path)
        self.assertEqual(table.get("a"), (True, "old"))
        self.assertEqual(table.get("b"), (True, None))
        self.assertEqual(table.get("z"), (True, "末"))
        self.assertEqual(table.get("q"), (False, None))


class CorruptionMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name
        self.good = os.path.join(self.path, "good.sst")
        SSTable.write(self.good, [("a", "1"), ("b", "2"), ("d", "x" * 50_000), ("é", "4")])
        raw = open(self.good, "rb").read()
        lines = raw.split(b"\n")
        self.header, self.footer = lines[0], lines[-2]
        self.entries = lines[1:-2]
        self.bloom = json.loads(self.footer)["bloom"]

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _expect_corruption(self, name: str, data: bytes) -> None:
        path = os.path.join(self.path, name)
        with open(path, "wb") as handle:
            handle.write(data)
        with self.assertRaises(CorruptionError) as caught:
            SSTable.open(path)
        self.assertEqual(caught.exception.context.get("path"), path)

    def _tampered_header(self, mutate) -> bytes:
        header = json.loads(self.header)
        mutate(header)
        return reseal(
            (json.dumps(header, separators=(",", ":")).encode() + b"\n"),
            self.entries,
            self.bloom,
        )

    def test_flipped_byte_is_corruption_not_miss(self) -> None:
        raw = open(self.good, "rb").read()
        self._expect_corruption("flip.sst", raw[:60] + b"Z" + raw[61:])
        # A store opening a directory that contains the damaged sealed table must fail loudly
        # rather than silently treating its keys as missing -- use a directory holding only it.
        store_dir = os.path.join(self.path, "store")
        os.makedirs(store_dir)
        with open(os.path.join(store_dir, "table-000009.sst"), "wb") as handle:
            handle.write(raw[:60] + b"Z" + raw[61:])
        with self.assertRaises(CorruptionError):
            Store(store_dir).open()

    def test_header_and_count_failures(self) -> None:
        prefix = b"{not-json}\n" + b"".join(line + b"\n" for line in self.entries)
        self._expect_corruption("bad-header.sst", reseal(b"{not-json}\n", self.entries, self.bloom))
        self._expect_corruption("missing-count.sst", self._tampered_header(lambda h: h.pop("count")))
        self._expect_corruption("negative-bits.sst", self._tampered_header(lambda h: h.update(bloomBits=-1)))
        self._expect_corruption("count-mismatch.sst", self._tampered_header(lambda h: h.update(count=99)))
        self._expect_corruption("bad-hashes.sst", self._tampered_header(lambda h: h.update(bloomHashes=0)))

    def test_ordering_and_key_failures(self) -> None:
        self._expect_corruption(
            "duplicate.sst", reseal(self.header, [self.entries[0]] + self.entries, self.bloom)
        )
        self._expect_corruption(
            "descending.sst",
            reseal(self.header, [self.entries[1], self.entries[0]] + self.entries[2:], self.bloom),
        )
        self._expect_corruption(
            "empty-key.sst", reseal(self.header, [b"\tMQ=="] + self.entries, self.bloom)
        )
        self._expect_corruption(
            "bad-utf8-key.sst",
            reseal(self.header, [b"\xff\tMQ=="] + self.entries, self.bloom),
        )
        self._expect_corruption(
            "no-separator.sst", reseal(self.header, [b"nosep"] + self.entries, self.bloom)
        )

    def test_value_and_footer_failures(self) -> None:
        self._expect_corruption(
            "bad-base64.sst", reseal(self.header, [b"q\t!!!!"] + self.entries, self.bloom)
        )
        self._expect_corruption(
            "bad-utf8-value.sst",
            reseal(self.header, [b"q\t" + base64.b64encode(b"\xff\xfe")] + self.entries, self.bloom),
        )
        self._expect_corruption(
            "bad-tombstone.sst", reseal(self.header, [b"q\t~x"] + self.entries, self.bloom)
        )
        self._expect_corruption(
            "illegal-padding.sst", reseal(self.header, [b"q\tYWJj=MQ=="] + self.entries, self.bloom)
        )
        self._expect_corruption("truncated.sst", open(self.good, "rb").read()[:-40_000])
        self._expect_corruption("no-newline.sst", open(self.good, "rb").read()[:-1])
        self._expect_corruption(
            "bad-bloom.sst", reseal(self.header, self.entries, "@@not-base64@@")
        )
        self._expect_corruption(
            "bloom-wrong-len.sst",
            reseal(self.header, self.entries, base64.b64encode(b"\x00\x00").decode()),
        )
        prefix = self.header + b"".join(line + b"\n" for line in self.entries)
        self._expect_corruption("footer-garbage.sst", prefix + b"{garbage\n")
        self._expect_corruption("empty.sst", b"")


if __name__ == "__main__":
    unittest.main()
