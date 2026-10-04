"""Lazy sealed-table reads and the incremental cross-layer merge.

These cover what changed from the resident-dictionary baseline:
  * opening a table keeps only bloom + count + a sparse block index, never keys or values;
  * a bloom-rejected get touches no entry bytes; a possible hit reads one bounded block;
  * scans merge ordered layers incrementally and stop reading once `limit` is satisfied;
  * tables written by the previous version (compact baseline header) open unchanged;
  * every structural failure surfaces as CorruptionError carrying the path -- never a raw
    UnicodeDecodeError/ValueError/KeyError -- and never as a miss;
  * large values no longer make point lookups or resident memory proportional to the payload.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
import zlib
from unittest import mock

from kvstore.errors import CorruptionError
from kvstore.sstable import BLOCK_ENTRIES, SSTable, SSTableWriter
from kvstore.store import Store


def legacy_table_bytes(items: list[tuple[str, str | None]]) -> bytes:
    """Reproduce the previous version's exact on-disk layout."""
    import hashlib

    def hashes(key: str, count: int, bits: int) -> list[int]:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return [int.from_bytes(digest[i * 4 : i * 4 + 4], "big") % bits for i in range(count)]

    keys = [key for key, _ in items]
    bits = max(64, 10 * max(1, len(keys)))
    payload = bytearray((bits + 7) // 8)
    for key in keys:
        for position in hashes(key, 4, bits):
            payload[position // 8] |= 1 << (position % 8)
    bloom = base64.b64encode(bytes(payload)).decode("ascii")
    body = [json.dumps({"count": len(items), "bloomBits": bits, "bloomHashes": 4}, separators=(",", ":")) + "\n"]
    for key, value in items:
        encoded = "~" if value is None else base64.b64encode(value.encode("utf-8")).decode("ascii")
        body.append(f"{key}\t{encoded}\n")
    prefix = "".join(body).encode("utf-8")
    footer = json.dumps({"bloom": bloom, "crc32": format(zlib.crc32(prefix) & 0xFFFFFFFF, "08x")}, separators=(",", ":"))
    return prefix + footer.encode("utf-8") + b"\n"


def seal(prefix: bytes, footer: dict[str, object]) -> bytes:
    """Rewrite a table's footer with a correct seal over a tampered prefix."""
    footer = dict(footer)
    footer["crc32"] = format(zlib.crc32(prefix) & 0xFFFFFFFF, "08x")
    return prefix + json.dumps(footer, separators=(",", ":")).encode("utf-8") + b"\n"


def split_table(raw: bytes) -> tuple[bytes, dict[str, object]]:
    assert raw.endswith(b"\n")
    footer_start = raw.rfind(b"\n", 0, len(raw) - 1)
    return raw[: footer_start + 1], json.loads(raw[footer_start + 1 : -1])


class SSTableLazyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "table-000001.sst")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write_many(self, count: int = 5 * BLOCK_ENTRIES + 7) -> SSTable:
        items = [(f"key-{index:05d}", f"value-{index:05d}") for index in range(count)]
        items[10] = ("key-00010", None)  # a tombstone among ordinary values
        return SSTable.write(self.path, items)

    def test_open_keeps_no_keys_or_values(self) -> None:
        table = self._write_many()
        self.assertFalse(hasattr(table, "values"))
        self.assertFalse(hasattr(table, "index"))
        self.assertEqual(table.count, 5 * BLOCK_ENTRIES + 7)
        # The sparse index holds one entry per block, not one per key.
        self.assertLess(len(table._blocks), table.count)
        self.assertGreaterEqual(len(table._blocks), table.count // BLOCK_ENTRIES)

    def test_get_and_scan_across_many_blocks_and_reopen(self) -> None:
        table = self._write_many()
        self.assertEqual(table.get("key-00000"), (True, "value-00000"))
        self.assertEqual(table.get("key-00010"), (True, None))  # tombstone, distinct from a miss
        self.assertEqual(table.get("key-00326"), (True, "value-00326"))
        self.assertEqual(table.get("key-00327"), (False, None))  # just past the end
        self.assertEqual(table.get("absent-zzz"), (False, None))
        reopened = SSTable.open(self.path)
        self.assertEqual(reopened.get("key-00200"), (True, "value-00200"))
        self.assertEqual(reopened.get("key-00010"), (True, None))
        rows = list(reopened.scan(start="key-00100", end="key-00103"))
        self.assertEqual(rows, [("key-00100", "value-00100"), ("key-00101", "value-00101"), ("key-00102", "value-00102")])
        self.assertEqual(len(list(reopened.scan(limit=10))), 10)

    def test_bloom_miss_reads_no_entry_bytes(self) -> None:
        table = self._write_many()
        # Pick a key the provably rejects on the bloom rather than gambling on a false positive.
        absent = next(f"absent-{i}" for i in range(10_000) if not table.bloom.might_contain(f"absent-{i}"))
        with mock.patch.object(SSTable, "_read_bytes", side_effect=AssertionError("entry region read on a bloom miss")):
            self.assertEqual(table.get(absent), (False, None))

    def test_possible_hit_reads_one_bounded_block(self) -> None:
        table = self._write_many()
        with mock.patch.object(SSTable, "_read_bytes", wraps=table._read_bytes) as spied:
            self.assertEqual(table.get("key-00200"), (True, "value-00200"))
        self.assertEqual(spied.call_count, 1)
        start, end = spied.call_args.args[0], spied.call_args.args[1]
        self.assertLessEqual(end - start, 32 << 10)  # only the block, never the whole table

    def test_limit_scan_reads_only_the_blocks_it_emits(self) -> None:
        table = self._write_many()
        with mock.patch.object(SSTable, "_parse_block", wraps=table._parse_block) as spied:
            rows = list(table.scan(limit=5))
        self.assertEqual([key for key, _ in rows], [f"key-{i:05d}" for i in range(5)])
        self.assertEqual(spied.call_count, 1)  # the first 64-entry block suffices

    def test_legacy_version_table_opens_without_migration(self) -> None:
        items = [("a", "1"), ("b", None), ("zé", "hé→")]
        with open(self.path, "wb") as handle:
            handle.write(legacy_table_bytes(items))
        table = SSTable.open(self.path)
        self.assertEqual(table.count, 3)
        self.assertEqual(table.get("a"), (True, "1"))
        self.assertEqual(table.get("b"), (True, None))
        self.assertEqual(table.get("missing"), (False, None))
        self.assertEqual([(k, v) for k, v in table.scan() if v is not None], [("a", "1"), ("zé", "hé→")])
        # It must also compact through the new write path: re-seal and read again.
        target = os.path.join(self.directory.name, "table-000002.sst")
        SSTable.write(target, [(k, v) for k, v in table.scan() if v is not None])
        self.assertEqual(SSTable.open(target).get("zé"), (True, "hé→"))

    def test_new_header_is_byte_compact_when_count_is_known(self) -> None:
        SSTable.write(self.path, [("a", "1"), ("b", "2")])
        with open(self.path, "rb") as handle:
            first = handle.readline()
        self.assertEqual(first, b'{"count":2,"bloomBits":64,"bloomHashes":4}\n')

    def test_every_structural_failure_is_corruption_with_path(self) -> None:
        table = self._write_many()
        with open(table.path, "rb") as handle:
            raw = handle.read()
        prefix, footer = split_table(raw)
        lines = prefix.splitlines()

        def expect_corrupt(rewritten: bytes) -> None:
            with open(self.path, "wb") as handle:
                handle.write(rewritten)
            with self.assertRaises(CorruptionError) as caught:
                SSTable.open(self.path)
            self.assertEqual(caught.exception.context.get("path"), self.path)

        # 1. value column that is not base64 (seal recomputed so structure itself is checked)
        lines_bad_b64 = list(lines)
        lines_bad_b64[1] = b"key-00000\t@@@@not-base64"
        expect_corrupt(seal(b"\n".join(lines_bad_b64) + b"\n", footer))

        # 2. legal base64 whose bytes are not UTF-8
        lines_bad_utf8 = list(lines)
        lines_bad_utf8[1] = b"key-00000\t" + base64.b64encode(b"\xff\xfe\xfd")
        expect_corrupt(seal(b"\n".join(lines_bad_utf8) + b"\n", footer))

        # 3. duplicate / non-ascending key
        lines_dup = list(lines)
        lines_dup[2] = lines_dup[1]
        expect_corrupt(seal(b"\n".join(lines_dup) + b"\n", footer))

        # 4. header count that lies about the entries
        header = json.loads(lines[0])
        header["count"] = len(lines) - 2
        lines_count = [json.dumps(header, separators=(",", ":")).encode()] + lines[1:]
        expect_corrupt(seal(b"\n".join(lines_count) + b"\n", footer))

        # 5. truncated body and missing trailing newline
        expect_corrupt(raw[: len(raw) // 2])
        expect_corrupt(raw[:-1])

        # 6. garbage footer
        expect_corrupt(prefix + b"not-json\n")

    def test_corruption_is_never_a_miss(self) -> None:
        self._write_many()
        with open(self.path, "r+b") as handle:
            handle.seek(40)
            handle.write(b"Z")
        with self.assertRaises(CorruptionError):
            SSTable.open(self.path).get("key-00000")


class StoreMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.directory.name

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_scan_merges_layers_without_materializing_tables(self) -> None:
        with Store(self.path).open() as store:
            for i in range(3 * BLOCK_ENTRIES):
                store.put(f"key-{i:05d}", f"old-{i:05d}")
            store.flush()
            store.put("key-00010", "new-10")  # newer layer overrides the sealed value
            store.delete("key-00011")  # tombstone in the memtable hides the sealed value
            rows = store.scan()
        keys = [key for key, _ in rows]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(rows), 3 * BLOCK_ENTRIES - 1)
        self.assertEqual(dict(rows)["key-00010"], "new-10")
        self.assertNotIn("key-00011", dict(rows))
        self.assertEqual(store.get("key-00011"), (True, None))
        self.assertEqual(store.get("key-09999"), (False, None))

    def test_limit_stops_reading_later_blocks_and_tables(self) -> None:
        with Store(self.path).open() as store:
            for i in range(2 * BLOCK_ENTRIES):
                store.put(f"a-{i:04d}", "v")
            store.flush()
            for i in range(2 * BLOCK_ENTRIES):
                store.put(f"z-{i:04d}", "v")
            store.flush()
            tables = list(store.tables)
            parse = SSTable._parse_block
            with mock.patch.object(SSTable, "_parse_block", autospec=True, side_effect=parse) as spied:
                rows = store.scan(limit=3)
            self.assertEqual([k for k, _ in rows], ["a-0000", "a-0001", "a-0002"])
            # Oldest blocks at the back of the range must never have been touched.
            self.assertLess(spied.call_count, sum(len(t._blocks) for t in tables))
            self.assertGreaterEqual(spied.call_count, 1)

    def test_unicode_ordering_matches_dictionary_order(self) -> None:
        with Store(self.path).open() as store:
            store.put("é", "1")
            store.flush()
            store.put("a", "2")
            store.put("字", "3")
            self.assertEqual([k for k, _ in store.scan()], ["a", "é", "字"])
            self.assertEqual([k for k, _ in store.scan(start="é", end="字")], ["é"])

    def test_compact_large_values_stays_lazy_and_round_trips(self) -> None:
        big = "x" * (200 << 10)
        with Store(self.path).open() as store:
            for i in range(40):
                store.put(f"k-{i:03d}", big if i % 2 == 0 else f"small-{i}")
                if store.memtable.is_full():
                    store.flush()
            store.delete("k-002")
            store.put("k-000", "winner")
            tables_before = len(store.tables)
            result = store.compact()
            self.assertGreaterEqual(tables_before, 2)
            self.assertEqual(result["keys"], 39)
            self.assertEqual(result["removedTables"], tables_before)
            self.assertEqual(store.get("k-000"), (True, "winner"))
            self.assertEqual(store.get("k-002"), (False, None))
            self.assertEqual(store.get("k-004"), (True, big))
        # Resident metadata must stay tiny against an 8 MiB payload, and a lookup must not pull
        # neighbouring large values off disk.
        with Store(self.path).open() as store:
            self.assertEqual(len(store.tables), 1)
            table = store.tables[0]
            resident = sum(len(block.first_key.encode()) for block in table._blocks) + len(table.bloom.payload)
            self.assertLess(resident, 128 << 10)
            with mock.patch.object(SSTable, "_read_bytes", wraps=table._read_bytes) as spied:
                self.assertEqual(store.get("k-038"), (True, big))
            start, end = spied.call_args.args[0], spied.call_args.args[1]
            self.assertLess(end - start, len(big) * 2 + 4096)
        self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(self.path)))

    def test_compact_then_verify_and_stats_agree(self) -> None:
        with Store(self.path).open() as store:
            for i in range(BLOCK_ENTRIES + 5):
                store.put(f"s-{i:03d}", str(i))
            store.flush()
            store.delete("s-000")
            store.flush()
            stats_before = store.stats()
            self.assertEqual(stats_before.sealed_entries, BLOCK_ENTRIES + 6)
            store.compact()
            stats = store.stats()
            self.assertEqual(stats.tables, 1)
            self.assertEqual(stats.sealed_entries, BLOCK_ENTRIES + 4)
            report = store.verify()
            self.assertEqual(len(report["tables"]), 1)
            self.assertEqual(report["tables"][0]["keys"], BLOCK_ENTRIES + 4)
            self.assertEqual(report["wal"]["truncatedBytes"], 0)
        with Store(self.path).open() as store:
            self.assertEqual(len(store.scan()), BLOCK_ENTRIES + 4)
            self.assertEqual(store.get("s-000"), (False, None))

    def test_snapshot_is_stable_after_lazy_flush_and_compact(self) -> None:
        with Store(self.path).open() as store:
            for i in range(BLOCK_ENTRIES + 10):
                store.put(f"snap-{i:03d}", "v")
            store.flush()
            snapshot = store.snapshot()
            store.put("snap-000", "changed")
            store.delete("snap-001")
            store.compact()
            self.assertEqual(snapshot.get("snap-000"), "v")
            self.assertEqual(snapshot.get("snap-001"), "v")
            self.assertIsNone(snapshot.get("nope"))
            self.assertEqual(len(snapshot.items()), BLOCK_ENTRIES + 10)

    def test_streaming_writer_rejects_overflowing_count(self) -> None:
        target = os.path.join(self.path, "table-000009.sst")
        writer = SSTableWriter(target, 1)
        writer.add("a", "1")
        writer.add("b", "2")
        with self.assertRaises(CorruptionError):
            writer.close(2)
        writer.abort()
        if os.path.exists(target):
            os.unlink(target)


if __name__ == "__main__":
    unittest.main()
