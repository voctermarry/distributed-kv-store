"""Sorted table on disk: one header, one line per key in ascending order, one footer.

Layout (UTF-8, LF):

    header  {"count":N,"bloomBits":M,"bloomHashes":K}
    entry   <key><TAB><base64 value>      (a tombstone stores the marker `~`)
    footer  {"bloom":"<base64 bits>","crc32":"<8 hex of all preceding bytes>"}

Opening a table reads and validates *every byte once* (the crc32 seal plus a full structural
check) but retains only the bloom filter, the entry count and a sparse block index -- the key
and byte offset of the first entry in each fixed-size block of entry lines. Neither keys nor
values stay resident, so an open table costs O(number of blocks + bloom bits) memory regardless
of the value payload on disk.

A point lookup that the bloom rejects touches no entry bytes at all; a possible hit reads just
the one block that could hold the key (BLOCK_ENTRIES lines), binary-searches its keys and decodes
only the matched value. Range scans walk blocks lazily in order; nothing materializes the table.
"""

from __future__ import annotations

import base64
import bisect
import hashlib
import json
import zlib
from dataclasses import dataclass
from typing import Iterator

from .errors import CorruptionError, ValidationError

TOMBSTONE = "~"
BLOOM_HASHES = 4
# How many entry lines make one index block. Smaller blocks mean a denser index (more resident
# memory) but fewer bytes read per lookup; 64 keeps a point lookup to a few KiB while the index
# stays well under 2% of the key count.
BLOCK_ENTRIES = 64
# A block also closes once it carries this many entry bytes, so a table whose values are large
# keeps each lookup's read to roughly one value instead of sixty-four.
BLOCK_BYTES = 32 << 10
# CRC the sealed prefix back in bounded chunks after the streaming count patch.
_CRC_CHUNK = 1 << 20


def _hashes(key: str, count: int, bits: int) -> list[int]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return [int.from_bytes(digest[index * 4 : index * 4 + 4], "big") % bits for index in range(count)]


@dataclass(slots=True)
class BloomFilter:
    bits: int
    hashes: int
    payload: bytearray

    @classmethod
    def build(cls, keys: list[str], bits_per_key: int = 10) -> "BloomFilter":
        bloom = cls.build_streaming(len(keys), bits_per_key)
        for key in keys:
            bloom.add(key)
        return bloom

    @classmethod
    def build_streaming(cls, count: int, bits_per_key: int = 10) -> "BloomFilter":
        """An empty filter sized for up to `count` keys; callers add keys with `add`."""
        bits = max(64, bits_per_key * max(1, count))
        return cls(bits=bits, hashes=BLOOM_HASHES, payload=bytearray((bits + 7) // 8))

    def add(self, key: str) -> None:
        for position in _hashes(key, self.hashes, self.bits):
            self.payload[position // 8] |= 1 << (position % 8)

    @classmethod
    def loads(cls, bits: int, hashes: int, encoded: str) -> "BloomFilter":
        return cls(bits=bits, hashes=hashes, payload=bytearray(base64.b64decode(encoded)))

    def dumps(self) -> str:
        return base64.b64encode(bytes(self.payload)).decode("ascii")

    def might_contain(self, key: str) -> bool:
        return all(self.payload[position // 8] & (1 << (position % 8)) for position in _hashes(key, self.hashes, self.bits))

    def to_document(self) -> dict[str, int]:
        return {"bits": self.bits, "hashes": self.hashes, "bytes": len(self.payload)}


def _decode_value(encoded: str, path: str) -> str | None:
    """Decode one base64 value column into the stored string (None marks a tombstone)."""
    if encoded == TOMBSTONE:
        return None
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise CorruptionError("entry value must be base64-encoded UTF-8", path=path) from error


@dataclass(slots=True)
class _BlockRef:
    """First key of a block and the byte range [start, end) of its lines within the file."""

    first_key: str
    start: int
    end: int


class SSTableWriter:
    """Stream sorted (key, value) pairs straight into a sealed table.

    Used by flush and compaction so the output never grows resident memory either. The entry
    count is only known once a merge finishes, so the header reserves a fixed-width numeric
    field (JSON whitespace after the colon is legal) and patches it in place before the footer --
    and its crc32 seal -- are written.
    """

    def __init__(self, path: str, count_estimate: int, *, bits_per_key: int = 10, exact: bool = False) -> None:
        self.path = path
        self.count_estimate = max(0, count_estimate)
        # `exact` knows the final entry count up front (flush), so the header stays byte-for-byte
        # the compact baseline shape; streaming callers (compaction) reserve a padded field.
        self._exact = exact
        self.bloom = BloomFilter.build_streaming(self.count_estimate, bits_per_key)
        self._width = max(1, len(str(self.count_estimate)))
        self._handle = open(path, "w+b")
        prefix = b'{"count":' + b" " * self._width + (
            f',"bloomBits":{self.bloom.bits},"bloomHashes":{self.bloom.hashes}}}\n'
        ).encode("ascii")
        self._handle.write(prefix)
        self._count_offset = len(b'{"count":')
        self._prefix_end = len(prefix)

    def add(self, key: str, value: str | None) -> None:
        self.bloom.add(key)
        encoded = TOMBSTONE if value is None else base64.b64encode(value.encode("utf-8")).decode("ascii")
        line = f"{key}\t{encoded}\n".encode("utf-8")
        self._handle.write(line)
        self._prefix_end += len(line)

    def close(self, count: int) -> "SSTable":
        if count < 0 or count > self.count_estimate or (self._exact and count != self.count_estimate):
            raise CorruptionError(
                "streaming table count disagrees with its reserved header field",
                path=self.path,
                count=count,
                capacity=self.count_estimate,
            )
        field = str(count).rjust(self._width)
        self._handle.seek(self._count_offset)
        self._handle.write(field.encode("ascii"))
        self._handle.flush()
        # Re-checksum the sealed prefix in bounded chunks after the in-place patch.
        self._handle.seek(0)
        checksum = 0
        remaining = self._prefix_end
        while remaining:
            chunk = self._handle.read(min(_CRC_CHUNK, remaining))
            if not chunk:
                raise CorruptionError("table body ended before its declared length", path=self.path)
            checksum = zlib.crc32(chunk, checksum)
            remaining -= len(chunk)
        footer = json.dumps(
            {"bloom": self.bloom.dumps(), "crc32": format(checksum & 0xFFFFFFFF, "08x")},
            separators=(",", ":"),
        )
        self._handle.seek(self._prefix_end)
        self._handle.write(footer.encode("utf-8"))
        self._handle.write(b"\n")
        self._handle.close()
        return SSTable.open(self.path)

    def abort(self) -> None:
        """Close the output after a failed stream without writing a seal."""
        if not self._handle.closed:
            self._handle.close()


class SSTable:
    """A sealed table: bloom filter, entry count and a sparse block index -- nothing else."""

    __slots__ = ("path", "bloom", "count", "_blocks", "_first_keys", "_min_key", "_max_key")

    def __init__(
        self,
        *,
        path: str,
        bloom: BloomFilter,
        count: int,
        blocks: list[_BlockRef],
        min_key: str | None,
        max_key: str | None,
    ) -> None:
        self.path = path
        self.bloom = bloom
        self.count = count
        self._blocks = blocks
        self._first_keys = tuple(block.first_key for block in blocks)
        self._min_key = min_key
        self._max_key = max_key

    @classmethod
    def write(cls, path: str, items: list[tuple[str, str | None]], *, bits_per_key: int = 10) -> "SSTable":
        keys = [key for key, _ in items]
        if keys != sorted(keys):
            raise ValidationError("sstable entries must be written in ascending key order")
        if len(set(keys)) != len(keys):
            raise ValidationError("sstable keys must be unique")
        writer = SSTableWriter(path, len(items), bits_per_key=bits_per_key, exact=True)
        try:
            for key, value in items:
                writer.add(key, value)
            return writer.close(len(items))
        except BaseException:
            writer.abort()
            raise

    @classmethod
    def open(cls, path: str) -> "SSTable":
        try:
            return cls._open(path)
        except CorruptionError:
            raise
        except (OSError, UnicodeError, ValueError, KeyError, IndexError) as error:
            # The on-disk contract only ever surfaces CorruptionError carrying the table path: a
            # torn byte must never leak as UnicodeDecodeError/KeyError/ValueError, nor read as a
            # miss (json.JSONDecodeError and binascii.Error are both ValueErrors).
            raise CorruptionError(f"unreadable table: {type(error).__name__}: {error}", path=path) from error

    @classmethod
    def _open(cls, path: str) -> "SSTable":
        with open(path, "rb") as handle:
            raw = handle.read()
        if not raw.endswith(b"\n"):
            raise CorruptionError("table is missing its trailing newline", path=path)
        footer_start = raw.rfind(b"\n", 0, len(raw) - 1)
        if footer_start <= 0:
            raise CorruptionError("table is too short to hold a footer", path=path)
        prefix = raw[: footer_start + 1]
        try:
            footer = json.loads(raw[footer_start + 1 : -1].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CorruptionError(f"unreadable footer: {error}", path=path) from error
        if not isinstance(footer, dict):
            raise CorruptionError("footer must be a JSON object", path=path)
        expected_crc = format(zlib.crc32(prefix) & 0xFFFFFFFF, "08x")
        if footer.get("crc32") != expected_crc:
            raise CorruptionError("table checksum mismatch", path=path)
        bloom_field = footer.get("bloom")
        if not isinstance(bloom_field, str):
            raise CorruptionError("footer bloom must be a base64 string", path=path)
        try:
            bloom_bytes = base64.b64decode(bloom_field, validate=True)
        except ValueError as error:
            raise CorruptionError("footer bloom is not valid base64", path=path) from error

        lines = prefix.splitlines()
        if not lines:
            raise CorruptionError("table is missing its header", path=path)
        try:
            header = json.loads(lines[0].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CorruptionError(f"unreadable header: {error}", path=path) from error
        if not isinstance(header, dict):
            raise CorruptionError("header must be a JSON object", path=path)
        count = header.get("count")
        bits = header.get("bloomBits")
        hashes = header.get("bloomHashes")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise CorruptionError("header count must be a non-negative integer", path=path)
        if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
            raise CorruptionError("header bloomBits must be a positive integer", path=path)
        if not isinstance(hashes, int) or isinstance(hashes, bool) or hashes <= 0:
            raise CorruptionError("header bloomHashes must be a positive integer", path=path)
        if len(bloom_bytes) != (bits + 7) // 8:
            raise CorruptionError("bloom payload length disagrees with bloomBits", path=path)
        bloom = BloomFilter(bits=bits, hashes=hashes, payload=bytearray(bloom_bytes))

        entry_lines = lines[1:]
        if len(entry_lines) != count:
            raise CorruptionError(
                "entry count differs from the header",
                path=path,
                expected=count,
                found=len(entry_lines),
            )

        blocks: list[_BlockRef] = []
        previous_key: str | None = None
        min_key: str | None = None
        max_key: str | None = None
        offset = len(lines[0]) + 1  # header bytes plus its LF
        block_due = True
        for position, line in enumerate(entry_lines):
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise CorruptionError("entry line is not valid UTF-8", path=path, line=position + 2) from error
            tab = text.find("\t")
            if tab <= 0:
                raise CorruptionError("entry without a key separator", path=path, line=position + 2)
            key = text[:tab]
            if previous_key is not None and key <= previous_key:
                raise CorruptionError(
                    "entries must be strictly ascending and unique",
                    path=path,
                    line=position + 2,
                    key=key,
                )
            _decode_value(text[tab + 1 :], path)
            if block_due or position % BLOCK_ENTRIES == 0:
                blocks.append(_BlockRef(first_key=key, start=offset, end=-1))
                block_due = False
            if min_key is None:
                min_key = key
            max_key = key
            previous_key = key
            offset += len(line) + 1  # LF stripped by splitlines
            if offset - blocks[-1].start >= BLOCK_BYTES:
                # Next entry opens a fresh block so a lookup never drags in a huge value's
                # neighbours; a single entry larger than the cap occupies a block on its own.
                block_due = True
        for index, block in enumerate(blocks):
            block.end = blocks[index + 1].start if index + 1 < len(blocks) else len(prefix)

        return cls(
            path=path,
            bloom=bloom,
            count=count,
            blocks=blocks,
            min_key=min_key,
            max_key=max_key,
        )

    # -- reads ---------------------------------------------------------------
    def __len__(self) -> int:
        return self.count

    def get(self, key: str) -> tuple[bool, str | None]:
        """Return `(found, value)`; a tombstone is found with value None.

        A bloom rejection touches no entry bytes. A possible hit reads just the one block whose
        first-key range brackets `key` -- bounded bytes, independent of table size -- and a bloom
        false positive is reported as an ordinary miss rather than as corruption.
        """
        if self.count == 0 or not self.bloom.might_contain(key):
            return False, None
        if key < self._min_key or key > self._max_key:
            return False, None
        index = bisect.bisect_right(self._first_keys, key) - 1
        if index < 0:
            return False, None
        return self._read_block(self._blocks[index], key)

    def scan(
        self,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> Iterator[tuple[str, str | None]]:
        """Yield `(key, value)` pairs in order, reading blocks lazily; None values are tombstones.

        One file handle serves the whole scan; blocks before `start` are never read and iteration
        stops at the block holding `end` (or after `limit` pairs).
        """
        if self.count == 0:
            return
        if start is not None and self._max_key < start:
            return
        if end is not None and self._min_key >= end:
            return
        produced = 0
        with open(self.path, "rb") as handle:
            for block in self._blocks_from(start):
                handle.seek(block.start)
                chunk = handle.read(block.end - block.start)
                if len(chunk) != block.end - block.start:
                    raise CorruptionError("table entry region is shorter than its index", path=self.path)
                for key, value in self._parse_block(chunk):
                    if start is not None and key < start:
                        continue
                    if end is not None and key >= end:
                        return
                    yield key, value
                    produced += 1
                    if limit is not None and produced >= limit:
                        return

    def items(self) -> list[tuple[str, str | None]]:
        """Materialize the whole table -- retained for small-table callers and tests only."""
        return list(self.scan())

    def to_document(self) -> dict[str, object]:
        return {"path": self.path, "keys": self.count, "bloom": self.bloom.to_document()}

    # -- internals -----------------------------------------------------------
    def _blocks_from(self, start: str | None) -> list[_BlockRef]:
        if start is None:
            return self._blocks
        index = bisect.bisect_right(self._first_keys, start) - 1
        return self._blocks[max(0, index) :]

    def _read_block(self, block: _BlockRef, key: str) -> tuple[bool, str | None]:
        chunk = self._read_bytes(block.start, block.end)
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CorruptionError("entry block is not valid UTF-8", path=self.path) from error
        found_line: str | None = None
        rows = 0
        for line in text.splitlines():
            rows += 1
            candidate, separator, encoded = line.partition("\t")
            if not separator or not candidate:
                raise CorruptionError("entry without a key separator", path=self.path)
            if candidate == key:
                found_line = encoded
                break
            if candidate > key:
                break  # strictly ordered: the key cannot appear later in the block
        if found_line is None:
            if not rows:
                raise CorruptionError("indexed block holds no entries", path=self.path)
            return False, None
        return True, _decode_value(found_line, self.path)

    def _read_bytes(self, start: int, end: int) -> bytes:
        try:
            with open(self.path, "rb") as handle:
                handle.seek(start)
                chunk = handle.read(end - start)
        except OSError as error:
            raise CorruptionError(f"table entry region unreadable: {error}", path=self.path) from error
        if len(chunk) != end - start:
            raise CorruptionError("table entry region is shorter than its index", path=self.path)
        return chunk

    def _parse_block(self, chunk: bytes) -> Iterator[tuple[str, str | None]]:
        """Yield `(key, value)` pairs from one block, decoding each value as it is yielded."""
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CorruptionError("entry block is not valid UTF-8", path=self.path) from error
        produced = 0
        for line in text.splitlines():
            key, separator, encoded = line.partition("\t")
            if not separator or not key:
                raise CorruptionError("entry without a key separator", path=self.path)
            yield key, _decode_value(encoded, self.path)
            produced += 1
        if not produced:
            raise CorruptionError("indexed block holds no entries", path=self.path)
