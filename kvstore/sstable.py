"""Sorted table on disk: one header, one line per key in ascending order, one footer.

Layout (UTF-8, LF):

    header  {"count":N,"bloomBits":M,"bloomHashes":K}
    entry   <key><TAB><base64 value>      (a tombstone stores the marker `~`)
    footer  {"bloom":"<base64 bits>","crc32":"<8 hex of all preceding bytes>"}

The bloom filter is checked before any entry is touched, so a miss on an absent key costs one hash
rather than a scan. `open()` indexes the table in memory -- a deliberate simplification for a
baseline: tables are small and the index keeps lookups O(1); the pair that extends this baseline is
expected to replace it with a sparse index over blocks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from dataclasses import dataclass

from .errors import CorruptionError, ParseError, ValidationError

TOMBSTONE = "~"
BLOOM_HASHES = 4


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
        bits = max(64, bits_per_key * max(1, len(keys)))
        payload = bytearray((bits + 7) // 8)
        for key in keys:
            for position in _hashes(key, BLOOM_HASHES, bits):
                payload[position // 8] |= 1 << (position % 8)
        return cls(bits=bits, hashes=BLOOM_HASHES, payload=payload)

    @classmethod
    def loads(cls, bits: int, hashes: int, encoded: str) -> "BloomFilter":
        return cls(bits=bits, hashes=hashes, payload=bytearray(base64.b64decode(encoded)))

    def dumps(self) -> str:
        return base64.b64encode(bytes(self.payload)).decode("ascii")

    def might_contain(self, key: str) -> bool:
        return all(self.payload[position // 8] & (1 << (position % 8)) for position in _hashes(key, self.hashes, self.bits))

    def to_document(self) -> dict[str, int]:
        return {"bits": self.bits, "hashes": self.hashes, "bytes": len(self.payload)}


@dataclass(slots=True)
class SSTable:
    path: str
    bloom: BloomFilter
    values: dict[str, str | None]
    index: dict[str, int]

    @classmethod
    def write(cls, path: str, items: list[tuple[str, str | None]], *, bits_per_key: int = 10) -> "SSTable":
        keys = [key for key, _ in items]
        if keys != sorted(keys):
            raise ValidationError("sstable entries must be written in ascending key order")
        if len(set(keys)) != len(keys):
            raise ValidationError("sstable keys must be unique")
        bloom = BloomFilter.build(keys, bits_per_key)
        body = [json.dumps({"count": len(items), "bloomBits": bloom.bits, "bloomHashes": bloom.hashes}, separators=(",", ":")) + "\n"]
        for key, value in items:
            encoded = TOMBSTONE if value is None else base64.b64encode(value.encode("utf-8")).decode("ascii")
            body.append(f"{key}\t{encoded}\n")
        prefix = "".join(body).encode("utf-8")
        footer = json.dumps(
            {"bloom": bloom.dumps(), "crc32": format(zlib.crc32(prefix) & 0xFFFFFFFF, "08x")},
            separators=(",", ":"),
        )
        with open(path, "wb") as handle:
            handle.write(prefix)
            handle.write(footer.encode("utf-8"))
            handle.write(b"\n")
        return cls.open(path)

    @classmethod
    def open(cls, path: str) -> "SSTable":
        with open(path, "rb") as handle:
            raw = handle.read()
        newline = raw.rfind(b"\n", 0, len(raw) - 1) if raw.endswith(b"\n") else -1
        if newline <= 0:
            raise CorruptionError("table is too short to hold a footer", path=path)
        prefix, footer_raw = raw[: newline + 1], raw[newline + 1 :].decode("utf-8").strip()
        try:
            footer = json.loads(footer_raw)
        except json.JSONDecodeError as error:
            raise CorruptionError(f"unreadable footer: {error.msg}", path=path) from error
        expected = format(zlib.crc32(prefix) & 0xFFFFFFFF, "08x")
        if footer.get("crc32") != expected:
            raise CorruptionError("table checksum mismatch", path=path)
        lines = prefix.decode("utf-8").splitlines()
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise CorruptionError(f"unreadable header: {error.msg}", path=path) from error
        bloom = BloomFilter.loads(int(header["bloomBits"]), int(header["bloomHashes"]), str(footer["bloom"]))
        values: dict[str, str | None] = {}
        index: dict[str, int] = {}
        for offset, line in enumerate(lines[1:], start=1):
            if not line:
                continue
            key, _, encoded = line.partition("\t")
            if not _:
                raise CorruptionError("entry without a separator", path=path, line=offset)
            values[key] = None if encoded == TOMBSTONE else base64.b64decode(encoded).decode("utf-8")
            index[key] = offset
        if len(values) != int(header["count"]):
            raise CorruptionError("entry count differs from the header", path=path, expected=header["count"], found=len(values))
        return cls(path=path, bloom=bloom, values=values, index=index)

    def get(self, key: str) -> tuple[bool, str | None]:
        if not self.bloom.might_contain(key):
            return False, None
        if key in self.values:
            return True, self.values[key]
        return False, None

    def items(self) -> list[tuple[str, str | None]]:
        return sorted(self.values.items())

    def to_document(self) -> dict[str, object]:
        return {"path": self.path, "keys": len(self.values), "bloom": self.bloom.to_document()}
