"""Sorted table on disk: one header, one line per key in ascending order, one footer.

Layout (UTF-8, LF):

    header  {"count":N,"bloomBits":M,"bloomHashes":K}
    entry   <key><TAB><base64 value>      (a tombstone stores the marker `~`)
    footer  {"bloom":"<base64 bits>","crc32":"<8 hex of all preceding bytes>"}

Opening a table never materialises its payload: the file is streamed once, verifying the crc32
seal and every structural rule, but only the bloom filter, the entry count and a compact array of
entry byte offsets survive in memory. Values stay on disk and are fetched on demand; resident
memory is O(N * word) plus the bloom bits, independent of the total value byte count.

Lookups exploit the sorted keys: the bloom filter settles an absent key without touching the
entry region at all; a possible hit binary-searches entries with probes that read through the
candidate key and exactly one value-marker byte, so the bytes read never grow with value length.
Only an exact match pays for a full value read. Range scans stream rows in key order through
``open_stream`` / ``scan_rows``.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import json
import os
import zlib
from array import array
from dataclasses import dataclass
from typing import Iterator

from .errors import CorruptionError, ValidationError
from .wal import fsync_directory

TOMBSTONE = "~"
TOMBSTONE_BYTE = ord(TOMBSTONE)
# A value encoded as the empty base64 string (the empty string itself: `key\t\n`) has no marker
# byte at all; zero never occurs in the base64 alphabet, so it doubles as the "empty" marker.
EMPTY_MARKER = 0
BLOOM_HASHES = 4
# Fixed block size for the open-time structural pass. It bounds the decode buffer to a constant
# regardless of how large the payload on disk is: residency never grows with the value byte count.
VALIDATE_BLOCK = 1 << 20
# First read of a locating probe: keys are normally short, so a small chunk usually reaches the
# tab and the value marker; a key longer than this keeps reading in the continuation size.
PROBE_INITIAL = 64
PROBE_BLOCK = 1 << 12
_TMP_ENTRY = ".tmp"
_TMP_FINAL = ".out"


def _hashes(key: str, count: int, bits: int) -> list[int]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return [int.from_bytes(digest[index * 4 : index * 4 + 4], "big") % bits for index in range(count)]


@dataclass(slots=True)
class BloomFilter:
    bits: int
    hashes: int
    payload: bytearray

    @classmethod
    def create(cls, expected_keys: int, bits_per_key: int = 10) -> "BloomFilter":
        bits = max(64, bits_per_key * max(1, expected_keys))
        return cls(bits=bits, hashes=BLOOM_HASHES, payload=bytearray((bits + 7) // 8))

    @classmethod
    def build(cls, keys: list[str], bits_per_key: int = 10) -> "BloomFilter":
        bloom = cls.create(len(keys), bits_per_key)
        for key in keys:
            bloom.add(key)
        return bloom

    @classmethod
    def loads(cls, bits: int, hashes: int, encoded: str) -> "BloomFilter":
        return cls(bits=bits, hashes=hashes, payload=bytearray(base64.b64decode(encoded)))

    def add(self, key: str) -> None:
        for position in _hashes(key, self.hashes, self.bits):
            self.payload[position // 8] |= 1 << (position % 8)

    def dumps(self) -> str:
        return base64.b64encode(bytes(self.payload)).decode("ascii")

    def might_contain(self, key: str) -> bool:
        return all(self.payload[position // 8] & (1 << (position % 8)) for position in _hashes(key, self.hashes, self.bits))

    def to_document(self) -> dict[str, int]:
        return {"bits": self.bits, "hashes": self.hashes, "bytes": len(self.payload)}


def _decode_value(encoded: str, *, path: str, line: int) -> str | None:
    """Decode one entry payload; the tombstone marker is None, anything else must be base64 UTF-8."""
    if encoded == TOMBSTONE:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise CorruptionError("entry value must be base64", path=path, line=line) from error
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorruptionError("entry value must decode to UTF-8", path=path, line=line) from error


class TableWriter:
    """Sink for a sorted row stream.

    Entry lines spool to a temp file while the running crc32 accumulates and the bloom filter is
    built incrementally, so callers (notably compaction) never hold an entire table -- or a merged
    copy of their input tables -- in memory. The header only exists once the count is known, so
    the final file is assembled in a second streaming pass and atomically renamed into place.
    """

    __slots__ = ("path", "entry_tmp", "final_tmp", "bits_per_key", "handle", "crc", "count", "previous")

    def __init__(self, path: str, *, bits_per_key: int = 10) -> None:
        self.path = path
        self.entry_tmp = path + _TMP_ENTRY
        self.final_tmp = path + _TMP_FINAL
        self.bits_per_key = bits_per_key
        self.handle = open(self.entry_tmp, "wb")
        self.crc = 0
        self.count = 0
        self.previous: str | None = None

    def add(self, key: str, value: str | None) -> None:
        if self.previous is not None and key <= self.previous:
            if key == self.previous:
                raise ValidationError("sstable keys must be unique", key=key)
            raise ValidationError("sstable entries must be written in ascending key order", key=key)
        encoded = TOMBSTONE if value is None else base64.b64encode(value.encode("utf-8")).decode("ascii")
        block = f"{key}\t{encoded}\n".encode("utf-8")
        self.handle.write(block)
        self.crc = zlib.crc32(block, self.crc)
        self.previous = key
        self.count += 1

    def finish(self) -> "SSTable":
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        # The count is only known once every row arrived, so size the bloom filter now and build
        # it while the spool is re-read for assembly: one entry line is resident at a time.
        bloom = BloomFilter.create(self.count, self.bits_per_key)
        header = json.dumps(
            {"count": self.count, "bloomBits": bloom.bits, "bloomHashes": bloom.hashes},
            separators=(",", ":"),
        ) + "\n"
        header_bytes = header.encode("utf-8")
        checksum = zlib.crc32(header_bytes)
        rows = 0
        with open(self.entry_tmp, "rb") as source, open(self.final_tmp, "wb") as target:
            target.write(header_bytes)
            while True:
                line = source.readline()
                if not line:
                    break
                rows += 1
                tab = line.find(b"\t")
                try:
                    bloom.add(line[:tab].decode("utf-8"))
                except UnicodeDecodeError as error:
                    raise CorruptionError("entry key must be UTF-8", path=self.path, line=rows + 1) from error
                checksum = zlib.crc32(line, checksum)
                target.write(line)
            footer = json.dumps(
                {"bloom": bloom.dumps(), "crc32": format(checksum & 0xFFFFFFFF, "08x")},
                separators=(",", ":"),
            )
            target.write(footer.encode("utf-8"))
            target.write(b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(self.final_tmp, self.path)
        # Make the new directory entry durable: without the directory fsync a crash could still
        # expose the pre-rename state (the spool present, the final table absent).
        fsync_directory(self.path)
        os.unlink(self.entry_tmp)
        return SSTable.open(self.path)

    def abort(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass
        for tmp in (self.entry_tmp, self.final_tmp):
            if os.path.exists(tmp):
                os.unlink(tmp)


@dataclass(slots=True)
class SSTable:
    path: str
    bloom: BloomFilter
    count: int
    # Byte offset of the start of every entry line, in key order. Eight bytes per entry regardless
    # of pointer width; this array is the table's only per-key resident structure.
    offsets: array

    @classmethod
    def write(cls, path: str, items: list[tuple[str, str | None]], *, bits_per_key: int = 10) -> "SSTable":
        keys = [key for key, _ in items]
        if keys != sorted(keys):
            raise ValidationError("sstable entries must be written in ascending key order")
        if len(set(keys)) != len(keys):
            raise ValidationError("sstable keys must be unique")
        writer = TableWriter(path, bits_per_key=bits_per_key)
        try:
            for key, value in items:
                writer.add(key, value)
            return writer.finish()
        except BaseException:
            writer.abort()
            raise

    @classmethod
    def open(cls, path: str) -> "SSTable":
        size = os.path.getsize(path)
        if size < 2:
            raise CorruptionError("table is too short to hold a footer", path=path)
        with open(path, "rb") as handle:
            handle.seek(size - 1)
            if handle.read(1) != b"\n":
                raise CorruptionError("table is not newline terminated", path=path)
            # The footer is one short JSON line; read only the tail that can contain it.
            chunk_start = max(0, size - 65536)
            handle.seek(chunk_start)
            tail = handle.read(size - chunk_start)
        rel = tail.rfind(b"\n", 0, len(tail) - 1)
        if rel < 0:
            raise CorruptionError("table is too short to hold a footer", path=path)
        footer_start = chunk_start + rel + 1
        prefix_len = footer_start
        try:
            footer = json.loads(tail[rel + 1 : -1].decode("utf-8").strip())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CorruptionError("unreadable footer", path=path) from error
        if not isinstance(footer, dict) or not isinstance(footer.get("crc32"), str) or not isinstance(footer.get("bloom"), str):
            raise CorruptionError("footer is missing crc32 or bloom", path=path)

        offsets: array = array("Q")
        checksum = 0
        seen = 0
        previous: str | None = None
        alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        try:
            with open(path, "rb") as handle:
                header_line = handle.readline()
                if not header_line or not header_line.endswith(b"\n") or handle.tell() > prefix_len:
                    raise CorruptionError("missing or overlong header", path=path)
                checksum = zlib.crc32(header_line)
                try:
                    header = json.loads(header_line[:-1].decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise CorruptionError("unreadable header", path=path) from error
                if not isinstance(header, dict):
                    raise CorruptionError("header must be an object", path=path)
                try:
                    bloom_bits = int(header["bloomBits"])  # type: ignore[arg-type]
                    bloom_hashes = int(header["bloomHashes"])  # type: ignore[arg-type]
                    expected_count = int(header["count"])  # type: ignore[arg-type]
                except (KeyError, TypeError, ValueError) as error:
                    raise CorruptionError("header field is missing or not an integer", path=path) from error
                if bloom_bits <= 0 or bloom_hashes <= 0 or expected_count < 0:
                    raise CorruptionError("header fields must be positive", path=path)

                # Stream the entry region in fixed blocks. The seal is updated per block, the
                # strict ordering is checked, and payloads are validated as base64-then-UTF-8 a
                # few decoded bytes at a time: the bytes of one value are never resident.
                buf = b""

                def pull() -> bytes:
                    nonlocal checksum
                    remaining = prefix_len - handle.tell()
                    if remaining <= 0:
                        return b""
                    chunk = handle.read(min(VALIDATE_BLOCK, remaining))
                    checksum = zlib.crc32(chunk, checksum)
                    return chunk

                while handle.tell() - len(buf) < prefix_len:
                    offsets.append(handle.tell() - len(buf))
                    # -- key up to the tab ----------------------------------------------
                    while b"\t" not in buf:
                        if b"\n" in buf:
                            raise CorruptionError("entry without a separator", path=path, line=seen + 1)
                        chunk = pull()
                        if not chunk:
                            raise CorruptionError("entry line is unterminated", path=path, line=seen + 1)
                        buf += chunk
                    tab_index = buf.find(b"\t")
                    if b"\n" in buf[:tab_index]:
                        raise CorruptionError("entry without a separator", path=path, line=seen + 1)
                    try:
                        key = bytes(buf[:tab_index]).decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise CorruptionError("entry key must be UTF-8", path=path, line=seen + 1) from error
                    if not key:
                        raise CorruptionError("entry has an empty key", path=path, line=seen + 1)
                    if previous is not None and key <= previous:
                        raise CorruptionError(
                            "entries must be strictly ascending and unique",
                            path=path,
                            line=seen + 1,
                            key=key,
                        )
                    buf = buf[tab_index + 1 :]
                    if not buf:
                        chunk = pull()
                        if not chunk:
                            raise CorruptionError("entry line is unterminated", path=path, line=seen + 1)
                        buf = chunk

                    # -- the payload: empty, tombstone, or base64 value ------------------
                    if buf[:1] == b"\n":
                        buf = buf[1:]  # empty string value
                    elif buf[:1] == b"~":
                        if len(buf) >= 2:
                            if buf[1] != 0x0A:
                                raise CorruptionError("invalid tombstone payload", path=path, line=seen + 1)
                            buf = buf[2:]
                        else:
                            chunk = pull()
                            if not chunk or chunk[0] != 0x0A:
                                raise CorruptionError("invalid tombstone payload", path=path, line=seen + 1)
                            buf = chunk[1:]
                    else:
                        pending = b""
                        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")

                        def feed(piece: bytes, *, final: bool) -> None:
                            """Validate base64 incrementally: between chunks keep at least the
                            last quad pending (the only quad that may carry padding), decode the
                            aligned prefix, and stream raw bytes through a UTF-8 decoder."""
                            nonlocal pending
                            data = pending + piece
                            if final:
                                groups, pending = data, b""
                                core = groups.rstrip(b"=")
                                pad_count = len(groups) - len(core)
                                if core.strip(alphabet) or pad_count > 2 or len(groups) % 4:
                                    raise CorruptionError("entry value must be base64", path=path, line=seen + 1)
                            else:
                                if len(data) < 4:
                                    pending = data
                                    return
                                # Reserve the trailing quad: padding ('=' / '==') can only occur
                                # in the very last quad, which the newline-bearing chunk finalises.
                                end = (len(data) - 4) & ~3
                                groups, pending = data[:end], data[end:]
                                if groups.strip(alphabet):
                                    raise CorruptionError("entry value must be base64", path=path, line=seen + 1)
                            if groups:
                                try:
                                    raw = base64.b64decode(groups, validate=True)
                                    decoder.decode(raw)
                                    if final:
                                        decoder.decode(b"", final=True)
                                except (binascii.Error, UnicodeDecodeError, ValueError) as error:
                                    raise CorruptionError("entry value must be base64 UTF-8", path=path, line=seen + 1) from error

                        while True:
                            newline = buf.find(b"\n")
                            if newline >= 0:
                                feed(buf[:newline], final=True)
                                buf = buf[newline + 1 :]
                                break
                            feed(buf, final=False)
                            buf = b""
                            chunk = pull()
                            if not chunk:
                                raise CorruptionError("entry line is unterminated", path=path, line=seen + 1)
                            buf = chunk
                    previous = key
                    seen += 1

            if seen != expected_count:
                raise CorruptionError(
                    "entry count differs from the header",
                    path=path,
                    expected=expected_count,
                    found=seen,
                )
            if format(checksum & 0xFFFFFFFF, "08x") != footer["crc32"]:
                raise CorruptionError("table checksum mismatch", path=path)
            try:
                bloom = BloomFilter.loads(bloom_bits, bloom_hashes, str(footer["bloom"]))
            except (binascii.Error, ValueError) as error:
                raise CorruptionError("bloom payload must be base64", path=path) from error
            if len(bloom.payload) != (bloom_bits + 7) // 8:
                raise CorruptionError("bloom payload length does not match bloomBits", path=path)
        except CorruptionError:
            raise
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise CorruptionError("table could not be parsed", path=path) from error
        return cls(path=path, bloom=bloom, count=seen, offsets=offsets)

    # -- internals -----------------------------------------------------------
    def _slot_at_or_after(self, handle: object, key: str | None) -> int:
        """First slot whose entry key is >= ``key`` (0 when ``key`` is None)."""
        if key is None:
            return 0
        lo, hi = 0, len(self.offsets)
        while lo < hi:
            mid = (lo + hi) // 2
            candidate, _ = self._probe(handle, mid)
            if candidate < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _probe(self, handle: object, slot: int) -> tuple[str, int]:
        """Read just enough to locate an entry: its key and one value-marker byte.

        Returns ``(key, marker)``: ``~`` means tombstone, :data:`EMPTY_MARKER` means the empty
        string, and anything else (a base64 alphabet byte) says "a value exists". The value bytes
        are never part of this read.
        """
        handle.seek(int(self.offsets[slot]))
        key_bytes = b""
        size = PROBE_INITIAL
        while True:
            block = handle.read(size)
            if not block:
                raise CorruptionError("entry line is unterminated", path=self.path, line=slot + 2)
            tab = block.find(b"\t")
            newline = block.find(b"\n")
            if tab >= 0 and (newline < 0 or tab < newline):
                key_bytes += block[:tab]
                marker = block[tab + 1 : tab + 2]
                if not marker:
                    marker = handle.read(1)
                if not marker:
                    raise CorruptionError("entry without a value", path=self.path, line=slot + 2)
                code = marker[0]
                if code == 0x0A:
                    code = EMPTY_MARKER
                break
            if newline >= 0:
                raise CorruptionError("entry without a separator", path=self.path, line=slot + 2)
            key_bytes += block
            size = PROBE_BLOCK  # long key: keep reading in modest chunks, never whole values
        try:
            return key_bytes.decode("utf-8"), code
        except UnicodeDecodeError as error:
            raise CorruptionError("entry key must be UTF-8", path=self.path, line=slot + 2) from error

    def _read_full(self, handle: object, slot: int) -> tuple[str, str | None]:
        handle.seek(int(self.offsets[slot]))
        line = handle.readline()
        if not line.endswith(b"\n"):
            raise CorruptionError("entry line is unterminated", path=self.path, line=slot + 2)
        entry = line[:-1]
        tab = entry.find(b"\t")
        if tab <= 0:
            raise CorruptionError("entry without a separator", path=self.path, line=slot + 2)
        try:
            key = entry[:tab].decode("utf-8")
            encoded = entry[tab + 1 :].decode("ascii")
        except UnicodeDecodeError as error:
            raise CorruptionError("entry must be UTF-8 key with ASCII base64 value", path=self.path, line=slot + 2) from error
        return key, _decode_value(encoded, path=self.path, line=slot + 2)

    # -- reads ---------------------------------------------------------------
    def get(self, key: str) -> tuple[bool, str | None]:
        if not self.bloom.might_contain(key):
            return False, None
        with open(self.path, "rb") as handle:
            lo, hi = 0, len(self.offsets)
            while lo < hi:
                mid = (lo + hi) // 2
                candidate, marker = self._probe(handle, mid)
                if candidate < key:
                    lo = mid + 1
                elif candidate > key:
                    hi = mid
                elif marker == TOMBSTONE_BYTE:
                    return True, None
                elif marker == EMPTY_MARKER:
                    return True, ""
                else:
                    _, value = self._read_full(handle, mid)
                    return True, value
        return False, None

    def open_stream(self, start: str | None = None) -> "RowStream":
        return RowStream(self, start)

    def scan_rows(self, start: str | None = None, end: str | None = None) -> Iterator[tuple[str, str | None]]:
        """Yield rows with ``start <= key < end`` (either bound optional), one line at a time."""
        stream = self.open_stream(start)
        try:
            while stream.head is not None:
                key, marker = stream.head
                if end is not None and key >= end:
                    return
                if marker == TOMBSTONE_BYTE:
                    value = None
                elif marker == EMPTY_MARKER:
                    value = ""
                else:
                    value = stream.value()
                yield key, value
                stream.advance()
        finally:
            stream.close()

    def items(self) -> list[tuple[str, str | None]]:
        return list(self.scan_rows())

    def to_document(self) -> dict[str, object]:
        return {"path": self.path, "keys": self.count, "bloom": self.bloom.to_document()}


class RowStream:
    """Forward cursor over a table.

    The head holds only ``(key, marker)``: the value bytes of a row are fetched on demand via
    :meth:`value`, so a k-way merge never reads values that a newer layer shadows (or that are
    tombstones). ``close`` releases the file handle.
    """

    __slots__ = ("table", "handle", "slot", "head")

    def __init__(self, table: SSTable, start: str | None = None) -> None:
        self.table = table
        self.handle = open(table.path, "rb")
        self.slot = table._slot_at_or_after(self.handle, start)
        self.head: tuple[str, int] | None = None
        self._load()

    def _load(self) -> None:
        if self.slot < len(self.table.offsets):
            self.head = self.table._probe(self.handle, self.slot)
            self.slot += 1
        else:
            self.head = None

    def value(self) -> str:
        """Full value of the current row; valid only when the marker is not a tombstone."""
        _, value = self.table._read_full(self.handle, self.slot - 1)
        return value  # type: ignore[return-value]

    def advance(self) -> None:
        self._load()

    def close(self) -> None:
        self.handle.close()
        self.head = None
