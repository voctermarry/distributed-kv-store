"""The store: one directory, an ordered set of sealed tables, and one live write-ahead log.

Read path: memtable -> newest sealed table -> ... -> oldest. A tombstone found on the way *is* the
answer (the key is deleted); only a full miss means the key never existed.

Write path: WAL first (so a crash cannot lose an acknowledged write), then the memtable. `flush`
seals the memtable into a new table and resets the log; `compact` merges every table into one and
drops tombstones that no longer hide anything.
"""

from __future__ import annotations

import heapq
import os
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Iterator

from .errors import ValidationError
from .memtable import MemTable
from .sstable import EMPTY_MARKER, TOMBSTONE_BYTE, SSTable, TableWriter
from .wal import DEL, PUT, WriteAheadLog, replay

TABLE_PATTERN = re.compile(r"^table-(\d+)\.sst$")


class _Cursor:
    """One ordered layer in the k-way merge. Lower rank wins a key tie (newer layer wins).

    The head is ``(key, marker_or_value)``: table cursors carry only the value-marker byte and
    fetch the value on demand, while the memtable cursor carries the value directly.
    """

    __slots__ = ("rank", "key", "payload")

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.key: str | None = None
        self.payload: object = None

    def __lt__(self, other: "_Cursor") -> bool:
        return (self.key, self.rank) < (other.key, other.rank)


class _MemTableCursor(_Cursor):
    __slots__ = ("entries", "positions", "index")

    def __init__(self, entries: dict[str, str | None], start: str | None) -> None:
        super().__init__(rank=0)  # the memtable is the newest layer
        self.entries = entries
        self.positions = sorted(entries)
        self.index = bisect_left(self.positions, start) if start is not None else 0
        self._load()

    def _load(self) -> None:
        if self.index < len(self.positions):
            self.key = self.positions[self.index]
            self.payload = self.entries[self.key]
        else:
            self.key = None

    @property
    def tombstone(self) -> bool:
        return self.payload is None

    def advance(self) -> None:
        self.index += 1
        self._load()

    def read_value(self) -> str:
        return self.payload  # type: ignore[return-value]


class _TableCursor(_Cursor):
    __slots__ = ("stream")

    def __init__(self, table: SSTable, rank: int, start: str | None) -> None:
        super().__init__(rank=rank)
        self.stream = table.open_stream(start)
        if self.stream.head is None:
            self.key = None
        else:
            self.key, self.payload = self.stream.head

    def advance(self) -> None:
        self.stream.advance()
        if self.stream.head is None:
            self.key = None
        else:
            self.key, self.payload = self.stream.head

    @property
    def tombstone(self) -> bool:
        return self.payload == TOMBSTONE_BYTE

    def read_value(self) -> str:
        if self.payload == EMPTY_MARKER:  # empty-string value: no value bytes to fetch
            return ""
        return self.stream.value()

    def close(self) -> None:
        self.stream.close()


@dataclass(frozen=True, slots=True)
class StoreStats:
    directory: str
    tables: int
    sealed_entries: int
    memtable_entries: int
    memtable_bytes: int
    tombstones: int
    wal_records: int
    wal_truncated_bytes: int
    next_table_number: int

    def to_document(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "tables": self.tables,
            "sealedEntries": self.sealed_entries,
            "memtableEntries": self.memtable_entries,
            "memtableBytes": self.memtable_bytes,
            "tombstones": self.tombstones,
            "walRecords": self.wal_records,
            "walTruncatedBytes": self.wal_truncated_bytes,
            "nextTableNumber": self.next_table_number,
        }


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A point-in-time view: tombstones are already resolved away."""

    sequence: int
    visible: tuple[tuple[str, str], ...]

    def get(self, key: str, default: str | None = None) -> str | None:
        for candidate, value in self.visible:
            if candidate == key:
                return value
        return default

    def items(self) -> list[tuple[str, str]]:
        return list(self.visible)

    def to_document(self) -> dict[str, object]:
        return {"sequence": self.sequence, "keys": len(self.visible)}


@dataclass(slots=True)
class Store:
    directory: str
    memtable_limit_bytes: int = 1 << 20
    memtable: MemTable = field(init=False)
    tables: list[SSTable] = field(default_factory=list, init=False)
    wal: WriteAheadLog = field(init=False)
    wal_records: int = field(default=0, init=False)
    wal_truncated_bytes: int = field(default=0, init=False)
    writes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.memtable_limit_bytes <= 0:
            raise ValidationError("memtable_limit_bytes must be > 0", value=self.memtable_limit_bytes)
        self.memtable = MemTable(limit_bytes=self.memtable_limit_bytes)
        self.wal = WriteAheadLog(os.path.join(self.directory, "wal.log"))

    # -- lifecycle -----------------------------------------------------------
    def open(self) -> "Store":
        os.makedirs(self.directory, exist_ok=True)
        self.tables = [SSTable.open(path) for path in self._table_paths()]
        self.tables.reverse()  # newest first
        self.wal.open()
        report = self.wal.recover()
        self.wal_records = len(report.records)
        self.wal_truncated_bytes = report.truncated_bytes
        for payload in report.records:
            if payload["op"] == PUT:
                self.memtable.put(str(payload["key"]), str(payload["value"]))
            else:
                self.memtable.delete(str(payload["key"]))
        return self

    def close(self) -> None:
        self.wal.close()

    def __enter__(self) -> "Store":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- writes --------------------------------------------------------------
    def put(self, key: str, value: str) -> None:
        self._check_key(key)
        self.wal.append(PUT, key, value)
        self.memtable.put(key, value)
        self.writes += 1
        if self.memtable.is_full():
            self.flush()

    def delete(self, key: str) -> None:
        self._check_key(key)
        self.wal.append(DEL, key)
        self.memtable.delete(key)
        self.writes += 1
        if self.memtable.is_full():
            self.flush()

    def flush(self) -> str | None:
        """Seal the memtable into a new table. Returns the path, or None when nothing was live."""
        if not self.memtable.entries:
            return None
        items = self.memtable.items()
        path = self._table_path(self.next_table_number)
        table = SSTable.write(path, items)
        self.tables.insert(0, table)
        self.memtable.clear()
        self.wal.reset()
        self.wal_records = 0
        return path

    def compact(self) -> dict[str, object]:
        """Merge every table plus the memtable into one, dropping unreachable tombstones.

        Order matters: oldest table first, newest last, memtable last of all -- the incremental
        merge lets each newer layer overwrite the key's previous owner, so the surviving value is
        the newest, and the surviving rows stream straight into the new table: no full copy of the
        sealed payload is ever built. (Writing this the other way round kept the *oldest* value,
        which the compaction test caught.)
        """
        removed = [table.path for table in self.tables]
        path = self._table_path(self.next_table_number)
        writer = TableWriter(path)
        keys = 0
        try:
            for key, value in self._merged_rows():
                if value is None:
                    continue  # nothing older remains below this table: the tombstone is unreachable
                writer.add(key, value)
                keys += 1
            new_table = writer.finish()
        except BaseException:
            writer.abort()
            raise
        for old in removed:
            if os.path.exists(old):
                os.unlink(old)
        self.tables = [new_table]
        self.memtable.clear()
        self.wal.reset()
        self.wal_records = 0
        return {"table": path, "keys": keys, "removedTables": len(removed)}

    # -- reads ---------------------------------------------------------------
    def get(self, key: str) -> tuple[bool, str | None]:
        found, value = self.memtable.get(key)
        if found:
            return True, value
        for table in self.tables:
            found, value = table.get(key)
            if found:
                return True, value
        return False, None

    def scan(self, start: str | None = None, end: str | None = None, limit: int | None = None) -> list[tuple[str, str]]:
        if limit is not None and limit <= 0:
            raise ValidationError("limit must be > 0", value=limit)
        if start and end and end < start:
            raise ValidationError("end must be >= start", value=f"{start}..{end}")
        rows: list[tuple[str, str]] = []
        merged = self._merged_rows(start=start, end=end)
        try:
            for key, value in merged:
                if value is None:
                    continue  # tombstone hides whatever an older layer still holds for this key
                rows.append((key, value))
                if limit is not None and len(rows) >= limit:
                    break  # close() below stops every cursor; no later entry is touched
        finally:
            merged.close()  # releases the tables' read handles deterministically
        return rows

    def snapshot(self) -> Snapshot:
        return Snapshot(sequence=self.writes, visible=tuple(self.scan()))

    def _merged_rows(self, start: str | None = None, end: str | None = None) -> Iterator[tuple[str, str | None]]:
        """Incrementally merge the memtable and every sealed table in global key order.

        Layers are never copied: each layer contributes one forward cursor (a probe-only disk
        cursor for tables), and a heap hands back one distinct key at a time together with the
        value of its *newest* owner -- a tombstone marker surfaces as ``None``. Older copies of the
        same key are drained (and their value bytes never read) before the row is yielded.
        """
        cursors: list[_Cursor] = []
        try:
            memtable_cursor = _MemTableCursor(self.memtable.entries, start)
            if memtable_cursor.key is not None:
                cursors.append(memtable_cursor)
            for rank, table in enumerate(self.tables, start=1):  # tables are ordered newest first
                cursor = _TableCursor(table, rank, start)
                if cursor.key is None:
                    cursor.close()
                    continue
                cursors.append(cursor)
            heap = [(cursor.key, cursor.rank, cursor) for cursor in cursors]
            heapq.heapify(heap)
            while heap:
                key, _, winner = heap[0]
                if end is not None and key >= end:
                    return
                value = None if winner.tombstone else winner.read_value()
                while heap and heap[0][0] == key:  # retire every older copy of this key
                    _, _, cursor = heapq.heappop(heap)
                    cursor.advance()
                    if cursor.key is not None:
                        heapq.heappush(heap, (cursor.key, cursor.rank, cursor))
                yield key, value
        finally:
            for cursor in cursors:
                close = getattr(cursor, "close", None)
                if close is not None:
                    close()

    # -- reporting -----------------------------------------------------------
    @property
    def next_table_number(self) -> int:
        numbers = [int(TABLE_PATTERN.match(os.path.basename(path)).group(1)) for path in self._table_paths()]
        return (max(numbers) + 1) if numbers else 1

    def stats(self) -> StoreStats:
        tombstones = sum(1 for value in self.memtable.entries.values() if value is None)
        return StoreStats(
            directory=os.path.abspath(self.directory),
            tables=len(self.tables),
            sealed_entries=sum(table.count for table in self.tables),
            memtable_entries=len(self.memtable.entries),
            memtable_bytes=self.memtable.bytes_used,
            tombstones=tombstones,
            wal_records=self.wal_records,
            wal_truncated_bytes=self.wal_truncated_bytes,
            next_table_number=self.next_table_number,
        )

    def verify(self) -> dict[str, object]:
        """Re-open every artifact from disk and report what recovery had to repair."""
        tables = []
        for path in self._table_paths():
            table = SSTable.open(path)  # raises CorruptionError when the seal is broken
            tables.append(table.to_document())
        with WriteAheadLog(self.wal.path) as log:
            report = log.recover()
        state = replay(report.records)
        return {
            "tables": tables,
            "wal": {
                "records": len(report.records),
                "truncatedBytes": report.truncated_bytes,
                "lastSequence": report.last_sequence,
                "putKeys": sum(1 for value in state.values() if value is not None),
                "deletedKeys": sum(1 for value in state.values() if value is None),
            },
        }

    # -- internals -----------------------------------------------------------
    def _table_paths(self) -> list[str]:
        if not os.path.isdir(self.directory):
            return []
        paths = [os.path.join(self.directory, name) for name in os.listdir(self.directory) if TABLE_PATTERN.match(name)]
        return sorted(paths, key=lambda path: int(TABLE_PATTERN.match(os.path.basename(path)).group(1)))

    def _table_path(self, number: int) -> str:
        return os.path.join(self.directory, f"table-{number:06d}.sst")

    @staticmethod
    def _check_key(key: str) -> None:
        if not key:
            raise ValidationError("key must be non-empty", value=key)
        if any(character in key for character in "\t\n"):
            raise ValidationError("key must not contain tabs or newlines", value=key)
