"""The store: one directory, an ordered set of sealed tables, and one live write-ahead log.

Read path: memtable -> newest sealed table -> ... -> oldest. A tombstone found on the way *is* the
answer (the key is deleted); only a full miss means the key never existed.

Write path: WAL first (so a crash cannot lose an acknowledged write), then the memtable. `flush`
seals the memtable into a new table and resets the log; `compact` merges every table into one and
drops tombstones that no longer hide anything.

Scans and compaction merge the layers incrementally: each sealed table is an ordered iterator
over lazily-read blocks and the memtable is one ordered iterator, combined with a small heap. No
layer is ever copied into a covering dictionary, so neither path grows resident memory with the
sealed value payload.
"""

from __future__ import annotations

import heapq
import os
import re
from dataclasses import dataclass, field
from typing import Iterator

from .errors import ValidationError
from .memtable import MemTable
from .sstable import SSTable, SSTableWriter
from .wal import DEL, PUT, WriteAheadLog, replay

TABLE_PATTERN = re.compile(r"^table-(\d+)\.sst$")


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

        Layers are streamed oldest -> newest -> memtable through the same incremental merge the
        read path uses, so the newest value wins and tombstones hide older ones without anyone
        buffering a copy of the database. The new table is fully written and sealed (its crc32
        checked by `SSTable.open`) before the old tables are removed: a failure mid-merge leaves
        the sealed inputs untouched on disk.
        """
        removed = [table.path for table in self.tables]
        path = self._table_path(self.next_table_number)
        estimate = sum(table.count for table in self.tables) + len(self.memtable.entries)
        temporary = path + ".tmp"
        if os.path.exists(temporary):
            os.unlink(temporary)
        writer = SSTableWriter(temporary, estimate)
        live = 0
        try:
            for key, value in self._visible_entries():
                if value is None:
                    continue
                writer.add(key, value)
                live += 1
            writer.close(live)  # validates the seal before anything old is touched
        except BaseException:
            writer.abort()
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
        # Publish first, delete after: a crash in between leaves the new higher-numbered table
        # next to the old ones, and layer ordering lets it win on reopen -- no lost writes.
        os.replace(temporary, path)
        for old in removed:
            if os.path.exists(old):
                os.unlink(old)
        # Re-open from its final name so the resident index carries the path callers see.
        table = SSTable.open(path)
        self.tables = [table]
        self.memtable.clear()
        self.wal.reset()
        self.wal_records = 0
        return {"table": path, "keys": live, "removedTables": len(removed)}

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
        for key, value in self._visible_entries(start=start, end=end):
            # Only visible (non-tombstone) rows come out of the merge, so each emitted row counts
            # toward the limit; reaching it closes the layer iterators and stops further reads.
            rows.append((key, value))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    def snapshot(self) -> Snapshot:
        return Snapshot(sequence=self.writes, visible=tuple(self.scan()))

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
    def _layer_iterators(self, start: str | None, end: str | None) -> list[Iterator[tuple[str, str | None]]]:
        """Ordered iterators from newest layer to oldest.

        The memtable is fully resident already; each sealed table contributes a lazy iterator
        that only reads the blocks its range touches. Nothing here copies a table.
        """
        layers: list[Iterator[tuple[str, str | None]]] = [iter(self.memtable.items())]
        for table in self.tables:
            layers.append(table.scan(start=start, end=end))
        return layers

    def _visible_entries(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> Iterator[tuple[str, str]]:
        """K-way merge of every layer, newest wins, yielding only visible (non-tombstone) rows.

        Heap entries are `(key, layer)`; equal keys collapse onto the newest layer (layer 0) and
        stale copies in older layers are consumed and discarded. Rows hidden by a tombstone are
        skipped but still pull every layer past that key, so iteration stays incremental.
        """
        layers = self._layer_iterators(start, end)

        heap: list[tuple[str, int]] = []
        current: list[tuple[str, str | None] | None] = [None] * len(layers)

        def advance(layer: int) -> None:
            try:
                key, value = next(layers[layer])
            except StopIteration:
                current[layer] = None
                return
            current[layer] = (key, value)
            heapq.heappush(heap, (key, layer))

        try:
            for layer in range(len(layers)):
                advance(layer)

            while heap:
                key, layer = heapq.heappop(heap)
                winner = current[layer]
                advance(layer)
                # Drain every older copy of this key: the newest layer already decided the outcome.
                while heap and heap[0][0] == key:
                    _, stale_layer = heapq.heappop(heap)
                    advance(stale_layer)
                if end is not None and key >= end:
                    return
                if start is not None and key < start:
                    continue
                if winner is not None and winner[1] is not None:
                    yield key, winner[1]
        finally:
            # An early `limit` break must release each table scan's file handle promptly rather
            # than waiting for garbage collection.
            for layer in layers:
                close = getattr(layer, "close", None)
                if close is not None:
                    close()

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
