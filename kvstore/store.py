"""The store: one directory, an ordered set of sealed tables, and one live write-ahead log.

Read path: memtable -> newest sealed table -> ... -> oldest. A tombstone found on the way *is* the
answer (the key is deleted); only a full miss means the key never existed.

Write path: WAL first (so a crash cannot lose an acknowledged write), then the memtable. `flush`
seals the memtable into a new table and resets the log; `compact` merges every table into one and
drops tombstones that no longer hide anything.
"""

from __future__ import annotations

import heapq
import json
import os
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Iterator

from .errors import CorruptionError, OutputError, ValidationError
from .memtable import MemTable
from .sstable import EMPTY_MARKER, TOMBSTONE_BYTE, SSTable, TableWriter
from .wal import DEL, PUT, WriteAheadLog, replay

TABLE_PATTERN = re.compile(r"^table-(\d+)\.sst$")
MANIFEST_PATTERN = re.compile(r"^compact-(\d+)\.manifest$")
# Compaction output is first assembled under this non-table name. Because it cannot match
# TABLE_PATTERN it is never mistaken for a live table; a manifest is the only thing that
# promotes it, so a crash before the manifest simply orphans it for the next open to sweep.
COMPACTION_TMP_SUFFIX = ".new"


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
        # Finish or undo any compaction whose process died mid-flight *before* tables are
        # loaded: the on-disk layout must be a single committed generation by the time reads
        # and the WAL replay see it.
        self._recover_compactions()
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

        Crash consistency: the merged rows are written to a non-table temp name while the old
        tables and the WAL stay exactly as they were. Success is only reported after a manifest
        has named the new generation and the install (rename the new table in, delete the old
        tables, clear the WAL) has completed durably. A crash at any earlier point leaves no
        manifest, and the next open sweeps the temp files: the pre-compaction layout (old tables
        plus the WAL) is then the truth. A crash after the manifest resumes the same idempotent
        install. Both layouts resolve to identical logical data.
        """
        removed = [table.path for table in self.tables]
        number = self.next_table_number
        final_path = self._table_path(number)
        temp_path = self._compaction_temp_path(number)
        manifest_path = self._manifest_path(number)
        keys = 0
        writer = TableWriter(temp_path)
        try:
            for key, value in self._merged_rows():
                if value is None:
                    continue  # nothing older remains below this table: the tombstone is unreachable
                writer.add(key, value)
                keys += 1
            writer.finish()  # lands only at temp_path; no live-table name exists yet
        except OSError as error:
            # Nothing has been committed yet (no manifest, no live-table mutation), so the old
            # tables and the WAL remain the recoverable truth; normalise the filesystem failure.
            try:
                writer.abort()
            except OSError:
                pass
            raise OutputError("compaction output could not be written", path=temp_path) from error
        except BaseException:
            writer.abort()
            raise
        # The commit point: once this file is durable the new generation is recoverable and the
        # next open rolls forward even if this process dies immediately afterwards.
        self._write_manifest(
            manifest_path,
            {"number": number, "temp": os.path.basename(temp_path), "final": os.path.basename(final_path),
             "old": [os.path.basename(path) for path in removed]},
        )
        self._install_compaction(number)
        # Re-point the in-memory view at the table the install finished. The install already
        # cleared and reopened the WAL durably, so only the in-memory bookkeeping remains.
        self.tables = [SSTable.open(final_path)]
        self.memtable.clear()
        self.wal_records = 0
        return {"table": final_path, "keys": keys, "removedTables": len(removed)}

    # -- compaction crash recovery ------------------------------------------
    def _recover_compactions(self) -> None:
        """Deterministically settle every interrupted compaction before the store opens.

        A durable manifest means the new table was complete and committed: roll forward through
        the idempotent install. Anything else -- temp tables or partials with no manifest -- can
        only be pre-commit debris, so it is removed and the committed (old) tables are untouched.
        """
        manifests = self._manifest_paths()
        manifest_numbers = {number for number, _ in manifests}
        for _, manifest_path in manifests:
            plan = self._load_manifest(manifest_path)
            temp_path = os.path.join(self.directory, str(plan["temp"]))
            final_path = os.path.join(self.directory, str(plan["final"]))
            # The output must be a complete, sealed table. It was fsynced before the manifest was
            # published, so a checksum failure here is genuine corruption, not a torn write; a
            # missing output means the committed plan lost its payload and recovery cannot honour
            # the commit.
            output_path = final_path if os.path.exists(final_path) else temp_path
            try:
                SSTable.open(output_path)
            except FileNotFoundError as error:
                raise CorruptionError("compaction output is missing", path=output_path) from error
            self._promote_new_table(temp_path, final_path)
            for old_name in plan["old"]:
                old_path = os.path.join(self.directory, str(old_name))
                if os.path.exists(old_path):
                    self._durably_unlink(old_path)
            self._clear_wal_file()
            self._durably_unlink(manifest_path)
        # Rollback phase: nothing references these, so they are pre-commit debris (a crash before
        # the manifest). The old tables and the intact WAL still hold every acknowledged write.
        # A manifest .tmp is never itself the commit record, so it is swept unconditionally; the
        # table-shaped temp output is only debris while no manifest references its generation.
        # TableWriter spool/assembly leftovers (table-N.sst.tmp / .out, also from an older
        # release that named its output directly) can never be live tables either.
        for name in os.listdir(self.directory):
            if re.match(r"^compact-\d+\.manifest\.tmp$", name) or re.match(r"^table-\d+\.sst(?:\.tmp|\.out)$", name):
                self._durably_unlink(os.path.join(self.directory, name))
                continue
            match = re.match(r"^compaction-(\d+)\.new(?:\.tmp|\.out)?$", name)
            if match and int(match.group(1)) not in manifest_numbers:
                self._durably_unlink(os.path.join(self.directory, name))

    def _install_compaction(self, number: int) -> None:
        """Live commit: run the same roll-forward the next open would, against the open WAL.

        The WAL is retired while the manifest is still on disk, so a filesystem failure at any
        point leaves the manifest behind and the next open finishes the install deterministically;
        compact() itself only reports success once this returns.
        """
        manifest_path = self._manifest_path(number)
        plan = self._load_manifest(manifest_path)
        temp_path = os.path.join(self.directory, str(plan["temp"]))
        final_path = os.path.join(self.directory, str(plan["final"]))
        self._promote_new_table(temp_path, final_path)
        # The new table is the whole sealed truth now: the records the WAL still holds are exactly
        # the memtable rows already merged into it, so retire the log rather than replay them.
        self._clear_wal_for_compact()
        for old_name in plan["old"]:
            old_path = os.path.join(self.directory, str(old_name))
            if os.path.exists(old_path):
                self._durably_unlink(old_path)
        self._durably_unlink(manifest_path)

    def _promote_new_table(self, temp_path: str, final_path: str) -> None:
        """Idempotently put the sealed output at its live-table name (both steps repeatable)."""
        if not os.path.exists(final_path):
            self._durably_replace(temp_path, final_path)
        elif os.path.exists(temp_path):
            self._durably_unlink(temp_path)

    def _load_manifest(self, manifest_path: str) -> dict[str, object]:
        try:
            with open(manifest_path, "rb") as handle:
                raw = handle.read()
        except OSError as error:
            raise OutputError("compaction manifest could not be read", path=manifest_path) from error
        try:
            plan = json.loads(raw.decode("utf-8"))
            temp_name = str(plan["temp"])
            final_name = str(plan["final"])
            old_names = [str(name) for name in plan["old"]]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise CorruptionError("compaction manifest is unreadable", path=manifest_path) from error
        # Every name must stay inside the store directory and keep the documented shape; a plan
        # that names anything else cannot be trusted to drive deletes.
        names = [temp_name, final_name, *old_names]
        well_formed = bool(temp_name) and bool(final_name) and isinstance(plan["old"], list)
        well_formed &= all(os.path.dirname(name) == "" and name not in ("", ".", "..") for name in names)
        well_formed &= bool(re.match(r"^compaction-\d{6}\.new$", temp_name))
        well_formed &= bool(TABLE_PATTERN.match(final_name))
        well_formed &= all(TABLE_PATTERN.match(name) for name in old_names)
        if not well_formed:
            raise CorruptionError("compaction manifest is incomplete", path=manifest_path)
        return {"temp": temp_name, "final": final_name, "old": old_names}

    def _write_manifest(self, path: str, plan: dict[str, object]) -> None:
        """Atomically publish the commit record: rename + fsync of the file and its directory."""
        tmp_path = path + ".tmp"
        body = (json.dumps(plan, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            self._fsync_directory()
        except OSError as error:
            if os.path.exists(tmp_path):
                self._durably_unlink(tmp_path)
            raise OutputError("compaction manifest could not be committed", path=path) from error

    def _clear_wal_file(self) -> None:
        """Truncate the WAL file to zero bytes durably; used by startup recovery (log not open)."""
        try:
            with open(self.wal.path, "wb") as emptied:
                emptied.flush()
                os.fsync(emptied.fileno())
            self._fsync_directory()
        except OSError as error:
            raise OutputError("the write-ahead log could not be cleared", path=self.wal.path) from error

    def _clear_wal_for_compact(self) -> None:
        """Drop the WAL durably while this store owns an open append handle on it."""
        handle = self.wal._handle
        try:
            if handle is not None:
                handle.close()
                self.wal._handle = None
            self._clear_wal_file()
            self.wal._sequence = 0
            self.wal.open()
        except OutputError:
            raise
        except OSError as error:
            raise OutputError("the write-ahead log could not be cleared", path=self.wal.path) from error

    def _durably_replace(self, source: str, destination: str) -> None:
        try:
            os.replace(source, destination)
            self._fsync_directory()
        except OSError as error:
            raise OutputError("compacted table could not be installed", path=destination) from error

    def _durably_unlink(self, path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            return
        except OSError as error:
            raise OutputError("an obsolete store file could not be removed", path=path) from error
        try:
            self._fsync_directory()
        except OSError as error:
            raise OutputError("an obsolete store file could not be removed", path=path) from error

    def _fsync_directory(self) -> None:
        handle = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)

    def _manifest_paths(self) -> list[tuple[int, str]]:
        if not os.path.isdir(self.directory):
            return []
        found = []
        for name in os.listdir(self.directory):
            match = MANIFEST_PATTERN.match(name)
            if match:
                found.append((int(match.group(1)), os.path.join(self.directory, name)))
        return sorted(found)

    @staticmethod
    def _manifest_name(number: int) -> str:
        return f"compact-{number:06d}.manifest"

    def _manifest_path(self, number: int) -> str:
        return os.path.join(self.directory, self._manifest_name(number))

    @staticmethod
    def _compaction_temp_name(number: int) -> str:
        return f"compaction-{number:06d}{COMPACTION_TMP_SUFFIX}"

    def _compaction_temp_path(self, number: int) -> str:
        return os.path.join(self.directory, self._compaction_temp_name(number))


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
        os.makedirs(self.directory, exist_ok=True)
        # verify runs without open(), so settle an interrupted compaction the same way open does:
        # partial output must never be reported as a (logically wrong) collection of live tables.
        self._recover_compactions()
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
