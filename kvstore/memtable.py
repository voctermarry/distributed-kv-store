"""In-memory table: the newest writes, held until they are sealed into a sorted table.

`None` is a tombstone, not an absent key -- the distinction is what lets compaction drop deletes
only once every older copy has been merged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MemTable:
    limit_bytes: int = 1 << 20
    entries: dict[str, str | None] = field(default_factory=dict)
    bytes_used: int = field(default=0, init=False)

    def put(self, key: str, value: str) -> None:
        self._store(key, value)

    def delete(self, key: str) -> None:
        self._store(key, None)

    def get(self, key: str) -> tuple[bool, str | None]:
        """Return `(found, value)`; a tombstone is found with value None."""
        if key in self.entries:
            return True, self.entries[key]
        return False, None

    def items(self) -> list[tuple[str, str | None]]:
        return sorted(self.entries.items())

    def is_full(self) -> bool:
        return self.bytes_used >= self.limit_bytes

    def clear(self) -> None:
        self.entries.clear()
        self.bytes_used = 0

    def _store(self, key: str, value: str | None) -> None:
        previous = self.entries.get(key)
        if previous is not None:
            self.bytes_used -= len(key) + len(previous)
        elif key in self.entries:
            self.bytes_used -= len(key)
        self.entries[key] = value
        self.bytes_used += len(key) + (len(value) if value is not None else 0)
