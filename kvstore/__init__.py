"""A small LSM key/value store: write-ahead log, sorted tables, snapshots.

The public surface is deliberately narrow: `Store` for reads and writes, `Snapshot` for consistent
reads, and one exception hierarchy whose `kind` is stable so callers never match on message text.
"""

from .errors import CorruptionError, KVError, OutputError, ParseError, ValidationError
from .store import Snapshot, Store, StoreStats

__all__ = [
    "CorruptionError",
    "KVError",
    "OutputError",
    "ParseError",
    "Snapshot",
    "Store",
    "StoreStats",
    "ValidationError",
]

__version__ = "0.1.0"
