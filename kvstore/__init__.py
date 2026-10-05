"""A small LSM key/value store: write-ahead log, sorted tables, snapshots.

The public surface is deliberately narrow: `Store` for reads and writes, `Snapshot` for consistent
reads, and one exception hierarchy whose `kind` is stable so callers never match on message text.
`HashRing` is an independent routing layer: it decides consistent-hash ownership and rebalance
plans without touching the on-disk store.
"""

from .errors import CorruptionError, KVError, OutputError, ParseError, ValidationError
from .routing import HashRing, KeyMigration, RebalancePlan
from .store import Snapshot, Store, StoreStats

__all__ = [
    "CorruptionError",
    "HashRing",
    "KVError",
    "KeyMigration",
    "OutputError",
    "ParseError",
    "RebalancePlan",
    "Snapshot",
    "Store",
    "StoreStats",
    "ValidationError",
]

__version__ = "0.1.0"
