"""Consistent-hash routing: a SHA-256 token ring with virtual nodes and ordered replicas.

The ring is a standalone, pure in-memory routing function. It never touches a `Store` directory,
reads or writes values, or otherwise participates in single-node persistence semantics: deriving
rings and planning rebalances only computes ownership, so a caller can review the plan before
moving any data.

Token mapping (fixed, UTF-8, identical on every platform):

* key token   -> ``SHA-256(key.encode("utf-8"))`` interpreted big-endian;
* vnode token -> ``SHA-256(node.encode("utf-8") + b"#" + str(index).encode("ascii"))``, with
  ``index`` running ``0 .. virtual_nodes-1`` per node.

Vnodes are kept in ascending token order. Two vnodes that hash to the same token never collapse:
they keep a deterministic tie-break on ``(node identifier, vnode index)``. A key belongs to the
first vnode at or clockwise after its token; walking the ring from there and skipping nodes
already seen yields the ordered, distinct replica list (wrapping past the highest token back to
the lowest), whose length is ``min(replicas, number of nodes)``.

Because construction sorts both the node set and the vnode table, the same set of nodes gives the
same primary and replica order regardless of input order, process, hash seed or platform.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections.abc import Iterable

from .errors import ValidationError

__all__ = ["HashRing"]


def _vnode_token(node: str, index: int) -> int:
    """Stable SHA-256 token of one virtual node (see module docstring for the wire format)."""
    digest = hashlib.sha256()
    digest.update(node.encode("utf-8"))
    digest.update(b"#")
    digest.update(str(index).encode("ascii"))
    return int.from_bytes(digest.digest(), "big")


def _key_token(key: str) -> int:
    """Stable SHA-256 token of a routed key."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")


class HashRing:
    """An immutable consistent-hash ring over non-empty string node identifiers.

    Build one from any iterable of distinct, non-empty node identifiers plus positive
    ``virtual_nodes`` and ``replicas`` counts; derive changed topologies with `with_node` and
    `without_node` (the receiver always stays untouched); route with `nodes_for`; compare two
    rings over a key set with `rebalance_plan`.
    """

    __slots__ = ("nodes", "virtual_nodes", "replicas", "_vnodes", "_tokens")

    def __init__(
        self,
        nodes: Iterable[str],
        virtual_nodes: int,
        replicas: int,
    ) -> None:
        if not _is_positive_int(virtual_nodes):
            raise ValidationError("virtual_nodes must be a positive integer", value=virtual_nodes)
        if not _is_positive_int(replicas):
            raise ValidationError("replicas must be a positive integer", value=replicas)
        members: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            if not isinstance(node, str) or not node:
                raise ValidationError("node identifiers must be non-empty strings", node=node)
            if node in seen:
                raise ValidationError("node identifiers must be unique", node=node)
            seen.add(node)
            members.append(node)
        if not members:
            raise ValidationError("nodes must contain at least one node")
        # Sorting the membership is what makes rings built from the same set in different input
        # orders equal and route identically on every platform.
        members.sort()
        self.nodes: tuple[str, ...] = tuple(members)
        self.virtual_nodes: int = virtual_nodes
        self.replicas: int = replicas
        vnodes = [
            (_vnode_token(node, index), node, index)
            for node in self.nodes
            for index in range(virtual_nodes)
        ]
        # Token first; (node, index) after it so a token collision keeps every vnode in one fixed
        # order instead of dropping the loser or relying on platform-dependent ordering.
        vnodes.sort(key=lambda item: (item[0], item[1], item[2]))
        self._vnodes: tuple[tuple[int, str, int], ...] = tuple(vnodes)
        self._tokens: tuple[int, ...] = tuple(token for token, _, _ in vnodes)

    # -- routing -------------------------------------------------------------
    def nodes_for(self, key: str) -> list[str]:
        """Return the ordered, distinct nodes responsible for ``key``.

        The first element is the primary; the following elements are the clockwise replicas. The
        result length is ``min(replicas, len(nodes))`` and the walk wraps from the highest token
        back to the lowest.
        """
        _require_non_empty_key(key)
        token = _key_token(key)
        wanted = min(self.replicas, len(self.nodes))
        start = bisect_left(self._tokens, token)
        if start == len(self._vnodes):  # past the last token: wrap to the ring head
            start = 0
        ordered: list[str] = []
        seen: set[str] = set()
        size = len(self._vnodes)
        for step in range(size):
            _, node, _ = self._vnodes[(start + step) % size]
            if node not in seen:
                seen.add(node)
                ordered.append(node)
                if len(ordered) == wanted:
                    break
        return ordered

    # -- topology derivation -------------------------------------------------
    def with_node(self, node: str) -> "HashRing":
        """Return a new ring that also contains ``node``; this ring is not modified."""
        _require_non_empty_node(node)
        if node in self.nodes:
            raise ValidationError("node is already a member of the ring", node=node)
        return HashRing((*self.nodes, node), self.virtual_nodes, self.replicas)

    def without_node(self, node: str) -> "HashRing":
        """Return a new ring without ``node``; this ring is not modified.

        Removing the last remaining node is refused so no unqueryable ring can be produced.
        """
        _require_non_empty_node(node)
        if node not in self.nodes:
            raise ValidationError("node is not a member of the ring", node=node)
        if len(self.nodes) == 1:
            raise ValidationError("cannot remove the last node from the ring", node=node)
        return HashRing(
            tuple(member for member in self.nodes if member != node),
            self.virtual_nodes,
            self.replicas,
        )

    # -- rebalance planning --------------------------------------------------
    def rebalance_plan(self, new_ring: "HashRing", keys: Iterable[str]) -> list[dict[str, object]]:
        """Compare this (old) ring against ``new_ring`` for ``keys`` and list what must move.

        Duplicate keys are merged and the plan is emitted in Unicode (code point) key order, one
        entry per key whose responsible sequence changed. Each entry records the old and new
        ordered node sequences plus the nodes that must receive a copy (``addedNodes``) and the
        nodes that must drop theirs (``revokedNodes``). Nothing is read or written: callers are
        expected to review this before transferring values themselves.
        """
        if not isinstance(new_ring, HashRing):
            raise TypeError("new_ring must be a HashRing")
        unique: set[str] = set()
        for key in keys:
            _require_non_empty_key(key)
            unique.add(key)
        plan: list[dict[str, object]] = []
        for key in sorted(unique):
            old_nodes = self.nodes_for(key)
            new_nodes = new_ring.nodes_for(key)
            if old_nodes == new_nodes:
                continue  # identical responsible set and order: nothing to transfer
            plan.append(
                {
                    "key": key,
                    "oldNodes": old_nodes,
                    "newNodes": new_nodes,
                    "addedNodes": sorted(set(new_nodes) - set(old_nodes)),
                    "revokedNodes": sorted(set(old_nodes) - set(new_nodes)),
                }
            )
        return plan

    # -- value semantics ------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashRing):
            return NotImplemented
        return (
            self.nodes == other.nodes
            and self.virtual_nodes == other.virtual_nodes
            and self.replicas == other.replicas
        )

    def __hash__(self) -> int:
        return hash((self.nodes, self.virtual_nodes, self.replicas))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"HashRing(nodes={list(self.nodes)!r}, virtual_nodes={self.virtual_nodes}, "
            f"replicas={self.replicas})"
        )


def _is_positive_int(value: object) -> bool:
    # bool is a subclass of int but is never a meaningful vnode/replica count.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require_non_empty_key(key: str) -> None:
    if not isinstance(key, str) or not key:
        raise ValidationError("key must be a non-empty string", key=key)


def _require_non_empty_node(node: str) -> None:
    if not isinstance(node, str) or not node:
        raise ValidationError("node identifiers must be non-empty strings", node=node)
