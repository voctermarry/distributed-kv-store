"""Consistent-hash routing: a pure, store-independent decision layer.

A :class:`HashRing` maps every non-empty key to an ordered list of *distinct* responsible nodes
(primary first, then clockwise replicas) using a virtual-node ring fixed by SHA-256. Nothing here
touches the on-disk store: rings are immutable value objects, :meth:`HashRing.with_node` and
:meth:`HashRing.without_node` derive new rings, and :meth:`HashRing.plan_rebalance` produces a
deterministic, review-before-transfer migration plan without moving a single value.

Stability contract:

* Node ids and keys enter SHA-256 as their UTF-8 bytes, so the same ring routes identically no
  matter the input ordering, process or platform.
* Virtual-node positions carry ``(node, replica_index)``; a hash-token collision is resolved by
  that pair's lexicographic order, so colliding vnodes are never dropped.
* The ring does not depend on insertion order -- construction deduplicates and sorts node ids.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass

from .errors import ValidationError


def _digest(value: str) -> int:
    """Map one string to a fixed 256-bit token via its UTF-8 bytes."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big")


def _virtual_token(node: str, index: int) -> int:
    # The index is decimal, plain ASCII for every non-negative int and unambiguous inside the
    # vnode namespace (keys are hashed bare, so no key token can depend on this spelling).
    return _digest(f"{node}#{index}")


@dataclass(frozen=True, slots=True)
class KeyMigration:
    """One key whose responsible sequence differs between an old ring and a new one."""

    key: str
    old_nodes: tuple[str, ...]
    new_nodes: tuple[str, ...]
    add_nodes: tuple[str, ...]
    remove_nodes: tuple[str, ...]

    def to_document(self) -> dict[str, object]:
        return {
            "key": self.key,
            "oldNodes": list(self.old_nodes),
            "newNodes": list(self.new_nodes),
            "addNodes": list(self.add_nodes),
            "removeNodes": list(self.remove_nodes),
        }


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    """The deterministic migration for a set of keys between two rings."""

    joined: tuple[str, ...]
    left: tuple[str, ...]
    migrations: tuple[KeyMigration, ...]

    @property
    def changes(self) -> int:
        return len(self.migrations)

    def __iter__(self):
        return iter(self.migrations)

    def __len__(self) -> int:
        return len(self.migrations)

    def to_document(self) -> dict[str, object]:
        return {
            "joined": list(self.joined),
            "left": list(self.left),
            "migrations": [migration.to_document() for migration in self.migrations],
        }


class HashRing:
    """An immutable consistent-hash ring over a set of non-empty node ids.

    Parameters are validated once at construction. Deriving a ring (:meth:`with_node`,
    :meth:`without_node`) re-validates the result, so no object can ever represent an empty or
    otherwise unqueryable ring.
    """

    __slots__ = ("_nodes", "_virtual_nodes", "_tokens", "_replicas")

    def __init__(
        self,
        nodes: Iterable[str],
        virtual_nodes: int,
        replicas: int,
        *,
        _nodes: frozenset[str] | None = None,
    ) -> None:
        if isinstance(nodes, str):
            # A bare string would otherwise be accepted as an iterable of one-character "nodes".
            raise ValidationError("nodes must be a non-empty collection of non-empty strings")
        try:
            node_list = list(nodes)
        except TypeError as error:
            raise ValidationError("nodes must be a non-empty collection of non-empty strings") from error
        if not all(isinstance(node, str) for node in node_list):
            raise ValidationError("every node id must be a string")
        if any(not node for node in node_list):
            raise ValidationError("node ids must be non-empty", node="")
        if len(node_list) != len(set(node_list)):
            duplicate = next(node for index, node in enumerate(node_list) if node in node_list[:index])
            raise ValidationError("node ids must be unique", node=duplicate)
        if not isinstance(virtual_nodes, int) or isinstance(virtual_nodes, bool) or virtual_nodes <= 0:
            raise ValidationError("virtual_nodes must be a positive integer", value=virtual_nodes)
        if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas <= 0:
            raise ValidationError("replicas must be a positive integer", value=replicas)

        ordered = frozenset(_nodes) if _nodes is not None else frozenset(node_list)
        if not ordered:
            raise ValidationError("at least one node is required")
        # Sorted, so two rings built from the same set in a different input order are identical.
        node_ids = sorted(ordered)
        # (token, node, index): sorting the full triple gives the stable collision order -- the
        # owning node id, then the vnode index -- without ever dropping a colliding vnode.
        points = sorted(
            (_virtual_token(node, index), node, index)
            for node in node_ids
            for index in range(virtual_nodes)
        )
        self._nodes: tuple[str, ...] = tuple(node_ids)
        self._virtual_nodes = virtual_nodes
        self._tokens: tuple[tuple[int, str, int], ...] = tuple(points)
        self._replicas = replicas

    # -- properties ----------------------------------------------------------
    @property
    def nodes(self) -> tuple[str, ...]:
        """The member node ids in sorted order."""
        return self._nodes

    @property
    def virtual_nodes(self) -> int:
        return self._virtual_nodes

    @property
    def replicas(self) -> int:
        return self._replicas

    @property
    def size(self) -> int:
        return len(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: object) -> bool:
        return isinstance(node, str) and node in self._nodes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashRing):
            return NotImplemented
        return (
            self._nodes == other._nodes
            and self._virtual_nodes == other._virtual_nodes
            and self._replicas == other._replicas
        )

    def __hash__(self) -> int:
        return hash((self._nodes, self._virtual_nodes, self._replicas))

    def __repr__(self) -> str:
        return f"HashRing(nodes={list(self._nodes)!r}, virtual_nodes={self._virtual_nodes}, replicas={self._replicas})"

    # -- routing -------------------------------------------------------------
    def nodes_for(self, key: str) -> list[str]:
        """Return the distinct responsible nodes for ``key``, primary first.

        The primary owns the first virtual node at or clockwise-after the key's token; the
        sequence then walks the ring clockwise, recording each *new* node, wrapping from the
        ring head as needed. Length is ``min(replicas, node_count)``.
        """
        if not isinstance(key, str) or not key:
            raise ValidationError("key must be a non-empty string", value=key)
        token = _digest(key)
        first = bisect_left(self._tokens, (token, "", -1)) % len(self._tokens)
        wanted = min(self._replicas, len(self._nodes))
        chosen: list[str] = []
        seen: set[str] = set()
        steps = 0
        while len(chosen) < wanted:
            _, node, _ = self._tokens[(first + steps) % len(self._tokens)]
            steps += 1
            if node not in seen:
                seen.add(node)
                chosen.append(node)
        return chosen

    def primary_node(self, key: str) -> str:
        """The single node owning ``key`` (the first entry of :meth:`nodes_for`)."""
        return self.nodes_for(key)[0]

    # -- derivation -----------------------------------------------------------
    def with_node(self, node: str) -> "HashRing":
        """Return a copy of this ring that also contains ``node``."""
        if not isinstance(node, str) or not node:
            raise ValidationError("node id must be a non-empty string", node=node if isinstance(node, str) else "")
        if node in self._nodes:
            raise ValidationError("node is already a member of the ring", node=node)
        return HashRing(
            self._nodes,
            self._virtual_nodes,
            self._replicas,
            _nodes=frozenset(self._nodes) | {node},
        )

    def without_node(self, node: str) -> "HashRing":
        """Return a copy of this ring with ``node`` removed.

        Removing the sole member fails: the result would not be queryable.
        """
        if not isinstance(node, str) or not node:
            raise ValidationError("node id must be a non-empty string", node=node if isinstance(node, str) else "")
        if node not in self._nodes:
            raise ValidationError("node is not a member of the ring", node=node)
        if len(self._nodes) == 1:
            raise ValidationError("cannot remove the last node", node=node)
        return HashRing(
            self._nodes,
            self._virtual_nodes,
            self._replicas,
            _nodes=frozenset(self._nodes) - {node},
        )

    # -- planning -------------------------------------------------------------
    def plan_rebalance(
        self,
        keys: Iterable[str],
        new_ring: "HashRing",
        *,
        joined: Iterable[str] | None = None,
        left: Iterable[str] | None = None,
    ) -> RebalancePlan:
        """Plan, deterministically, how ``keys`` move from this ring to ``new_ring``.

        Keys are deduplicated and sorted by Unicode code-point order; each key appears at most
        once. Empty keys raise the same :class:`ValidationError` as a routing query. A key enters
        the plan only when its responsible sequence (members *and* order) changes; each migration
        records both sequences plus the nodes to add and to withdraw.
        """
        try:
            key_list = list(keys)
        except TypeError as error:
            raise ValidationError("keys must be an iterable of non-empty strings") from error
        if not all(isinstance(key, str) for key in key_list):
            raise ValidationError("every key must be a string")
        if any(not key for key in key_list):
            raise ValidationError("key must be a non-empty string", value="")
        if self._virtual_nodes != new_ring._virtual_nodes or self._replicas != new_ring._replicas:
            raise ValidationError(
                "rebalance rings must share virtual_nodes and replicas",
                virtualNodes=self._virtual_nodes,
                newVirtualNodes=new_ring._virtual_nodes,
                replicas=self._replicas,
                newReplicas=new_ring._replicas,
            )

        migrations: list[KeyMigration] = []
        for key in sorted(set(key_list)):
            old = tuple(self.nodes_for(key))
            new = tuple(new_ring.nodes_for(key))
            if old == new:
                continue  # identical responsible set and order: nothing to transfer
            migrations.append(
                KeyMigration(
                    key=key,
                    old_nodes=old,
                    new_nodes=new,
                    add_nodes=tuple(node for node in new if node not in old),
                    remove_nodes=tuple(node for node in old if node not in new),
                )
            )
        return RebalancePlan(
            joined=tuple(sorted(joined)) if joined is not None else (),
            left=tuple(sorted(left)) if left is not None else (),
            migrations=tuple(migrations),
        )
