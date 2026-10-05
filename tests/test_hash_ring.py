"""HashRing routing: fixed SHA-256 mapping, stable replica order, deterministic rebalance plans.

These tests pin the contract that future multi-node replication will rely on: the same node set
routes identically regardless of input order, process or platform; membership changes are pure
derivations that do not touch disk; and rebalance plans are the exact minimal set of ownership
changes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

from kvstore import HashRing
from kvstore import hash_ring as hash_ring_module
from kvstore.errors import ValidationError

KEY_SPACE = [f"key-{i:04d}" for i in range(400)]
UNICODE_KEYS = ["z", "é", "中", "A", "a"]


def sha_token(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")


def reference_vnode_token(node: str, index: int) -> int:
    """Independent copy of the documented vnode wire format: SHA-256(node + '#' + str(index))."""
    return int.from_bytes(
        hashlib.sha256(node.encode("utf-8") + b"#" + str(index).encode("ascii")).digest(), "big"
    )


def reference_route(nodes: list[str], virtual_nodes: int, replicas: int, key: str) -> list[str]:
    """A straightforward re-implementation of the clockwise walk, used to pin the exact mapping."""
    points = sorted(
        (reference_vnode_token(node, index), node, index)
        for node in nodes
        for index in range(virtual_nodes)
    )
    key_token = sha_token(key)
    starts = [i for i, point in enumerate(points) if point[0] >= key_token]
    cursor = starts[0] if starts else 0
    ordered: list[str] = []
    wanted = min(replicas, len(nodes))
    for step in range(len(points)):
        node = points[(cursor + step) % len(points)][1]
        if node not in ordered:
            ordered.append(node)
            if len(ordered) == wanted:
                break
    return ordered


class ConstructionTests(unittest.TestCase):
    def test_membership_is_sorted_and_input_order_is_irrelevant(self) -> None:
        ring = HashRing(["gamma", "alpha", "beta"], 32, 2)
        self.assertEqual(ring.nodes, ("alpha", "beta", "gamma"))
        self.assertEqual(
            HashRing(["gamma", "alpha", "beta"], 32, 2),
            HashRing(["beta", "gamma", "alpha"], 32, 2),
        )

    def test_accepts_generators(self) -> None:
        ring = HashRing(iter(["x", "y"]), 4, 1)
        self.assertEqual(ring.nodes, ("x", "y"))

    def test_vnode_count_matches_nodes_times_virtual_nodes(self) -> None:
        ring = HashRing(["a", "b", "c"], 7, 2)
        self.assertEqual(len(ring._vnodes), 21)
        self.assertEqual(ring._tokens, tuple(token for token, _, _ in ring._vnodes))
        tokens = list(ring._tokens)
        self.assertEqual(tokens, sorted(tokens))  # tokens are kept in ascending ring order

    def test_routing_matches_the_reference_mapping(self) -> None:
        nodes = ["n0", "n1", "n2", "n3"]
        ring = HashRing(nodes, 19, 3)
        for key in KEY_SPACE:
            self.assertEqual(ring.nodes_for(key), reference_route(nodes, 19, 3, key))

    def test_first_result_is_the_clockwise_primary(self) -> None:
        ring = HashRing(["p", "q", "r"], 40, 1)
        for key in KEY_SPACE:
            self.assertEqual(ring.nodes_for(key), reference_route(["p", "q", "r"], 40, 1, key))


class ReplicaTests(unittest.TestCase):
    def test_replicas_are_distinct_and_clockwise(self) -> None:
        nodes = ["a", "b", "c", "d"]
        ring = HashRing(nodes, 32, 4)
        for key in KEY_SPACE:
            result = ring.nodes_for(key)
            self.assertEqual(result, reference_route(nodes, 32, 4, key))
            self.assertEqual(len(result), len(set(result)))

    def test_length_is_min_of_replicas_and_nodes_with_wraparound(self) -> None:
        ring = HashRing(["a", "b"], 8, 5)
        for key in KEY_SPACE[:25]:
            result = ring.nodes_for(key)
            self.assertEqual(len(result), 2)
            self.assertEqual(set(result), {"a", "b"})
        single = HashRing(["only"], 4, 3)
        self.assertEqual(single.nodes_for("anything"), ["only"])

    def test_utf8_keys_and_nodes_route_stably(self) -> None:
        ring = HashRing(["节点一", "node-ä", "plain"], 16, 2)
        again = HashRing(["plain", "node-ä", "节点一"], 16, 2)
        for key in ["ünïcode", "中文键", "ascii", "😀"]:
            self.assertEqual(ring.nodes_for(key), again.nodes_for(key))
            self.assertEqual(ring.nodes_for(key), reference_route(list(ring.nodes), 16, 2, key))


class CollisionTieBreakTests(unittest.TestCase):
    def test_token_collisions_keep_every_vnode_in_node_index_order(self) -> None:
        # Force a#3 and b#0 onto the same token; the collision must neither drop a vnode nor let
        # ordering depend on insertion order: (node, index) is the fixed tie-break.
        real = hash_ring_module._vnode_token
        sentinel = 1 << 200

        def forced(node: str, index: int) -> int:
            if (node, index) in {("a", 3), ("b", 0)}:
                return sentinel
            return real(node, index)

        with patch.object(hash_ring_module, "_vnode_token", forced):
            ring = HashRing(["c", "b", "a"], 4, 3)
        self.assertEqual(len(ring._vnodes), 12)  # no vnode lost in the collision
        collided = [entry for entry in ring._vnodes if entry[0] == sentinel]
        self.assertEqual(collided, [(sentinel, "a", 3), (sentinel, "b", 0)])
        # Routing still returns distinct, bounded replicas with both colliding owners alive.
        for key in KEY_SPACE[:50]:
            result = ring.nodes_for(key)
            self.assertEqual(len(result), len(set(result)))
            self.assertLessEqual(len(result), 3)


class DerivationTests(unittest.TestCase):
    def test_with_node_returns_a_new_ring_and_keeps_the_old_one(self) -> None:
        old = HashRing(["a", "b", "c"], 32, 2)
        new = old.with_node("d")
        self.assertEqual(old.nodes, ("a", "b", "c"))
        self.assertEqual(new.nodes, ("a", "b", "c", "d"))
        self.assertEqual((new.virtual_nodes, new.replicas), (32, 2))
        self.assertIsNot(old, new)

    def test_without_node_returns_a_new_ring_and_keeps_the_old_one(self) -> None:
        old = HashRing(["a", "b", "c"], 32, 2)
        new = old.without_node("b")
        self.assertEqual(old.nodes, ("a", "b", "c"))
        self.assertEqual(new.nodes, ("a", "c"))
        self.assertEqual((new.virtual_nodes, new.replicas), (32, 2))

    def test_removal_is_order_independent_derivation(self) -> None:
        one = HashRing(["a", "b", "c"], 16, 2).without_node("a")
        other = HashRing(("c", "b"), 16, 2)
        self.assertEqual(one, other)

    def test_join_and_remove_round_trip_to_an_equal_ring(self) -> None:
        ring = HashRing(["a", "b", "c"], 32, 2)
        self.assertEqual(ring.with_node("d").without_node("d"), ring)


class JoinSemanticsTests(unittest.TestCase):
    def test_only_keys_taken_over_by_the_new_node_change_primary(self) -> None:
        old = HashRing(["n1", "n2", "n3", "n4"], 64, 1)
        new = old.with_node("n5")
        plan = old.rebalance_plan(new, KEY_SPACE)
        moved = {entry["key"] for entry in plan}
        for key in KEY_SPACE:
            before, after = old.nodes_for(key), new.nodes_for(key)
            if before != after:
                self.assertIn(key, moved)
                self.assertEqual(after, ["n5"])  # the new node is the only possible new primary
            else:
                self.assertNotIn(key, moved)

    def test_join_only_affects_the_new_vnodes_clockwise_intervals(self) -> None:
        old = HashRing(["h1", "h2", "h3"], 48, 1)
        joined = "h4"
        new = old.with_node(joined)
        # Merge every vnode into one ring. Each joining vnode takes over the half-open arc from
        # its clockwise predecessor (exclusive) to its own token (inclusive); a key changes
        # primary precisely when its token falls in the union of those arcs (wrapping included).
        merged = sorted(
            list(old._vnodes)
            + [(reference_vnode_token(joined, i), joined, i) for i in range(48)]
        )
        takeover: list[tuple[int, int]] = []
        for position, (token, node, _) in enumerate(merged):
            if node != joined:
                continue
            predecessor = merged[position - 1][0]
            takeover.append((predecessor, token))

        def in_takeover(token: int) -> bool:
            for lower, upper in takeover:
                if lower < upper:  # ordinary arc
                    if lower < token <= upper:
                        return True
                else:  # the arc wraps past the top of the ring
                    if token > lower or token <= upper:
                        return True
            return False

        for key in KEY_SPACE:
            token = sha_token(key)
            self.assertEqual(new.nodes_for(key)[0] == joined, in_takeover(token))

    def test_join_with_replicas_recomputes_the_whole_sequence(self) -> None:
        old = HashRing(["a", "b", "c"], 32, 3)
        new = old.with_node("d")
        for entry in old.rebalance_plan(new, KEY_SPACE):
            self.assertEqual(entry["newNodes"], new.nodes_for(entry["key"]))
            self.assertEqual(entry["oldNodes"], old.nodes_for(entry["key"]))
            self.assertEqual(
                entry["addedNodes"], sorted(set(entry["newNodes"]) - set(entry["oldNodes"]))
            )
            self.assertEqual(
                entry["revokedNodes"], sorted(set(entry["oldNodes"]) - set(entry["newNodes"]))
            )


class RemoveSemanticsTests(unittest.TestCase):
    def test_removed_primaries_transfer_clockwise_and_others_do_not_drift(self) -> None:
        ring = HashRing(["a", "b", "c", "d"], 64, 1)
        survivor = ring.without_node("c")
        for key in KEY_SPACE:
            before, after = ring.nodes_for(key), survivor.nodes_for(key)
            if before == ["c"]:
                self.assertNotEqual(after, ["c"])
                self.assertIn(after[0], {"a", "b", "d"})
                self.assertEqual(after, _clockwise_skipping(ring, key, {"c"}))
            else:
                self.assertEqual(before, after, (key, before, after))

    def test_replicas_after_removal_are_distinct_survivors(self) -> None:
        ring = HashRing(["a", "b", "c"], 32, 3)
        survivor = ring.without_node("b")
        for key in KEY_SPACE:
            result = survivor.nodes_for(key)
            self.assertEqual(len(result), 2)
            self.assertEqual(set(result), {"a", "c"})
            self.assertEqual(result, _clockwise_skipping(ring, key, {"b"}))


class RebalancePlanTests(unittest.TestCase):
    def test_plan_is_in_unicode_key_order_with_duplicates_merged(self) -> None:
        old = HashRing(["a", "b", "c"], 32, 2)
        new = old.with_node("d")
        shuffled = ["中", "a", "z", "a", "é", "中", "A"]
        plan = old.rebalance_plan(new, shuffled)
        keys = [entry["key"] for entry in plan]
        self.assertEqual(keys, sorted(keys))  # Unicode code point order
        self.assertEqual(len(keys), len(set(keys)))  # each key appears at most once
        self.assertTrue(set(keys) <= set(shuffled))
        # The plan contains exactly the unique keys whose responsible sequence changed.
        changed = {key for key in set(shuffled) if old.nodes_for(key) != new.nodes_for(key)}
        self.assertEqual(set(keys), changed)

    def test_unicode_ordering_predates_changes(self) -> None:
        old = HashRing(["a", "b", "c", "d", "e"], 64, 1)
        new = old.with_node("f")
        plan = old.rebalance_plan(new, UNICODE_KEYS * 2)
        keys = [entry["key"] for entry in plan]
        self.assertEqual(keys, sorted(keys))
        self.assertTrue(set(keys) <= set(UNICODE_KEYS))

    def test_unchanged_sequences_are_excluded(self) -> None:
        old = HashRing(["a", "b", "c"], 32, 2)
        same = HashRing(["c", "b", "a"], 32, 2)
        self.assertEqual(old.rebalance_plan(same, KEY_SPACE), [])

    def test_entry_shape_and_diff_fields(self) -> None:
        old = HashRing(["a", "b", "c"], 16, 3)
        new = old.without_node("c")
        entries = old.rebalance_plan(new, ["k1", "k2", "k3", "k4"])
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(
                set(entry), {"key", "oldNodes", "newNodes", "addedNodes", "revokedNodes"}
            )
            self.assertEqual(
                entry["addedNodes"], sorted(set(entry["newNodes"]) - set(entry["oldNodes"]))
            )
            self.assertEqual(
                entry["revokedNodes"], sorted(set(entry["oldNodes"]) - set(entry["newNodes"]))
            )
            self.assertEqual(entry["oldNodes"], old.nodes_for(entry["key"]))
            self.assertEqual(entry["newNodes"], new.nodes_for(entry["key"]))

    def test_duplicate_plan_keys_report_one_entry(self) -> None:
        old = HashRing(["a", "b"], 16, 1)
        new = old.with_node("c")
        plan = old.rebalance_plan(new, ["only", "only", "only"])
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["key"], "only")

    def test_empty_key_in_plan_input_raises_like_a_query(self) -> None:
        old = HashRing(["a", "b"], 16, 1)
        new = old.with_node("c")
        with self.assertRaises(ValidationError) as caught:
            old.rebalance_plan(new, ["ok", "", "also"])
        self.assertEqual(caught.exception.context.get("key"), "")

    def test_planning_and_derivation_write_nothing(self) -> None:
        old = HashRing(["a", "b", "c"], 16, 2)
        with tempfile.TemporaryDirectory() as empty:
            cwd = os.getcwd()
            os.chdir(empty)
            try:
                new = old.with_node("d").without_node("a")
                old.rebalance_plan(new, KEY_SPACE)
            finally:
                os.chdir(cwd)
            self.assertEqual(os.listdir(empty), [])


class ValidationTests(unittest.TestCase):
    def assert_validation(self, callable_: object, **context: object) -> ValidationError:
        with self.assertRaises(ValidationError) as caught:
            callable_()  # type: ignore[operator]
        for key, value in context.items():
            self.assertEqual(caught.exception.context.get(key), value, caught.exception.context)
        return caught.exception

    def test_empty_node_set(self) -> None:
        self.assert_validation(lambda: HashRing([], 1, 1))

    def test_empty_node_identifier(self) -> None:
        self.assert_validation(lambda: HashRing(["a", ""], 1, 1), node="")

    def test_non_string_node_identifier(self) -> None:
        self.assert_validation(lambda: HashRing(["a", 3], 1, 1), node=3)  # type: ignore[list-item]

    def test_duplicate_node_identifier(self) -> None:
        self.assert_validation(lambda: HashRing(["a", "b", "a"], 1, 1), node="a")

    def test_non_positive_virtual_nodes(self) -> None:
        self.assert_validation(lambda: HashRing(["a"], 0, 1))
        self.assert_validation(lambda: HashRing(["a"], -3, 1))

    def test_non_positive_replicas(self) -> None:
        self.assert_validation(lambda: HashRing(["a"], 1, 0))
        self.assert_validation(lambda: HashRing(["a"], 1, -2))

    def test_counts_must_be_real_integers(self) -> None:
        self.assert_validation(lambda: HashRing(["a"], True, 1))  # bool is not a count
        self.assert_validation(lambda: HashRing(["a"], 1.0, 1))
        self.assert_validation(lambda: HashRing(["a"], 1, False))
        self.assert_validation(lambda: HashRing(["a"], "2", 1))  # type: ignore[arg-type]

    def test_empty_query_key(self) -> None:
        ring = HashRing(["a"], 4, 1)
        self.assert_validation(lambda: ring.nodes_for(""), key="")

    def test_join_existing_node(self) -> None:
        ring = HashRing(["a", "b"], 4, 1)
        self.assert_validation(lambda: ring.with_node("a"), node="a")

    def test_join_empty_node(self) -> None:
        ring = HashRing(["a"], 4, 1)
        self.assert_validation(lambda: ring.with_node(""), node="")

    def test_remove_unknown_node(self) -> None:
        ring = HashRing(["a", "b"], 4, 1)
        self.assert_validation(lambda: ring.without_node("ghost"), node="ghost")

    def test_remove_empty_node(self) -> None:
        ring = HashRing(["a", "b"], 4, 1)
        self.assert_validation(lambda: ring.without_node(""), node="")

    def test_removing_the_last_node_fails_and_keeps_the_ring(self) -> None:
        ring = HashRing(["solo"], 4, 1)
        self.assert_validation(lambda: ring.without_node("solo"), node="solo")
        self.assertEqual(ring.nodes, ("solo",))
        self.assertEqual(ring.nodes_for("still-queryable"), ["solo"])


def _clockwise_skipping(ring: HashRing, key: str, banned: set[str]) -> list[str]:
    """Reference: walk the *old* ring clockwise, dropping vnodes of removed node(s)."""
    key_token = sha_token(key)
    points = ring._vnodes
    starts = [i for i, point in enumerate(points) if point[0] >= key_token]
    cursor = starts[0] if starts else 0
    ordered: list[str] = []
    wanted = min(ring.replicas, len(ring.nodes) - len(banned))
    for step in range(len(points)):
        node = points[(cursor + step) % len(points)][1]
        if node in banned or node in ordered:
            continue
        ordered.append(node)
        if len(ordered) == wanted:
            break
    return ordered


if __name__ == "__main__":
    unittest.main()
