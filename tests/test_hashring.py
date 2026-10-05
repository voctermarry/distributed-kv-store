"""Consistent-hash routing: determinism, ownership order, derivation and rebalance plans."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
import unittest
from unittest import mock

from kvstore import HashRing
from kvstore.cli import main as cli_main
from kvstore.errors import ValidationError


def digest(value: str) -> int:
    """An independent reimplementation of the fixed token mapping, for cross-checking the ring."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big")


def reference_points(nodes, virtual_nodes):
    return sorted(
        (digest(f"{node}#{index}"), node, index)
        for node in nodes
        for index in range(virtual_nodes)
    )


def reference_route(points, key, replicas):
    """Clockwise walk from the key token, recording each distinct node; wraps at ring head."""
    start = __import__("bisect").bisect_left(points, (digest(key), "", -1)) % len(points)
    chosen: list[str] = []
    steps = 0
    while len(chosen) < min(replicas, len({point[1] for point in points})):
        _, node, _ = points[(start + steps) % len(points)]
        steps += 1
        if node not in chosen:
            chosen.append(node)
    return chosen


class ConstructionValidationTests(unittest.TestCase):
    def test_rejects_empty_node_collection(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            HashRing([], virtual_nodes=4, replicas=1)
        self.assertEqual(caught.exception.kind, "validation_error")

    def test_rejects_bare_string_as_nodes(self) -> None:
        with self.assertRaises(ValidationError):
            HashRing("alpha", virtual_nodes=4, replicas=1)  # type: ignore[arg-type]

    def test_rejects_non_iterable_nodes(self) -> None:
        with self.assertRaises(ValidationError):
            HashRing(7, virtual_nodes=4, replicas=1)  # type: ignore[arg-type]

    def test_rejects_empty_node_id_with_node_context(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            HashRing(["alpha", ""], virtual_nodes=4, replicas=1)
        self.assertEqual(caught.exception.context.get("node"), "")

    def test_rejects_non_string_node(self) -> None:
        with self.assertRaises(ValidationError):
            HashRing(["alpha", 3], virtual_nodes=4, replicas=1)  # type: ignore[list-item]

    def test_rejects_duplicate_nodes_with_node_context(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            HashRing(["alpha", "beta", "alpha"], virtual_nodes=4, replicas=2)
        self.assertEqual(caught.exception.context.get("node"), "alpha")

    def test_rejects_non_positive_virtual_nodes(self) -> None:
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    HashRing(["alpha"], virtual_nodes=bad, replicas=1)
        with self.assertRaises(ValidationError):
            HashRing(["alpha"], virtual_nodes=True, replicas=1)  # type: ignore[arg-type]

    def test_rejects_non_positive_replicas(self) -> None:
        for bad in (0, -3):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    HashRing(["alpha"], virtual_nodes=4, replicas=bad)
        with self.assertRaises(ValidationError):
            HashRing(["alpha"], virtual_nodes=4, replicas=False)  # type: ignore[arg-type]

    def test_rejects_empty_query_key(self) -> None:
        ring = HashRing(["alpha"], virtual_nodes=4, replicas=1)
        with self.assertRaises(ValidationError):
            ring.nodes_for("")

    def test_accepts_generator_of_nodes(self) -> None:
        ring = HashRing((node for node in ["alpha", "beta"]), virtual_nodes=2, replicas=1)
        self.assertEqual(ring.nodes, ("alpha", "beta"))


class RoutingDeterminismTests(unittest.TestCase):
    def test_input_order_does_not_matter(self) -> None:
        first = HashRing(["alpha", "beta", "gamma"], virtual_nodes=8, replicas=3)
        second = HashRing(["gamma", "alpha", "beta"], virtual_nodes=8, replicas=3)
        self.assertEqual(first, second)
        for index in range(200):
            key = f"key-{index:04d}"
            self.assertEqual(first.nodes_for(key), second.nodes_for(key))

    def test_fixed_sha256_golden_assignment(self) -> None:
        # Pins the UTF-8/SHA-256 mapping and the "{node}#{index}" vnode spelling.
        ring = HashRing(["alpha", "beta", "gamma"], virtual_nodes=3, replicas=3)
        self.assertEqual(ring.primary_node("user:1001"), "beta")
        self.assertEqual(ring.nodes_for("user:1001"), ["beta", "gamma", "alpha"])

    def test_routes_match_independent_reference(self) -> None:
        nodes = ["n1", "n2", "n3", "n4", "n5"]
        ring = HashRing(nodes, virtual_nodes=16, replicas=3)
        points = reference_points(nodes, 16)
        for index in range(300):
            key = f"object/{index:04d}"
            self.assertEqual(ring.nodes_for(key), reference_route(points, key, 3))

    def test_length_is_min_of_replicas_and_nodes(self) -> None:
        ring = HashRing(["a", "b", "c", "d", "e"], virtual_nodes=8, replicas=3)
        for index in range(100):
            route = ring.nodes_for(f"k-{index}")
            self.assertEqual(len(route), 3)
            self.assertEqual(len(set(route)), 3)
        small = HashRing(["a", "b"], virtual_nodes=8, replicas=5)
        for index in range(100):
            route = small.nodes_for(f"k-{index}")
            self.assertEqual(len(route), 2)
            self.assertEqual(set(route), {"a", "b"})

    def test_single_node_owns_every_key_including_utf8(self) -> None:
        ring = HashRing(["节点α"], virtual_nodes=4, replicas=3)
        self.assertEqual(ring.nodes_for("数据β"), ["节点α"])
        self.assertEqual(ring.primary_node("日本語キー"), "节点α")

    def test_wraps_from_ring_head(self) -> None:
        # Find, deterministically, a key whose token is above every vnode token: its primary must
        # be the ring head, reached only by wrapping.
        nodes = ["wrap-a", "wrap-b"]
        ring = HashRing(nodes, virtual_nodes=1, replicas=2)
        points = reference_points(nodes, 1)
        ceiling = max(point[0] for point in points)
        key = next(f"wrap-key-{i}" for i in range(1000) if digest(f"wrap-key-{i}") > ceiling)
        self.assertGreater(digest(key), ceiling)
        self.assertEqual(ring.primary_node(key), points[0][1])
        self.assertEqual(ring.nodes_for(key), [points[0][1], points[1][1]])

    def test_token_collisions_keep_every_vnode_in_stable_order(self) -> None:
        maximum = (1 << 256) - 1
        with mock.patch("kvstore.routing._virtual_token", return_value=maximum):
            ring_a = HashRing(["a", "b"], virtual_nodes=2, replicas=2)
            ring_b = HashRing(["b", "a"], virtual_nodes=2, replicas=2)
        # No vnode is lost to the collision ...
        self.assertEqual(len(ring_a._tokens), 4)
        # ... and the (node, index) tie-break is independent of input order: a#0, a#1, b#0, b#1.
        self.assertEqual(
            [(node, index) for _, node, index in ring_a._tokens],
            [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
        )
        self.assertEqual(ring_a, ring_b)
        self.assertEqual(ring_a.nodes_for("anything"), ["a", "b"])

    def test_collision_order_uses_index_before_a_new_node(self) -> None:
        # A later index of the same node sorts before another node at the same token: both vnodes
        # survive and the primary is stable even when every position collides.
        with mock.patch("kvstore.routing._virtual_token", return_value=1 << 255):
            ring = HashRing(["a", "b"], virtual_nodes=3, replicas=1)
        self.assertEqual(
            [(node, index) for _, node, index in ring._tokens],
            [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)],
        )
        self.assertEqual(ring.primary_node("key"), "a")


class DerivationTests(unittest.TestCase):
    def test_join_and_leave_equal_fresh_rings(self) -> None:
        base = HashRing(["a", "b", "c"], virtual_nodes=8, replicas=2)
        joined = base.with_node("d")
        self.assertEqual(joined, HashRing(["c", "d", "b", "a"], virtual_nodes=8, replicas=2))
        self.assertEqual(joined.without_node("d"), base)
        self.assertEqual(joined.without_node("a"), HashRing(["b", "c", "d"], virtual_nodes=8, replicas=2))

    def test_derivation_leaves_original_untouched(self) -> None:
        base = HashRing(["a", "b", "c"], virtual_nodes=8, replicas=2)
        before = base.nodes_for("stable-key")
        base.with_node("d")
        base.with_node("e").without_node("a")
        self.assertEqual(base.nodes, ("a", "b", "c"))
        self.assertEqual(base.nodes_for("stable-key"), before)

    def test_join_existing_node_is_validation_error_with_node(self) -> None:
        ring = HashRing(["a", "b"], virtual_nodes=4, replicas=1)
        with self.assertRaises(ValidationError) as caught:
            ring.with_node("b")
        self.assertEqual(caught.exception.context.get("node"), "b")

    def test_leave_missing_node_is_validation_error_with_node(self) -> None:
        ring = HashRing(["a", "b"], virtual_nodes=4, replicas=1)
        with self.assertRaises(ValidationError) as caught:
            ring.without_node("ghost")
        self.assertEqual(caught.exception.context.get("node"), "ghost")

    def test_removing_last_node_fails(self) -> None:
        ring = HashRing(["solo"], virtual_nodes=4, replicas=1)
        with self.assertRaises(ValidationError) as caught:
            ring.without_node("solo")
        self.assertEqual(caught.exception.context.get("node"), "solo")

    def test_join_leave_reject_empty_or_non_string_id(self) -> None:
        ring = HashRing(["a", "b"], virtual_nodes=4, replicas=1)
        for call in (ring.with_node, ring.without_node):
            with self.subTest(call=call.__name__):
                with self.assertRaises(ValidationError):
                    call("")
                with self.assertRaises(ValidationError):
                    call(5)  # type: ignore[arg-type]

    def test_join_only_new_node_can_take_primary(self) -> None:
        base = HashRing(["n1", "n2", "n3", "n4"], virtual_nodes=32, replicas=3)
        grown = base.with_node("newcomer")
        for index in range(500):
            key = f"obj-{index:04d}"
            old_primary = base.primary_node(key)
            new_primary = grown.primary_node(key)
            self.assertIn(new_primary, {old_primary, "newcomer"})

    def test_leave_primary_drift_rules(self) -> None:
        full = HashRing(["n1", "n2", "n3", "n4"], virtual_nodes=32, replicas=3)
        points = reference_points(["n1", "n2", "n3", "n4"], 32)
        survivors = [point for point in points if point[1] != "n4"]
        reduced = full.without_node("n4")
        for index in range(500):
            key = f"obj-{index:04d}"
            old_primary = full.primary_node(key)
            new_primary = reduced.primary_node(key)
            if old_primary != "n4":
                # Keys owned by survivors never drift to another primary.
                self.assertEqual(new_primary, old_primary, key)
            else:
                # Keys the departed node owned move clockwise to the first surviving vnode's node.
                self.assertNotEqual(new_primary, "n4")
                self.assertIn(new_primary, {"n1", "n2", "n3"})
                expected = reference_route(survivors, key, 3)[0]
                self.assertEqual(new_primary, expected, key)

    def test_removed_node_vanishes_from_every_sequence(self) -> None:
        full = HashRing(["n1", "n2", "n3", "n4"], virtual_nodes=16, replicas=3)
        reduced = full.without_node("n2")
        for index in range(200):
            route = reduced.nodes_for(f"k-{index}")
            self.assertNotIn("n2", route)
            self.assertEqual(len(route), len(set(route)))


class RebalancePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old = HashRing(["a", "b", "c"], virtual_nodes=32, replicas=2)

    def test_plan_keys_are_unicode_sorted_deduped_and_unique(self) -> None:
        new = self.old.with_node("d")
        keys = ["中", "b", "a", "é", "a", "b"]
        plan = self.old.plan_rebalance(keys, new, joined=["d"])
        changed = sorted({key for key in set(keys) if self.old.nodes_for(key) != new.nodes_for(key)})
        planned = [migration.key for migration in plan]
        # Unchanged keys never appear; the rest keep Unicode code-point order with one entry each.
        self.assertEqual(planned, changed)
        self.assertEqual(len(planned), len(set(planned)))
        self.assertEqual(planned, sorted(planned))
        self.assertEqual(plan.joined, ("d",))
        self.assertEqual(plan.left, ())
        self.assertEqual(plan.changes, len(plan.migrations))
        self.assertEqual(len(plan), len(plan.migrations))
        self.assertEqual(list(plan), list(plan.migrations))

    def test_identical_rings_produce_empty_plan(self) -> None:
        same = HashRing(["c", "a", "b"], virtual_nodes=32, replicas=2)
        plan = self.old.plan_rebalance(["a", "b", "c"], same)
        self.assertEqual(plan.migrations, ())

    def test_unchanged_keys_are_absent_even_after_join(self) -> None:
        new = self.old.with_node("d")
        stable_keys = [f"stay-{i}" for i in range(300) if new.nodes_for(f"stay-{i}") == self.old.nodes_for(f"stay-{i}")]
        moved_keys = [f"move-{i}" for i in range(300)
                      if new.nodes_for(f"move-{i}") != self.old.nodes_for(f"move-{i}")]
        self.assertTrue(stable_keys)
        self.assertTrue(moved_keys)
        plan = self.old.plan_rebalance(stable_keys + moved_keys, new)
        planned = {migration.key for migration in plan.migrations}
        self.assertTrue(planned.isdisjoint(stable_keys))
        self.assertTrue(planned.issuperset(moved_keys))

    def test_migration_records_sequences_and_add_remove_nodes(self) -> None:
        new = self.old.without_node("c")
        plan = self.old.plan_rebalance([f"k-{i}" for i in range(200)], new, left=["c"])
        self.assertTrue(plan.migrations)
        for migration in plan.migrations:
            self.assertIn("c", migration.old_nodes)
            self.assertNotIn("c", migration.new_nodes)
            self.assertEqual(migration.add_nodes, tuple(n for n in migration.new_nodes if n not in migration.old_nodes))
            self.assertEqual(migration.remove_nodes, tuple(n for n in migration.old_nodes if n not in migration.new_nodes))
            self.assertEqual(len(set(migration.new_nodes)), len(migration.new_nodes))

    def test_join_plan_adds_only_newcomer_as_a_primary_owner(self) -> None:
        new = self.old.with_node("c9")
        plan = self.old.plan_rebalance([f"j-{i}" for i in range(400)], new, joined=["c9"])
        primaries_changed = [m for m in plan.migrations if m.old_nodes[0] != m.new_nodes[0]]
        self.assertTrue(primaries_changed)
        for migration in primaries_changed:
            self.assertEqual(migration.new_nodes[0], "c9")
            self.assertEqual(migration.remove_nodes, tuple(n for n in migration.old_nodes if n not in migration.new_nodes))

    def test_empty_and_duplicate_plan_keys(self) -> None:
        new = self.old.with_node("d")
        with self.assertRaises(ValidationError):
            self.old.plan_rebalance(["ok", ""], new)
        with self.assertRaises(ValidationError):
            self.old.plan_rebalance(["ok", 2], new)  # type: ignore[list-item]
        plan = self.old.plan_rebalance(["one", "one", "one"], new)
        self.assertEqual([m.key for m in plan], ["one"] * (1 if plan.migrations else 0))
        self.assertLessEqual(len(plan), 1)

    def test_rejects_rings_with_different_shape(self) -> None:
        with self.assertRaises(ValidationError):
            self.old.plan_rebalance(["k"], HashRing(["a", "b", "c"], virtual_nodes=16, replicas=2))
        with self.assertRaises(ValidationError):
            self.old.plan_rebalance(["k"], HashRing(["a", "b", "c"], virtual_nodes=32, replicas=1))

    def test_plan_is_deterministic(self) -> None:
        new = self.old.without_node("b")
        keys = [f"det-{i}" for i in range(100)]
        first = self.old.plan_rebalance(reversed(keys), new, left=["b"])
        second = self.old.plan_rebalance(list(reversed(keys)), new, left=["b"])
        self.assertEqual(first, second)
        self.assertEqual(first.to_document(), second.to_document())

    def test_to_document_shape(self) -> None:
        new = self.old.with_node("d")
        plan = self.old.plan_rebalance(["alpha"], new, joined=["d"])
        if plan.migrations:
            document = plan.migrations[0].to_document()
            self.assertEqual(
                set(document), {"key", "oldNodes", "newNodes", "addNodes", "removeNodes"}
            )
        self.assertEqual(set(plan.to_document()), {"joined", "left", "migrations"})

    def test_planning_touches_no_files(self) -> None:
        # Routing is independent of the Store: deriving rings and planning must never touch disk.
        with tempfile.TemporaryDirectory() as empty:
            cwd = os.getcwd()
            os.chdir(empty)
            try:
                new = self.old.with_node("d").without_node("a")
                self.old.plan_rebalance([f"x-{i}" for i in range(50)], new, joined=["d"], left=["a"])
            finally:
                os.chdir(cwd)
            self.assertEqual(os.listdir(empty), [])


class DescribeUnchangedTests(unittest.TestCase):
    def test_describe_still_lists_the_nine_store_operations_only(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(["describe"])
        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        import json

        document = json.loads(out.getvalue())
        self.assertEqual(
            document["operations"],
            ["compact", "delete", "describe", "flush", "get", "put", "scan", "stats", "verify"],
        )
        self.assertNotIn("ring", document)
        self.assertNotIn("hashRing", document)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
