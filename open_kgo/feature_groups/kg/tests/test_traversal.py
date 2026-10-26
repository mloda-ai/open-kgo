"""Direct unit tests for the shared BFS helpers in ``traversal.py``.

These exercise ``bfs_frontier_walk`` and ``bfs_collect_ids`` in isolation
on a small toy graph, covering the invariants called out in their
docstrings and in the issue that requested these tests.
"""

from __future__ import annotations

from open_kgo.feature_groups.kg.traversal import bfs_collect_ids, bfs_frontier_walk


def _neighbors(graph: dict[str, list[str]]):
    def neighbors(node: str) -> list[str]:
        return graph.get(node, [])

    return neighbors


def _emit(node: str) -> dict[str, str]:
    return {"id": node}


class TestBfsFrontierWalk:
    def test_depth_or_remaining_zero_returns_empty(self) -> None:
        graph = {"a": ["b"]}
        assert bfs_frontier_walk(_neighbors(graph), "a", depth=0, remaining=5, emit=_emit) == []
        assert bfs_frontier_walk(_neighbors(graph), "a", depth=1, remaining=0, emit=_emit) == []

    def test_start_is_never_emitted(self) -> None:
        graph = {"a": ["b"], "b": ["c"]}
        rows = bfs_frontier_walk(_neighbors(graph), "a", depth=2, remaining=10, emit=_emit)
        assert [r["id"] for r in rows] == ["b", "c"]
        assert "a" not in {r["id"] for r in rows}

    def test_cycle_terminates(self) -> None:
        graph = {"a": ["b"], "b": ["a"]}
        rows = bfs_frontier_walk(_neighbors(graph), "a", depth=5, remaining=10, emit=_emit)
        assert [r["id"] for r in rows] == ["b"]

    def test_stops_once_remaining_rows_emitted(self) -> None:
        graph = {"a": ["b", "c", "d"]}
        rows = bfs_frontier_walk(_neighbors(graph), "a", depth=1, remaining=2, emit=_emit)
        assert len(rows) == 2
        assert {r["id"] for r in rows}.issubset({"b", "c", "d"})

    def test_shared_emitted_set_emits_overlap_once_but_still_expands(self) -> None:
        # a -> b -> c, and b is already in emitted
        graph = {"a": ["b"], "b": ["c"]}
        emitted = {"a", "b"}
        rows = bfs_frontier_walk(
            _neighbors(graph), "a", depth=2, remaining=10, emit=_emit, emitted=emitted
        )
        # b is not re-emitted, but c is still reached via expansion of b
        assert [r["id"] for r in rows] == ["c"]
        assert "b" in emitted
        assert "c" in emitted

    def test_should_traverse_rejects_node(self) -> None:
        graph = {"a": ["b", "c"], "b": ["d"]}
        rows = bfs_frontier_walk(
            _neighbors(graph),
            "a",
            depth=2,
            remaining=10,
            emit=_emit,
            should_traverse=lambda n: n != "b",
        )
        # b is neither emitted nor expanded, so d is unreachable
        assert [r["id"] for r in rows] == ["c"]


class TestBfsCollectIds:
    def test_start_is_included(self) -> None:
        graph = {"a": ["b"], "b": []}
        ids = bfs_collect_ids(lambda n: n in graph, _neighbors(graph), "a", depth=1)
        assert ids == ["a", "b"]

    def test_missing_start_returns_empty(self) -> None:
        graph = {"a": ["b"], "b": []}
        ids = bfs_collect_ids(lambda n: n in graph, _neighbors(graph), "missing")
        assert ids == []

    def test_depth_limits_hops(self) -> None:
        graph = {"a": ["b"], "b": ["c"], "c": ["d"], "d": []}
        ids = bfs_collect_ids(lambda n: n in graph, _neighbors(graph), "a", depth=1)
        assert ids == ["a", "b"]

    def test_depth_none_walks_whole_graph(self) -> None:
        graph = {"a": ["b"], "b": ["c"], "c": ["d"], "d": []}
        ids = bfs_collect_ids(lambda n: n in graph, _neighbors(graph), "a", depth=None)
        assert ids == ["a", "b", "c", "d"]

    def test_max_nodes_caps_result(self) -> None:
        graph = {"a": ["b", "c", "d"], "b": [], "c": [], "d": []}
        ids = bfs_collect_ids(lambda n: n in graph, _neighbors(graph), "a", depth=1, max_nodes=2)
        assert len(ids) == 2
        assert "a" in ids

    def test_nodes_failing_contains_are_skipped(self) -> None:
        graph = {"a": ["b", "missing"], "b": ["c"], "c": []}
        ids = bfs_collect_ids(lambda n: n in {"a", "b", "c"}, _neighbors(graph), "a", depth=2)
        assert ids == ["a", "b", "c"]
