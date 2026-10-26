"""Direct unit tests for the shared bounded BFS helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import pytest

from open_kgo.feature_groups.kg.traversal import bfs_collect_ids, bfs_frontier_walk


def _neighbors(graph: dict[str, list[str]]) -> Callable[[str], Iterable[str]]:
    """Return the graph's adjacency lookup in the shape both helpers accept."""

    def neighbors(node: str) -> list[str]:
        return graph.get(node, [])

    return neighbors


def _emit(node: str) -> dict[str, str]:
    """Use a small, inspectable row shape for the frontier walker."""
    return {"node": node}


@pytest.mark.parametrize(("depth", "remaining"), [(0, 3), (3, 0)])
def test_bfs_frontier_walk_returns_empty_at_a_zero_boundary(depth: int, remaining: int) -> None:
    """Zero depth or budget must avoid emitting even immediate neighbours."""
    rows = bfs_frontier_walk(
        _neighbors({"start": ["child"]}),
        "start",
        depth=depth,
        remaining=remaining,
        emit=_emit,
    )
    assert rows == []


def test_bfs_frontier_walk_terminates_cycles_without_emitting_start() -> None:
    """The local visited set stops a cycle and the caller owns the start row."""
    rows = bfs_frontier_walk(
        _neighbors({"start": ["first"], "first": ["second"], "second": ["start"]}),
        "start",
        depth=10,
        remaining=10,
        emit=_emit,
    )
    assert rows == [{"node": "first"}, {"node": "second"}]


def test_bfs_frontier_walk_stops_when_remaining_rows_are_emitted() -> None:
    """The result budget limits traversal work rather than only the returned slice."""
    consumed: list[str] = []

    def children() -> Iterator[str]:
        for child in ["one", "two", "three"]:
            consumed.append(child)
            yield child

    def neighbors(node: str) -> Iterable[str]:
        return children() if node == "start" else []

    rows = bfs_frontier_walk(neighbors, "start", depth=2, remaining=2, emit=_emit)
    assert rows == [{"node": "one"}, {"node": "two"}]
    assert consumed == ["one", "two"]


def test_bfs_frontier_walk_expands_shared_nodes_without_emitting_them_twice() -> None:
    """A caller-shared emitted set deduplicates output but not the second walk's frontier."""
    emitted = {"start"}
    first = bfs_frontier_walk(
        _neighbors({"start": ["shared"]}),
        "start",
        depth=1,
        remaining=10,
        emit=_emit,
        emitted=emitted,
    )
    second = bfs_frontier_walk(
        _neighbors({"start": ["shared"], "shared": ["second-only"]}),
        "start",
        depth=2,
        remaining=10,
        emit=_emit,
        emitted=emitted,
    )
    assert first == [{"node": "shared"}]
    assert second == [{"node": "second-only"}]


def test_bfs_frontier_walk_does_not_emit_or_expand_rejected_nodes() -> None:
    """A rejected intermediary makes nodes behind it unreachable."""
    visited_by_neighbors: list[str] = []

    def neighbors(node: str) -> list[str]:
        visited_by_neighbors.append(node)
        return {"start": ["missing"], "missing": ["reachable-only-through-missing"]}.get(node, [])

    rows = bfs_frontier_walk(
        neighbors,
        "start",
        depth=2,
        remaining=10,
        emit=_emit,
        should_traverse=lambda node: node != "missing",
    )
    assert rows == []
    assert visited_by_neighbors == ["start"]


def test_bfs_collect_ids_includes_start_and_respects_depth() -> None:
    """A finite depth counts hops from the included start node."""
    rows = bfs_collect_ids(
        lambda node: node in {"start", "one", "two"},
        _neighbors({"start": ["one"], "one": ["two"]}),
        "start",
        depth=1,
    )
    assert rows == ["start", "one"]


def test_bfs_collect_ids_returns_empty_for_a_missing_start() -> None:
    """An absent start is an empty reachability result, not an exception."""
    rows = bfs_collect_ids(lambda node: node != "missing", _neighbors({}), "missing")
    assert rows == []


def test_bfs_collect_ids_walks_the_whole_graph_when_depth_is_unbounded() -> None:
    """``depth=None`` reaches every node exactly once even through a cycle."""
    rows = bfs_collect_ids(
        lambda node: node in {"start", "one", "two"},
        _neighbors({"start": ["one"], "one": ["two"], "two": ["start"]}),
        "start",
        depth=None,
    )
    assert rows == ["start", "one", "two"]


def test_bfs_collect_ids_stops_at_max_nodes() -> None:
    """The cap bounds the collected breadth-first result."""
    rows = bfs_collect_ids(
        lambda node: True,
        _neighbors({"start": ["one", "two"], "one": ["three"]}),
        "start",
        depth=None,
        max_nodes=2,
    )
    assert rows == ["start", "one"]


def test_bfs_collect_ids_skips_and_does_not_expand_missing_intermediaries() -> None:
    """A node excluded by ``contains`` cannot provide a path to later nodes."""
    visited_by_neighbors: list[str] = []

    def neighbors(node: str) -> list[str]:
        visited_by_neighbors.append(node)
        return {"start": ["missing"], "missing": ["reachable-only-through-missing"]}.get(node, [])

    rows = bfs_collect_ids(
        lambda node: node != "missing",
        neighbors,
        "start",
        depth=None,
    )
    assert rows == ["start"]
    assert visited_by_neighbors == ["start"]
