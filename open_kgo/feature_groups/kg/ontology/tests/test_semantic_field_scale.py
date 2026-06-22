"""Scale tests for SemanticField against the full MetaQA knowledge base.

Requires the full MetaQA GML at the path below (not committed to the repo).
Tests are skipped automatically when the file is absent so CI stays green.

Strategy: stream-parse the full GML into an adjacency list, then extract
the 2-hop subgraph around each anchor pair (~1 000 nodes) before passing
to the solver.  This keeps memory low and the solver tractable.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import pytest

from open_kgo.feature_groups.kg.ontology.semantic_field import SemanticField

METAQA_GML = Path("/Volumes/ExtraStorage/mloda_New_arch/prototypes/demo/data/metaqa_full.gml")

pytestmark = pytest.mark.skipif(
    not METAQA_GML.exists(),
    reason="Full MetaQA GML not present — skipped in CI",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_parse_gml(path: Path) -> tuple[set[str], list[tuple[str, str, str]]]:
    """Stream-parse GML line by line — never loads the full file into memory."""
    id_to_label: dict[int, str] = {}
    edges: list[tuple[str, str, str]] = []

    in_node = False
    in_edge = False
    cur_id: int | None = None
    cur_label: str | None = None
    cur_src: int | None = None
    cur_tgt: int | None = None
    cur_rel: str | None = None

    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if line == "node [":
                in_node, in_edge = True, False
                cur_id = cur_label = None
            elif line == "edge [":
                in_edge, in_node = True, False
                cur_src = cur_tgt = cur_rel = None
            elif line == "]":
                if in_node and cur_id is not None and cur_label is not None:
                    id_to_label[cur_id] = cur_label
                elif in_edge and cur_src is not None and cur_tgt is not None and cur_rel is not None:
                    s = id_to_label.get(cur_src)
                    t = id_to_label.get(cur_tgt)
                    if s and t:
                        edges.append((s, cur_rel, t))
                in_node = in_edge = False
            elif in_node:
                if line.startswith("id "):
                    cur_id = int(line[3:])
                elif line.startswith('label "'):
                    cur_label = line[7:-1]
            elif in_edge:
                if line.startswith("source "):
                    cur_src = int(line[7:])
                elif line.startswith("target "):
                    cur_tgt = int(line[7:])
                elif line.startswith('relation "'):
                    cur_rel = line[10:-1]

    return set(id_to_label.values()), edges


def _build_adj(edges: list[tuple[str, str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Undirected adjacency list: node → [(neighbour, relation), ...]."""
    adj: dict[str, list[tuple[str, str]]] = {}
    for s, r, t in edges:
        adj.setdefault(s, []).append((t, r))
        adj.setdefault(t, []).append((s, r))
    return adj


def _subgraph(
    adj: dict[str, list[tuple[str, str]]],
    seeds: list[str],
    hops: int = 2,
    max_nodes: int = 5000,
) -> list[tuple[str, str, str]]:
    """BFS from seeds up to `hops` away; return edges among collected nodes."""
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds if s in adj)
    while queue and len(visited) < max_nodes:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if depth < hops:
            for nbr, _ in adj.get(node, []):
                if nbr not in visited and len(visited) < max_nodes:
                    queue.append((nbr, depth + 1))

    # collect edges where both endpoints are in the subgraph
    sub: list[tuple[str, str, str]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for node in visited:
        for nbr, rel in adj.get(node, []):
            if nbr in visited:
                key = (min(node, nbr), rel, max(node, nbr))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    sub.append((node, rel, nbr))
    return sub


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def metaqa_data() -> tuple[set[str], list[tuple[str, str, str]]]:
    return _stream_parse_gml(METAQA_GML)


@pytest.fixture(scope="session")
def metaqa_adj(metaqa_data: tuple[set[str], list[tuple[str, str, str]]]) -> dict[str, list[tuple[str, str]]]:
    _, edges = metaqa_data
    return _build_adj(edges)


# ---------------------------------------------------------------------------
# Parametrised AND-query cases
# ---------------------------------------------------------------------------

AND_CASES = [
    ("Christopher Nolan", "Sci-Fi"),
    ("Steven Spielberg", "Action"),
    ("James Cameron", "Thriller"),
    ("Ridley Scott", "Drama"),
]


@pytest.mark.parametrize("director,genre", AND_CASES, ids=[c[0] for c in AND_CASES])
def test_compute_and_scale(
    director: str,
    genre: str,
    metaqa_data: tuple[set[str], list[tuple[str, str, str]]],
    metaqa_adj: dict[str, list[tuple[str, str]]],
) -> None:
    labels, _ = metaqa_data

    if director not in labels:
        pytest.skip(f"{director!r} not in dataset")
    if genre not in labels:
        pytest.skip(f"{genre!r} not in dataset")

    sub_edges = _subgraph(metaqa_adj, seeds=[director, genre], hops=2, max_nodes=10000)
    node_count = len({n for s, _, t in sub_edges for n in (s, t)})
    print(f"\n[{director} + {genre}] subgraph: {node_count} nodes, {len(sub_edges)} edges")

    t0 = time.perf_counter()
    scores = SemanticField.compute_and(
        namespace="metaqa",
        edges=sub_edges,
        source={director: 1.0},
        sink={genre: 0.0},
    )
    elapsed = time.perf_counter() - t0

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  solved in {elapsed:.2f}s — top 10: {[(e, round(s, 4)) for e, s in top]}")

    assert len(scores) > 0


def test_scores_differ_across_queries(
    metaqa_adj: dict[str, list[tuple[str, str]]],
    metaqa_data: tuple[set[str], list[tuple[str, str, str]]],
) -> None:
    """Two different director+genre pairs should produce different rankings."""
    labels, _ = metaqa_data
    pairs = [
        ("Christopher Nolan", "Sci-Fi"),
        ("Steven Spielberg", "Action"),
    ]
    for d, g in pairs:
        if d not in labels or g not in labels:
            pytest.skip(f"{d} or {g} not in dataset")

    results = []
    for director, genre in pairs:
        sub = _subgraph(metaqa_adj, seeds=[director, genre], hops=2, max_nodes=10000)
        scores = SemanticField.compute_and(
            namespace="metaqa",
            edges=sub,
            source={director: 1.0},
            sink={genre: 0.0},
        )
        top5 = frozenset(e for e, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5])
        results.append(top5)

    assert results[0] != results[1], "Different queries returned identical top-5"
