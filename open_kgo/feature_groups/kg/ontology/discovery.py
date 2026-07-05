"""DC circuit beam search over the EM potential landscape (Layer 3).

SemanticField (Layer 2) pre-computes:

  - V[e]: electric potential at every entity (solved via graph Laplacian)
  - Implied edge current: G(i,j) × |V(i) - V(j)|

The edge current IS the beam search heuristic.  No LLM calls, no oracle
scores, no sampling.  The field gradient already encodes which traversal
directions are semantically live.

Two operations form the public API:

``find_paths`` — beam search from source to sink following edge currents.
    Score of a path = bottleneck current: the minimum edge current along
    the path (weakest link in the conducting chain).  Paths are returned
    sorted descending by score.

``extract_circuit`` — return only edges whose current exceeds a threshold.
    Dead-end branches have zero potential difference (both endpoints float
    to the same voltage when disconnected from the opposing anchor), so
    they are automatically excluded — no filtering logic required.

These two primitives act as a backbone for context engineering
(``extract_circuit`` returns the minimal query-specific subgraph that
replaces static GraphRAG community partitions) and loop engineering
(``find_paths`` with updated anchors between agent iterations produces the
carry-forward relevance structure for the next step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from open_kgo.feature_groups.kg.ontology.semantic_field import _compute_field, _conductance

# ---------------------------------------------------------------------------
# Optional numba acceleration for batch edge-current computation.
# Falls back to pure Python when numba / numpy are not installed.
# ---------------------------------------------------------------------------

_np_mod: Any = None
_nb_edge_currents: Any = None  # compiled kernel: (v_src, v_tgt, conductances) -> currents

try:
    import numpy as _numba_np
    from numba import njit as _njit

    @_njit(cache=True)
    def _nb_edge_currents_kernel(v_src, v_tgt, conductances):  # type: ignore[no-untyped-def]
        """Vectorised edge current: conductances[i] * abs(v_src[i] - v_tgt[i])."""
        n = v_src.shape[0]
        out = _numba_np.zeros(n)
        for i in range(n):
            out[i] = conductances[i] * abs(v_src[i] - v_tgt[i])
        return out

    _np_mod = _numba_np
    _nb_edge_currents = _nb_edge_currents_kernel

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredPath:
    """A single path discovered by beam search through the EM potential landscape.

    ``len(nodes) == len(relations) + 1`` always holds.
    ``score`` is the bottleneck current — minimum edge current along the path.
    Higher score means the path's weakest link carries more current.
    Trivial paths (source node is also a sink node) have ``score = math.inf``.
    """

    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    score: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_neighbor_edges(
    edges: list[tuple[str, str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Undirected adjacency: node → [(neighbour, relation), ...].

    A directed edge (s, r, t) contributes both (s→t via r) and (t→s via r),
    consistent with the DC circuit model where current flows either way.
    """
    adj: dict[str, list[tuple[str, str]]] = {}
    for s, r, t in edges:
        adj.setdefault(s, []).append((t, r))
        adj.setdefault(t, []).append((s, r))
    return adj


def _compute_edge_currents(
    namespace: str,
    edges: list[tuple[str, str, str]],
    voltages: dict[str, float],
) -> dict[tuple[str, str], float]:
    """Batch-compute edge current for every edge: G(s,r,t) × |V(s) - V(t)|.

    Returns an undirected dict — both (s,t) and (t,s) map to the same value.
    When multiple edges connect the same (s,t) pair, the maximum current over
    all such edges is kept (follow the strongest signal).
    Uses the numba kernel when available; falls back to pure Python otherwise.
    """
    if not edges:
        return {}

    result: dict[tuple[str, str], float] = {}

    if _nb_edge_currents is not None and _np_mod is not None:
        v_src_list = [voltages.get(e[0], 0.0) for e in edges]
        v_tgt_list = [voltages.get(e[2], 0.0) for e in edges]
        cond_list = [_conductance(namespace, e[1]) for e in edges]
        v_src_arr = _np_mod.array(v_src_list, dtype=_np_mod.float64)
        v_tgt_arr = _np_mod.array(v_tgt_list, dtype=_np_mod.float64)
        cond_arr = _np_mod.array(cond_list, dtype=_np_mod.float64)
        currents_arr = _nb_edge_currents(v_src_arr, v_tgt_arr, cond_arr)
        for i, (s, _, t) in enumerate(edges):
            c = float(currents_arr[i])
            if c > result.get((s, t), 0.0):
                result[(s, t)] = c
            if c > result.get((t, s), 0.0):
                result[(t, s)] = c
    else:
        for s, r, t in edges:
            g = _conductance(namespace, r)
            c = g * abs(voltages.get(s, 0.0) - voltages.get(t, 0.0))
            if c > result.get((s, t), 0.0):
                result[(s, t)] = c
            if c > result.get((t, s), 0.0):
                result[(t, s)] = c

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DiscoveryEngine:
    """Beam-search discovery over the DC circuit potential landscape (Layer 3).

    Requires SemanticField (Layer 2) for the field computation and
    OntologyRegistry (Layer 1) for conductance look-up.  Both are used
    internally; callers need only pass a namespace and an edge list.

    ``find_paths`` — ranked typed paths from source to sink guided by the
    EM field gradient.  The beam expands along highest-current edges; the
    score of each path is its bottleneck current (minimum edge current along
    the path).  Dead-end branches carry zero current and never enter the beam.

    ``extract_circuit`` — minimal current-carrying subgraph above a threshold.
    Useful as a context engineering primitive: hand the LLM a compact,
    query-specific typed subgraph rather than a static community partition.
    """

    @staticmethod
    def find_paths(
        namespace: str,
        edges: list[tuple[str, str, str]],
        source: dict[str, float],
        sink: dict[str, float],
        *,
        beam_width: int = 5,
        max_depth: int = 4,
        max_paths: int = 10,
    ) -> list[DiscoveredPath]:
        """Beam search from source to sink guided by the EM field gradient.

        Parameters
        ----------
        namespace:
            Ontology namespace for conductance look-up.
        edges:
            Instance-graph edges as ``(source_id, relation_type, target_id)``.
        source:
            High-voltage anchors, e.g. ``{"Christopher Nolan": 1.0}``.
        sink:
            Low-voltage anchors, e.g. ``{"Sci-Fi": 0.0}``.
        beam_width:
            Number of candidate paths kept alive after each expansion step.
        max_depth:
            Maximum number of hops from a source node to a sink node.
        max_paths:
            Maximum number of completed paths to return.

        Returns
        -------
        list[DiscoveredPath]
            Completed paths sorted descending by bottleneck current score.
            Each path starts at a source-group node and ends at a sink-group
            node.  Intermediate nodes are excluded from both groups.
            Empty list when source or sink is empty, or when no path exists
            within ``max_depth`` hops.

        Notes
        -----
        Cycle prevention: a node already on the current path is never revisited.
        Beam pruning is global across all live paths after each expansion step.
        Paths are collected as they are completed; duplicates (same node and
        relation sequence) are discarded.
        """
        if not source or not sink:
            return []

        combined_anchors: dict[str, float] = {**sink, **source}
        voltages = _compute_field(namespace, edges, combined_anchors)
        neighbor_edges = _build_neighbor_edges(edges)
        edge_current = _compute_edge_currents(namespace, edges, voltages)

        completed: list[DiscoveredPath] = []
        seen_completed: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

        # Trivial paths: source node is also a sink node.
        # Empty tuple must be explicitly typed to satisfy tuple[str, ...] for mypy.
        _empty: tuple[str, ...] = ()
        frontier: list[tuple[tuple[str, ...], tuple[str, ...], float]] = []
        for src in source:
            if src in sink:
                trivial_nodes: tuple[str, ...] = (src,)
                trivial_key = (trivial_nodes, _empty)
                if trivial_key not in seen_completed:
                    seen_completed.add(trivial_key)
                    completed.append(DiscoveredPath(nodes=trivial_nodes, relations=_empty, score=math.inf))
            elif src in neighbor_edges or src in voltages:
                src_nodes: tuple[str, ...] = (src,)
                frontier.append((src_nodes, _empty, math.inf))

        for _ in range(max_depth):
            if not frontier or len(completed) >= max_paths:
                break

            candidates: list[tuple[tuple[str, ...], tuple[str, ...], float]] = []
            for path_nodes, path_relations, path_score in frontier:
                terminal = path_nodes[-1]
                visited = set(path_nodes)
                for neighbour, relation in neighbor_edges.get(terminal, []):
                    if neighbour in visited:
                        continue
                    ec = edge_current.get((terminal, neighbour), 0.0)
                    new_score = min(path_score, ec)
                    new_nodes = path_nodes + (neighbour,)
                    new_relations = path_relations + (relation,)
                    if neighbour in sink:
                        key = (new_nodes, new_relations)
                        if key not in seen_completed:
                            seen_completed.add(key)
                            completed.append(DiscoveredPath(nodes=new_nodes, relations=new_relations, score=new_score))
                    else:
                        candidates.append((new_nodes, new_relations, new_score))

            candidates.sort(key=lambda x: x[2], reverse=True)
            frontier = candidates[:beam_width]

        completed.sort(key=lambda p: p.score, reverse=True)
        return completed[:max_paths]

    @staticmethod
    def extract_circuit(
        namespace: str,
        edges: list[tuple[str, str, str]],
        source: dict[str, float],
        sink: dict[str, float],
        *,
        current_threshold: float = 0.01,
    ) -> list[tuple[str, str, str]]:
        """Return the minimal current-carrying subgraph above a threshold.

        Computes the EM field for the (source, sink) query and filters edges
        to those where ``G(s,r,t) × |V(s) - V(t)| > current_threshold``.

        Dead-end branches are excluded for free: a node connected to only one
        voltage side floats to that side's potential.  Both endpoints share the
        same voltage, giving zero potential difference and zero current.

        Parameters
        ----------
        namespace:
            Ontology namespace for conductance look-up.
        edges:
            Instance-graph edges as ``(source_id, relation_type, target_id)``.
        source:
            High-voltage anchors.
        sink:
            Low-voltage anchors.
        current_threshold:
            Edges with current at or below this value are excluded.
            Default ``0.01`` removes floating-point noise from near-dead-ends.

        Returns
        -------
        list[tuple[str, str, str]]
            Subset of ``edges`` carrying current above ``current_threshold``,
            in the same order as the input.
        """
        if not edges:
            return []

        combined_anchors: dict[str, float] = {**sink, **source}
        voltages = _compute_field(namespace, edges, combined_anchors)

        result: list[tuple[str, str, str]] = []
        for s, r, t in edges:
            g = _conductance(namespace, r)
            c = g * abs(voltages.get(s, 0.0) - voltages.get(t, 0.0))
            if c > current_threshold:
                result.append((s, r, t))
        return result
