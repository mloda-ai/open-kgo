"""DC circuit semantic field for ontology-grounded query scoring (Layer 2).

Models the knowledge graph as a resistor network:

  - Edge conductance  = ontology-declared relationship weight  (G = w)
  - Edge resistance   = inverse weight                         (R = 1/w)
  - Query anchors     = Dirichlet boundary conditions          (fixed voltage)
  - Semantic field    = electric potential V(entity) at every node

The potential is solved via the conductance-weighted graph Laplacian:

    L · V = s

where L[i,i] = Σ G(i,j) over all neighbours j,  L[i,j] = -G(i,j),
and the known anchor voltages are applied as Dirichlet boundary conditions
(standard impressed-voltage treatment from electrical network theory).

For a multi-constraint AND query (``compute_and``) each constraint specifies
a relationship-type filter together with its anchor set.  Each constraint is
solved on the subgraph that contains *only* edges of that relationship type.
An entity that is not connected to a constraint's anchor in that subgraph
receives potential 0.0 — it lies in a disconnected component.  The final
score is the element-wise product of per-constraint potentials: if any
single constraint gives 0.0 the overall score is 0.0.  This is the
series-circuit AND semantics: current cannot flow when any segment of the
series path is an open circuit.

Ontology weights shape the potential when an entity sits between two anchors
at different voltages.  A higher-conductance path (higher weight) pulls the
intermediate node's potential closer to the high-voltage anchor.

Reference: Kirchhoff matrix / graph Laplacian for resistor networks
(Strang, Introduction to Linear Algebra ch. 10; verified against
MIT OCW 18.06 and Berkeley Stat260/CS294 lecture notes on spectral graph
methods; effective-resistance formulation from Ghosh & Boyd, Stanford 2006).
"""

from __future__ import annotations

from typing import Any

from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

# ---------------------------------------------------------------------------
# Optional numba acceleration for the Gaussian elimination kernel.
# Falls back to pure Python when numba / numpy are not installed.
# ---------------------------------------------------------------------------

_np_mod: Any = None
_nb_solve: Any = None  # compiled kernel: (aug: ndarray) -> ndarray

try:
    import numpy as _numba_np
    from numba import njit as _njit

    @_njit(cache=True)
    def _nb_gauss_elim(aug):  # type: ignore[no-untyped-def]
        """Gaussian elimination with partial pivoting on an augmented matrix.

        Operates in-place on ``aug`` (shape m × m+1, float64).
        Returns the solution vector V of length m.
        """
        m = aug.shape[0]
        for col in range(m):
            pivot = col
            for r in range(col + 1, m):
                if abs(aug[r, col]) > abs(aug[pivot, col]):
                    pivot = r
            for c in range(m + 1):
                tmp = aug[col, c]
                aug[col, c] = aug[pivot, c]
                aug[pivot, c] = tmp
            if abs(aug[col, col]) < 1e-12:
                continue
            for r in range(col + 1, m):
                f = aug[r, col] / aug[col, col]
                for c in range(col, m + 1):
                    aug[r, c] -= f * aug[col, c]
        V = _numba_np.zeros(m)
        for row in range(m - 1, -1, -1):
            if abs(aug[row, row]) < 1e-12:
                V[row] = 0.0
                continue
            V[row] = aug[row, m]
            for c in range(row + 1, m):
                V[row] -= aug[row, c] * V[c]
            V[row] /= aug[row, row]
        return V

    _np_mod = _numba_np
    _nb_solve = _nb_gauss_elim

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _conductance(namespace: str, relation: str) -> float:
    """Return the conductance for ``relation`` in ``namespace``.

    Conductance = declared ontology weight.  Falls back to 1.0 when no
    ontology is registered or the relation is not declared (open-world
    assumption — unknown edges are treated as unit conductors).
    """
    ontology = OntologyRegistry.get(namespace)
    if ontology is None:
        return 1.0
    rule = ontology.relationships.get(relation)
    return rule.weight if rule is not None else 1.0


def _filter_edges(
    edges: list[tuple[str, str, str]],
    relation_type: str | None,
) -> list[tuple[str, str, str]]:
    """Return edges restricted to ``relation_type``, or all edges if None."""
    if relation_type is None:
        return list(edges)
    return [(s, r, t) for s, r, t in edges if r == relation_type]


def _build_laplacian(
    nodes: list[str],
    edges: list[tuple[str, str, str]],
    namespace: str,
) -> list[list[float]]:
    """Return the conductance-weighted graph Laplacian as a 2-D list.

    Edges are treated as undirected — current flows both ways through a
    resistor regardless of the semantic direction of the relationship.
    """
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    L: list[list[float]] = [[0.0] * n for _ in range(n)]
    for src, relation, tgt in edges:
        if src not in idx or tgt not in idx:
            continue
        g = _conductance(namespace, relation)
        i, j = idx[src], idx[tgt]
        L[i][j] -= g
        L[j][i] -= g
        L[i][i] += g
        L[j][j] += g
    return L


def _solve_dirichlet(
    L: list[list[float]],
    interior: list[int],
    boundary: dict[int, float],
) -> list[float]:
    """Solve L_II · V_I = -L_IB · V_B via Gaussian elimination.

    Implements the standard Dirichlet decomposition for resistor networks:
    partition nodes into boundary B (known voltage) and interior I (unknown),
    then solve the reduced system.  Zero-pivot rows (disconnected nodes)
    receive potential 0.0.
    """
    m = len(interior)
    if m == 0:
        return []

    aug: list[list[float]] = []
    for row_idx in interior:
        aug_row: list[float] = [L[row_idx][col_idx] for col_idx in interior]
        rhs_val = -sum(L[row_idx][b] * v for b, v in boundary.items())
        aug_row.append(rhs_val)
        aug.append(aug_row)

    # Numba-compiled path (when numba + numpy are available)
    if _nb_solve is not None:
        aug_arr = _np_mod.array(aug, dtype=_np_mod.float64)
        return list(_nb_solve(aug_arr))

    # Pure-Python fallback — forward elimination with partial pivoting
    for col in range(m):
        pivot_row = col
        for r in range(col + 1, m):
            if abs(aug[r][col]) > abs(aug[pivot_row][col]):
                pivot_row = r
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            continue
        for r in range(col + 1, m):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, m + 1):
                aug[r][c] -= factor * aug[col][c]

    # Back substitution
    V: list[float] = [0.0] * m
    for row in range(m - 1, -1, -1):
        if abs(aug[row][row]) < 1e-12:
            V[row] = 0.0
            continue
        V[row] = aug[row][m]
        for col in range(row + 1, m):
            V[row] -= aug[row][col] * V[col]
        V[row] /= aug[row][row]

    return V


def _compute_field(
    namespace: str,
    edges: list[tuple[str, str, str]],
    anchors: dict[str, float],
) -> dict[str, float]:
    """Core field computation (no edge filtering — callers filter first)."""
    nodes_set: set[str] = set(anchors)
    for src, _, tgt in edges:
        nodes_set.add(src)
        nodes_set.add(tgt)
    nodes = sorted(nodes_set)
    idx = {node: i for i, node in enumerate(nodes)}

    boundary = {idx[e]: v for e, v in anchors.items() if e in idx}
    interior = [i for i in range(len(nodes)) if i not in boundary]

    L = _build_laplacian(nodes, edges, namespace)
    V_interior = _solve_dirichlet(L, interior, boundary)

    result: dict[str, float] = {}
    for e, v in anchors.items():
        if e in idx:
            result[e] = v
    for k, i in enumerate(interior):
        result[nodes[i]] = V_interior[k]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SemanticField:
    """Query-induced DC circuit field over a knowledge graph.

    Two methods form the public API:

    ``compute`` — single-anchor potential field.  Place one or more voltage
    sources at anchor entities and solve the Dirichlet problem.  Each entity
    receives a potential proportional to how well it is connected to the
    anchors through the conductance network.

    ``compute_and`` — series-circuit AND query.  Place a *source* group at
    high voltage and a *sink* group at low voltage in one combined circuit.
    Score each interior entity by the current it carries between source and
    sink.  Entities that bridge both groups carry current (score > 0);
    entities hanging off only one side float to that side's extreme potential
    and carry zero current.  No relation-type filtering is needed — the
    circuit topology enforces the AND constraint.
    """

    @staticmethod
    def compute(
        namespace: str,
        edges: list[tuple[str, str, str]],
        anchors: dict[str, float],
        *,
        relation_type: str | None = None,
    ) -> dict[str, float]:
        """Compute the electric potential at every entity for one anchor set.

        Parameters
        ----------
        namespace:
            Ontology namespace used to look up relationship weights.
        edges:
            Instance-graph edges as ``(source_id, relation_type, target_id)``.
        anchors:
            Fixed-voltage boundary conditions: ``{entity_id: voltage}``.
        relation_type:
            When given, only edges of this type are included in the circuit.
            Entities unreachable from the anchor in the filtered subgraph
            receive potential 0.0.

        Returns
        -------
        dict[str, float]
            Potential at every entity appearing in ``edges`` or ``anchors``.
            Anchors carry their declared voltage exactly.
            Unreachable entities receive 0.0.
        """
        filtered = _filter_edges(edges, relation_type)
        return _compute_field(namespace, filtered, anchors)

    @staticmethod
    def compute_and(
        namespace: str,
        edges: list[tuple[str, str, str]],
        source: dict[str, float],
        sink: dict[str, float],
    ) -> dict[str, float]:
        """Series-circuit AND query: score entities by the current they carry.

        Sets up one circuit with ``source`` entities at high voltage and
        ``sink`` entities at low voltage.  Solves the Dirichlet problem, then
        scores each interior entity by the current flowing through it:

            I(e) = sum over neighbors j where V(j) > V(e):  G(e,j) * (V(j) - V(e))

        Entities on a conducting path between source and sink carry current
        and score > 0.  Entities whose only connections lead toward one side
        (a dead-end branch) float to that side's extreme potential, yielding
        zero potential difference and thus zero current.

        No relation-type filtering is required.  The circuit structure itself
        enforces the AND constraint: an entity must be connected through the
        graph to *both* a source-group node and a sink-group node to carry
        current.

        Parameters
        ----------
        namespace:
            Ontology namespace for conductance lookup.
        edges:
            Instance-graph edges as ``(source_id, relation_type, target_id)``.
        source:
            High-voltage anchors, e.g. ``{"Nolan": 1.0}``.
        sink:
            Low-voltage anchors, e.g. ``{"Sci-Fi": 0.0}``.

        Returns
        -------
        dict[str, float]
            Current score for every non-anchor entity.  Anchor entities are
            excluded from the result.

        Example
        -------
        Find sci-fi movies directed by Nolan::

            SemanticField.compute_and(
                "movie", edges,
                source={"Nolan": 1.0},
                sink={"Sci-Fi": 0.0},
            )

        Dark Knight: connected to Nolan (V=1.0) via directed_by but its only
        other connection is to Action — a dead end with no path to Sci-Fi.
        Dark Knight and Action both float to V=1.0.  Zero potential difference,
        zero current.  Score = 0.0.

        Interstellar: connected to Nolan (1.0) via directed_by AND to Sci-Fi
        (0.0) via has_genre.  Sits between source and sink at V=0.5625.
        Carries current 0.9 * (1.0 - 0.5625) = 0.394.  Score > 0.
        """
        combined_anchors: dict[str, float] = {**sink, **source}
        potentials = _compute_field(namespace, edges, combined_anchors)

        adj: dict[str, dict[str, float]] = {}
        for src, relation, tgt in edges:
            g = _conductance(namespace, relation)
            adj.setdefault(src, {})[tgt] = adj.get(src, {}).get(tgt, 0.0) + g
            adj.setdefault(tgt, {})[src] = adj.get(tgt, {}).get(src, 0.0) + g

        scores: dict[str, float] = {}
        for entity, v_e in potentials.items():
            if entity in combined_anchors:
                continue
            current = sum(
                g * (potentials.get(neighbor, 0.0) - v_e)
                for neighbor, g in adj.get(entity, {}).items()
                if potentials.get(neighbor, 0.0) > v_e
            )
            scores[entity] = current

        return scores
