"""Shared DC circuit primitives for the ontology scoring layers.

Solves the conductance-weighted graph Laplacian ``L * V = s`` with anchor
voltages as Dirichlet boundary conditions. Used by both
``semantic_field.SemanticField`` (Layer 2) and ``discovery.DiscoveryEngine``
(Layer 3).
"""

from __future__ import annotations

from typing import Any

from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

# Optional numba acceleration for the Gaussian elimination kernel; falls back
# to pure Python when numba / numpy are not installed.
_np_mod: Any = None
_nb_solve: Any = None  # compiled kernel: (aug: ndarray) -> ndarray

try:
    import numpy as _numba_np
    from numba import njit as _njit

    @_njit(cache=True)
    def _nb_gauss_elim(aug):  # type: ignore[no-untyped-def]
        """Gaussian elimination with partial pivoting on an augmented matrix (in-place, m x m+1)."""
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


def conductance(namespace: str, relation: str) -> float:
    """Return the declared ontology weight for ``relation``, or 1.0 if unknown/unregistered."""
    ontology = OntologyRegistry.get(namespace)
    if ontology is None:
        return 1.0
    rule = ontology.relationships.get(relation)
    return rule.weight if rule is not None else 1.0


def filter_edges(
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
    """Return the conductance-weighted graph Laplacian as a 2-D list (edges treated as undirected)."""
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    L: list[list[float]] = [[0.0] * n for _ in range(n)]
    for src, relation, tgt in edges:
        if src not in idx or tgt not in idx:
            continue
        g = conductance(namespace, relation)
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
    """Solve L_II * V_I = -L_IB * V_B via Gaussian elimination (boundary = known voltages, interior = unknowns)."""
    m = len(interior)
    if m == 0:
        return []

    aug: list[list[float]] = []
    for row_idx in interior:
        aug_row: list[float] = [L[row_idx][col_idx] for col_idx in interior]
        rhs_val = -sum(L[row_idx][b] * v for b, v in boundary.items())
        aug_row.append(rhs_val)
        aug.append(aug_row)

    if _nb_solve is not None:
        aug_arr = _np_mod.array(aug, dtype=_np_mod.float64)
        return list(_nb_solve(aug_arr))

    # Pure-Python fallback: forward elimination with partial pivoting, then back substitution.
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


def compute_field(
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
