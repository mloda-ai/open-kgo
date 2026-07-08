"""DC circuit semantic field for ontology-grounded query scoring (Layer 2).

Models the knowledge graph as a resistor network; the Laplacian construction
and Dirichlet solve live in the sibling ``dc_solver`` module (shared with
``discovery.DiscoveryEngine``, Layer 3, which reuses the same solved
potentials). This module owns the public scoring API:

  - Semantic field    = electric potential V(entity) at every node

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
"""

from __future__ import annotations

from open_kgo.feature_groups.kg.ontology.dc_solver import compute_field, conductance, filter_edges

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
        filtered = filter_edges(edges, relation_type)
        return compute_field(namespace, filtered, anchors)

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
        potentials = compute_field(namespace, edges, combined_anchors)

        adj: dict[str, dict[str, float]] = {}
        for src, relation, tgt in edges:
            g = conductance(namespace, relation)
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
