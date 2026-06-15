"""Tests for SemanticField — DC circuit model (Layer 2).

Key circuit properties tested:

Single-anchor (``compute``):
  - With one voltage source and no ground reference, all nodes in the same
    connected component float to the source voltage.  Disconnected nodes
    receive 0.0.
  - Ontology weights (conductances) shape the potential when an entity sits
    *between* two anchors at different voltages.

Series-circuit AND (``compute_and``):
  - Source anchor at high voltage, sink anchor at low voltage, one circuit.
  - Score = current carried by each interior entity.
  - Entities that bridge source and sink carry current (score > 0).
  - Entities hanging off only one side float to that side's extreme potential
    and carry zero current — no relation-type filtering required.
  - Dark Knight: connected to Nolan (high) via directed_by, to Action (dead
    end) via has_genre.  Floats to V=1.0.  Zero current.  Score = 0.0.

Layer-1 (OntologyRegistry) is reset before every test by the conftest
autouse fixture.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry
from open_kgo.feature_groups.kg.ontology.semantic_field import SemanticField, _conductance

FIXTURE_DIR = Path(__file__).parent / "fixtures"
METAQA_YAML = FIXTURE_DIR / "metaqa_ontology.yaml"

# (source, relation_type, target)
MOVIE_EDGES: list[tuple[str, str, str]] = [
    ("Interstellar", "directed_by", "Nolan"),
    ("Interstellar", "has_genre", "Sci-Fi"),
    ("DarkKnight", "directed_by", "Nolan"),
    ("DarkKnight", "has_genre", "Action"),
    ("Inception", "directed_by", "Nolan"),
    ("Inception", "has_genre", "Sci-Fi"),
    ("Arrival", "directed_by", "Villeneuve"),
    ("Arrival", "has_genre", "Sci-Fi"),
]


class TestConductance:
    def test_declared_weight_returned(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert _conductance("movie", "directed_by") == pytest.approx(0.9)
        assert _conductance("movie", "has_genre") == pytest.approx(0.7)
        assert _conductance("movie", "has_tags") == pytest.approx(0.2)

    def test_unknown_relation_defaults_to_one(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert _conductance("movie", "nonexistent_rel") == pytest.approx(1.0)

    def test_no_ontology_defaults_to_one(self) -> None:
        assert _conductance("movie", "directed_by") == pytest.approx(1.0)


class TestComputeSingleAnchor:
    def test_anchor_entity_carries_its_declared_voltage(self) -> None:
        potentials = SemanticField.compute("movie", MOVIE_EDGES, {"Nolan": 1.0})
        assert potentials["Nolan"] == pytest.approx(1.0)

    def test_directly_connected_entity_has_nonzero_potential(self) -> None:
        potentials = SemanticField.compute("movie", MOVIE_EDGES, {"Nolan": 1.0})
        assert potentials["Interstellar"] > 0.0
        assert potentials["DarkKnight"] > 0.0

    def test_disconnected_entity_has_zero_potential(self) -> None:
        isolated_edges = [
            ("MovieX", "directed_by", "DirectorY"),
            ("Interstellar", "directed_by", "Nolan"),
        ]
        potentials = SemanticField.compute("movie", isolated_edges, {"Nolan": 1.0})
        assert potentials.get("MovieX", 0.0) == pytest.approx(0.0)

    def test_anchor_at_zero_voltage_gives_zero_everywhere(self) -> None:
        potentials = SemanticField.compute("movie", MOVIE_EDGES, {"Nolan": 0.0})
        for v in potentials.values():
            assert v == pytest.approx(0.0, abs=1e-9)

    def test_relation_type_filter_isolates_subgraph(self) -> None:
        potentials = SemanticField.compute("movie", MOVIE_EDGES, {"Nolan": 1.0}, relation_type="directed_by")
        assert potentials["Interstellar"] > 0.0
        assert potentials.get("Arrival", 0.0) == pytest.approx(0.0)

    def test_ontology_weights_shape_intermediate_potential(self) -> None:
        """Weights as conductances determine the voltage divider ratio.

        Circuit: Nolan (1.0) --directed_by (G=0.9)--> Movie_A --has_genre (G=0.7)--> Genre (0.0)
        V(Movie_A) = G_left / (G_left + G_right) = 0.9 / 1.6 = 0.5625

        Circuit: Nolan (1.0) --has_tags (G=0.2)--> Movie_B --has_genre (G=0.7)--> Genre (0.0)
        V(Movie_B) = 0.2 / (0.2 + 0.7) = 0.222...
        """
        OntologyRegistry.load_file(str(METAQA_YAML))
        edges = [
            ("Movie_A", "directed_by", "Nolan"),
            ("Movie_A", "has_genre", "Genre"),
            ("Movie_B", "has_tags", "Nolan"),
            ("Movie_B", "has_genre", "Genre"),
        ]
        potentials = SemanticField.compute("movie", edges, {"Nolan": 1.0, "Genre": 0.0})
        assert potentials["Movie_A"] == pytest.approx(0.9 / (0.9 + 0.7), abs=1e-6)
        assert potentials["Movie_B"] == pytest.approx(0.2 / (0.2 + 0.7), abs=1e-6)
        assert potentials["Movie_A"] > potentials["Movie_B"]


class TestComputeAnd:
    """Series-circuit AND: source (V=high) and sink (V=low) in one circuit.

    Score = current through each interior entity.  Entities that bridge
    source and sink carry current.  Dead-end branches carry zero.
    """

    def test_entity_bridging_source_and_sink_scores_nonzero(self) -> None:
        # Interstellar: Nolan (source) via directed_by AND Sci-Fi (sink) via has_genre
        scores = SemanticField.compute_and("movie", MOVIE_EDGES, source={"Nolan": 1.0}, sink={"Sci-Fi": 0.0})
        assert scores["Interstellar"] > 0.0
        assert scores["Inception"] > 0.0

    def test_darkknight_carries_zero_current(self) -> None:
        # DarkKnight: Nolan (1.0) via directed_by, Action (dead end) via has_genre.
        # No path from DarkKnight to Sci-Fi (0.0).
        # DarkKnight and Action both float to V=1.0 — zero potential difference.
        scores = SemanticField.compute_and("movie", MOVIE_EDGES, source={"Nolan": 1.0}, sink={"Sci-Fi": 0.0})
        assert scores["DarkKnight"] == pytest.approx(0.0, abs=1e-9)

    def test_arrival_carries_zero_current(self) -> None:
        # Arrival: Sci-Fi (0.0) via has_genre, Villeneuve (dead end) via directed_by.
        # No path from Arrival to Nolan (1.0).
        # Arrival and Villeneuve both float to V=0.0.
        scores = SemanticField.compute_and("movie", MOVIE_EDGES, source={"Nolan": 1.0}, sink={"Sci-Fi": 0.0})
        assert scores["Arrival"] == pytest.approx(0.0, abs=1e-9)

    def test_interstellar_scores_higher_than_darkknight_and_arrival(self) -> None:
        scores = SemanticField.compute_and("movie", MOVIE_EDGES, source={"Nolan": 1.0}, sink={"Sci-Fi": 0.0})
        assert scores["Interstellar"] > scores["DarkKnight"]
        assert scores["Interstellar"] > scores["Arrival"]

    def test_current_matches_verified_calculation(self) -> None:
        # Verified analytically: V(Interstellar) = 0.9/1.6 = 0.5625
        # I(Interstellar) = G(Nolan,I) * (V(Nolan) - V(I)) = 0.9 * (1.0 - 0.5625) = 0.39375
        OntologyRegistry.load_file(str(METAQA_YAML))
        scores = SemanticField.compute_and("movie", MOVIE_EDGES, source={"Nolan": 1.0}, sink={"Sci-Fi": 0.0})
        assert scores["Interstellar"] == pytest.approx(0.9 * (1.0 - 0.9 / 1.6), abs=1e-6)

    def test_no_ontology_still_computes(self) -> None:
        scores = SemanticField.compute_and(
            "unknown_namespace", MOVIE_EDGES, source={"Nolan": 1.0}, sink={"Sci-Fi": 0.0}
        )
        assert scores["Interstellar"] > 0.0
        assert scores["DarkKnight"] == pytest.approx(0.0, abs=1e-9)

    def test_anchors_not_in_result(self) -> None:
        scores = SemanticField.compute_and("movie", MOVIE_EDGES, source={"Nolan": 1.0}, sink={"Sci-Fi": 0.0})
        assert "Nolan" not in scores
        assert "Sci-Fi" not in scores


class TestRelationshipWeight:
    def test_weight_loaded_from_yaml(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        ontology = OntologyRegistry.get("movie")
        assert ontology is not None
        assert ontology.relationships["directed_by"].weight == pytest.approx(0.9)
        assert ontology.relationships["has_tags"].weight == pytest.approx(0.2)

    def test_missing_weight_defaults_to_one(self) -> None:
        yaml_content = (
            "namespace: test_ns\n"
            "entities:\n"
            "  A:\n"
            "    valid_outgoing: [rel]\n"
            "relationships:\n"
            "  rel:\n"
            "    domain: A\n"
            "    range: B\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp = f.name

        OntologyRegistry.load_file(tmp)
        ontology = OntologyRegistry.get("test_ns")
        assert ontology is not None
        assert ontology.relationships["rel"].weight == pytest.approx(1.0)
