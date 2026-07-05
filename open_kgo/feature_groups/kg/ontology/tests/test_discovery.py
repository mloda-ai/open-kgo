"""Tests for DiscoveryEngine — beam search over EM potential landscape (Layer 3).

The beam search heuristic is the edge current G(i,j) × |V(i) - V(j)|.
Paths are scored by bottleneck current (minimum edge current along the path).
Dead-end branches carry zero current and are naturally excluded.

Layer-1 (OntologyRegistry) is reset before every test by the conftest
autouse fixture.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from open_kgo.feature_groups.kg.ontology.discovery import DiscoveredPath, DiscoveryEngine
from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

FIXTURE_DIR = Path(__file__).parent / "fixtures"
METAQA_YAML = FIXTURE_DIR / "metaqa_ontology.yaml"

# Mirrors metaqa_tiny.gml (10 nodes, 11 edges including the intentionally invalid edge).
GML_EDGES: list[tuple[str, str, str]] = [
    ("The Dark Knight", "directed_by", "Christopher Nolan"),
    ("The Dark Knight", "starred_actors", "Christian Bale"),
    ("The Dark Knight", "has_genre", "Action"),
    ("The Dark Knight", "has_genre", "Crime"),
    ("Inception", "directed_by", "Christopher Nolan"),
    ("Inception", "starred_actors", "Leonardo DiCaprio"),
    ("Inception", "has_genre", "Action"),
    ("The Godfather", "directed_by", "Francis Ford Coppola"),
    ("The Godfather", "has_genre", "Crime"),
    ("The Godfather", "has_genre", "Drama"),
    ("Crime", "directed_by", "Christopher Nolan"),  # intentionally invalid in ontology
]

# Minimal 3-node circuit for pinned numerical tests.
# Nolan --directed_by(G=0.9)-- Movie_A --has_genre(G=0.7)-- Action
THREE_NODE_EDGES: list[tuple[str, str, str]] = [
    ("Movie_A", "directed_by", "Nolan"),
    ("Movie_A", "has_genre", "Action"),
]


class TestDiscoveredPathDataclass:
    def test_frozen_fields_cannot_be_mutated(self) -> None:
        p = DiscoveredPath(nodes=("A", "B"), relations=("rel",), score=0.5)
        with pytest.raises(Exception):
            p.score = 1.0  # type: ignore[misc]

    def test_nodes_and_relations_length_invariant(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert len(paths) > 0
        for p in paths:
            assert len(p.nodes) == len(p.relations) + 1

    def test_score_is_float(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert len(paths) > 0
        assert isinstance(paths[0].score, float)

    def test_score_nonnegative(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert all(p.score >= 0.0 for p in paths)


class TestFindPathsBasic:
    def test_nolan_to_action_finds_paths(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert len(paths) >= 1
        path_node_sets = [set(p.nodes) for p in paths]
        assert any("The Dark Knight" in ns or "Inception" in ns for ns in path_node_sets)

    def test_paths_sorted_descending_by_score(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        scores = [p.score for p in paths]
        assert scores == sorted(scores, reverse=True)

    def test_first_node_is_source(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert len(paths) > 0
        assert all(p.nodes[0] == "Christopher Nolan" for p in paths)

    def test_last_node_is_sink(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert len(paths) > 0
        assert all(p.nodes[-1] == "Action" for p in paths)

    def test_relations_include_directed_by_and_has_genre(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        all_relations: set[str] = set()
        for p in paths:
            all_relations.update(p.relations)
        assert "directed_by" in all_relations
        assert "has_genre" in all_relations


class TestFindPathsEdgeCases:
    def test_empty_edges_returns_empty(self) -> None:
        paths = DiscoveryEngine.find_paths(
            "movie",
            [],
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert paths == []

    def test_empty_source_returns_empty(self) -> None:
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={},
            sink={"Action": 0.0},
        )
        assert paths == []

    def test_empty_sink_returns_empty(self) -> None:
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={},
        )
        assert paths == []

    def test_source_with_no_edges_returns_empty(self) -> None:
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"NoSuchEntity": 1.0},
            sink={"Action": 0.0},
        )
        assert paths == []

    def test_disconnected_components_returns_empty(self) -> None:
        # Two completely separate components; no path exists between them.
        edges: list[tuple[str, str, str]] = [
            ("Movie_A", "directed_by", "Director_A"),
            ("Movie_B", "has_genre", "Genre_B"),
        ]
        paths = DiscoveryEngine.find_paths(
            "movie",
            edges,
            source={"Director_A": 1.0},
            sink={"Genre_B": 0.0},
        )
        assert paths == []

    def test_source_equals_sink_returns_trivial_path(self) -> None:
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Christopher Nolan": 0.0},
        )
        assert len(paths) == 1
        assert paths[0].nodes == ("Christopher Nolan",)
        assert paths[0].relations == ()
        assert paths[0].score == math.inf

    def test_max_depth_one_does_not_reach_two_hop_sink(self) -> None:
        # Action is 2 hops from Nolan (Nolan→Movie→Action); max_depth=1 should miss it.
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
            max_depth=1,
        )
        assert paths == []

    def test_max_paths_limits_output(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
            max_paths=1,
        )
        assert len(paths) <= 1

    def test_beam_width_one_still_finds_path(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
            beam_width=1,
        )
        assert len(paths) >= 1


class TestFindPathsCyclePrevention:
    def test_no_path_visits_same_node_twice(self) -> None:
        # Crime ↔ Christopher Nolan creates a potential cycle.
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
            max_depth=6,
        )
        for p in paths:
            assert len(p.nodes) == len(set(p.nodes)), f"cycle detected in path: {p.nodes}"

    def test_max_depth_caps_path_length(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
            max_depth=2,
        )
        assert all(len(p.nodes) <= 3 for p in paths)


class TestFindPathsOntologyWeights:
    def test_ontology_shapes_scores(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths_with_ontology = DiscoveryEngine.find_paths(
            "movie",
            THREE_NODE_EDGES,
            source={"Nolan": 1.0},
            sink={"Action": 0.0},
        )
        paths_no_ontology = DiscoveryEngine.find_paths(
            "unknown_ns",
            THREE_NODE_EDGES,
            source={"Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert len(paths_with_ontology) == 1
        assert len(paths_no_ontology) == 1
        # With equal conductances (G=1.0 fallback) the voltage divider is symmetric;
        # with unequal weights it is not — scores differ.
        assert paths_with_ontology[0].score != pytest.approx(paths_no_ontology[0].score, abs=1e-6)

    def test_unknown_namespace_uses_unit_conductance(self) -> None:
        # All conductances = 1.0 → voltage divider is symmetric → V(Movie_A) = 0.5
        # → both edge currents = 1.0 × 0.5 = 0.5 → bottleneck = 0.5
        paths = DiscoveryEngine.find_paths(
            "unknown_ns",
            THREE_NODE_EDGES,
            source={"Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert len(paths) == 1
        assert paths[0].score == pytest.approx(0.5, abs=1e-6)

    def test_bottleneck_score_verified_calculation(self) -> None:
        """Pinned numerical test for the 3-node circuit.

        Nolan (V=1.0) --directed_by(G=0.9)-- Movie_A --has_genre(G=0.7)-- Action (V=0.0)

        V(Movie_A) = 0.9 / (0.9 + 0.7) = 0.5625
        Both edges carry current = 0.39375 (KCL: in == out).
        Bottleneck = 0.39375.
        """
        OntologyRegistry.load_file(str(METAQA_YAML))
        paths = DiscoveryEngine.find_paths(
            "movie",
            THREE_NODE_EDGES,
            source={"Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert len(paths) == 1
        assert paths[0].nodes == ("Nolan", "Movie_A", "Action")
        assert paths[0].relations == ("directed_by", "has_genre")
        # V(Movie_A) = 0.9/1.6; both edges carry 0.9 * (1.0 - 0.9/1.6)
        expected = 0.9 * (1.0 - 0.9 / 1.6)
        assert paths[0].score == pytest.approx(expected, abs=1e-6)


class TestExtractCircuit:
    def test_returns_subset_of_input_edges(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        circuit = DiscoveryEngine.extract_circuit(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        for edge in circuit:
            assert edge in GML_EDGES

    def test_dead_end_actors_excluded(self) -> None:
        # Bale and DiCaprio each connect to exactly one movie — they float to that
        # movie's voltage, giving zero potential difference, zero current.
        OntologyRegistry.load_file(str(METAQA_YAML))
        circuit = DiscoveryEngine.extract_circuit(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        dead_ends = {"Christian Bale", "Leonardo DiCaprio"}
        assert not any(s in dead_ends or t in dead_ends for s, _, t in circuit)

    def test_dead_end_explicitly_excluded(self) -> None:
        # Actor_X connects only to Movie_A — guaranteed dead end.
        edges: list[tuple[str, str, str]] = [
            ("Movie_A", "directed_by", "Nolan"),
            ("Movie_A", "has_genre", "Action"),
            ("Movie_A", "starred_actors", "Actor_X"),
        ]
        OntologyRegistry.load_file(str(METAQA_YAML))
        circuit = DiscoveryEngine.extract_circuit(
            "movie",
            edges,
            source={"Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert not any("Actor_X" in (s, t) for s, _, t in circuit)

    def test_high_threshold_returns_empty(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        circuit = DiscoveryEngine.extract_circuit(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
            current_threshold=999.0,
        )
        assert circuit == []

    def test_zero_threshold_includes_current_carrying_edges(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        circuit = DiscoveryEngine.extract_circuit(
            "movie",
            GML_EDGES,
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
            current_threshold=0.0,
        )
        assert len(circuit) > 0

    def test_empty_edges_returns_empty(self) -> None:
        circuit = DiscoveryEngine.extract_circuit(
            "movie",
            [],
            source={"Christopher Nolan": 1.0},
            sink={"Action": 0.0},
        )
        assert circuit == []

    def test_three_node_circuit_pinned(self) -> None:
        """Pinned: only the two conducting edges survive; both carry 0.39375 > default threshold."""
        OntologyRegistry.load_file(str(METAQA_YAML))
        circuit = DiscoveryEngine.extract_circuit(
            "movie",
            THREE_NODE_EDGES,
            source={"Nolan": 1.0},
            sink={"Action": 0.0},
        )
        # The two edges (Movie_A→Nolan via directed_by) and (Movie_A→Action via has_genre)
        # both carry current. Both should survive the default threshold.
        assert len(circuit) == 2
        edge_set = {(min(s, t), r, max(s, t)) for s, r, t in circuit}
        assert ("Movie_A", "directed_by", "Nolan") in edge_set or (
            "Movie_A",
            "directed_by",
            "Nolan",
        ) in {(s, r, t) for s, r, t in circuit}
        assert any(r == "has_genre" for _, r, _ in circuit)
