"""Integration tests: ontology layer wired through a real KG connector.

These tests exercise the full path:
  credential slot (ontology key) → _prepare_load → OntologyRegistry → ctx.ontology_namespace
  → typed traversal validation on a real NetworkX graph.

The fixture graph (metaqa_tiny.gml) is a 10-node, 11-edge directed graph with
real MetaQA relation names, node ``type`` attributes, and edge ``relation``
attributes. It includes one intentionally invalid edge (Crime → directed_by →
Christopher Nolan) to exercise ontology violation detection.

All tests run in-process with no network, no Docker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TINY_GML = FIXTURE_DIR / "metaqa_tiny.gml"
METAQA_YAML = FIXTURE_DIR / "metaqa_ontology.yaml"


# ---------------------------------------------------------------------------
# Shared traversal helper
# ---------------------------------------------------------------------------


def traverse(
    graph: nx.DiGraph,
    start: str,
    relationship: str,
    namespace: str | None = None,
) -> list[str]:
    """Return nodes reachable from ``start`` via edges with ``relation == relationship``.

    When ``namespace`` is supplied the traversal is ontology-validated:
    - Source entity type is read from the node's ``type`` attribute.
    - ``OntologyRegistry.is_valid_edge`` is checked before following edges.
    - If the edge is ontology-invalid, ``ValueError`` is raised rather than
      silently returning empty results.
    - Each target node's type is checked against the relationship's declared
      range; a mismatch raises ``ValueError``.

    When ``namespace`` is None no ontology checks are applied (plain graph walk).
    This makes the ontology layer strictly opt-in and backward-compatible.
    """
    if namespace is not None:
        entity_type: str = graph.nodes[start].get("type", "Unknown")
        if not OntologyRegistry.is_valid_edge(namespace, entity_type, relationship):
            raise ValueError(
                f"Ontology violation: '{relationship}' is not a valid outgoing "
                f"relationship from entity type '{entity_type}' in namespace '{namespace}'."
            )
        expected_range = OntologyRegistry.get_range_type(namespace, relationship)

    results: list[str] = []
    for _, target, data in graph.out_edges(start, data=True):
        if data.get("relation") != relationship:
            continue
        if namespace is not None and expected_range is not None:
            target_type: str = graph.nodes[target].get("type", "Unknown")
            if target_type != expected_range:
                raise ValueError(
                    f"Ontology range violation: '{relationship}' expects range "
                    f"'{expected_range}' but reached node '{target}' of type '{target_type}'."
                )
        results.append(target)
    return sorted(results)


# ---------------------------------------------------------------------------
# Credential → LoadContext pipeline
# ---------------------------------------------------------------------------


class TestCredentialPipeline:
    """The ontology key in a credential slot must reach ctx.ontology_namespace."""

    def test_ontology_namespace_set_when_key_present(self) -> None:
        from open_kgo.feature_groups.kg.embedded.networkx_embedded import NetworkxEmbeddedReader

        slot: dict[str, Any] = {
            "locator": str(TINY_GML),
            "graph_file_format": "gml",
            "read_only": True,
            "max_threads": 1,
            "ontology": str(METAQA_YAML),
        }
        ctx = NetworkxEmbeddedReader._prepare_load(slot)
        assert ctx.ontology_namespace == "movie"

    def test_ontology_namespace_none_when_key_absent(self) -> None:
        from open_kgo.feature_groups.kg.embedded.networkx_embedded import NetworkxEmbeddedReader

        slot: dict[str, Any] = {
            "locator": str(TINY_GML),
            "graph_file_format": "gml",
            "read_only": True,
            "max_threads": 1,
        }
        ctx = NetworkxEmbeddedReader._prepare_load(slot)
        assert ctx.ontology_namespace is None

    def test_same_file_loaded_twice_is_idempotent(self) -> None:
        from open_kgo.feature_groups.kg.embedded.networkx_embedded import NetworkxEmbeddedReader

        slot: dict[str, Any] = {
            "locator": str(TINY_GML),
            "graph_file_format": "gml",
            "read_only": True,
            "max_threads": 1,
            "ontology": str(METAQA_YAML),
        }
        ctx1 = NetworkxEmbeddedReader._prepare_load(slot)
        ctx2 = NetworkxEmbeddedReader._prepare_load(slot)
        assert ctx1.ontology_namespace == ctx2.ontology_namespace == "movie"


# ---------------------------------------------------------------------------
# 1-hop validation
# ---------------------------------------------------------------------------


class TestOneHopValidation:
    @pytest.fixture(autouse=True)
    def load_graph(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        self.g: nx.DiGraph = nx.read_gml(str(TINY_GML))

    def test_movie_directed_by_person_is_valid(self) -> None:
        results = traverse(self.g, "The Dark Knight", "directed_by", namespace="movie")
        assert results == ["Christopher Nolan"]

    def test_movie_starred_actors_is_valid(self) -> None:
        results = traverse(self.g, "Inception", "starred_actors", namespace="movie")
        assert results == ["Leonardo DiCaprio"]

    def test_movie_has_genre_is_valid(self) -> None:
        results = traverse(self.g, "The Dark Knight", "has_genre", namespace="movie")
        assert sorted(results) == ["Action", "Crime"]

    def test_genre_outgoing_raises_ontology_violation(self) -> None:
        # Genre has no valid outgoing edges — any relationship attempt is blocked
        with pytest.raises(ValueError, match="Ontology violation"):
            traverse(self.g, "Action", "directed_by", namespace="movie")

    def test_person_outgoing_raises_ontology_violation(self) -> None:
        # Person has no valid outgoing in MetaQA KB — graph is movie-centric
        with pytest.raises(ValueError, match="Ontology violation"):
            traverse(self.g, "Christopher Nolan", "directed_by", namespace="movie")

    def test_invalid_edge_in_graph_caught_by_ontology(self) -> None:
        # Crime → directed_by → Christopher Nolan exists in the fixture
        # but Crime is Genre which has no valid outgoing — ontology catches it
        with pytest.raises(ValueError, match="Ontology violation"):
            traverse(self.g, "Crime", "directed_by", namespace="movie")

    def test_no_namespace_follows_invalid_edge_silently(self) -> None:
        # Without ontology the bad edge is followed without complaint
        results = traverse(self.g, "Crime", "directed_by", namespace=None)
        assert "Christopher Nolan" in results


# ---------------------------------------------------------------------------
# Multi-hop validation
# ---------------------------------------------------------------------------


class TestMultiHopValidation:
    @pytest.fixture(autouse=True)
    def load_graph(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        self.g: nx.DiGraph = nx.read_gml(str(TINY_GML))

    def test_2hop_find_movies_sharing_director(self) -> None:
        """2-hop: find all movies that share a director with The Dark Knight.

        Hop 1: The Dark Knight → directed_by → [Christopher Nolan]
        Hop 2 (reverse): find all movies where directed_by target == Christopher Nolan
        """
        directors = traverse(self.g, "The Dark Knight", "directed_by", namespace="movie")
        assert directors == ["Christopher Nolan"]

        shared = [
            m
            for m in self.g.nodes()
            if self.g.nodes[m].get("type") == "Movie"
            and any(t in directors and d.get("relation") == "directed_by" for _, t, d in self.g.out_edges(m, data=True))
        ]
        assert "Inception" in shared

    def test_3hop_genres_of_movies_by_same_director(self) -> None:
        """3-hop: genres of all movies by the director of The Dark Knight.

        Hop 1: Movie → directed_by → Person
        Hop 2: find all Movies with same director (reverse lookup)
        Hop 3: Movie → has_genre → Genre
        """
        directors = traverse(self.g, "The Dark Knight", "directed_by", namespace="movie")
        all_genres: list[str] = []
        for movie in self.g.nodes():
            if self.g.nodes[movie].get("type") != "Movie":
                continue
            movie_directors = traverse(self.g, movie, "directed_by", namespace="movie")
            if any(d in directors for d in movie_directors):
                genres = traverse(self.g, movie, "has_genre", namespace="movie")
                all_genres.extend(genres)

        assert "Action" in all_genres
        assert "Crime" in all_genres

    def test_ontology_blocks_invalid_hop_mid_chain(self) -> None:
        """Ontology must block an invalid hop anywhere in the chain, not just at start."""
        genres = traverse(self.g, "The Dark Knight", "has_genre", namespace="movie")
        assert len(genres) > 0
        for genre in genres:
            with pytest.raises(ValueError, match="Ontology violation"):
                traverse(self.g, genre, "directed_by", namespace="movie")


# ---------------------------------------------------------------------------
# Standalone API (importable outside the repo)
# ---------------------------------------------------------------------------


class TestStandaloneAPI:
    """OntologyRegistry is usable with zero mloda imports.

    A user outside this repo can do:
        from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry
        OntologyRegistry.load_file("my_ontology.yaml")
        OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by")
    """

    def test_load_and_query_without_connector(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by") is True
        assert OntologyRegistry.is_valid_edge("movie", "Movie", "has_genre") is True
        assert OntologyRegistry.is_valid_edge("movie", "Genre", "directed_by") is False
        assert OntologyRegistry.is_valid_edge("movie", "Person", "directed_by") is False
        assert OntologyRegistry.get_range_type("movie", "directed_by") == "Person"
        assert OntologyRegistry.get_range_type("movie", "has_genre") == "Genre"
        assert "directed_by" in OntologyRegistry.valid_next_hops("movie", "Movie")
        assert OntologyRegistry.valid_next_hops("movie", "Genre") == frozenset()
        assert OntologyRegistry.valid_next_hops("movie", "Person") == frozenset()

    def test_two_namespaces_independent(self, tmp_path: Path) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        bio = tmp_path / "bio.yaml"
        bio.write_text(
            "namespace: biomedical\n"
            "entities:\n"
            "  Protein:\n"
            "    valid_outgoing: [interacts_with]\n"
            "relationships:\n"
            "  interacts_with:\n"
            "    domain: Protein\n"
            "    range: Protein\n",
            encoding="utf-8",
        )
        OntologyRegistry.load_file(str(bio))

        assert OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by") is True
        assert OntologyRegistry.is_valid_edge("biomedical", "Protein", "interacts_with") is True
        assert OntologyRegistry.is_valid_edge("biomedical", "Protein", "directed_by") is False
        assert OntologyRegistry.is_valid_edge("movie", "Protein", "interacts_with") is True  # unknown type passes
