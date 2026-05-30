"""Tests for OntologyRegistry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

FIXTURE_DIR = Path(__file__).parent / "fixtures"
METAQA_YAML = FIXTURE_DIR / "metaqa_ontology.yaml"


class TestLoadFile:
    def test_loads_and_returns_namespace(self) -> None:
        ns = OntologyRegistry.load_file(str(METAQA_YAML))
        assert ns == "movie"

    def test_same_file_twice_is_noop(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        ns = OntologyRegistry.load_file(str(METAQA_YAML))
        assert ns == "movie"

    def test_different_file_same_namespace_raises(self, tmp_path: Path) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        duplicate = tmp_path / "other.yaml"
        duplicate.write_text("namespace: movie\nentities: {}\nrelationships: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="already registered"):
            OntologyRegistry.load_file(str(duplicate))

    def test_missing_namespace_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("entities: {}\nrelationships: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing the required 'namespace' key"):
            OntologyRegistry.load_file(str(bad))

    def test_non_mapping_top_level_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_list.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            OntologyRegistry.load_file(str(bad))

    def test_relationship_missing_domain_or_range_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_rel.yaml"
        bad.write_text(
            "namespace: movie\nentities: {}\nrelationships:\n  directed_by:\n    domain: Movie\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="must be a mapping declaring both 'domain' and 'range'"):
            OntologyRegistry.load_file(str(bad))

    def test_empty_entities_and_relationships_loads_cleanly(self, tmp_path: Path) -> None:
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text("namespace: empty_domain\nentities: {}\nrelationships: {}\n", encoding="utf-8")
        ns = OntologyRegistry.load_file(str(minimal))
        assert ns == "empty_domain"


class TestIsValidEdge:
    def test_valid_edge_returns_true(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by") is True

    def test_invalid_edge_returns_false(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        # Genre has no valid outgoing edges
        assert OntologyRegistry.is_valid_edge("movie", "Genre", "directed_by") is False

    def test_person_valid_edges(self) -> None:
        # Person has no valid outgoing edges in MetaQA — the KB is movie-centric
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.is_valid_edge("movie", "Person", "directed_by") is False
        assert OntologyRegistry.is_valid_edge("movie", "Person", "has_genre") is False

    def test_unknown_entity_type_passes_through(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        # Unknown entity types must not block traversal
        assert OntologyRegistry.is_valid_edge("movie", "UnknownType", "directed_by") is True

    def test_no_ontology_registered_passes_through(self) -> None:
        # No file loaded: registry has no entry for this namespace
        assert OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by") is True

    def test_unknown_namespace_passes_through(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.is_valid_edge("biomedical", "Protein", "interacts_with") is True


class TestGetRangeType:
    def test_known_relationship_returns_range(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.get_range_type("movie", "directed_by") == "Person"
        assert OntologyRegistry.get_range_type("movie", "has_genre") == "Genre"
        assert OntologyRegistry.get_range_type("movie", "starred_actors") == "Person"

    def test_unknown_relationship_returns_none(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.get_range_type("movie", "nonexistent_rel") is None

    def test_no_ontology_returns_none(self) -> None:
        assert OntologyRegistry.get_range_type("movie", "directed_by") is None


class TestValidNextHops:
    def test_movie_has_expected_outgoing(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        hops = OntologyRegistry.valid_next_hops("movie", "Movie")
        assert "directed_by" in hops
        assert "has_genre" in hops
        assert "starred_actors" in hops
        # Person/Genre have no valid outgoing — those relations must not appear here
        assert "in_language" in hops

    def test_genre_has_no_outgoing(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.valid_next_hops("movie", "Genre") == frozenset()

    def test_person_has_no_outgoing(self) -> None:
        # Person is terminal in MetaQA — the KB is movie-centric with no inverse edges
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.valid_next_hops("movie", "Person") == frozenset()

    def test_unknown_entity_returns_empty(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.valid_next_hops("movie", "UnknownType") == frozenset()

    def test_no_ontology_returns_empty(self) -> None:
        assert OntologyRegistry.valid_next_hops("movie", "Movie") == frozenset()


class TestMultiNamespace:
    def test_two_namespaces_isolated(self, tmp_path: Path) -> None:
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

        # movie namespace unaffected
        assert OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by") is True
        assert OntologyRegistry.is_valid_edge("movie", "Protein", "interacts_with") is True  # unknown type passes

        # biomedical namespace independent
        assert OntologyRegistry.is_valid_edge("biomedical", "Protein", "interacts_with") is True
        assert OntologyRegistry.is_valid_edge("biomedical", "Movie", "directed_by") is True  # unknown type passes
        assert OntologyRegistry.is_valid_edge("biomedical", "Protein", "directed_by") is False


class TestClear:
    def test_clear_removes_all_entries(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        assert OntologyRegistry.get("movie") is not None
        OntologyRegistry._clear()
        assert OntologyRegistry.get("movie") is None

    def test_reload_after_clear(self) -> None:
        OntologyRegistry.load_file(str(METAQA_YAML))
        OntologyRegistry._clear()
        # Must be loadable again after clear
        ns = OntologyRegistry.load_file(str(METAQA_YAML))
        assert ns == "movie"


def test_registry_get_returns_none_for_unregistered(tmp_path: Any) -> None:
    assert OntologyRegistry.get("unregistered") is None
