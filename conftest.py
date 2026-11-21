"""Test fixtures shared across the repo; lives at the root so the published wheel never ships it.

``kg.fixtures`` memoises file-backed parses and FDs in process-wide ``lru_cache`` instances, and the
path-only kuzu cache cannot self-invalidate, so every KG test starts and ends with the caches and the
ontology registry cleared. Tests outside ``open_kgo/feature_groups/kg/`` keep their state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from open_kgo.feature_groups.kg.fixtures import (
    _open_kuzu_database_cached,
    _read_json_cached,
    _read_oxigraph_store_cached,
    _read_rdf_graph_cached,
)
from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

_KG_DIR = Path(__file__).resolve().parent / "open_kgo" / "feature_groups" / "kg"


def _clear() -> None:
    _read_json_cached.cache_clear()
    _read_rdf_graph_cached.cache_clear()
    _read_oxigraph_store_cached.cache_clear()
    _open_kuzu_database_cached.cache_clear()
    OntologyRegistry._clear()


@pytest.fixture(autouse=True)
def _clear_kg_resource_caches(request: pytest.FixtureRequest) -> Iterator[None]:
    """Clear every resource cache before and after each test under ``open_kgo/feature_groups/kg/``."""
    kg_test = Path(request.node.path).resolve().is_relative_to(_KG_DIR)
    if kg_test:
        _clear()
    yield
    if kg_test:
        _clear()
