"""KG-test fixtures: keep the shared resource caches isolated across tests.

The ``kg.fixtures`` module memoises file-backed parses and FDs in
process-wide ``lru_cache`` instances. Across tests this can produce two
distinct kinds of cross-test contamination:

- mtime-keyed caches (``_read_json_cached``, ``_read_rdf_graph_cached``)
  auto-invalidate when the underlying file changes, but a test that
  inspects identity / call counts can still see entries seeded by an
  earlier test if it points at the same canonical fixture.
- the path-only kuzu cache (``_open_kuzu_database_cached``) cannot
  self-invalidate, so a test that rotates the directory at the same
  path as an earlier test would otherwise get the stale ``Database``
  handle back. ``pytest``'s ``tmp_path`` is unique per test in normal
  runs, but the safety property is "no test re-uses a kuzu locator
  path that an earlier test cleaned up" — relying on that as an
  implicit invariant is what the autouse clear here removes.

Applies to every test under ``open_kgo/feature_groups/kg/``
(pytest discovers ``conftest.py`` recursively) so existing per-family
contract tests benefit, not only ``tests/test_resource_cache.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from open_kgo.feature_groups.kg.fixtures import (
    _open_kuzu_database_cached,
    _read_json_cached,
    _read_rdf_graph_cached,
)
from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry


@pytest.fixture(autouse=True)
def _clear_kg_resource_caches() -> Any:
    """Clear every resource cache before and after each KG test."""
    _read_json_cached.cache_clear()
    _read_rdf_graph_cached.cache_clear()
    _open_kuzu_database_cached.cache_clear()
    OntologyRegistry._clear()
    yield
    _read_json_cached.cache_clear()
    _read_rdf_graph_cached.cache_clear()
    _open_kuzu_database_cached.cache_clear()
    OntologyRegistry._clear()
