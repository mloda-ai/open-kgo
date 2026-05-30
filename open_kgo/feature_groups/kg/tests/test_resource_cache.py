"""Resource-lifecycle / shared-cache contract tests (issue #32 item 3).

Pins the behavior of ``kg.fixtures.load_json_fixture``,
``kg.fixtures.load_rdf_graph``, and ``kg.fixtures.load_kuzu_database`` and
the file-backed concretes that route through them. A future refactor that
silently drops caching (e.g. inlines ``json.load`` back into a
``_connect_from_slot``) passes the existing functional tests today; the
identity / call-count checks below are what make the caching contract
load-bearing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import rdflib

from open_kgo.feature_groups.kg.errors import FixtureLoadError
from open_kgo.feature_groups.kg.fixtures import (
    copy_cached_row,
    load_json_fixture,
    load_kuzu_database,
    load_rdf_graph,
)


# Resource-cache clearing is provided by ``kg/conftest.py``'s autouse
# fixture: it applies to every KG test (this module included), so the
# duplicate fixture that previously lived here was removed when the
# clearing was hoisted up so existing per-family contract tests benefit
# from the same isolation.


def test_load_json_fixture_returns_identical_object_across_calls(tmp_path: Path) -> None:
    """Two calls with the same locator return the same dict instance.

    Identity (``is``), not equality: a future refactor that silently
    rebuilds the dict per call would still satisfy ``==`` but the
    cache would not actually be saving work.
    """
    path = tmp_path / "x.json"
    path.write_text('{"k": "v"}', encoding="utf-8")
    first = load_json_fixture("test", path)
    second = load_json_fixture("test", path)
    assert first is second


def test_load_json_fixture_invalidates_on_mtime_change(tmp_path: Path) -> None:
    """Modifying the underlying file evicts the cached parse via mtime keying."""
    path = tmp_path / "x.json"
    path.write_text('{"k": "v"}', encoding="utf-8")
    first = load_json_fixture("test", path)
    # Bump mtime explicitly so a same-second rewrite still rotates the key.
    new_mtime_ns = path.stat().st_mtime_ns + 1_000_000_000
    path.write_text('{"k": "w"}', encoding="utf-8")
    os.utime(path, ns=(new_mtime_ns, new_mtime_ns))
    second = load_json_fixture("test", path)
    assert first is not second
    assert second == {"k": "w"}


def test_load_json_fixture_rejects_remote_scheme() -> None:
    with pytest.raises(FixtureLoadError):
        load_json_fixture("test", "http://example.invalid/x.json")


def test_load_json_fixture_typed_error_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FixtureLoadError):
        load_json_fixture("test", tmp_path / "nope.json")


def test_load_json_fixture_typed_error_on_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json {", encoding="utf-8")
    with pytest.raises(FixtureLoadError):
        load_json_fixture("test", path)


def test_load_json_fixture_typed_error_on_non_dict_top_level(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(FixtureLoadError):
        load_json_fixture("test", path)


def test_load_rdf_graph_returns_identical_object_across_calls(tmp_path: Path) -> None:
    path = tmp_path / "x.ttl"
    path.write_text(
        "@prefix ex: <http://example.org/> .\nex:a ex:b ex:c .\n",
        encoding="utf-8",
    )
    first = load_rdf_graph("test", path)
    second = load_rdf_graph("test", path)
    assert first is second
    assert isinstance(first, rdflib.Graph)


def test_load_rdf_graph_rejects_remote_scheme() -> None:
    with pytest.raises(FixtureLoadError):
        load_rdf_graph("test", "https://example.invalid/x.ttl")


def test_load_rdf_graph_typed_error_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FixtureLoadError):
        load_rdf_graph("test", tmp_path / "nope.ttl")


def test_load_rdf_graph_typed_error_on_unparseable_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.ttl"
    path.write_text("this is not turtle ::: @@@ ###", encoding="utf-8")
    with pytest.raises(FixtureLoadError):
        load_rdf_graph("test", path)


def test_load_kuzu_database_returns_identical_object_across_calls(tmp_path: Path) -> None:
    """The cached ``kuzu.Database`` is the FD holder; identity confirms no re-open."""
    import kuzu

    db_path = tmp_path / "graph.kuzu"
    seed_db = kuzu.Database(str(db_path))
    seed_conn = kuzu.Connection(seed_db)
    seed_conn.execute("CREATE NODE TABLE T(id STRING, PRIMARY KEY(id))")
    # Drop our seeding handles before exercising the cache: a stale local
    # reference would mask a regression where the cache silently rebuilt.
    del seed_conn
    del seed_db

    first = load_kuzu_database("test", db_path)
    second = load_kuzu_database("test", db_path)
    assert first is second


def test_load_kuzu_database_typed_error_on_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FixtureLoadError):
        load_kuzu_database("test", tmp_path / "nope.kuzu")


def test_load_kuzu_database_rejects_remote_scheme() -> None:
    with pytest.raises(FixtureLoadError):
        load_kuzu_database("test", "http://example.invalid/db.kuzu")


def test_dbt_manifest_reader_uses_shared_cache() -> None:
    """Two ``_connect_from_slot`` calls return the same dict object."""
    from open_kgo.feature_groups.kg.lineage.dbt_manifest import DbtManifestReader

    fixture = Path(__file__).parent.parent / "lineage" / "tests" / "fixtures" / "manifest.json"
    slot = {"locator": str(fixture)}
    first = DbtManifestReader._connect_from_slot(slot)
    second = DbtManifestReader._connect_from_slot(slot)
    assert first is second


def test_cyclonedx_sbom_reader_uses_shared_cache() -> None:
    from open_kgo.feature_groups.kg.code_build.cyclonedx_sbom import CycloneDxSbomReader

    fixture = Path(__file__).parent.parent / "code_build" / "tests" / "fixtures" / "sample.cdx.json"
    slot = {"manifest_path": str(fixture)}
    first = CycloneDxSbomReader._connect_from_slot(slot)
    second = CycloneDxSbomReader._connect_from_slot(slot)
    assert first is second


def test_file_fixture_citation_reader_uses_shared_cache() -> None:
    from open_kgo.feature_groups.kg.citation_rest.file_fixture_citation import (
        FileFixtureCitationReader,
    )

    fixture = Path(__file__).parent.parent / "citation_rest" / "tests" / "fixtures" / "reactome.json"
    slot = {"locator": str(fixture)}
    first = FileFixtureCitationReader._connect_from_slot(slot)
    second = FileFixtureCitationReader._connect_from_slot(slot)
    assert first is second


def test_file_fixture_citation_reader_does_not_leak_cached_row_to_caller(tmp_path: Path) -> None:
    """``load_data`` must shallow-copy each appended row.

    The cache returns a shared dict; if ``load_data`` appended
    ``catalog[node_id]`` directly, a downstream consumer mutating the
    row would poison subsequent loads. The shallow-copy at the row
    boundary keeps the cache read-only at the surface level.
    """
    from mloda.core.abstract_plugins.components.feature_set import FeatureSet
    from mloda.user import Feature, Options

    from open_kgo.feature_groups.kg.citation_rest.file_fixture_citation import (
        FileFixtureCitationReader,
    )

    catalog_path = tmp_path / "tiny_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "X-1": {"stable_id": "X-1", "name": "X1", "ancestors": []},
            }
        ),
        encoding="utf-8",
    )
    creds = {"file_fixture_citation": {"locator": str(catalog_path), "auth_method": "none"}}
    feat = Feature(
        "file_fixture_citation__probe",
        options=Options(context={"stable_id": "X-1", "hierarchy_depth": 0}),
    )
    fs = FeatureSet()
    fs.add(feat)

    rows = FileFixtureCitationReader.load_data(creds, fs)
    assert len(rows) == 1
    rows[0]["name"] = "MUTATED"

    # The cache entry must not have been mutated by the caller-side write above.
    cached = FileFixtureCitationReader._connect_from_slot(creds["file_fixture_citation"])
    assert cached["X-1"]["name"] == "X1", (
        "citation reader leaked a cached row to the caller (mutation surfaced in cache)."
    )


def test_citation_shallow_copy_is_shell_only_not_deep(tmp_path: Path) -> None:
    """Pins the shell-only nature of the row shallow copy (base.py contract).

    The shallow copy at the row boundary (``{**catalog[node_id]}``)
    protects top-level field assignments but NOT mutations through
    nested mutable refs: ``row["ancestors"].append(...)`` mutates the
    SAME list object that lives in the cache. ``base.py``'s connection-
    lifecycle docstring documents this limitation explicitly; this test
    pins the behavior so a future refactor that promises "deep
    isolation" (or, conversely, drops the shell copy entirely) surfaces
    here and forces the docstring to stay honest.
    """
    from open_kgo.feature_groups.kg.citation_rest.file_fixture_citation import (
        FileFixtureCitationReader,
    )

    catalog_path = tmp_path / "tiny_catalog.json"
    catalog_path.write_text(
        json.dumps({"X-1": {"stable_id": "X-1", "name": "X1", "ancestors": ["Y-1"]}}),
        encoding="utf-8",
    )
    slot = {"locator": str(catalog_path), "auth_method": "none"}
    row = FileFixtureCitationReader._connect_from_slot(slot)["X-1"]
    # The cache row's ancestors list is the same object the shallow shell
    # copy would share with the caller. Mutating through that nested ref
    # is the known limitation; this assertion documents it.
    cached_ancestors = row["ancestors"]
    cached_ancestors.append("Z-1")
    reloaded = FileFixtureCitationReader._connect_from_slot(slot)
    assert reloaded["X-1"]["ancestors"] == ["Y-1", "Z-1"], (
        "nested-list mutation isolation changed (cache no longer shares the ancestors list); "
        "if this is intentional, update base.py's shell-only docstring accordingly."
    )


def test_file_fixture_rest_reader_caches_each_page(tmp_path: Path) -> None:
    """A second ``load_data`` call against the same pages directory does not re-parse.

    Pins the per-page caching: the original implementation re-globbed
    and re-parsed every page on every call (issue #32 item 3). The
    glob itself stays uncached (cheap); the per-page parse is what
    the mtime-keyed cache amortises.

    Uses ``_read_json_cached.cache_info().misses`` rather than
    monkeypatching ``json.load``: lru_cache exposes a miss counter that
    only ticks on a true cache miss, which is exactly the parse we want
    to gate, and the assertion is local to the JSON cache instead of
    touching the global ``json`` module.
    """
    from mloda.core.abstract_plugins.components.feature_set import FeatureSet
    from mloda.user import Feature, Options

    from open_kgo.feature_groups.kg.fixtures import _read_json_cached
    from open_kgo.feature_groups.kg.rest_public.file_fixture_rest import (
        FileFixtureRestReader,
    )

    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_1.json").write_text(
        json.dumps({"results": [{"id": "a"}, {"id": "b"}], "meta": {"next_cursor": None}}),
        encoding="utf-8",
    )

    creds = {"file_fixture_rest": {"locator": str(pages_dir), "auth_method": "none", "pagination_style": "cursor"}}
    feat = Feature("file_fixture_rest__probe", options=Options(context={}))
    fs = FeatureSet()
    fs.add(feat)

    FileFixtureRestReader.load_data(creds, fs)
    misses_after_first = _read_json_cached.cache_info().misses
    assert misses_after_first >= 1, "first load_data call did not exercise the JSON parser"
    FileFixtureRestReader.load_data(creds, fs)
    misses_after_second = _read_json_cached.cache_info().misses
    assert misses_after_second == misses_after_first, (
        f"second load_data call re-parsed {misses_after_second - misses_after_first} page(s); cache did not engage."
    )


def test_kuzu_reader_uses_cached_database_and_fresh_connection(tmp_path: Path) -> None:
    """Cached Database (FD holder) + transient Connection (caller-owned)."""
    import kuzu

    from open_kgo.feature_groups.kg.network_pg.kuzu_cypher import KuzuCypherReader

    db_path = tmp_path / "graph.kuzu"
    seed_db = kuzu.Database(str(db_path))
    seed_conn = kuzu.Connection(seed_db)
    seed_conn.execute("CREATE NODE TABLE Person(name STRING, PRIMARY KEY(name))")
    del seed_conn
    del seed_db

    slot = {"locator": str(db_path)}
    conn1 = KuzuCypherReader._connect_from_slot(slot)
    conn2 = KuzuCypherReader._connect_from_slot(slot)
    # Connections are fresh and caller-owned; Database is shared.
    assert conn1 is not conn2
    # Identity of the underlying cached Database survives both Connection builds.
    assert load_kuzu_database("kuzu_cypher", str(db_path)) is load_kuzu_database("kuzu_cypher", str(db_path))


def test_rdflib_reader_uses_shared_graph_cache() -> None:
    from open_kgo.feature_groups.kg.rdf.rdflib_sparql import RdfLibSparqlReader

    fixture = Path(__file__).parent.parent / "rdf" / "tests" / "fixtures" / "sample.ttl"
    slot = {"locator": str(fixture)}
    first = RdfLibSparqlReader._connect_from_slot(slot)
    second = RdfLibSparqlReader._connect_from_slot(slot)
    assert first is second


def test_rdflib_reader_empty_locator_returns_fresh_graph_each_call() -> None:
    """The no-locator empty-graph path is not cached: fresh per call.

    Documents the carve-out in ``RdfLibSparqlReader._connect_from_slot``:
    a falsy ``locator`` builds an empty ``rdflib.Graph`` on the spot
    rather than routing through ``load_rdf_graph``. Identity divergence
    here pins that contract so a future refactor that tries to memoize
    "empty graph" (a single shared instance) would have to be explicit.
    """
    from open_kgo.feature_groups.kg.rdf.rdflib_sparql import RdfLibSparqlReader

    first = RdfLibSparqlReader._connect_from_slot({"locator": None})
    second = RdfLibSparqlReader._connect_from_slot({"locator": None})
    assert first is not second


def test_copy_cached_row_returns_independent_dict_copy() -> None:
    """A dict row is shallow-copied so mutating the copy cannot poison the cached source."""
    cached = {"id": "W001", "title": "Paper One"}
    copy = copy_cached_row(cached)
    assert copy == cached
    assert copy is not cached
    copy["title"] = "MUTATED"
    assert cached["title"] == "Paper One"


def test_copy_cached_row_is_shallow() -> None:
    """Nested mutable values are shared (shallow copy): only the top level is duplicated."""
    nested = {"deps": ["a", "b"]}
    cached = {"component": nested}
    copy = copy_cached_row(cached)
    assert copy["component"] is nested


def test_copy_cached_row_passes_non_dict_through_unchanged() -> None:
    """Non-dict values have nothing to alias-protect and are returned as-is."""
    for value in ("a string", 42, ["a", "list"], None):
        assert copy_cached_row(value) is value
