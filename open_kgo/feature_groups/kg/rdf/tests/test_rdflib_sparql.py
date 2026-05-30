"""Concrete tests for RdfLibSparqlReader.

Wires the 5 adapter methods + inherits the universal + per-family contract
tests from ``KgConnectorContractBase`` and ``RdfContractTestBase``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.errors import FixtureLoadError
from open_kgo.feature_groups.kg.rdf.rdflib_sparql import RdfLibSparqlReader
from open_kgo.feature_groups.kg.rdf.tests.kg_rdf_contract import RdfContractTestBase


_FIXTURE_TTL = Path(__file__).parent / "fixtures" / "sample.ttl"


class TestRdfLibSparqlReader(RdfContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[RdfLibSparqlReader]:
        return RdfLibSparqlReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return {
            "rdflib_sparql": {
                "locator": str(_FIXTURE_TTL),
                "result_format": "application/sparql-results+json",
                "reasoning_profile": "none",
                "result_limit": 100,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        # ``reasoning_profile`` is still narrowed to ``{"none"}`` on this
        # concrete; any other family-allowed value (e.g. ``"rdfs"``) is
        # outside the narrowed set and rejects. Replaces the earlier
        # ``auth_method="evil"`` seed (issue #32 item 2).
        return {"rdflib_sparql": {"locator": str(_FIXTURE_TTL), "reasoning_profile": "rdfs"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "rdflib_sparql__select_knows",
            options=Options(
                context={
                    "query_text": (
                        "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?s ?o WHERE { ?s foaf:knows ?o } LIMIT 10"
                    ),
                }
            ),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        def _check(result: Any) -> bool:
            if not isinstance(result, list):
                return False
            if len(result) == 0:
                return False
            for row in result:
                if not isinstance(row, dict):
                    return False
                if "s" not in row or "o" not in row:
                    return False
            return True

        return _check

    def test_query_returns_three_knows_triples(self) -> None:
        """The fixture has 3 foaf:knows triples; query should return all 3."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("rdflib_sparql", self.valid_credentials()["rdflib_sparql"], feat)
        assert len(rows) == 3

    def test_result_limit_truncates_emitted_rows(self) -> None:
        """``result_limit`` caps the number of emitted rows, not the iteration index.

        The fixture yields 3 ``foaf:knows`` rows; a limit of 2 must return
        exactly 2. The limit counts appended rows, so a (hypothetical) skipped
        non-row result would not consume the budget.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["rdflib_sparql"])
        slot["result_limit"] = 2
        feat = self.feature_under_test()
        rows = run_query("rdflib_sparql", slot, feat)
        assert len(rows) == 2

    def test_http_locator_rejected(self) -> None:
        """connect() must refuse remote schemes (no network IO at fetch time)."""
        cls = self.connector_reader_class()
        creds = {cls.CONNECTOR_ID: {"locator": "http://example.com/evil.ttl"}}
        with pytest.raises(ValueError, match="scheme"):
            cls.connect(creds)

    def test_windows_drive_locator_passes_scheme_guard(self) -> None:
        """Windows-style drive letter prefixes must not be flagged as remote schemes.

        ``urlparse(r"C:\\foo.ttl").scheme == "c"`` (and the same for forward
        slashes), so without the heuristic the guard would reject valid local
        paths. The fix is positively observed by failing *only* on the
        downstream stat call (``FixtureLoadError`` from the mtime-keyed
        ``load_rdf_graph`` cache, since the path does not exist on a
        non-Windows host): a regression that re-broadens the scheme guard
        would surface here as a ``ValueError`` mentioning the scheme,
        which the assertion below distinguishes from the expected
        file-not-found ``FixtureLoadError``.
        """
        from urllib.parse import urlparse

        # Pin the assumption the guard depends on; flag if a future Python changes it.
        assert urlparse(r"C:\nonexistent\sample.ttl").scheme == "c"
        assert urlparse("C:/nonexistent/sample.ttl").scheme == "c"

        cls = self.connector_reader_class()
        for locator in (r"C:\nonexistent\sample.ttl", "C:/nonexistent/sample.ttl"):
            creds = {cls.CONNECTOR_ID: {"locator": locator}}
            # Expect FixtureLoadError (file not found via stat) — proves the
            # scheme guard passed the locator through to the loader. A
            # regression that re-rejects Windows drive locators would raise
            # ValueError("scheme ...") from _connect_from_slot instead.
            with pytest.raises(FixtureLoadError):
                cls.connect(creds)
