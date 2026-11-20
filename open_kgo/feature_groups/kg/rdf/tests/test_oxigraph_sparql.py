"""Concrete tests for OxigraphSparqlReader.

Reuses the committed ``sample.ttl`` fixture (shared with the rdflib reader) so
both SPARQL engines are asserted against the same triples, demonstrating the
RDF family base generalises across implementations.
"""

from __future__ import annotations

import pickle
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import pyoxigraph
import pytest

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.errors import FixtureLoadError
from open_kgo.feature_groups.kg.rdf.oxigraph_sparql import OxigraphSparqlReader, _lexical
from open_kgo.feature_groups.kg.rdf.tests.kg_rdf_contract import RdfContractTestBase
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials, run_query


_FIXTURE_TTL = Path(__file__).parent / "fixtures" / "sample.ttl"
_XSD = "http://www.w3.org/2001/XMLSchema#"


class TestOxigraphSparqlReader(RdfContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[OxigraphSparqlReader]:
        return OxigraphSparqlReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return make_valid_credentials(cls.connector_reader_class(), locator=str(_FIXTURE_TTL), result_limit=100)

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        # ``reasoning_profile`` is narrowed to ``{"none"}``; any other
        # family-allowed value (e.g. ``"rdfs"``) is outside the narrowed set.
        return {"oxigraph_sparql": {"locator": str(_FIXTURE_TTL), "reasoning_profile": "rdfs"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "oxigraph_sparql__select_knows",
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
            if not isinstance(result, list) or len(result) == 0:
                return False
            return all(isinstance(row, dict) and "s" in row and "o" in row for row in result)

        return _check

    def test_query_returns_three_knows_triples(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("oxigraph_sparql", self.valid_credentials()["oxigraph_sparql"], feat)
        assert len(rows) == 3

    def test_result_limit_truncates_emitted_rows(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["oxigraph_sparql"])
        slot["result_limit"] = 2
        feat = self.feature_under_test()
        rows = run_query("oxigraph_sparql", slot, feat)
        assert len(rows) == 2

    def test_ask_query_returns_boolean_row(self) -> None:
        """An ASK query maps to a single ``{"boolean": ...}`` row."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "oxigraph_sparql__ask_knows",
            options=Options(
                context={
                    "query_text": "PREFIX foaf: <http://xmlns.com/foaf/0.1/> ASK { ?s foaf:knows ?o }",
                }
            ),
        )
        rows = run_query("oxigraph_sparql", self.valid_credentials()["oxigraph_sparql"], feat)
        assert rows == [{"boolean": True}]

    def test_optional_unbound_variable_key_is_omitted(self) -> None:
        """An OPTIONAL variable that never binds is OMITTED from the row, not set to None.

        No subject in ``sample.ttl`` has ``foaf:nick``, so ``?nick`` is unbound
        in every solution. The emitted rows must carry only the bound ``?s`` key
        (key set ``{"s"}``), matching the rdflib sibling's ``asdict()`` which
        omits unbound variables — a cross-backend SELECT-schema parity assertion.
        """
        import open_kgo.feature_groups.kg.rdf.rdflib_sparql  # noqa: F401
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        query = (
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/> "
            "SELECT ?s ?nick WHERE { ?s foaf:knows ?o . OPTIONAL { ?s foaf:nick ?nick } }"
        )
        ox_feat = Feature("oxigraph_sparql__optional", options=Options(context={"query_text": query}))
        ox_rows = run_query("oxigraph_sparql", self.valid_credentials()["oxigraph_sparql"], ox_feat)
        assert len(ox_rows) == 3
        assert all(set(row) == {"s"} for row in ox_rows)

        # Cross-backend parity: the rdflib sibling omits the unbound key too, so
        # the per-row key sets match (values differ by the documented term-vs-str
        # divergence, hence key-set comparison rather than full-row equality).
        rdflib_slot = {
            "locator": str(_FIXTURE_TTL),
            "result_format": "application/sparql-results+json",
            "reasoning_profile": "none",
            "result_limit": 100,
        }
        rdflib_feat = Feature("rdflib_sparql__optional", options=Options(context={"query_text": query}))
        rdflib_rows = run_query("rdflib_sparql", rdflib_slot, rdflib_feat)
        ox_key_sets = sorted(tuple(sorted(row)) for row in ox_rows)
        rdflib_key_sets = sorted(tuple(sorted(row)) for row in rdflib_rows)
        assert ox_key_sets == rdflib_key_sets

    def test_missing_locator_file_is_typed(self) -> None:
        """A ``locator`` pointing at a non-existent file raises ``FixtureLoadError``."""
        cls = self.connector_reader_class()
        creds = {cls.CONNECTOR_ID: {"locator": "/nonexistent/path/to/graph.ttl"}}
        with pytest.raises(FixtureLoadError):
            cls.connect(creds)

    def test_malformed_rdf_locator_is_typed(self, tmp_path: Path) -> None:
        """A present-but-unparseable Turtle file raises ``FixtureLoadError``.

        Exercises the parse-failure branch in ``load_oxigraph_store`` (pyoxigraph
        raises the builtin ``SyntaxError``, wrapped as ``FixtureLoadError``), so a
        future pyoxigraph exception-type change surfaces as a test failure rather
        than a leaked untyped error.
        """
        cls = self.connector_reader_class()
        bad = tmp_path / "bad.ttl"
        bad.write_text("@prefix ex: <http://example.org/> .\nex:s ex:p <<< not turtle", encoding="utf-8")
        creds = {cls.CONNECTOR_ID: {"locator": str(bad)}}
        with pytest.raises(FixtureLoadError):
            cls.connect(creds)

    def test_unknown_suffix_parse_failure_names_assumed_format(self, tmp_path: Path) -> None:
        """A parse failure on an unmapped suffix names the Turtle assumption and the suffix.

        ``.owl`` is not in the suffix-to-format map, so the loader falls back
        to the documented Turtle default; an RDF/XML-bodied ``.owl`` file then
        fails to parse. The error message must say it was parsed as TURTLE
        because the suffix is unknown, instead of the bare misleading "not
        parseable as RDF".
        """
        cls = self.connector_reader_class()
        owl = tmp_path / "ontology.owl"
        # RDF/XML body: valid RDF for the right parser, guaranteed not Turtle.
        owl.write_text(
            '<?xml version="1.0"?>\n'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            '  <rdf:Description rdf:about="http://example.org/s"/>\n'
            "</rdf:RDF>\n",
            encoding="utf-8",
        )
        creds = {cls.CONNECTOR_ID: {"locator": str(owl)}}
        with pytest.raises(FixtureLoadError, match=r"parsed as TURTLE based on suffix '\.owl' which is not in"):
            cls.connect(creds)

    def test_named_graph_triples_visible_to_plain_select(self, tmp_path: Path) -> None:
        """Triples in a TriG NAMED graph are visible to a SELECT without a GRAPH clause.

        The loader advertises quad-bearing formats (.trig/.nq/.jsonld) whose
        triples land in named graphs; oxigraph scopes a plain pattern to the
        default graph only, so without ``use_default_graph_as_union=True``
        this query returns zero rows for the named-graph triple.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        trig = tmp_path / "graphs.trig"
        trig.write_text(
            "@prefix ex: <http://example.org/> .\n"
            "ex:g {\n"
            "  ex:alice ex:knows ex:bob .\n"
            "}\n"
            "ex:carol ex:knows ex:dave .\n",
            encoding="utf-8",
        )
        slot = dict(self.valid_credentials()["oxigraph_sparql"])
        slot["locator"] = str(trig)
        feat = Feature(
            "oxigraph_sparql__trig_union",
            options=Options(
                context={
                    "query_text": (
                        "PREFIX ex: <http://example.org/> SELECT ?s ?o WHERE { ?s ex:knows ?o } ORDER BY ?s"
                    ),
                }
            ),
        )
        rows = run_query("oxigraph_sparql", slot, feat)
        assert rows == [
            {"s": "http://example.org/alice", "o": "http://example.org/bob"},
            {"s": "http://example.org/carol", "o": "http://example.org/dave"},
        ]

    def test_construct_query_yields_spo_triple_rows(self) -> None:
        """CONSTRUCT maps to ``{"s": ..., "p": ..., "o": ...}`` rows of plain strings.

        Pins the documented divergence from the rdflib sibling (which yields no
        rows for non-SELECT shapes): the triple branch emits each constructed
        triple's terms via ``.value``, so IRIs and literals are plain strings.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "oxigraph_sparql__construct_knows",
            options=Options(
                context={
                    "query_text": (
                        "PREFIX foaf: <http://xmlns.com/foaf/0.1/> "
                        "CONSTRUCT { ?s foaf:knows ?o } WHERE { ?s foaf:knows ?o }"
                    ),
                }
            ),
        )
        rows = run_query("oxigraph_sparql", self.valid_credentials()["oxigraph_sparql"], feat)
        assert len(rows) == 3
        for row in rows:
            assert set(row) == {"s", "p", "o"}
            assert all(isinstance(value, str) for value in row.values())
            assert row["p"] == "http://xmlns.com/foaf/0.1/knows"

    @pytest.mark.parametrize(
        ("term", "expected"),
        [
            (pyoxigraph.Literal("3", datatype=pyoxigraph.NamedNode(f"{_XSD}integer")), 3),
            (pyoxigraph.Literal("1.5", datatype=pyoxigraph.NamedNode(f"{_XSD}decimal")), Decimal("1.5")),
            (pyoxigraph.Literal("true", datatype=pyoxigraph.NamedNode(f"{_XSD}boolean")), True),
            (pyoxigraph.Literal("abc", datatype=pyoxigraph.NamedNode(f"{_XSD}integer")), "abc"),
            (pyoxigraph.Literal(str(2**70), datatype=pyoxigraph.NamedNode(f"{_XSD}integer")), str(2**70)),
            (pyoxigraph.Literal("x", language="en"), "x"),
            (pyoxigraph.Literal("plain"), "plain"),
            (pyoxigraph.NamedNode("http://example.org/a"), "http://example.org/a"),
        ],
    )
    def test_project_value_converts_typed_literals(self, term: Any, expected: Any) -> None:
        """A row cell keeps its lexical string (also after pickling) and projects like on the rdflib reader."""
        cell = pickle.loads(pickle.dumps(_lexical(term)))
        projected = OxigraphSparqlReader._project_value(cell)
        assert cell == term.value and isinstance(cell, str)
        assert projected == expected and type(projected) is type(expected)

    def test_construct_object_projection_is_typed(self) -> None:
        """A CONSTRUCT row's ``~o`` cell converts a typed literal object too."""
        query = (
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX ex: <http://example.org/> "
            "CONSTRUCT { ?s ex:n ?n } WHERE { SELECT ?s (COUNT(?o) AS ?n) WHERE { ?s foaf:knows ?o } GROUP BY ?s }"
        )
        feat = Feature("oxigraph_sparql__construct_counts~o", options=Options(context={"query_text": query}))
        cells = run_query("oxigraph_sparql", self.valid_credentials()["oxigraph_sparql"], feat)
        assert sorted(cells) == [1, 1, 1] and all(type(cell) is int for cell in cells)

    def test_result_limit_truncates_construct_branch(self) -> None:
        """``result_limit`` short-circuits the CONSTRUCT/DESCRIBE triple branch too.

        The SELECT branch's truncation is covered elsewhere; this pins the
        same contract on the triple iterator (3 constructed triples, limit 2).
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["oxigraph_sparql"])
        slot["result_limit"] = 2
        feat = Feature(
            "oxigraph_sparql__construct_capped",
            options=Options(
                context={
                    "query_text": (
                        "PREFIX foaf: <http://xmlns.com/foaf/0.1/> "
                        "CONSTRUCT { ?s foaf:knows ?o } WHERE { ?s foaf:knows ?o }"
                    ),
                }
            ),
        )
        rows = run_query("oxigraph_sparql", slot, feat)
        assert len(rows) == 2
