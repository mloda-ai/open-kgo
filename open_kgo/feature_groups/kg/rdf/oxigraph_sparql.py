"""Concrete RDF/SPARQL connector backed by an in-memory ``pyoxigraph.Store``.

Second concrete in the ``rdf`` family alongside ``RdfLibSparqlReader``: parses
a Turtle/N-Triples/RDF-XML file at ``locator`` into an embedded oxigraph store
and runs SPARQL 1.1 queries against it. oxigraph is a different SPARQL engine
(Rust-backed) than rdflib, so this proves the family base generalises across
implementations; it does not unlock new family surface (oxigraph has no RDFS
inference and we return JSON-binding-shaped rows, so ``reasoning_profile`` and
``result_format`` stay narrowed exactly as on ``RdfLibSparqlReader``).

The query text comes from the Feature's options context under ``query_text``.
``result_limit`` is enforced by breaking out of solution iteration once the cap
is reached (short-circuit, not slice-at-end). SELECT results map to
JSON-binding-shaped dicts; CONSTRUCT/DESCRIBE map to ``s``/``p``/``o`` triple
rows; ASK maps to a single ``{"boolean": ...}`` row.

Queries run with ``use_default_graph_as_union=True``: the loader accepts
quad-bearing serialisations (TriG / N-Quads / JSON-LD) whose triples land in
NAMED graphs, and oxigraph's default query semantics scope a pattern without a
``GRAPH`` clause to the default graph only, which would make every such triple
invisible to every SELECT. The union semantics treat the default graph as the
union of all graphs, matching rdflib's default-union behaviour so the two
backends agree on what a plain query sees.

Backend divergence (deliberate, additive): unlike the rdflib sibling, which
yields no rows for non-SELECT shapes (its ``asdict``-skip path), this reader
emits boolean/triple rows for ASK/CONSTRUCT/DESCRIBE. oxigraph is the more
complete backend here; the richer output is documented rather than regressed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import pyoxigraph

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.fixtures import _rejected_scheme, load_oxigraph_store
from open_kgo.feature_groups.kg.rdf.base import RdfSparqlFeatureGroup, RdfSparqlReader


class OxigraphSparqlReader(RdfSparqlReader):
    """pyoxigraph in-memory SPARQL reader.

    Accepts an optional ``locator`` (path to a turtle/n-triples/rdf-xml file).
    With ``locator=None`` the reader runs against an empty store, which is only
    useful for shape tests.
    """

    CONNECTOR_ID: ClassVar[str] = "oxigraph_sparql"
    # Strict-enum narrowings (identical disposition to RdfLibSparqlReader):
    #   - result_format: SELECT solutions are converted to JSON-binding-shaped
    #     dicts; XML / Turtle / N-Triples serialisations are not honored.
    #   - reasoning_profile: oxigraph has no inference engine, so only "none".
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "result_format": frozenset({"application/sparql-results+json"}),
        "reasoning_profile": frozenset({"none"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> pyoxigraph.Store:
        """Return a shared, path-cached ``pyoxigraph.Store`` populated from ``slot['locator']``.

        Routes through the shared ``load_oxigraph_store`` cache (mtime-keyed)
        so a 100-feature ``mloda.run_all`` pays one parse instead of one
        hundred. The returned store is shared across calls and MUST be treated
        as read-only. ``locator=None``/falsy stays out of the cache: a fresh
        empty store is built per call, which is what the empty-store shape
        tests document. The scheme guard keeps the file-only contract (and its
        ``ValueError`` flavour) uniform with ``RdfLibSparqlReader``.
        """
        locator = slot.get("locator")
        if not locator:
            return pyoxigraph.Store()
        bad = _rejected_scheme(locator)
        if bad is not None:
            raise ValueError(
                f"{cls.CONNECTOR_ID}: locator scheme {bad!r} is not permitted; "
                f"only local file paths or file:// URLs are allowed."
            )
        # ``load_oxigraph_store`` is typed ``-> Any`` (it defers the pyoxigraph
        # import to keep fixtures.py importable without the kg-rdf extra); bind
        # through a typed local so the declared ``Store`` return holds under
        # mypy --strict rather than leaking ``Any``.
        store: pyoxigraph.Store = load_oxigraph_store(cls.CONNECTOR_ID, locator)
        return store

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        """Run the SPARQL query and return up to result_limit rows as list-of-dicts."""
        store = connection

        query_text = cls.build_query(features)
        # use_default_graph_as_union: triples loaded from quad-bearing formats
        # (TriG / N-Quads / JSON-LD) live in named graphs; without the union
        # they are invisible to any query lacking a GRAPH clause (see module
        # docstring).
        result = store.query(query_text, use_default_graph_as_union=True)
        rows: list[dict[str, Any]] = []

        if isinstance(result, pyoxigraph.QuerySolutions):
            variables = result.variables
            for solution in result:
                # Backend divergence (deliberate): SELECT bindings are emitted
                # as their lexical ``.value`` (a plain ``str``), whereas the
                # rdflib sibling yields rdflib term objects (URIRef / Literal)
                # via ``row.asdict()``. oxigraph's ``.value`` drops datatype and
                # language tags, so a consumer that needs typed terms must use
                # the rdflib reader. The same ``.value`` flattening applies to
                # the CONSTRUCT/DESCRIBE triple branch below: blank nodes emit
                # their bare id (no ``_:`` prefix), and IRIs vs literals are
                # indistinguishable plain strings in the emitted s/p/o rows.
                # Unbound bindings (e.g. an OPTIONAL variable
                # with no match) are OMITTED from the row, matching the rdflib
                # sibling's ``asdict()`` which only carries bound variables, so
                # the two backends agree on the SELECT row schema.
                row: dict[str, Any] = {}
                for var in variables:
                    term = solution[var]
                    if term is not None:
                        row[var.value] = term.value
                rows.append(row)
                if len(rows) >= ctx.result_limit:
                    break
            return rows

        if isinstance(result, pyoxigraph.QueryBoolean):
            return [{"boolean": bool(result)}]

        # CONSTRUCT / DESCRIBE -> iterator of pyoxigraph.Triple.
        for triple in result:
            rows.append({"s": triple.subject.value, "p": triple.predicate.value, "o": triple.object.value})
            if len(rows) >= ctx.result_limit:
                break
        return rows


class OxigraphSparqlFeatureGroup(RdfSparqlFeatureGroup):
    READER_CLASS: ClassVar[type[OxigraphSparqlReader]] = OxigraphSparqlReader
