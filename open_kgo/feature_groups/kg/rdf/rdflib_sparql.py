"""Concrete RDF/SPARQL connector backed by an in-memory rdflib Graph.

Loads triples from a Turtle/N-Triples/RDF-XML file at ``locator`` (or accepts
``locator=None`` for an empty graph) and runs SPARQL queries against it.

The query text comes from the Feature's options context under the key
``query_text``. ``result_limit`` is enforced by breaking out of result
iteration as soon as the cap is reached (short-circuit, not slice-at-end).
``reasoning_profile`` is validated as an enum but only ``"none"`` is
implemented in this prototype.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, ClassVar, Mapping

import rdflib
from rdflib.term import IdentifiedNode, Identifier, Literal

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.fixtures import _rejected_scheme, load_rdf_graph
from open_kgo.feature_groups.kg.rdf.base import RdfSparqlFeatureGroup, RdfSparqlReader


# toPython() results a projected column keeps; anything else (XML documents, year-month durations) projects as str.
_SCALARS = (bool, int, float, Decimal, str, bytes, date, datetime, time, timedelta)


class RdfLibSparqlReader(RdfSparqlReader):
    """rdflib in-memory SPARQL reader.

    Accepts an optional ``locator`` (path to a turtle/n-triples/rdf-xml file).
    With ``locator=None`` the reader runs against an empty graph, which is
    only useful for shape tests.
    """

    CONNECTOR_ID: ClassVar[str] = "rdflib_sparql"
    # Strict-enum narrowings:
    #   - result_format: ``graph.query()`` always returns rdflib row objects
    #     converted via ``.asdict()`` into JSON-binding-shaped dicts; XML /
    #     Turtle / N-Triples are not honored.
    #   - reasoning_profile: prototype docstring pins "only ``none`` is
    #     implemented"; rdflib in-memory has no inference engine.
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "result_format": frozenset({"application/sparql-results+json"}),
        "reasoning_profile": frozenset({"none"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> rdflib.Graph:
        """Return an rdflib.Graph populated from slot['locator']; parsed once and shared.

        A real Turtle file is 1-100MB; rdflib's parse pass is the
        expensive step. Routes through the shared
        ``load_rdf_graph`` cache (mtime-keyed) so a 100-feature
        ``mloda.run_all`` pays one parse instead of one hundred. The
        returned graph is shared across calls and MUST be treated as
        read-only — ``add`` / ``remove`` calls would corrupt subsequent
        loads in the same process. ``rdflib.Graph.close()`` is a no-op
        on the default Memory store, so the matcher-path contract test
        that closes ``connect()``'s return survives this caching change.

        ``locator=None``/falsy stays out of the cache: we build a fresh
        empty graph each call, which is what the empty-graph shape tests
        document. The scheme guard remains here (rather than only in the
        shared loader) so the non-locator validation keeps its existing
        ``ValueError`` contract (separate from ``FixtureLoadError``),
        but it delegates the parsing rule to ``_rejected_scheme`` so the
        rule itself lives in one place.
        """
        locator = slot.get("locator")
        if not locator:
            return rdflib.Graph()
        bad = _rejected_scheme(locator)
        if bad is not None:
            raise ValueError(
                f"{cls.CONNECTOR_ID}: locator scheme {bad!r} is not permitted; "
                f"only local file paths or file:// URLs are allowed."
            )
        return load_rdf_graph(cls.CONNECTOR_ID, locator)

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        """Run the SPARQL query and return up to result_limit rows as list-of-dicts."""
        graph = connection

        query_text = cls.build_query(features)
        rows: list[dict[str, Any]] = []
        # Limit on the number of rows actually emitted, not the iteration
        # index: a result that yields a row without ``asdict`` (e.g. a
        # non-SELECT shape) is skipped via ``continue`` and must not count
        # against ``result_limit``, or fewer than ``result_limit`` rows
        # would be returned.
        for row in graph.query(query_text):
            asdict = getattr(row, "asdict", None)
            if asdict is None:
                continue
            rows.append({str(k): v for k, v in asdict().items()})
            if len(rows) >= ctx.result_limit:
                break
        return rows

    @classmethod
    def _project_value(cls, value: Any) -> Any:
        """Literals become Python scalars; IRIs, blank nodes and any other term become ``str``."""
        if not isinstance(value, Identifier):
            return value
        converted = value.toPython() if isinstance(value, (IdentifiedNode, Literal)) else value
        if isinstance(converted, int) and not isinstance(converted, bool) and not -(2**63) <= converted < 2**63:
            return str(value)
        return converted if isinstance(converted, _SCALARS) and not isinstance(converted, Identifier) else str(value)


class RdfLibSparqlFeatureGroup(RdfSparqlFeatureGroup):
    READER_CLASS: ClassVar[type[RdfLibSparqlReader]] = RdfLibSparqlReader  # type: ignore[assignment]
