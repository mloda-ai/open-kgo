"""Family base for RDF / SPARQL KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase, QueryReader
from open_kgo.feature_groups.kg.mixins import InferenceMixin
from open_kgo.feature_groups.kg.spec import property_spec


_RESULT_FORMATS: dict[str, str] = {
    "application/sparql-results+json": "SPARQL JSON results format (SELECT/ASK).",
    "application/sparql-results+xml": "SPARQL XML results format (SELECT/ASK).",
    "text/turtle": "Turtle serialisation (CONSTRUCT/DESCRIBE).",
    "application/n-triples": "N-Triples serialisation (CONSTRUCT/DESCRIBE).",
}

_FAMILY_PROPERTIES: dict[str, Any] = {
    "default_graph_uris": property_spec(
        "List of named graph URIs to merge into the default graph (FROM clauses).",
        default=(),
    ),
    "named_graph_uris": property_spec(
        "List of named graphs available via FROM NAMED.",
        default=(),
    ),
    "update_endpoint": property_spec(
        "Optional separate URL for SPARQL UPDATE; null means same as locator.",
    ),
    "result_format": property_spec(
        "MIME type the SPARQL endpoint should return results in.",
        strict=True,
        allowed_values=_RESULT_FORMATS,
        default="application/sparql-results+json",
    ),
}


class RdfSparqlReader(InferenceMixin, QueryReader, family_properties=_FAMILY_PROPERTIES):
    """Family base for SPARQL endpoints (Network) and in-memory triple stores.

    Concrete plugins (RdfLibSparqlReader, GraphDbReader, ...) override
    ``CONNECTOR_ID``, ``REQUIRED_KEYS``, and the abstract ``connect`` /
    ``build_query`` / ``load_data`` methods.

    The doc-recommended properties for this family land here so every concrete
    plugin can reuse them. ``PROPERTY_MAPPING`` is auto-composed from the
    parent reader, the declared mixins, and ``_FAMILY_PROPERTIES`` (see
    ``KgConnectorReaderBase._compose_family_surface``).
    """

    # Honest surface (option 3, see base.py): SPARQL graph-dataset selectors and
    # the separate UPDATE endpoint the concretes ignore (single in-memory triple
    # store, no dataset scoping or UPDATE), reserved for a networked concrete.
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"default_graph_uris", "named_graph_uris", "update_endpoint"}
    )


class RdfSparqlFeatureGroup(KgConnectorFeatureGroupBase):
    """Abstract family base for RDF/SPARQL connectors; each concrete pins ``READER_CLASS``."""
