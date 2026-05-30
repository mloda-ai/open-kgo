"""Family base for RDF / SPARQL KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import (
    KgConnectorFeatureGroupBase,
    QueryReader,
    compose_property_mapping,
)
from open_kgo.feature_groups.kg.mixins import InferenceMixin


_RESULT_FORMATS: dict[str, str] = {
    "application/sparql-results+json": "SPARQL JSON results format (SELECT/ASK).",
    "application/sparql-results+xml": "SPARQL XML results format (SELECT/ASK).",
    "text/turtle": "Turtle serialisation (CONSTRUCT/DESCRIBE).",
    "application/n-triples": "N-Triples serialisation (CONSTRUCT/DESCRIBE).",
}


class RdfSparqlReader(InferenceMixin, QueryReader):
    """Family base for SPARQL endpoints (Network) and in-memory triple stores.

    Concrete plugins (RdfLibSparqlReader, GraphDbReader, ...) override
    ``CONNECTOR_ID``, ``REQUIRED_KEYS``, and the abstract ``connect`` /
    ``build_query`` / ``load_data`` methods.

    The doc-recommended properties for this family land here so every concrete
    plugin can reuse them.
    """

    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        QueryReader.PROPERTY_MAPPING,
        InferenceMixin.PROPERTY_MAPPING_DELTA,
        {
            "default_graph_uris": {
                "explanation": "List of named graph URIs to merge into the default graph (FROM clauses).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: (),
            },
            "named_graph_uris": {
                "explanation": "List of named graphs available via FROM NAMED.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: (),
            },
            "update_endpoint": {
                "explanation": "Optional separate URL for SPARQL UPDATE; null means same as locator.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "result_format": {
                "explanation": "MIME type the SPARQL endpoint should return results in.",
                "allowed_values": _RESULT_FORMATS,
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                DefaultOptionKeys.default: "application/sparql-results+json",
            },
        },
        context="RdfSparqlReader",
    )


class RdfSparqlFeatureGroup(KgConnectorFeatureGroupBase):
    """Family-base FG for RDF/SPARQL connectors. Concrete subclasses pin READER_CLASS."""

    READER_CLASS = None
