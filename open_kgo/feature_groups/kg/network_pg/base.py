"""Family base for network property-graph KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import (
    KgConnectorFeatureGroupBase,
    QueryReader,
    compose_property_mapping,
)


_READ_CONSISTENCY: dict[str, str] = {
    "read": "Read against any replica (Neo4j default).",
    "write": "Read after write on the leader.",
    "linearizable": "Linearizable read (e.g. Memgraph SYNC).",
}


_TRANSACTION_MODE: dict[str, str] = {
    "auto": "Auto-commit (per-statement transactions).",
    "explicit": "Explicit BEGIN/COMMIT.",
    "schema": "Schema-mutating transaction (TypeDB SCHEMA mode).",
}


class NetworkPropertyGraphReader(QueryReader):
    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        QueryReader.PROPERTY_MAPPING,
        {
            "dataset": {
                "explanation": "Database / graph / space name on the endpoint.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "read_consistency": {
                "explanation": "Read consistency level the connector should request.",
                "allowed_values": _READ_CONSISTENCY,
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                DefaultOptionKeys.default: "read",
            },
            "transaction_mode": {
                "explanation": "Transaction handling mode used by the engine.",
                "allowed_values": _TRANSACTION_MODE,
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                DefaultOptionKeys.default: "auto",
            },
        },
        context="NetworkPropertyGraphReader",
    )


class NetworkPropertyGraphFeatureGroup(KgConnectorFeatureGroupBase):
    READER_CLASS = None
