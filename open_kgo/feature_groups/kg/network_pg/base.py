"""Family base for network property-graph KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase, QueryReader
from open_kgo.feature_groups.kg.spec import property_spec


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

_FAMILY_PROPERTIES: dict[str, Any] = {
    "dataset": property_spec(
        "Database / graph / space name on the endpoint.",
    ),
    "read_consistency": property_spec(
        "Read consistency level the connector should request.",
        strict=True,
        allowed_values=_READ_CONSISTENCY,
        default="read",
    ),
    "transaction_mode": property_spec(
        "Transaction handling mode used by the engine.",
        strict=True,
        allowed_values=_TRANSACTION_MODE,
        default="auto",
    ),
}


class NetworkPropertyGraphReader(QueryReader, family_properties=_FAMILY_PROPERTIES):
    # Honest surface (option 3, see base.py): ``dataset`` names a database on a
    # networked endpoint; both concretes run over a single in-memory/embedded
    # graph with no such concept. Reserved for a networked concrete (Neo4j).
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"dataset"})


class NetworkPropertyGraphFeatureGroup(KgConnectorFeatureGroupBase):
    """Abstract family base; each concrete pins ``READER_CLASS``."""
