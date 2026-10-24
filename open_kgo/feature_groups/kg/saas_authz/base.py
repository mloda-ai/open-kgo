"""Family base for SaaS / authz / wiki KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase, ParamReader
from open_kgo.feature_groups.kg.mixins import EntityFilterPropertyMixin, PaginationMixin
from open_kgo.feature_groups.kg.spec import property_spec


_CONSISTENCY_MODES: dict[str, str] = {
    "minimize_latency": "Eventually consistent (default; SpiceDB MINIMIZE_LATENCY).",
    "at_least_as_fresh": "At-least-as-fresh as a token (SpiceDB AT_LEAST_AS_FRESH).",
    "at_exact_snapshot": "At an exact ZedToken snapshot (SpiceDB AT_EXACT_SNAPSHOT).",
    "fully_consistent": "Fully consistent / strong (SpiceDB FULLY_CONSISTENT).",
    "eventual": "OData eventual consistency.",
    "strong": "OData strong consistency.",
    "HIGHER_CONSISTENCY": "OpenFGA higher-consistency request.",
}

_FAMILY_PROPERTIES: dict[str, Any] = {
    "tenant": property_spec(
        (
            "Tenant identifier; six observed shapes: subdomain, instance_url, store_id, "
            "token-implicit, wiki_url, vault_path."
        ),
    ),
    "api_version": property_spec(
        "API version pin (e.g. v1.0, 2026-04, v62.0).",
    ),
    "consistency_token": property_spec(
        "Opaque ZedToken / OData consistency token.",
    ),
    "consistency_mode": property_spec(
        "Consistency semantics requested for authz reads.",
        strict=True,
        allowed_values=_CONSISTENCY_MODES,
        default="minimize_latency",
    ),
    "authorization_model_id": property_spec(
        "OpenFGA authorization model id (also: Salesforce permset, etc.).",
    ),
}


class SaasAuthzReader(
    EntityFilterPropertyMixin,
    PaginationMixin,
    ParamReader,
    family_properties=_FAMILY_PROPERTIES,
):
    # Honest surface (option 3, see base.py): SpiceDB/OpenFGA/OData knobs the
    # in-process fakes ignore (plain dict ops, no versioning/consistency),
    # reserved for a real authz backend.
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"api_version", "consistency_token", "authorization_model_id"}
    )


class SaasAuthzFeatureGroup(KgConnectorFeatureGroupBase):
    """Abstract family base; each concrete pins ``READER_CLASS``."""
