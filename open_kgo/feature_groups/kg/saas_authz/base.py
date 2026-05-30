"""Family base for SaaS / authz / wiki KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import (
    KgConnectorFeatureGroupBase,
    ParamReader,
    compose_property_mapping,
)
from open_kgo.feature_groups.kg.mixins import EntityFilterPropertyMixin, PaginationMixin


_CONSISTENCY_MODES: dict[str, str] = {
    "minimize_latency": "Eventually consistent (default; SpiceDB MINIMIZE_LATENCY).",
    "at_least_as_fresh": "At-least-as-fresh as a token (SpiceDB AT_LEAST_AS_FRESH).",
    "at_exact_snapshot": "At an exact ZedToken snapshot (SpiceDB AT_EXACT_SNAPSHOT).",
    "fully_consistent": "Fully consistent / strong (SpiceDB FULLY_CONSISTENT).",
    "eventual": "OData eventual consistency.",
    "strong": "OData strong consistency.",
    "HIGHER_CONSISTENCY": "OpenFGA higher-consistency request.",
}


class SaasAuthzReader(EntityFilterPropertyMixin, PaginationMixin, ParamReader):
    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        ParamReader.PROPERTY_MAPPING,
        PaginationMixin.PROPERTY_MAPPING_DELTA,
        EntityFilterPropertyMixin.PROPERTY_MAPPING_DELTA,
        {
            "tenant": {
                "explanation": (
                    "Tenant identifier; six observed shapes: subdomain, instance_url, store_id, "
                    "token-implicit, wiki_url, vault_path."
                ),
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "api_version": {
                "explanation": "API version pin (e.g. v1.0, 2026-04, v62.0).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "consistency_token": {
                "explanation": "Opaque ZedToken / OData consistency token.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "consistency_mode": {
                "explanation": "Consistency semantics requested for authz reads.",
                "allowed_values": _CONSISTENCY_MODES,
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                DefaultOptionKeys.default: "minimize_latency",
            },
            "authorization_model_id": {
                "explanation": "OpenFGA authorization model id (also: Salesforce permset, etc.).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
        },
        context="SaasAuthzReader",
    )

    PARAMS_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        PaginationMixin.PARAMS_MAPPING_DELTA,
        context="SaasAuthzReader.PARAMS_MAPPING",
    )


class SaasAuthzFeatureGroup(KgConnectorFeatureGroupBase):
    READER_CLASS = None
