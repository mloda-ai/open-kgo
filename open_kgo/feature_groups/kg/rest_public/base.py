"""Family base for REST non-SPARQL public KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import (
    KgConnectorFeatureGroupBase,
    ParamReader,
    compose_property_mapping,
)
from open_kgo.feature_groups.kg.mixins import PaginationMixin


_PER_CALL_ENTITY: dict[str, Any] = {
    "entity_type": {
        "explanation": "Resource type for the request (e.g. 'works', 'authors').",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: None,
    },
}


class RestPublicReader(PaginationMixin, ParamReader):
    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        ParamReader.PROPERTY_MAPPING,
        PaginationMixin.PROPERTY_MAPPING_DELTA,
        {
            "dataset_version": {
                "explanation": "Optional dataset/release version pin for reproducibility.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "user_agent": {
                "explanation": "User-Agent string (often required by polite-pool endpoints).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "rate_limit_pace": {
                "explanation": "Soft pace cap (requests/min); concrete plugin enforces.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: 100,
            },
        },
        context="RestPublicReader",
    )

    PARAMS_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        PaginationMixin.PARAMS_MAPPING_DELTA,
        _PER_CALL_ENTITY,
        context="RestPublicReader.PARAMS_MAPPING",
    )


class RestPublicFeatureGroup(KgConnectorFeatureGroupBase):
    READER_CLASS = None
