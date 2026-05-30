"""Family base for citation / scientific REST connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import (
    KgConnectorFeatureGroupBase,
    ParamReader,
    compose_property_mapping,
)
from open_kgo.feature_groups.kg.mixins import PaginationMixin


_PER_CALL_KEYS: dict[str, Any] = {
    "entity_type": {
        "explanation": "Resource type (e.g. 'pathway', 'work').",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: None,
    },
    "stable_id": {
        "explanation": "System-stable identifier of the entity to fetch (e.g. R-HSA-1640170).",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: None,
    },
    "hierarchy_depth": {
        "explanation": "Depth limit for ancestors/descendants traversal.",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: 1,
    },
}


class CitationRestReader(PaginationMixin, ParamReader):
    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        ParamReader.PROPERTY_MAPPING,
        PaginationMixin.PROPERTY_MAPPING_DELTA,
        {
            "species_prefix": {
                "explanation": "Species prefix (e.g. HSA for human Reactome IDs).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "dataset_version": {
                "explanation": "Release version pin (e.g. 'v90' for Reactome).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
        },
        context="CitationRestReader",
    )

    PARAMS_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        PaginationMixin.PARAMS_MAPPING_DELTA,
        _PER_CALL_KEYS,
        context="CitationRestReader.PARAMS_MAPPING",
    )

    REQUIRED_PARAMS: ClassVar[tuple[tuple[str, ...], ...]] = (("stable_id",),)


class CitationRestFeatureGroup(KgConnectorFeatureGroupBase):
    READER_CLASS = None
