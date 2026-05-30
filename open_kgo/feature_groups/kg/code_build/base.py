"""Family base for code / build / SBOM KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import (
    KgConnectorFeatureGroupBase,
    ParamReader,
    compose_property_mapping,
)
from open_kgo.feature_groups.kg.mixins import EntityFilterParamMixin, TraversalMixin


class CodeBuildReader(ParamReader):
    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        ParamReader.PROPERTY_MAPPING,
        {
            "manifest_path": {
                "explanation": "Path to the manifest/database/SBOM artifact (replaces locator).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "commit_sha": {
                "explanation": "Source commit SHA the artifact was produced from.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "branch": {
                "explanation": "Source branch the artifact was produced on.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "language_code": {
                "explanation": "Language code (e.g. 'java', 'python') for language-scoped artifacts.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
        },
        context="CodeBuildReader",
    )

    PARAMS_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        TraversalMixin.PARAMS_MAPPING_DELTA,
        EntityFilterParamMixin.PARAMS_MAPPING_DELTA,
        context="CodeBuildReader.PARAMS_MAPPING",
    )


class CodeBuildFeatureGroup(KgConnectorFeatureGroupBase):
    READER_CLASS = None
