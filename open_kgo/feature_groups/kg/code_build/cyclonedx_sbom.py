"""CycloneDX SBOM parser as a code_build source.

CycloneDX is a real SBOM format; ``components`` array has packages with names
+ versions; ``dependencies`` array has package-ref edges. This concrete
returns the ``components`` list as-is (sliced by ``result_limit``); the
``dependencies`` graph is not walked, so all family-base traversal /
entity-filter per-call keys (``lineage_direction``, ``upstream_depth``,
``downstream_depth``, ``entity_type``, ``relationship_type``,
``expand_paths``) are dropped from this plugin's ``PARAMS_MAPPING`` (option 1
of the "Honest credential surface" rule in base.py). Setting any of them in
``feature.options`` is rejected per-call via the ``_STRIPPED_PARAMS`` hook on
``ParamReader``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.code_build.base import (
    CodeBuildFeatureGroup,
    CodeBuildReader,
)
from open_kgo.feature_groups.kg.fixtures import copy_cached_rows, load_json_fixture


class CycloneDxSbomReader(CodeBuildReader):
    CONNECTOR_ID: ClassVar[str] = "cyclonedx_sbom"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("manifest_path", "locator"),)

    PARAMS_MAPPING: ClassVar[dict[str, Any]] = {}
    ROWS_FROM_SLOT: ClassVar[bool] = True

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> dict[str, Any]:
        """Return the parsed SBOM dict; mtime-cached so repeated loads skip the JSON parse.

        Returned dict is shared across calls and MUST be treated as
        read-only. ``load_data`` shallow-copies each component row before
        returning so the cached component dicts never escape as mutable
        references (mirrors ``FileFixtureCitationReader.load_data``;
        keeps the base-class mutation-safety contract uniform across
        cached readers).
        """
        manifest_path = slot.get("manifest_path") or slot.get("locator")
        return load_json_fixture(cls.CONNECTOR_ID, manifest_path)

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        sbom = connection

        # copy_cached_rows copies only the rows it emits and keeps the shared
        # cached SBOM read-only when a component is returned to a caller (see
        # ``_connect_from_slot``).
        return copy_cached_rows(sbom.get("components", []), ctx.result_limit)


class CycloneDxSbomFeatureGroup(CodeBuildFeatureGroup):
    READER_CLASS: ClassVar[type[CycloneDxSbomReader]] = CycloneDxSbomReader  # type: ignore[assignment]
