"""mloda entry-point manifest for the code_build family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.code_build.cyclonedx_sbom import CycloneDxSbomFeatureGroup
from open_kgo.feature_groups.kg.code_build.spdx_sbom import SpdxSbomFeatureGroup

FEATURE_GROUPS = (CycloneDxSbomFeatureGroup, SpdxSbomFeatureGroup)
