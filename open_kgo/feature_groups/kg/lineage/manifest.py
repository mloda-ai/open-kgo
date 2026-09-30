"""mloda entry-point manifest for the lineage family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.lineage.dbt_manifest import DbtManifestFeatureGroup
from open_kgo.feature_groups.kg.lineage.openlineage_events import OpenLineageFeatureGroup

FEATURE_GROUPS = (DbtManifestFeatureGroup, OpenLineageFeatureGroup)
