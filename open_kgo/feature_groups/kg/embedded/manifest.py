"""mloda entry-point manifest for the embedded family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.embedded.igraph_embedded import IGraphEmbeddedFeatureGroup
from open_kgo.feature_groups.kg.embedded.networkx_embedded import NetworkxEmbeddedFeatureGroup

FEATURE_GROUPS = (IGraphEmbeddedFeatureGroup, NetworkxEmbeddedFeatureGroup)
