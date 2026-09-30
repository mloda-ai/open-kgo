"""mloda entry-point manifest for the agent_memory family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.agent_memory.graph_walk_memory import GraphWalkMemoryFeatureGroup
from open_kgo.feature_groups.kg.agent_memory.networkx_memory import NetworkxMemoryFeatureGroup

FEATURE_GROUPS = (GraphWalkMemoryFeatureGroup, NetworkxMemoryFeatureGroup)
