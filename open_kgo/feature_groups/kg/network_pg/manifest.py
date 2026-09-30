"""mloda entry-point manifest for the network_pg family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.network_pg.grand_cypher import GrandCypherFeatureGroup
from open_kgo.feature_groups.kg.network_pg.kuzu_cypher import KuzuCypherFeatureGroup

FEATURE_GROUPS = (GrandCypherFeatureGroup, KuzuCypherFeatureGroup)
