"""mloda entry-point manifest for the saas_authz family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.saas_authz.in_process_tuple_store import InProcessTupleStoreFeatureGroup
from open_kgo.feature_groups.kg.saas_authz.paginated_tuple_store import PaginatedTupleStoreFeatureGroup

FEATURE_GROUPS = (InProcessTupleStoreFeatureGroup, PaginatedTupleStoreFeatureGroup)
