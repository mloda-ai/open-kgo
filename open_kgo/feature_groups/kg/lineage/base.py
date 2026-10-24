"""Family base for metadata / lineage KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase, ParamReader
from open_kgo.feature_groups.kg.mixins import EntityFilterParamMixin, TraversalMixin
from open_kgo.feature_groups.kg.spec import property_spec


_FAMILY_PARAMS: dict[str, Any] = {
    "asset_urn": property_spec(
        "URN of the starting asset (DataHub/Atlas style).",
    ),
}


class LineageReader(
    TraversalMixin,
    EntityFilterParamMixin,
    ParamReader,
    family_properties={},
    family_params=_FAMILY_PARAMS,
):
    REQUIRED_PARAMS: ClassVar[tuple[tuple[str, ...], ...]] = (("asset_urn",),)

    # Honest surface (option 3, see base.py): entity/relation filter keys (from
    # EntityFilterParamMixin) the concretes don't apply (they walk by
    # direction/depth from asset_urn), reserved for a traversal-scoping concrete.
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"entity_type", "relationship_type", "expand_paths"})


class LineageFeatureGroup(KgConnectorFeatureGroupBase):
    """Abstract family base; each concrete pins ``READER_CLASS``."""
