"""Family base for REST non-SPARQL public KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase, ParamReader
from open_kgo.feature_groups.kg.mixins import PaginationMixin
from open_kgo.feature_groups.kg.spec import property_spec


_FAMILY_PROPERTIES: dict[str, Any] = {
    "dataset_version": property_spec(
        "Optional dataset/release version pin for reproducibility.",
    ),
    "user_agent": property_spec(
        "User-Agent string (often required by polite-pool endpoints).",
    ),
    "rate_limit_pace": property_spec(
        "Soft pace cap (requests/min); concrete plugin enforces.",
        default=100,
    ),
}

_FAMILY_PARAMS: dict[str, Any] = {
    "entity_type": property_spec(
        "Resource type for the request (e.g. 'works', 'authors').",
    ),
}


class RestPublicReader(
    PaginationMixin, ParamReader, family_properties=_FAMILY_PROPERTIES, family_params=_FAMILY_PARAMS
):
    # Honest surface (option 3, see base.py): HTTP-endpoint knobs the file-fixture
    # concretes ignore (they read canned JSON from disk), reserved for a networked
    # REST concrete.
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"dataset_version", "user_agent", "rate_limit_pace"})


class RestPublicFeatureGroup(KgConnectorFeatureGroupBase):
    """Abstract family base; each concrete pins ``READER_CLASS``."""
