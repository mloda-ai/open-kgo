"""File-fixture citation connector: serves canned Reactome-shaped JSON.

Looks up an entity by ``stable_id`` from a JSON file that maps stable IDs to
records, plus a hierarchy that lets ``hierarchy_depth`` walk ancestors.

The concrete walker is single-shot (no pagination) and does not dispatch on
``entity_type``. Surface narrowing:

- ``pagination_style`` and ``page_size`` are dropped from ``PROPERTY_MAPPING``
  and rejected by the closed-world credential check.
- ``cursor_token`` and ``entity_type`` are dropped from ``PARAMS_MAPPING``;
  setting either in ``feature.options`` is rejected per-call via the
  ``_STRIPPED_PARAMS`` hook on ``ParamReader``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext, narrow_property_mapping
from open_kgo.feature_groups.kg.citation_rest.base import (
    CitationRestFeatureGroup,
    CitationRestReader,
)
from open_kgo.feature_groups.kg.fixtures import copy_cached_row, load_json_fixture
from open_kgo.feature_groups.kg.traversal import bfs_collect_ids


class FileFixtureCitationReader(CitationRestReader):
    CONNECTOR_ID: ClassVar[str] = "file_fixture_citation"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",),)

    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = narrow_property_mapping(
        CitationRestReader.PROPERTY_MAPPING, "pagination_style", "page_size"
    )
    PARAMS_MAPPING: ClassVar[dict[str, Any]] = {
        k: v for k, v in CitationRestReader.PARAMS_MAPPING.items() if k in {"stable_id", "hierarchy_depth"}
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> dict[str, Any]:
        """Return the parsed citation catalog; mtime-cached.

        Returned dict is shared across calls and MUST be treated as
        read-only. ``load_data`` shallow-copies each row before
        appending so the cached entry never escapes as a mutable
        reference.
        """
        return load_json_fixture(cls.CONNECTOR_ID, slot["locator"])

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        catalog = connection
        # Thread ctx.slot through so the cross-layer hook engages. cursor_token
        # is stripped from PARAMS_MAPPING here, so _reject_stripped_params
        # short-circuits before _validate_cross_layer; the slot is still passed
        # for forward compatibility with future concretes that retain cursor_token.
        params = cls.build_params(features, ctx.slot)
        stable_id = params.get("stable_id")
        depth = int(params.get("hierarchy_depth", 1))

        if stable_id is None:
            raise ValueError(f"{cls.CONNECTOR_ID}: stable_id is required for citation lookup.")

        # BFS the ancestor hierarchy, bounded by result_limit so the cap
        # bounds the walk rather than slicing a fully-expanded set.
        collected = bfs_collect_ids(
            lambda node_id: node_id in catalog,
            lambda node_id: catalog[node_id].get("ancestors", []),
            stable_id,
            depth=depth,
            max_nodes=ctx.result_limit,
        )
        # ``catalog`` is shared across calls (see ``_connect_from_slot``);
        # copy_cached_row keeps the cache read-only at the row level.
        return [copy_cached_row(catalog[node_id]) for node_id in collected]


class FileFixtureCitationFeatureGroup(CitationRestFeatureGroup):
    READER_CLASS: ClassVar[type[FileFixtureCitationReader]] = FileFixtureCitationReader
