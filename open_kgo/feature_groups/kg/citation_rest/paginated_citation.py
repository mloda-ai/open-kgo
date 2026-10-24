"""Paginated citation connector serving a canned citation graph.

Second concrete in the ``citation_rest`` family alongside
``FileFixtureCitationReader`` (single-shot Reactome ancestor walk). This reader
*honors* the family surface the Reactome concrete drops/strips: cursor
pagination (``pagination_style=cursor`` + ``page_size`` + per-call
``cursor_token``) and the ``entity_type`` per-call filter. It models a
citation-graph API (OpenCitations / Semantic Scholar shape) where a work cites
other works via ``references``.

Walk semantics: BFS the citation graph from ``stable_id`` out to
``hierarchy_depth`` hops, optionally filter the collected works by
``entity_type``, sort by ``stableId`` for a stable page order, then return the
``page_size`` slice starting at the ``cursor_token`` offset (an
``"offset:<N>"`` token), capped at ``result_limit``. Parsed with stdlib JSON
(no new dependency).

HONESTY NOTE: the reader never emits a next-cursor; ``cursor_token`` values
are positional ``"offset:<N>"`` offsets computed by the caller, not opaque
server-issued cursors. The offset indexes into the sorted, filtered walk of
the *current* call, so changing ``entity_type`` (or the fixture contents)
between continuation calls shifts which rows a given offset returns.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.citation_rest.base import (
    CitationRestFeatureGroup,
    CitationRestReader,
)
from open_kgo.feature_groups.kg.fixtures import copy_cached_row, load_json_fixture
from open_kgo.feature_groups.kg.mixins import cursor_page_slice
from open_kgo.feature_groups.kg.traversal import bfs_collect_ids
from open_kgo.feature_groups.kg.validation import parse_bounded_int


class PaginatedCitationReader(CitationRestReader):
    CONNECTOR_ID: ClassVar[str] = "paginated_citation"
    # pagination_style is REQUIRED (not just narrowed): the family default is
    # "none", which this reader does not honor, and SUPPORTED_VALUES only
    # validates keys present in the slot. An omitted pagination_style would
    # otherwise pass is_valid_credentials, serve a cursor-paginated page 1,
    # and then reject its own continuation token (cursor_token + the
    # defaulted "none" fails PaginationMixin._validate_cross_layer).
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",), ("pagination_style",))

    # pagination_style is narrowed to the cursor style this reader implements;
    # page_size, cursor_token, and entity_type are all RETAINED (the new surface
    # this concrete honors, vs. the Reactome concrete which drops/strips them).
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "pagination_style": frozenset({"cursor"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> dict[str, Any]:
        """Return the parsed citation catalog; mtime-cached. Shared, read-only."""
        return load_json_fixture(cls.CONNECTOR_ID, slot["locator"])

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        catalog = connection
        params = cls.build_params(features, ctx.slot)

        # stable_id presence is enforced by REQUIRED_PARAMS inside build_params
        # (MissingRequiredParamsError), so no None re-check is needed here.
        stable_id = params["stable_id"]
        depth = parse_bounded_int(
            cls.CONNECTOR_ID, "hierarchy_depth", params.get("hierarchy_depth"), min_value=0, default=1
        )
        entity_type = params.get("entity_type")

        # BFS the citation graph from stable_id out to ``depth`` hops. The
        # walk is NOT capped at result_limit: the cursor offset indexes into
        # the full sorted walk, so a capped walk would shift saved tokens.
        collected = bfs_collect_ids(
            lambda node_id: node_id in catalog,
            lambda node_id: catalog[node_id].get("references", []),
            stable_id,
            depth=depth,
        )

        # entity_type filter, then deterministic order for stable pagination.
        if entity_type is not None:
            collected = [cid for cid in collected if catalog[cid].get("entityType") == entity_type]
        collected.sort()

        # Cursor page slice, bounded by result_limit.
        page_ids = cursor_page_slice(
            cls.CONNECTOR_ID,
            collected,
            cursor_token=params.get("cursor_token"),
            page_size_value=ctx.slot.get("page_size"),
            result_limit=ctx.result_limit,
        )
        return [copy_cached_row(catalog[cid]) for cid in page_ids]


class PaginatedCitationFeatureGroup(CitationRestFeatureGroup):
    READER_CLASS: ClassVar[type[PaginatedCitationReader]] = PaginatedCitationReader
