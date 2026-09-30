"""File-fixture REST connector using page-number pagination.

Second concrete in the ``rest_public`` family alongside ``FileFixtureRestReader``
(which uses opaque-cursor pagination). This reader exercises the family's
*counter* pagination branch: ``pagination_style=page`` plus ``page_size``, the
two surfaces the cursor concrete narrows away (it pins ``pagination_style`` to
``cursor`` and drops ``page_size``). It therefore proves the PaginationMixin's
page/page_size contract is real, modelling page-number APIs such as GBIF or the
GitHub REST API rather than OpenAlex cursors.

The ``locator`` points to a directory of ``page_<N>.json`` files, each shaped
like ``{"results": [...rows...]}``. The reader walks pages in numeric order and
stops at the first page returning fewer than ``page_size`` rows (the standard
page-number end-of-collection signal), or when ``result_limit`` is reached.

``page_size`` is a termination threshold, NOT a per-page row cap:

- A page file holding MORE rows than ``page_size`` emits all of its rows;
  the reader never truncates a page down to ``page_size``.
- A ``page_size`` LARGER than a page's row count makes that page look like
  the final (short) page, so the walk stops there even when later
  ``page_<N>.json`` files exist on disk. This simulates the standard "server
  returned fewer rows than requested means last page" heuristic; the flip
  side is that a misconfigured ``page_size`` silently truncates the corpus.
- A last page holding exactly ``page_size`` rows does not trip the
  short-page signal; the walk then terminates cleanly via file exhaustion
  (no successor ``page_<N>.json`` file).

Because the threshold must match the fixture's authored page size, both
``pagination_style`` and ``page_size`` are in ``REQUIRED_KEYS``: an omitted
``pagination_style`` would bypass the ``SUPPORTED_VALUES`` narrowing (only
keys present in the slot are validated) and run under the family default
``none``, and an omitted ``page_size`` would default to the family's 100,
which makes any committed fixture page look final and silently drops every
later page. Omitting either key is rejected at ``is_valid_credentials`` /
``_validate_shape`` time via ``MissingRequiredKeysError``.

There is no per-call page selector (the concrete strips ``cursor_token`` /
``entity_type`` per-call params and the family declares no page-number
param): every call concatenates rows from page 1 until termination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.fixtures import copy_cached_rows, load_json_fixture
from open_kgo.feature_groups.kg.mixins import parse_page_size
from open_kgo.feature_groups.kg.rest_public.base import (
    RestPublicFeatureGroup,
    RestPublicReader,
)

# Same page_<N>.json filename-to-index rule as the cursor concrete; imported
# from the sibling rather than duplicated so the sort key cannot drift.
from open_kgo.feature_groups.kg.rest_public.file_fixture_rest import _page_index


class FileFixturePagedRestReader(RestPublicReader):
    CONNECTOR_ID: ClassVar[str] = "file_fixture_paged_rest"
    # pagination_style is REQUIRED (not just narrowed): the family default is
    # "none", which this reader does not honor, and SUPPORTED_VALUES only
    # validates keys present in the slot. An omitted pagination_style would
    # otherwise pass is_valid_credentials and silently run the page walk
    # under a defaulted "none" label.
    # page_size is REQUIRED too (this reader only): it is the walk's
    # termination threshold and must match the fixture's authored page size;
    # the family default of 100 makes any authored page look like the final
    # (short) page and silently truncates fixture corpora to page 1.
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",), ("pagination_style",), ("page_size",))

    # ``page_size`` is RETAINED (the new surface this concrete honors, vs. the
    # cursor concrete which drops it). ``cursor_token`` and ``entity_type`` are
    # dropped from PARAMS_MAPPING and rejected per-call via ``_STRIPPED_PARAMS``.
    PARAMS_MAPPING: ClassVar[dict[str, Any]] = {}
    ROWS_FROM_SLOT: ClassVar[bool] = True

    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "pagination_style": frozenset({"page"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> Path:
        """Return the locator path as-is; pages are read lazily in load_data.

        The returned ``Path`` is a transient, no-resource handle (category 3
        in the ``KgConnectorReaderBase._connect_from_slot`` lifecycle
        contract), mirroring the cursor sibling. It is consumed by the
        contract test that calls ``connect()`` and gives the caller something
        to assert on; ``load_data`` re-derives the path from the slot (falling
        back to ``path.parent`` when the locator is a file) and routes each
        page through ``load_json_fixture`` directly, so this return value is
        intentionally not threaded into the page-walk loop.
        """
        path = Path(str(slot["locator"]))
        if not path.exists():
            raise FileNotFoundError(f"{cls.CONNECTOR_ID}: locator path {path} does not exist.")
        return path

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        path = connection

        page_size = parse_page_size(cls.CONNECTOR_ID, ctx.slot.get("page_size"), 100)
        pages_dir = path if path.is_dir() else path.parent
        page_files = sorted(pages_dir.glob("page_*.json"), key=_page_index)

        rows: list[dict[str, Any]] = []
        for page_file in page_files:
            body = load_json_fixture(cls.CONNECTOR_ID, page_file)
            results = body.get("results", [])
            rows.extend(copy_cached_rows(results, ctx.result_limit - len(rows)))
            if len(rows) >= ctx.result_limit:
                return rows
            # Page-number termination: a page shorter than ``page_size`` is the
            # last page of the collection, so stop rather than reading further
            # files (mirrors how a real page-number API signals exhaustion).
            if len(results) < page_size:
                break
        return rows


class FileFixturePagedRestFeatureGroup(RestPublicFeatureGroup):
    READER_CLASS: ClassVar[type[FileFixturePagedRestReader]] = FileFixturePagedRestReader  # type: ignore[assignment]
