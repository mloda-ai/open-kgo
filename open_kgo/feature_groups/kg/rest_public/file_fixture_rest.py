"""File-fixture REST connector: serves canned JSON pages from local files.

Validates the rest_public contract (cursor pagination, dataset_version)
without making real HTTP calls. The ``locator`` points to a directory
containing ``page_<N>.json`` files; each file looks like an OpenAlex page:

    {"results": [...rows...], "meta": {"next_cursor": "..."}}

The reader walks pages until ``next_cursor`` is null or ``result_limit`` is
reached.

Surface narrowing:

- ``pagination_style`` is narrowed via ``SUPPORTED_VALUES`` to ``cursor`` only;
  any other value is rejected at ``is_valid_credentials`` time.
- ``page_size`` is dropped from ``PROPERTY_MAPPING``; the fixture walker
  reads whole pages, so a credential setting it would be a surface lie. The
  closed-world credential check rejects it.
- ``cursor_token`` and ``entity_type`` are dropped from ``PARAMS_MAPPING``;
  setting either in ``feature.options`` is rejected per-call via the
  ``_STRIPPED_PARAMS`` hook on ``ParamReader``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import narrow_property_mapping
from open_kgo.feature_groups.kg.fixtures import copy_cached_row, load_json_fixture
from open_kgo.feature_groups.kg.rest_public.base import (
    RestPublicFeatureGroup,
    RestPublicReader,
)


def _page_index(page_file: Path) -> int:
    """Return the integer ``<N>`` from a ``page_<N>.json`` filename for numeric sort."""
    match = re.search(r"\d+", page_file.stem)
    return int(match.group()) if match else 0


class FileFixtureRestReader(RestPublicReader):
    CONNECTOR_ID: ClassVar[str] = "file_fixture_rest"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",),)

    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = narrow_property_mapping(RestPublicReader.PROPERTY_MAPPING, "page_size")
    PARAMS_MAPPING: ClassVar[dict[str, Any]] = {}

    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "pagination_style": frozenset({"cursor"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> Path:
        """Return the locator path; pages are read lazily in load_data.

        The returned ``Path`` is a transient, no-resource handle (category 3
        in the ``KgConnectorReaderBase._connect_from_slot`` lifecycle
        contract). It is consumed by the contract test that calls
        ``connect()`` and gives the caller something to assert on;
        ``load_data`` re-derives the same path from the slot and routes
        each page through ``load_json_fixture`` directly, so the
        ``_connect_from_slot`` return value is intentionally not threaded
        into the page-walk loop.
        """
        path = Path(str(slot["locator"]))
        if not path.exists():
            raise FileNotFoundError(f"{cls.CONNECTOR_ID}: locator path {path} does not exist.")
        return path

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> list[dict[str, Any]]:
        ctx = cls._prepare_load(data_access)
        path = cls._connect_from_slot(ctx.slot)

        pages_dir = path if path.is_dir() else path.parent
        # Sort numerically on the ``<N>`` in ``page_<N>.json``: a lexical
        # ``sorted()`` orders ``page_10`` before ``page_2``, which both
        # walks pages out of order (wrong rows under ``result_limit``) and
        # can trip the ``next_cursor`` break on the wrong page, truncating
        # the walk. Files without a digit sort first (key 0).
        page_files = sorted(pages_dir.glob("page_*.json"), key=_page_index)
        rows: list[dict[str, Any]] = []
        for page_file in page_files:
            # Per-page parse routed through the mtime-keyed cache: a real
            # OpenAlex page is 100KB-1MB and the per-call dispatch reads
            # the same pages on every load (issue #32 item 3). The glob
            # itself is cheap and stays uncached.
            body = load_json_fixture(cls.CONNECTOR_ID, page_file)
            for row in body.get("results", []):
                # ``body`` is the cached page dict; copy_cached_row keeps the
                # cache read-only when the row is handed to a downstream consumer.
                rows.append(copy_cached_row(row))
                if len(rows) >= ctx.result_limit:
                    return rows
            if not body.get("meta", {}).get("next_cursor"):
                break
        return rows


class FileFixtureRestFeatureGroup(RestPublicFeatureGroup):
    READER_CLASS: ClassVar[type[FileFixtureRestReader]] = FileFixtureRestReader  # type: ignore[assignment]
