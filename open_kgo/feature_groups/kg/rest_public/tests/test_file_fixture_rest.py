"""Concrete tests for FileFixtureRestReader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from mloda.user import Feature, Options

import pytest

from mloda.provider import HashableDict

from open_kgo.feature_groups.kg.errors import InvalidCredentialShape
from open_kgo.feature_groups.kg.rest_public.file_fixture_rest import (
    FileFixtureRestReader,
)
from open_kgo.feature_groups.kg.rest_public.tests.kg_rest_public_contract import (
    RestPublicContractTestBase,
)


_FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestFileFixtureRestReader(RestPublicContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[FileFixtureRestReader]:
        return FileFixtureRestReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return {
            "file_fixture_rest": {
                "locator": str(_FIXTURE_DIR),
                "pagination_style": "cursor",
                "rate_limit_pace": 100,
                "result_limit": 100,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        return {"file_fixture_rest": {"locator": str(_FIXTURE_DIR), "pagination_style": "evil"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature("file_fixture_rest__list_works", options=Options(context={}))

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "id" in result[0]

    def test_pagination_yields_three_total_rows(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("file_fixture_rest", self.valid_credentials()["file_fixture_rest"], feat)
        assert len(rows) == 3
        assert {r["id"] for r in rows} == {"W001", "W002", "W003"}

    @pytest.mark.parametrize("style", ["page", "offset", "odata-nextLink", "cursorMark", "start_rows", "none"])
    def test_unsupported_pagination_styles_rejected_at_validate_time(self, style: str) -> None:
        slot = dict(self.valid_credentials()["file_fixture_rest"])
        slot["pagination_style"] = style
        creds = HashableDict({"file_fixture_rest": slot})
        assert FileFixtureRestReader.is_valid_credentials(creds) is False
        with pytest.raises(InvalidCredentialShape):
            FileFixtureRestReader._validate_shape(slot)

    def test_stripped_keys_rejected_by_closed_world(self) -> None:
        slot = dict(self.valid_credentials()["file_fixture_rest"])
        slot["page_size"] = 2
        creds = HashableDict({"file_fixture_rest": slot})
        assert FileFixtureRestReader.is_valid_credentials(creds) is False
        with pytest.raises(InvalidCredentialShape):
            FileFixtureRestReader._validate_shape(slot)

    def test_pages_walked_in_numeric_not_lexical_order(self, tmp_path: Path) -> None:
        """11 pages must walk page_1..page_11, not the lexical page_1, page_10, page_11, page_2.

        Only the last page (page_11) carries a null ``next_cursor``; every
        earlier page chains forward. A lexical sort would visit page_11
        (null cursor) third and break early, dropping page_2..page_10. A
        numeric sort reads all 11 rows in order.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        total = 11
        for n in range(1, total + 1):
            next_cursor = None if n == total else f"cursor_to_page_{n + 1}"
            (tmp_path / f"page_{n}.json").write_text(
                json.dumps(
                    {
                        "results": [{"id": f"W{n:03d}", "title": f"Paper {n}"}],
                        "meta": {"next_cursor": next_cursor},
                    }
                ),
                encoding="utf-8",
            )

        slot = dict(self.valid_credentials()["file_fixture_rest"])
        slot["locator"] = str(tmp_path)
        feat = self.feature_under_test()
        rows = run_query("file_fixture_rest", slot, feat)

        assert [r["id"] for r in rows] == [f"W{n:03d}" for n in range(1, total + 1)]
