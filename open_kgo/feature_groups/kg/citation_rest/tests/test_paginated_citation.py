"""Concrete tests for PaginatedCitationReader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.citation_rest.paginated_citation import (
    PaginatedCitationReader,
)
from open_kgo.feature_groups.kg.citation_rest.tests.kg_citation_rest_contract import (
    CitationRestContractTestBase,
)
from open_kgo.feature_groups.kg.errors import InvalidCredentialShape, MissingRequiredKeysError
from open_kgo.feature_groups.kg.mixins import parse_offset_cursor
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials


_FIXTURE = Path(__file__).parent / "fixtures" / "citations.json"


class TestPaginatedCitationReader(CitationRestContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[PaginatedCitationReader]:
        return PaginatedCitationReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return make_valid_credentials(
            cls.connector_reader_class(),
            locator=str(_FIXTURE),
            pagination_style="cursor",
            page_size=2,
            species_prefix="HSA",
            dataset_version="v1",
            result_limit=100,
        )

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        # pagination_style="page" is in the family set but outside this
        # concrete's SUPPORTED_VALUES narrowing ({"cursor"}), so it rejects.
        return {"paginated_citation": {"locator": str(_FIXTURE), "pagination_style": "page"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "paginated_citation__page1",
            options=Options(context={"stable_id": "W1", "hierarchy_depth": 3, "entity_type": "article"}),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "stableId" in result[0]

    def test_first_page_honors_page_size(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("paginated_citation", self.valid_credentials()["paginated_citation"], feat)
        assert [r["stableId"] for r in rows] == ["W1", "W2"]

    def test_second_page_via_cursor_token(self) -> None:
        """A cursor_token offset returns the next page of the article-filtered walk."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "paginated_citation__page2",
            options=Options(
                context={
                    "stable_id": "W1",
                    "hierarchy_depth": 3,
                    "entity_type": "article",
                    "cursor_token": "offset:2",
                }
            ),
        )
        rows = run_query("paginated_citation", self.valid_credentials()["paginated_citation"], feat)
        assert [r["stableId"] for r in rows] == ["W4", "W5"]

    def test_entity_type_filter_selects_books(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "paginated_citation__books",
            options=Options(context={"stable_id": "W1", "hierarchy_depth": 3, "entity_type": "book"}),
        )
        rows = run_query("paginated_citation", self.valid_credentials()["paginated_citation"], feat)
        assert [r["stableId"] for r in rows] == ["W3"]

    def test_short_final_page_returns_tail(self) -> None:
        """A cursor offset landing one short of the end returns just the tail id.

        The article-filtered, sorted walk is ``[W1, W2, W4, W5]``; ``offset:3``
        with ``page_size=2`` slices ``[3:5]`` -> the single remaining id ``W5``.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "paginated_citation__tail",
            options=Options(
                context={
                    "stable_id": "W1",
                    "hierarchy_depth": 3,
                    "entity_type": "article",
                    "cursor_token": "offset:3",
                }
            ),
        )
        rows = run_query("paginated_citation", self.valid_credentials()["paginated_citation"], feat)
        assert [r["stableId"] for r in rows] == ["W5"]

    @pytest.mark.parametrize("bad", [0, -1, "abc", True, False])
    def test_invalid_page_size_rejected(self, bad: object) -> None:
        """A non-positive-int / non-numeric / bool page_size raises InvalidCredentialShape at load time."""
        slot = dict(self.valid_credentials()["paginated_citation"])
        slot["page_size"] = bad
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        with pytest.raises(InvalidCredentialShape):
            PaginatedCitationReader.load_data({"paginated_citation": slot}, fs)

    @pytest.mark.parametrize(
        "bad",
        [
            "garbage",
            True,
            "offset:-1",
            # int() leniencies outside the documented ^offset:[0-9]+$ grammar:
            "offset:5_0",  # underscore separator; bare int() would parse 50
            "offset:+5",  # explicit sign
            "offset: 5",  # whitespace
            "offset:٥",  # Arabic-Indic digit five; isdecimal-true but not ASCII
        ],
    )
    def test_malformed_cursor_token_rejected(self, bad: object) -> None:
        """Anything outside the exact ``offset:`` + ASCII-digits grammar raises InvalidCredentialShape."""
        with pytest.raises(InvalidCredentialShape):
            parse_offset_cursor("paginated_citation", bad)

    def test_malformed_cursor_token_rejected_through_reader(self) -> None:
        """The typed error also surfaces through the reader's load path, not just the helper."""
        slot = dict(self.valid_credentials()["paginated_citation"])
        fs = FeatureSet()
        fs.add(
            Feature(
                "paginated_citation__bad_cursor",
                options=Options(context={"stable_id": "W1", "cursor_token": "garbage"}),
            )
        )
        with pytest.raises(InvalidCredentialShape, match="malformed"):
            PaginatedCitationReader.load_data({"paginated_citation": slot}, fs)

    def test_none_cursor_is_first_page(self) -> None:
        assert parse_offset_cursor("paginated_citation", None) == 0

    def test_leading_zero_cursor_offset_is_plain_decimal(self) -> None:
        """Leading zeros stay valid: ``offset:007`` is decimal 7, not octal or an error."""
        assert parse_offset_cursor("paginated_citation", "offset:007") == 7

    @pytest.mark.parametrize("bad", ["abc", -1, True, False, 2.5])
    def test_invalid_hierarchy_depth_rejected(self, bad: object) -> None:
        """A non-int / bool / negative hierarchy_depth raises InvalidCredentialShape, not a raw ValueError."""
        slot = dict(self.valid_credentials()["paginated_citation"])
        fs = FeatureSet()
        fs.add(
            Feature(
                "paginated_citation__bad_depth",
                options=Options(context={"stable_id": "W1", "hierarchy_depth": bad}),
            )
        )
        with pytest.raises(InvalidCredentialShape, match="hierarchy_depth"):
            PaginatedCitationReader.load_data({"paginated_citation": slot}, fs)

    def test_omitted_pagination_style_rejected_at_validate_time(self) -> None:
        """A full valid slot minus ``pagination_style`` fails ``is_valid_credentials``.

        ``SUPPORTED_VALUES`` only validates keys present in the slot, and the
        family default is ``none`` (a style this reader does not honor), so
        the omission case must be closed by ``REQUIRED_KEYS``: without it, the
        slot would validate, serve a cursor-paginated page 1, and then reject
        its own continuation token at the cross-layer guard.
        """

        slot = dict(self.valid_credentials()["paginated_citation"])
        del slot["pagination_style"]
        creds = {"paginated_citation": slot}
        assert PaginatedCitationReader.is_valid_credentials(creds) is False
        with pytest.raises(MissingRequiredKeysError):
            PaginatedCitationReader._validate_shape(slot)

    def test_cursor_token_requires_cursor_family_style(self) -> None:
        """The PaginationMixin cross-layer guard fires through ``build_params``.

        Exercises the real rejection path, not just ``is_valid_credentials``
        (which never runs ``_validate_cross_layer``). The reader narrows
        ``pagination_style`` to ``{"cursor"}`` AND requires it via
        ``REQUIRED_KEYS``, so no slot that passes validation can omit it (see
        ``test_omitted_pagination_style_rejected_at_validate_time``); this
        test bypasses slot validation deliberately: ``build_params`` does not
        run ``_validate_shape``, so a partial slot without ``pagination_style``
        (the cross-layer hook defaults it to ``"none"``) paired with a
        ``cursor_token`` on the feature still exercises the guard's rejection
        branch directly.
        """
        fs = FeatureSet()
        fs.add(
            Feature(
                "paginated_citation__cursor",
                options=Options(context={"stable_id": "W1", "cursor_token": "offset:2"}),
            )
        )
        creds = {"locator": str(_FIXTURE)}
        with pytest.raises(InvalidCredentialShape, match="cursor-family"):
            PaginatedCitationReader.build_params(fs, creds)

    def test_result_limit_caps_below_page_size(self) -> None:
        """result_limit smaller than page_size truncates the page to result_limit rows."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["paginated_citation"])
        slot["page_size"] = 2
        slot["result_limit"] = 1
        feat = self.feature_under_test()
        rows = run_query("paginated_citation", slot, feat)
        assert [r["stableId"] for r in rows] == ["W1"]

    def test_overflow_cursor_offset_yields_empty_page(self) -> None:
        """A cursor offset past the end of the collected walk returns an empty page."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "paginated_citation__overflow",
            options=Options(
                context={
                    "stable_id": "W1",
                    "hierarchy_depth": 3,
                    "entity_type": "article",
                    "cursor_token": "offset:99",
                }
            ),
        )
        rows = run_query("paginated_citation", self.valid_credentials()["paginated_citation"], feat)
        assert rows == []
