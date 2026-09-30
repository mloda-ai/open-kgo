"""Concrete tests for FileFixturePagedRestReader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.errors import InvalidCredentialShape, MissingRequiredKeysError
from open_kgo.feature_groups.kg.rest_public.file_fixture_paged_rest import (
    FileFixturePagedRestReader,
)
from open_kgo.feature_groups.kg.rest_public.tests.kg_rest_public_contract import (
    RestPublicContractTestBase,
)
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "paged"


class TestFileFixturePagedRestReader(RestPublicContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[FileFixturePagedRestReader]:
        return FileFixturePagedRestReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return make_valid_credentials(
            cls.connector_reader_class(),
            locator=str(_FIXTURE_DIR),
            pagination_style="page",
            page_size=2,
            result_limit=100,
        )

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        return {"file_fixture_paged_rest": {"locator": str(_FIXTURE_DIR), "pagination_style": "evil"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature("file_fixture_paged_rest__list", options=Options(context={}))

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "id" in result[0]

    def test_page_pagination_yields_three_total_rows(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("file_fixture_paged_rest", self.valid_credentials()["file_fixture_paged_rest"], feat)
        # Assert exact count alongside the set so a duplicate-emission regression
        # (which a set-only assertion would mask) is caught.
        assert len(rows) == 3
        assert {r["id"] for r in rows} == {"P001", "P002", "P003"}

    def test_result_limit_short_circuits_across_pages(self, tmp_path: Path) -> None:
        """result_limit caps rows across page boundaries; later pages are never read.

        page_1 and page_2 are both *full* (``page_size`` rows each, so the
        end-of-collection break does not fire), and result_limit=3 is reached
        partway through page_2. page_3 is malformed JSON: if the reader walked
        into it, ``load_json_fixture`` would raise loudly, so a passing test
        proves the short-circuit prevented the read.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        (tmp_path / "page_1.json").write_text(json.dumps({"results": [{"id": "A1"}, {"id": "A2"}]}), encoding="utf-8")
        (tmp_path / "page_2.json").write_text(json.dumps({"results": [{"id": "B1"}, {"id": "B2"}]}), encoding="utf-8")
        (tmp_path / "page_3.json").write_text("{ not valid json", encoding="utf-8")

        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        slot["locator"] = str(tmp_path)
        slot["page_size"] = 2
        slot["result_limit"] = 3
        rows = run_query("file_fixture_paged_rest", slot, self.feature_under_test())
        assert [r["id"] for r in rows] == ["A1", "A2", "B1"]

    def test_page_size_terminates_walk(self, tmp_path: Path) -> None:
        """A page shorter than ``page_size`` ends the walk; later pages are not read.

        page_1 has page_size rows (full) and page_2 has fewer (partial → last),
        so page_3 must never be reached even though the file exists.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        (tmp_path / "page_1.json").write_text(json.dumps({"results": [{"id": "A1"}, {"id": "A2"}]}), encoding="utf-8")
        (tmp_path / "page_2.json").write_text(json.dumps({"results": [{"id": "B1"}]}), encoding="utf-8")
        (tmp_path / "page_3.json").write_text(json.dumps({"results": [{"id": "C1"}]}), encoding="utf-8")

        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        slot["locator"] = str(tmp_path)
        rows = run_query("file_fixture_paged_rest", slot, self.feature_under_test())
        assert [r["id"] for r in rows] == ["A1", "A2", "B1"]

    def test_page_size_above_page_rows_stops_after_first_page(self, tmp_path: Path) -> None:
        """page_size larger than a page's row count makes it look final; later pages are skipped.

        Pins the documented "termination threshold, not a cap" semantics:
        page_1 has 2 rows < page_size=3, so the walk treats page_1 as the
        last page and silently ignores page_2 even though the file exists
        (the misconfigured-page_size corpus-truncation behavior).
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        (tmp_path / "page_1.json").write_text(json.dumps({"results": [{"id": "A1"}, {"id": "A2"}]}), encoding="utf-8")
        (tmp_path / "page_2.json").write_text(json.dumps({"results": [{"id": "B1"}, {"id": "B2"}]}), encoding="utf-8")

        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        slot["locator"] = str(tmp_path)
        slot["page_size"] = 3
        rows = run_query("file_fixture_paged_rest", slot, self.feature_under_test())
        assert [r["id"] for r in rows] == ["A1", "A2"]

    def test_exact_multiple_corpus_terminates_via_file_exhaustion(self, tmp_path: Path) -> None:
        """A last page with exactly page_size rows and no successor file terminates cleanly.

        Every page is full (the short-page break never fires), so the walk
        ends only because the page_<N>.json files run out, returning the
        whole corpus on the exact-multiple boundary.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        (tmp_path / "page_1.json").write_text(json.dumps({"results": [{"id": "A1"}, {"id": "A2"}]}), encoding="utf-8")
        (tmp_path / "page_2.json").write_text(json.dumps({"results": [{"id": "B1"}, {"id": "B2"}]}), encoding="utf-8")

        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        slot["locator"] = str(tmp_path)
        slot["page_size"] = 2
        rows = run_query("file_fixture_paged_rest", slot, self.feature_under_test())
        assert [r["id"] for r in rows] == ["A1", "A2", "B1", "B2"]

    @pytest.mark.parametrize("style", ["cursor", "offset", "odata-nextLink", "cursorMark", "start_rows", "none"])
    def test_unsupported_pagination_styles_rejected_at_validate_time(self, style: str) -> None:
        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        slot["pagination_style"] = style
        creds = {"file_fixture_paged_rest": slot}
        assert FileFixturePagedRestReader.is_valid_credentials(creds) is False
        with pytest.raises(InvalidCredentialShape):
            FileFixturePagedRestReader._validate_shape(slot)

    def test_omitted_pagination_style_rejected_at_validate_time(self) -> None:
        """A full valid slot minus ``pagination_style`` fails ``is_valid_credentials``.

        ``SUPPORTED_VALUES`` only validates keys present in the slot, and the
        family default is ``none`` (a style this reader does not honor), so
        the omission case must be closed by ``REQUIRED_KEYS``: without it, the
        slot would validate and the page walk would run under a defaulted
        ``none`` label.
        """
        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        del slot["pagination_style"]
        creds = {"file_fixture_paged_rest": slot}
        assert FileFixturePagedRestReader.is_valid_credentials(creds) is False
        with pytest.raises(MissingRequiredKeysError):
            FileFixturePagedRestReader._validate_shape(slot)

    def test_omitted_page_size_rejected_at_validate_time(self) -> None:
        """A full valid slot minus ``page_size`` fails ``is_valid_credentials``.

        ``page_size`` is this reader's termination threshold and must match
        the fixture's authored page size. The family default of 100 makes any
        committed fixture page look like the final (short) page, so an
        omitted ``page_size`` would silently truncate the corpus to page 1
        (the committed 2-row-per-page fixture would drop page_2 entirely).
        ``REQUIRED_KEYS`` closes that omission hole.
        """
        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        del slot["page_size"]
        creds = {"file_fixture_paged_rest": slot}
        assert FileFixturePagedRestReader.is_valid_credentials(creds) is False
        with pytest.raises(MissingRequiredKeysError):
            FileFixturePagedRestReader._validate_shape(slot)

    def test_page_size_is_honored_in_property_mapping(self) -> None:
        """Unlike the cursor concrete, ``page_size`` is a valid credential key here."""
        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        slot["page_size"] = 5
        creds = {"file_fixture_paged_rest": slot}
        assert FileFixturePagedRestReader.is_valid_credentials(creds) is True

    @pytest.mark.parametrize("bad", [0, -1, "abc"])
    def test_invalid_page_size_rejected(self, bad: object) -> None:
        """A non-positive-int / non-numeric page_size raises InvalidCredentialShape at load time.

        ``page_size`` is ``strict_validation=False`` (no enum), so the guard
        lives at the runtime read in ``load_data``, not in ``_validate_shape``.
        """
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet

        slot = dict(self.valid_credentials()["file_fixture_paged_rest"])
        slot["page_size"] = bad
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        with pytest.raises(InvalidCredentialShape):
            FileFixturePagedRestReader.load_data({"file_fixture_paged_rest": slot}, fs)
