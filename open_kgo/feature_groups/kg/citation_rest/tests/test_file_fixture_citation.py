"""Concrete tests for FileFixtureCitationReader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mloda.user import Feature, Options

import pytest

from mloda.provider import HashableDict

from open_kgo.feature_groups.kg.citation_rest.file_fixture_citation import (
    FileFixtureCitationReader,
)
from open_kgo.feature_groups.kg.citation_rest.tests.kg_citation_rest_contract import (
    CitationRestContractTestBase,
)
from open_kgo.feature_groups.kg.errors import InvalidCredentialShape


_FIXTURE = Path(__file__).parent / "fixtures" / "reactome.json"


class TestFileFixtureCitationReader(CitationRestContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[FileFixtureCitationReader]:
        return FileFixtureCitationReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return {
            "file_fixture_citation": {
                "locator": str(_FIXTURE),
                "species_prefix": "HSA",
                "dataset_version": "v90",
                "result_limit": 100,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        # No strict-validation enum on this concrete after the universal
        # auth surface was removed; the closed-world
        # unknown-key rejection still exercises the contract.
        return {"file_fixture_citation": {"locator": str(_FIXTURE), "definitely_not_a_kg_key": "x"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "file_fixture_citation__pathway",
            options=Options(context={"stable_id": "R-HSA-1640170", "hierarchy_depth": 1}),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "stableId" in result[0]

    def test_hierarchy_depth_includes_ancestor(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("file_fixture_citation", self.valid_credentials()["file_fixture_citation"], feat)
        ids = {r["stableId"] for r in rows}
        assert "R-HSA-1640170" in ids
        assert "R-HSA-1640171" in ids

    def test_hierarchy_depth_one_excludes_grandparent(self) -> None:
        """Depth=1 must NOT include the grandparent (BFS is bounded, not unbounded).

        ``entity_type`` is omitted from context: this concrete strips it from
        ``PARAMS_MAPPING`` so the per-call validator rejects it. The BFS test
        only needs ``stable_id`` + ``hierarchy_depth`` to exercise the
        depth-bounding behaviour.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "file_fixture_citation__pathway_d1",
            options=Options(context={"stable_id": "R-HSA-1640170", "hierarchy_depth": 1}),
        )
        rows = run_query("file_fixture_citation", self.valid_credentials()["file_fixture_citation"], feat)
        ids = {r["stableId"] for r in rows}
        assert "R-HSA-1640172" not in ids

    def test_hierarchy_depth_two_includes_grandparent(self) -> None:
        """Depth=2 must reach grandparents (BFS recurses, not flat slice).

        See ``test_hierarchy_depth_one_excludes_grandparent`` for the
        rationale on omitting ``entity_type``.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "file_fixture_citation__pathway_d2",
            options=Options(context={"stable_id": "R-HSA-1640170", "hierarchy_depth": 2}),
        )
        rows = run_query("file_fixture_citation", self.valid_credentials()["file_fixture_citation"], feat)
        ids = {r["stableId"] for r in rows}
        assert ids == {"R-HSA-1640170", "R-HSA-1640171", "R-HSA-1640172"}

    @pytest.mark.parametrize("key", ["pagination_style", "page_size"])
    def test_stripped_keys_rejected_by_closed_world(self, key: str) -> None:
        slot = dict(self.valid_credentials()["file_fixture_citation"])
        slot[key] = "none" if key == "pagination_style" else 100
        creds = HashableDict({"file_fixture_citation": slot})
        assert FileFixtureCitationReader.is_valid_credentials(creds) is False
        with pytest.raises(InvalidCredentialShape):
            FileFixtureCitationReader._validate_shape(slot)
