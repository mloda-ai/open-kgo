"""Concrete tests for CycloneDxSbomReader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.code_build.cyclonedx_sbom import CycloneDxSbomReader
from open_kgo.feature_groups.kg.code_build.tests.kg_code_build_contract import (
    CodeBuildContractTestBase,
)


_FIXTURE = Path(__file__).parent / "fixtures" / "sample.cdx.json"


class TestCycloneDxSbomReader(CodeBuildContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[CycloneDxSbomReader]:
        return CycloneDxSbomReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        # REQUIRED_KEYS = (("manifest_path", "locator"),) lists both keys as
        # alternatives; the per-alternative coherence contract requires
        # valid_credentials() to set every alternative. _connect_from_slot
        # reads `manifest_path or locator`, so a slot with BOTH keys set
        # only exercises the manifest_path branch from connect() directly;
        # the coherence test pops one alternative at a time and re-calls
        # connect(), which is where the locator branch is exercised.
        return {
            "cyclonedx_sbom": {
                "manifest_path": str(_FIXTURE),
                "locator": str(_FIXTURE),
                "commit_sha": "a1b2c3d4",
                "branch": "main",
                "language_code": "python",
                "result_limit": 100,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        # The concrete declares no strict-validation enums after the universal
        # auth surface was removed (issue #32 item 2). Trigger the closed-world
        # unknown-key rejection instead.
        return {"cyclonedx_sbom": {"manifest_path": str(_FIXTURE), "definitely_not_a_kg_key": "x"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature("cyclonedx_sbom__components", options=Options(context={}))

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) == 3 and "name" in result[0]
