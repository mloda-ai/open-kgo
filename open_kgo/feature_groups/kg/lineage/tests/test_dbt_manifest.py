"""Concrete tests for DbtManifestReader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.lineage.dbt_manifest import DbtManifestReader
from open_kgo.feature_groups.kg.lineage.tests.kg_lineage_contract import (
    LineageContractTestBase,
)


_FIXTURE = Path(__file__).parent / "fixtures" / "manifest.json"


class TestDbtManifestReader(LineageContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[DbtManifestReader]:
        return DbtManifestReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return {
            "dbt_manifest": {
                "locator": str(_FIXTURE),
                "result_limit": 100,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        # Bad ``lineage_direction`` value triggers the SUPPORTED_VALUES
        # narrowing on this concrete; the earlier ``auth_method="evil"``
        # seed went away with the universal auth surface.
        return {"dbt_manifest": {"locator": str(_FIXTURE), "lineage_direction": "SIDEWAYS"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "dbt_manifest__upstream",
            options=Options(
                context={
                    "asset_urn": "model.shop.fct_orders",
                    "lineage_direction": "UPSTREAM",
                    "upstream_depth": 2,
                    "downstream_depth": 0,
                }
            ),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "urn" in result[0]

    def test_upstream_walk_two_hops(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("dbt_manifest", self.valid_credentials()["dbt_manifest"], feat)
        urns = [r["urn"] for r in rows]
        assert "model.shop.fct_orders" in urns
        assert "model.shop.stg_orders" in urns
        assert "source.shop.raw.orders" in urns
