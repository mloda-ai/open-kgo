"""Concrete tests for NetworkxEmbeddedReader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.embedded.networkx_embedded import (
    NetworkxEmbeddedReader,
)
from open_kgo.feature_groups.kg.embedded.tests.kg_embedded_contract import (
    EmbeddedContractTestBase,
)


_FIXTURE_GML = Path(__file__).parent / "fixtures" / "triangle.gml"


class TestNetworkxEmbeddedReader(EmbeddedContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[NetworkxEmbeddedReader]:
        return NetworkxEmbeddedReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return {
            "networkx_embedded": {
                "locator": str(_FIXTURE_GML),
                "graph_file_format": "gml",
                "read_only": True,
                "max_threads": 1,
                "result_limit": 100,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        return {
            "networkx_embedded": {
                "locator": str(_FIXTURE_GML),
                "graph_file_format": "evil_format",
            }
        }

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "networkx_embedded__nodes",
            options=Options(context={"operation": "nodes"}),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) > 0

    def test_neighbors_operation(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "networkx_embedded__neighbors",
            options=Options(context={"operation": "neighbors", "start_node": "alice"}),
        )
        rows = run_query("networkx_embedded", self.valid_credentials()["networkx_embedded"], feat)
        assert sorted(r["node"] for r in rows) == ["bob", "carol"]

    def test_http_locator_rejected(self) -> None:
        """Remote schemes must be refused, like the other file-backed readers.

        networkx's loaders only open local paths, so a remote locator could
        never fetch anyway — but rejecting it up front keeps the file-only
        contract uniform and the error message consistent across families.
        """
        cls = self.connector_reader_class()
        creds = {cls.CONNECTOR_ID: {"locator": "http://example.com/evil.gml"}}
        with pytest.raises(ValueError, match="scheme"):
            cls.connect(creds)
