"""Concrete tests for NetworkxEmbeddedReader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.embedded.networkx_embedded import (
    NetworkxEmbeddedReader,
)
from open_kgo.feature_groups.kg.embedded.tests.kg_embedded_contract import (
    EmbeddedContractTestBase,
)
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials


_FIXTURE_GML = Path(__file__).parent / "fixtures" / "triangle.gml"
_FIXTURE_DIRECTED_GML = Path(__file__).parent / "fixtures" / "directed_chain.gml"

# Same document the igraph sibling uses, so a difference between the two
# backends cannot be blamed on the input.
_GRAPHML_CHAIN = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph id="G" edgedefault="directed">
    <node id="alice"/>
    <node id="bob"/>
    <node id="carol"/>
    <edge source="alice" target="bob"/>
    <edge source="bob" target="carol"/>
  </graph>
</graphml>
"""


class TestNetworkxEmbeddedReader(EmbeddedContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[NetworkxEmbeddedReader]:
        return NetworkxEmbeddedReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return make_valid_credentials(cls.connector_reader_class(), locator=str(_FIXTURE_GML), result_limit=100)

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

    def test_edges_operation(self) -> None:
        """``operation=edges`` is advertised in SUPPORTED_VALUES and dispatched in _load_rows, but was never asserted."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = {"locator": str(_FIXTURE_DIRECTED_GML), "graph_file_format": "gml"}
        feat = Feature("networkx_embedded__edges", options=Options(context={"operation": "edges"}))
        rows = run_query("networkx_embedded", slot, feat)
        assert sorted((r["src"], r["dst"]) for r in rows) == [("alice", "bob"), ("bob", "carol")]

    def test_graphml_format_loads_and_returns_rows(self, tmp_path: Path) -> None:
        """``graph_file_format=graphml`` maps to nx.read_graphml in _LOADERS; prove that path works.

        Asserts nodes and edges from one tmp_path-written GraphML document, so
        the advertised format is exercised end to end rather than only declared.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        graphml_path = tmp_path / "chain.graphml"
        graphml_path.write_text(_GRAPHML_CHAIN, encoding="utf-8")
        slot = {"locator": str(graphml_path), "graph_file_format": "graphml"}

        nodes_feat = Feature("networkx_embedded__graphml_nodes", options=Options(context={"operation": "nodes"}))
        node_rows = run_query("networkx_embedded", slot, nodes_feat)
        assert sorted(r["node"] for r in node_rows) == ["alice", "bob", "carol"]

        edges_feat = Feature("networkx_embedded__graphml_edges", options=Options(context={"operation": "edges"}))
        edge_rows = run_query("networkx_embedded", slot, edges_feat)
        assert sorted((r["src"], r["dst"]) for r in edge_rows) == [("alice", "bob"), ("bob", "carol")]
