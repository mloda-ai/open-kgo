"""NetworkX-backed embedded graph reader.

Loads a graph from a fixture file (.gml / .graphml / etc.) at ``locator``,
then runs a small "operation" defined by the Feature's options:

- ``operation=neighbors`` + ``start_node``: returns the neighbors list.
- ``operation=nodes``: returns all nodes.
- ``operation=edges``: returns all edges.

Demonstrates the embedded-family contract without real Cypher/SPARQL.
"""

from __future__ import annotations

from itertools import islice
from typing import Any, ClassVar, Mapping

import networkx as nx

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.embedded.base import (
    EmbeddedGraphFeatureGroup,
    EmbeddedGraphReader,
)
from open_kgo.feature_groups.kg.fixtures import _rejected_scheme


_LOADERS: dict[str, Any] = {
    "gml": nx.read_gml,
    "graphml": nx.read_graphml,
    "edgelist": nx.read_edgelist,
}


class NetworkxEmbeddedReader(EmbeddedGraphReader):
    CONNECTOR_ID: ClassVar[str] = "networkx_embedded"
    # Strict-enum dispositions:
    #   - graph_file_format: every family-advertised format is honored via
    #     ``_LOADERS``. Mirrored explicitly (rather than left implicit) so a
    #     future family-level addition surfaces here as a contract failure.
    #   - operation: every family-advertised operation is dispatched in
    #     ``load_data``. Same rationale.
    # Maintainer note: the two mirrored sets below are coupled to the
    # family-level ``allowed_values`` for ``graph_file_format`` /
    # ``operation``. Adding a value at the family level requires either
    # wiring it in (``_LOADERS`` for formats, the ``load_data`` dispatch
    # for operations) and adding it to the mirror, or excluding it here
    # so the gap surfaces as a contract failure rather than a silent drop.
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "graph_file_format": frozenset({"gml", "graphml", "edgelist"}),
        "operation": frozenset({"nodes", "edges", "neighbors"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> nx.Graph:
        locator = slot.get("locator")
        if not locator:
            return nx.Graph()
        # Reject remote locators (http://, ftp://, ...) for parity with the
        # other file-backed readers; the networkx loaders only open local
        # paths, so this keeps the file-only contract uniform across families.
        bad = _rejected_scheme(locator)
        if bad is not None:
            raise ValueError(
                f"{cls.CONNECTOR_ID}: locator scheme {bad!r} is not permitted; "
                f"only local file paths or file:// URLs are allowed."
            )
        fmt = slot.get("graph_file_format", "gml")
        loader = _LOADERS.get(fmt)
        if loader is None:
            raise ValueError(f"{cls.CONNECTOR_ID}: unsupported graph_file_format={fmt!r}")
        return loader(locator)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> list[dict[str, Any]]:
        ctx = cls._prepare_load(data_access)
        graph = cls._connect_from_slot(ctx.slot)

        params = cls.build_params(features, ctx.slot)
        op = params["operation"]

        if op == "nodes":
            return [{"node": n} for n in islice(graph.nodes(), ctx.result_limit)]
        if op == "edges":
            return [{"src": u, "dst": v} for (u, v) in islice(graph.edges(), ctx.result_limit)]
        if op == "neighbors":
            start = params.get("start_node")
            if start is None:
                raise ValueError(f"{cls.CONNECTOR_ID}: operation=neighbors requires 'start_node'.")
            return [{"node": n} for n in islice(graph.neighbors(start), ctx.result_limit)]
        raise ValueError(f"{cls.CONNECTOR_ID}: unsupported operation={op!r}")


class NetworkxEmbeddedFeatureGroup(EmbeddedGraphFeatureGroup):
    READER_CLASS: ClassVar[type[NetworkxEmbeddedReader]] = NetworkxEmbeddedReader  # type: ignore[assignment]
