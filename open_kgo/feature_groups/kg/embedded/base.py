"""Family base for embedded / in-memory graph connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase, ParamReader
from open_kgo.feature_groups.kg.spec import property_spec


_GRAPH_FILE_FORMATS: dict[str, str] = {
    "gml": "Graph Modeling Language.",
    "graphml": "GraphML XML.",
    "edgelist": "Plain edge list.",
}


_OPERATIONS: dict[str, str] = {
    "nodes": "Return all node ids.",
    "edges": "Return all edges as [u, v] pairs.",
    "neighbors": "Return neighbors of `start_node`.",
}


class UnknownStartNodeError(ValueError):
    """Raised by embedded connectors when ``start_node`` is not present in the loaded graph.

    Shaped like ``UnknownTenantError`` / ``UnknownMemoryScopeError`` in
    ``kg.errors``: per-call validation can enforce that ``start_node`` is
    *present* for ``operation=neighbors``, but cannot enforce that the value
    *exists* in a given graph file, so the concrete reader resolves it at
    runtime and raises this typed error. Every embedded concrete MUST raise
    this same error on an unknown ``start_node``; the family thesis is
    backend variety with identical behavior, and before this contract the
    two backends diverged (igraph silently returned ``[]``, NetworkX leaked
    a raw ``networkx.NetworkXError``). Enforced family-wide by the
    ``test_unknown_start_node_raises_typed_error`` contract test.
    """

    def __init__(self, connector_id: str, start_node: Any) -> None:
        super().__init__(f"{connector_id}: start_node {start_node!r} is not present in the loaded graph.")
        self.connector_id = connector_id
        self.start_node = start_node


_FAMILY_PROPERTIES: dict[str, Any] = {
    "graph_file_format": property_spec(
        "Graph serialisation format the locator points at.",
        strict=True,
        allowed_values=_GRAPH_FILE_FORMATS,
        default="gml",
    ),
    "read_only": property_spec(
        "Open the graph in read-only mode (advisory; concrete plugin enforces).",
        default=True,
    ),
    "max_threads": property_spec(
        "Soft cap on background worker threads; concrete plugin honors if relevant.",
        default=1,
    ),
}

_FAMILY_PARAMS: dict[str, Any] = {
    "operation": property_spec(
        "Per-call operation against the embedded graph.",
        strict=True,
        allowed_values=_OPERATIONS,
        default="nodes",
    ),
    "start_node": property_spec(
        "Starting node id for `operation=neighbors`; ignored for nodes/edges.",
    ),
}


class EmbeddedGraphReader(ParamReader, family_properties=_FAMILY_PROPERTIES, family_params=_FAMILY_PARAMS):
    """Family base for embedded graph backends.

    Concrete plugins (NetworkxEmbeddedReader, IGraphReader, ...) load a graph
    object from a filesystem path or accept ``locator=None`` for empty graph.
    The backend is in-process, so there is no network surface to configure.

    Per-call inputs (``operation``, ``start_node``) live on
    ``PARAMS_MAPPING`` rather than being read raw from
    ``feature.options.context``. ``operation`` is strict-enum-validated;
    ``start_node`` is free-form (only required when ``operation=neighbors``,
    which the concrete reader enforces at runtime; this is a one-of-N param
    dependency that REQUIRED_PARAMS can't express directly).
    """

    # Honest surface (option 3, see base.py): advisory backend knobs neither
    # in-process concrete enforces (whole-graph, single-threaded), reserved for a
    # backend that can.
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"read_only", "max_threads"})

    REQUIRED_PARAMS: ClassVar[tuple[tuple[str, ...], ...]] = (("operation",),)


class EmbeddedGraphFeatureGroup(KgConnectorFeatureGroupBase):
    """Abstract family base; each concrete pins ``READER_CLASS``."""
