"""python-igraph-backed embedded graph reader.

Second concrete in the ``embedded`` family alongside ``NetworkxEmbeddedReader``:
loads a graph from a fixture file (.gml / .graphml) at ``locator`` via
python-igraph and runs the same family operations (``nodes`` / ``edges`` /
``neighbors``). Demonstrates that the embedded family base generalises across a
second in-memory graph library; no new family surface is unlocked (igraph,
like NetworkX, is an embedded in-process backend), so this is backend variety,
not a new contract branch.

The family-advertised ``edgelist`` format is deliberately NOT supported here
(see the ``SUPPORTED_VALUES`` narrowing note below): igraph's ``Read_Edgelist``
yields integer-indexed vertices with no name/label, whereas the NetworkX
sibling's ``nx.read_edgelist`` yields string node tokens, so the same file
would produce different row identities across the two embedded backends.

The vertex identity exposed in rows is the first present of ``name`` / ``label``
/ ``id``, falling back to the integer vertex index. The committed GML fixtures
carry a ``label`` per node, so rows key on the label (e.g. ``"alice"``), which
mirrors what ``nx.read_gml`` produces (it relabels nodes to their ``label``).
A ``start_node`` absent from the graph raises the family-typed
``UnknownStartNodeError`` (identical behavior to the NetworkX sibling).
"""

from __future__ import annotations

from itertools import islice
from typing import Any, ClassVar, Mapping

import igraph

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.embedded.base import (
    EmbeddedGraphFeatureGroup,
    EmbeddedGraphReader,
    UnknownStartNodeError,
)
from open_kgo.feature_groups.kg.fixtures import _rejected_scheme


_LOADERS: dict[str, Any] = {
    "gml": igraph.Graph.Read_GML,
    "graphml": igraph.Graph.Read_GraphML,
}


def _vertex_key(vertex: igraph.Vertex) -> Any:
    """Return a stable identifier for ``vertex``: first present of name/label/id, else index.

    igraph vertices are integer-indexed; GML/GraphML readers surface the
    source identifiers as the ``name`` / ``label`` / ``id`` attributes. The
    NetworkX reader keys rows on the node id (its GML loader relabels nodes
    to their ``label``), so this preference order keeps the two embedded
    backends' row shapes aligned for LABEL-ONLY GML fixtures like the
    committed ones. The parity claim is scoped to that shape: on a GML file
    carrying BOTH ``name`` and ``label``, the backends diverge (igraph keys
    on ``name``, e.g. ``"Alice"``, while ``nx.read_gml`` keys on ``label``,
    e.g. ``"alice"``); and igraph surfaces the GML ``id`` fallback as a
    float (e.g. ``0.0``), not an int. The integer-index fallback only guards
    attribute-less vertices. Do not reorder the preference: the committed
    fixtures only carry ``label``, and changing the key order risks breaking
    row parity.
    """
    attrs = vertex.attributes()
    for attr in ("name", "label", "id"):
        value = attrs.get(attr)
        if value is not None:
            return value
    return vertex.index


class IGraphEmbeddedReader(EmbeddedGraphReader):
    CONNECTOR_ID: ClassVar[str] = "igraph_embedded"
    # Strict-enum dispositions:
    #   - graph_file_format: NARROWED to {gml, graphml}. The family also
    #     advertises ``edgelist``, but it is deliberately dropped here:
    #     igraph's ``Read_Edgelist`` produces integer-indexed, attribute-less
    #     vertices, so ``_vertex_key`` would return the integer index, whereas
    #     the NetworkX sibling's ``nx.read_edgelist`` yields string node tokens.
    #     The same edgelist file would therefore key rows differently across the
    #     two embedded backends, so honest narrowing (a non-empty subset of the
    #     family enum) is preferred over a parity lie. ``operation`` still
    #     mirrors the full family set (every value is dispatched in ``load_data``).
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "graph_file_format": frozenset({"gml", "graphml"}),
        "operation": frozenset({"nodes", "edges", "neighbors"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> igraph.Graph:
        locator = slot.get("locator")
        if not locator:
            return igraph.Graph()
        # Reject remote locators for parity with the other file-backed readers;
        # igraph's loaders only open local paths, so this keeps the file-only
        # contract uniform across families.
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
        return loader(str(locator))

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        graph = connection

        params = cls.build_params(features, ctx.slot)
        op = params["operation"]

        if op == "nodes":
            return [{"node": _vertex_key(v)} for v in islice(graph.vs, ctx.result_limit)]
        if op == "edges":
            return [
                {"src": _vertex_key(graph.vs[e.source]), "dst": _vertex_key(graph.vs[e.target])}
                for e in islice(graph.es, ctx.result_limit)
            ]
        if op == "neighbors":
            start = params.get("start_node")
            if start is None:
                raise ValueError(f"{cls.CONNECTOR_ID}: operation=neighbors requires 'start_node'.")
            start_index = next((v.index for v in graph.vs if _vertex_key(v) == start), None)
            if start_index is None:
                # Family-typed error (instead of the prior silent ``[]``) so
                # both embedded backends fail identically on an unknown
                # start_node.
                raise UnknownStartNodeError(cls.CONNECTOR_ID, start)
            # ``mode="out"`` matches NetworkxEmbeddedReader, which uses
            # ``DiGraph.neighbors`` (successors). On an undirected graph igraph
            # ignores ``mode`` and returns all neighbors, so the two embedded
            # backends agree on both directed and undirected inputs.
            neighbors = islice(graph.neighbors(start_index, mode="out"), ctx.result_limit)
            return [{"node": _vertex_key(graph.vs[n])} for n in neighbors]
        raise ValueError(f"{cls.CONNECTOR_ID}: unsupported operation={op!r}")


class IGraphEmbeddedFeatureGroup(EmbeddedGraphFeatureGroup):
    READER_CLASS: ClassVar[type[IGraphEmbeddedReader]] = IGraphEmbeddedReader
