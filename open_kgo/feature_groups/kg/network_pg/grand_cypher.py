"""GrandCypher-backed embedded Cypher reader.

Second concrete in the ``network_pg`` family alongside ``KuzuCypherReader``:
runs openCypher queries against an in-memory NetworkX graph via the pure-Python
``grand-cypher`` library. Kuzu is an embedded C++ engine; GrandCypher is a
Python Cypher interpreter over NetworkX, so the two share the family's
Cypher-shaped contract on different substrates. Like Kuzu, this backend is
single-process and embedded, so ``read_consistency`` / ``transaction_mode``
are no-ops (waived, not narrowed: real network_pg backends will honor them).

The graph is loaded from a GML / GraphML file at ``locator`` (NetworkX relabels
GML nodes to their ``label``, keeping other attributes such as ``name``). The
Cypher text comes from the Feature's options context under ``query_text``.
GrandCypher returns a columnar ``{column: [values...]}`` dict, which we pivot
into the family's row-oriented ``list[dict]`` shape, capped at ``result_limit``.

Label mapping: grand-cypher resolves ``[:TYPE]`` and ``(n:Label)`` patterns
against a ``__labels__`` set attribute that ``nx.read_gml`` / ``nx.read_graphml``
never produce, so without an explicit mapping every typed pattern silently
returns zero rows. After loading, this reader therefore maps (see
``_apply_grand_labels``):

- the edge ``label`` attribute (present in GML fixtures such as ``org.gml``)
  to ``__labels__`` on each edge, so ``[:MANAGES]`` matches; and
- the node ``type`` attribute to ``__labels__`` on each node, so ``(p:Person)``
  matches when the source file carries one.

Node-label limitation: the committed ``org.gml`` fixture carries NO node
``type`` attribute (its GML node ``label`` is consumed by ``nx.read_gml`` as
the node id), so node-label filters match nothing against the committed
fixtures. Only a per-node ``type`` attribute is honored as the node label
source; other attribute names (``labels``, ``category``, ...) are NOT
consulted.

Engine quirks (grand-cypher 1.0.x, pinned here so callers are not surprised):

- A bare ``RETURN n`` yields a column keyed by a ``lark.Token`` (a ``str``
  subclass), with whole node-attribute dicts as values (including the
  injected ``__labels__`` set).
- Aggregations return nested per-alias dicts rather than flat scalars, e.g.
  ``RETURN COUNT(n)`` produces ``{'COUNT(n)': [{'e': 3, 'a': 3}]}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Mapping

import networkx as nx
from grandcypher import GrandCypher

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.errors import FixtureLoadError
from open_kgo.feature_groups.kg.fixtures import _rejected_scheme
from open_kgo.feature_groups.kg.network_pg.base import (
    NetworkPropertyGraphFeatureGroup,
    NetworkPropertyGraphReader,
)


_LOADERS: dict[str, Any] = {
    ".gml": nx.read_gml,
    ".graphml": nx.read_graphml,
}


def _apply_grand_labels(graph: nx.Graph) -> None:
    """Map source attributes onto the ``__labels__`` sets grand-cypher matches against.

    grand-cypher resolves ``[:TYPE]`` / ``(n:Label)`` patterns exclusively
    against a ``__labels__`` set attribute that NetworkX's file loaders never
    produce; without this mapping every typed pattern silently returns zero
    rows. The mapping is explicit and closed: edge ``label`` attributes and
    node ``type`` attributes are the only sources consulted (see the module
    docstring for the node-label limitation on the committed fixtures).
    """
    for _, node_data in graph.nodes(data=True):
        node_type = node_data.get("type")
        if node_type is not None:
            node_data["__labels__"] = {str(node_type)}
    for _, _, edge_data in graph.edges(data=True):
        edge_label = edge_data.get("label")
        if edge_label is not None:
            edge_data["__labels__"] = {str(edge_label)}


class GrandCypherReader(NetworkPropertyGraphReader):
    CONNECTOR_ID: ClassVar[str] = "grand_cypher"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",),)
    # Waive read_consistency / transaction_mode: GrandCypher runs over an
    # in-memory NetworkX graph (single process, no replicas / transactions), so
    # both are no-ops here. Narrowing would lock the family contract to the
    # embedded backends' single honored value and force future networked
    # concretes (Neo4j, Memgraph) to widen — the same disposition as Kuzu.
    _WAIVED_ENUM_KEYS: ClassVar[frozenset[str]] = frozenset({"read_consistency", "transaction_mode"})

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> nx.Graph:
        """Load the NetworkX graph from ``slot['locator']`` (GML / GraphML).

        Format is selected from the file extension. Remote locators are
        rejected for parity with the other file-backed readers (NetworkX's
        loaders only open local paths). Load failures raise the typed
        ``FixtureLoadError`` for parity with the kuzu sibling: an empty
        locator, a missing file, and unparseable GML/GraphML all surface
        loudly (there is deliberately no empty-graph fallback, which would
        silently answer every query with zero rows). A fresh graph is parsed
        per call, matching ``NetworkxEmbeddedReader`` (NetworkX parsing of
        the small fixtures is cheap; no shared cache is introduced for this
        family). After parsing, ``_apply_grand_labels`` maps source
        attributes onto ``__labels__`` (see module docstring).
        """
        locator = slot.get("locator")
        if not locator:
            raise FixtureLoadError(
                cls.CONNECTOR_ID,
                str(locator),
                "locator is missing or empty; a local GML/GraphML file path is required.",
            )
        bad = _rejected_scheme(locator)
        if bad is not None:
            raise ValueError(
                f"{cls.CONNECTOR_ID}: locator scheme {bad!r} is not permitted; "
                f"only local file paths or file:// URLs are allowed."
            )
        loader = _LOADERS.get(Path(str(locator)).suffix.lower())
        if loader is None:
            raise ValueError(
                f"{cls.CONNECTOR_ID}: unsupported graph file extension for locator {locator!r}; "
                f"expected one of {sorted(_LOADERS)}."
            )
        try:
            graph: nx.Graph = loader(str(locator))
        except OSError as exc:
            raise FixtureLoadError(cls.CONNECTOR_ID, str(locator), f"cannot open locator file: {exc}") from exc
        except (nx.NetworkXError, SyntaxError, ValueError, UnicodeDecodeError) as exc:
            # nx.read_gml raises NetworkXError on bad GML; nx.read_graphml
            # raises xml.etree.ElementTree.ParseError (a SyntaxError subclass)
            # on bad XML; ValueError / UnicodeDecodeError cover bad bytes.
            raise FixtureLoadError(
                cls.CONNECTOR_ID, str(locator), f"locator is not parseable as GML/GraphML: {exc}"
            ) from exc
        _apply_grand_labels(graph)
        return graph

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        graph = connection

        query_text = cls.build_query(features)
        # GrandCypher's constructor ``limit`` is best-effort only: an in-query
        # ``LIMIT`` overrides it (verified on grand-cypher 1.0.x), so the
        # trailing ``rows[: ctx.result_limit]`` slice below is the actual
        # enforcement of the result_limit contract.
        columns: dict[str, list[Any]] = GrandCypher(graph, limit=ctx.result_limit).run(query_text)

        if not columns:
            return []
        # Columns are normally equal-length; the min() is defensive against
        # engine shapes where they are not (e.g. the aggregation quirk in the
        # module docstring) so the pivot below never IndexErrors.
        row_count = min(len(values) for values in columns.values())
        rows = [{column: values[i] for column, values in columns.items()} for i in range(row_count)]
        return rows[: ctx.result_limit]


class GrandCypherFeatureGroup(NetworkPropertyGraphFeatureGroup):
    READER_CLASS: ClassVar[type[GrandCypherReader]] = GrandCypherReader
