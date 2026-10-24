"""NetworkX-backed graph-walk agent memory store.

Second concrete in the ``agent_memory`` family alongside ``NetworkxMemoryReader``
(lexical search). This reader implements ``retrieval_mode=graph`` — the pure
graph-walk retrieval branch the lexical concrete narrows away — by treating the
``query_text`` as a seed node id and returning the memories reachable from it
(BFS over the directed memory graph). It is the family's proof that the
``retrieval_mode`` enum's ``graph`` value is real, and it adds no new dependency
(NetworkX is already a family dependency).

Memory data is loaded from a JSON file at ``locator`` (shape:
``{user_id: {"nodes": [...], "edges": [...]}}``), the same shape the lexical
concrete consumes. ``retrieval_mode`` is narrowed to ``graph`` and
``pagination_style`` to ``none`` via ``SUPPORTED_VALUES``; ``retrieval_mode``
is additionally listed in ``REQUIRED_KEYS`` because the family default is
``lexical`` (a value this reader does not honor) and ``SUPPORTED_VALUES``
only checks keys present in the slot, so an omitted ``retrieval_mode`` would
otherwise validate and silently run graph retrieval under a defaulted
``lexical`` label.

The family-level bi-temporal keys (``valid_at_range`` / ``invalid_at_range`` /
``reference_time``) are ACCEPTED but NOT APPLIED by this graph-walk concrete:
the BFS returns reachable memories regardless of temporal validity. The lexical
sibling (``NetworkxMemoryReader``) honors ``valid_at_range``; this one does not
yet. This mirrors how the family base documents the ``memory_scope_*`` aliases
as accepted-but-no-op until a concrete owns the corresponding store.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import networkx as nx

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.agent_memory.base import (
    AgentMemoryFeatureGroup,
    AgentMemoryReader,
)
from open_kgo.feature_groups.kg.agent_memory.shared import build_memory_graph, validate_user_data
from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.errors import UnknownMemoryScopeError
from open_kgo.feature_groups.kg.fixtures import load_json_fixture
from open_kgo.feature_groups.kg.traversal import bfs_collect_ids


class GraphWalkMemoryReader(AgentMemoryReader):
    CONNECTOR_ID: ClassVar[str] = "graph_walk_memory"
    # retrieval_mode is REQUIRED (not just narrowed): the family default is
    # "lexical", which this reader does not honor, and SUPPORTED_VALUES only
    # validates keys present in the slot. An omitted retrieval_mode would
    # otherwise pass is_valid_credentials and silently run graph retrieval
    # under a defaulted "lexical" label.
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("locator",),
        ("memory_scope_user_id",),
        ("retrieval_mode",),
    )
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "pagination_style": frozenset({"none"}),
        "retrieval_mode": frozenset({"graph"}),
    }
    # Honest surface (option 3, see base.py): the BFS walk ignores temporal
    # validity, so ``valid_at_range`` is accepted but not applied here (the
    # lexical sibling reads it; the family base waives the rest).
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"valid_at_range"})

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> nx.MultiDiGraph:
        locator = str(slot["locator"])
        store = load_json_fixture(cls.CONNECTOR_ID, locator)
        user_id = str(slot["memory_scope_user_id"])
        if user_id not in store:
            raise UnknownMemoryScopeError(cls.CONNECTOR_ID, user_id)
        # require_graph_integrity: this concrete's rows come straight from the
        # BFS walk, so dangling edge endpoints must be rejected at connect time.
        user_data = validate_user_data(cls.CONNECTOR_ID, locator, user_id, store[user_id], require_graph_integrity=True)
        return build_memory_graph(user_data)

    @classmethod
    def build_query(cls, features: FeatureSet) -> str:
        feature = next(iter(features.features))
        text = feature.options.get("query_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{cls.CONNECTOR_ID}: 'query_text' (the seed node id) is required.")
        # Return the stripped seed: validation already requires non-whitespace
        # content, and an unstripped "  m1 " would validate here yet match no
        # node in the walk, silently returning [].
        return text.strip()

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        graph = connection
        seed = cls.build_query(features)

        # BFS over reachable memories (depth-unbounded); short-circuits at
        # result_limit so the cap bounds the walk rather than slicing a
        # fully-expanded reachable set. A seed absent from the graph yields
        # an empty walk (the contains gate skips it without expanding).
        collected = bfs_collect_ids(
            lambda node_id: node_id in graph,
            graph.successors,
            seed,
            max_nodes=ctx.result_limit,
        )
        return [{"id": node_id, **graph.nodes[node_id]} for node_id in collected]


class GraphWalkMemoryFeatureGroup(AgentMemoryFeatureGroup):
    READER_CLASS: ClassVar[type[GraphWalkMemoryReader]] = GraphWalkMemoryReader
