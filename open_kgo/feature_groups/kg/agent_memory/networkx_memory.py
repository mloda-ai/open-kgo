"""NetworkX-backed agent memory store.

In-process MultiDiGraph keyed by (node, valid_at, invalid_at). Supports
``retrieval_mode=lexical`` (string-match against node labels) without an LLM
or embedding service. Demonstrates the agent_memory contract shape.

PROTOTYPE NOTE: ``retrieval_mode`` is narrowed to ``lexical`` via
``SUPPORTED_VALUES``; ``vector`` / ``hybrid`` / ``graph`` are rejected at
``is_valid_credentials`` time rather than passing validation and raising
``NotImplementedError`` at load. Bi-temporal filtering uses simple
lexicographic comparison on ISO timestamps.

Memory data is loaded from a JSON file pointed to by the ``locator``
credential slot (shape: ``{user_id: {"nodes": [...], "edges": [...]}}``).
A user_id absent from the loaded fixture raises ``UnknownMemoryScopeError``
at ``connect()`` time; an unreadable / malformed locator raises
``FixtureLoadError``. Both surface the gap as a typed credential-shape
error rather than silently emptying the result set or leaking raw IO
exceptions. Fixture loads are mtime-cached via
``kg.fixtures.load_json_fixture``, so repeated ``load_data`` calls do
not re-read disk.
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


class NetworkxMemoryReader(AgentMemoryReader):
    CONNECTOR_ID: ClassVar[str] = "networkx_memory"
    # Narrowed from the family-level OR-group ({memory_scope_user_id,
    # memory_scope_agent_id, ..._session_id, ..._run_id, ..._group_ids}) to
    # ``("memory_scope_user_id",)`` only: the JSON fixture is keyed by
    # user_id, so the other aliases would either silently no-op or surface
    # a misleading "user_id not provisioned" error at connect time. Future
    # concrete readers (Mem0, Zep+Graphiti, Letta) honor different scope
    # keys; this concrete pins the one its substrate supports. The
    # ``MEMORY_SCOPE_KEYS`` constant in ``agent_memory/base.py`` remains the
    # canonical scope list for those future readers.
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("locator",),
        ("memory_scope_user_id",),
    )
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "pagination_style": frozenset({"none"}),
        "retrieval_mode": frozenset({"lexical"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> nx.MultiDiGraph:
        locator = str(slot["locator"])
        store = load_json_fixture(cls.CONNECTOR_ID, locator)
        # REQUIRED_KEYS now requires memory_scope_user_id specifically (this
        # concrete narrows the family OR-group), so the slot subscript is
        # safe — the validator already enforced presence + truthiness.
        user_id = str(slot["memory_scope_user_id"])
        if user_id not in store:
            raise UnknownMemoryScopeError(cls.CONNECTOR_ID, user_id)
        user_data = validate_user_data(cls.CONNECTOR_ID, locator, user_id, store[user_id])
        return build_memory_graph(user_data)

    @classmethod
    def build_query(cls, features: FeatureSet) -> str:
        feature = next(iter(features.features))
        text = feature.options.get("query_text")
        if not isinstance(text, str):
            raise ValueError(f"{cls.CONNECTOR_ID}: 'query_text' is required.")
        return text

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        graph = connection
        query_text = cls.build_query(features).lower()
        valid_range = ctx.slot.get("valid_at_range") or ()

        rows: list[dict[str, Any]] = []
        for node_id, attrs in graph.nodes(data=True):
            label = str(attrs.get("label", "")).lower()
            if query_text and query_text not in label:
                continue
            if valid_range and len(valid_range) == 2:
                start, end = valid_range[0], valid_range[1]
                node_valid = attrs.get("valid_at")
                if node_valid is not None and not (start <= node_valid <= end):
                    continue
            rows.append({"id": node_id, **attrs})
            if len(rows) >= ctx.result_limit:
                break
        return rows


class NetworkxMemoryFeatureGroup(AgentMemoryFeatureGroup):
    READER_CLASS: ClassVar[type[NetworkxMemoryReader]] = NetworkxMemoryReader
