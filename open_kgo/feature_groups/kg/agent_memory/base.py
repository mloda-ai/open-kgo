"""Family base for agent memory / GraphRAG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import (
    KgConnectorFeatureGroupBase,
    QueryReader,
    compose_property_mapping,
)
from open_kgo.feature_groups.kg.mixins import PaginationMixin


_RETRIEVAL_MODES: dict[str, str] = {
    "lexical": "BM25-style lexical search.",
    "vector": "Embedding similarity search.",
    "hybrid": "Combined lexical + vector with mmr_lambda blend.",
    "graph": "Pure graph-walk retrieval.",
}


# Single source of truth for the memory-scope key family. Drives both
# ``MEMORY_SCOPE_KEYS`` and the property-mapping entries below, so renaming
# or extending the scope happens in exactly one place.
#
# Note on consumers: the existing ``NetworkxMemoryReader`` narrows
# ``REQUIRED_KEYS`` to ``("memory_scope_user_id",)`` only — its JSON fixture
# is keyed by user_id and the other scope aliases would silently no-op. The
# canonical OR-group ``MEMORY_SCOPE_KEYS`` is reserved for future concretes
# (Mem0, Letta, Zep+Graphiti) whose backends honor the full scope; keeping
# the constant exported documents that contract for the family.
#
# Uniformity assumption: every scope key gets ``context=True`` and
# ``strict_validation=False``. The first scope key that needs a different
# pair (e.g. ``strict_validation=True`` for an enum scope) will force this
# tuple shape to grow — at that point switch to a per-key spec dict
# constant rather than threading more positional fields through.
_MEMORY_SCOPE_SPECS: tuple[tuple[str, str, None | tuple[()]], ...] = (
    ("memory_scope_user_id", "User identifier scope (Mem0 user_id, Letta user_id).", None),
    ("memory_scope_agent_id", "Agent identifier scope.", None),
    ("memory_scope_session_id", "Session identifier scope (e.g. LangGraph thread_id).", None),
    ("memory_scope_run_id", "Run identifier scope.", None),
    ("memory_scope_group_ids", "Graphiti-style group_ids (list).", ()),
)

MEMORY_SCOPE_KEYS: tuple[str, ...] = tuple(name for name, _, _ in _MEMORY_SCOPE_SPECS)

_MEMORY_SCOPE_PROPERTY_MAPPING: dict[str, Any] = {
    name: {
        "explanation": explanation,
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: default,
    }
    for name, explanation, default in _MEMORY_SCOPE_SPECS
}


class AgentMemoryReader(PaginationMixin, QueryReader):
    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = compose_property_mapping(
        QueryReader.PROPERTY_MAPPING,
        PaginationMixin.PROPERTY_MAPPING_DELTA,
        _MEMORY_SCOPE_PROPERTY_MAPPING,
        {
            "reference_time": {
                "explanation": "Bi-temporal reference time (ISO 8601).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: None,
            },
            "valid_at_range": {
                "explanation": "[start, end] for valid_at filter.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: (),
            },
            "invalid_at_range": {
                "explanation": "[start, end] for invalid_at filter.",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: (),
            },
            "retrieval_mode": {
                "explanation": "Retrieval strategy used to score candidate memories.",
                "allowed_values": _RETRIEVAL_MODES,
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                DefaultOptionKeys.default: "lexical",
            },
            "mmr_lambda": {
                "explanation": "MMR lambda for hybrid retrieval blend (0.0-1.0).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: 0.5,
            },
            "threshold": {
                "explanation": "Similarity threshold (0.0-1.0).",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.default: 0.0,
            },
        },
        context="AgentMemoryReader",
    )


class AgentMemoryFeatureGroup(KgConnectorFeatureGroupBase):
    READER_CLASS = None
