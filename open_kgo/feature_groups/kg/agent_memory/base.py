"""Family base for agent memory / GraphRAG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase, QueryReader
from open_kgo.feature_groups.kg.mixins import PaginationMixin
from open_kgo.feature_groups.kg.spec import property_spec


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
    name: property_spec(explanation, default=default) for name, explanation, default in _MEMORY_SCOPE_SPECS
}

_FAMILY_PROPERTIES: dict[str, Any] = {
    **_MEMORY_SCOPE_PROPERTY_MAPPING,
    "reference_time": property_spec(
        "Bi-temporal reference time (ISO 8601).",
    ),
    "valid_at_range": property_spec(
        "[start, end] for valid_at filter.",
        default=(),
    ),
    "invalid_at_range": property_spec(
        "[start, end] for invalid_at filter.",
        default=(),
    ),
    "retrieval_mode": property_spec(
        "Retrieval strategy used to score candidate memories.",
        strict=True,
        allowed_values=_RETRIEVAL_MODES,
        default="lexical",
    ),
    "mmr_lambda": property_spec(
        "MMR lambda for hybrid retrieval blend (0.0-1.0).",
        default=0.5,
    ),
    "threshold": property_spec(
        "Similarity threshold (0.0-1.0).",
        default=0.0,
    ),
}


class AgentMemoryReader(PaginationMixin, QueryReader, family_properties=_FAMILY_PROPERTIES):
    # Honest surface (option 3, see base.py): forward-compat GraphRAG menu the
    # in-process concretes don't read (alternate scope aliases, bi-temporal and
    # scoring knobs, pagination), reserved for future backends (Mem0, Letta,
    # Zep). ``valid_at_range`` is waived per-concrete on GraphWalkMemoryReader,
    # since the lexical sibling reads it.
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "page_size",
            "memory_scope_agent_id",
            "memory_scope_session_id",
            "memory_scope_run_id",
            "memory_scope_group_ids",
            "reference_time",
            "invalid_at_range",
            "mmr_lambda",
            "threshold",
        }
    )


class AgentMemoryFeatureGroup(KgConnectorFeatureGroupBase):
    """Abstract family base; each concrete pins ``READER_CLASS``."""
