"""Per-family contract base for agent memory connectors.

Behavior coverage comes from the universal end-to-end test (which validates
that REQUIRED_KEYS for ``memory_scope_*`` is enforced via run_query) plus
the per-concrete ``test_lexical_search_finds_two_coffee_memories``.
"""

from __future__ import annotations

from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class AgentMemoryContractTestBase(KgConnectorContractBase):
    pass
