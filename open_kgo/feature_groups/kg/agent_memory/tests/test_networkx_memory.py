"""Concrete tests for NetworkxMemoryReader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mloda.user import Feature, Options

import pytest

from mloda.provider import HashableDict

from open_kgo.feature_groups.kg.agent_memory.networkx_memory import NetworkxMemoryReader
from open_kgo.feature_groups.kg.agent_memory.tests.kg_agent_memory_contract import (
    AgentMemoryContractTestBase,
)
from open_kgo.feature_groups.kg.errors import (
    FixtureLoadError,
    InvalidCredentialShape,
    UnknownMemoryScopeError,
)


_FIXTURE = Path(__file__).parent / "fixtures" / "memories.json"


class TestNetworkxMemoryReader(AgentMemoryContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[NetworkxMemoryReader]:
        return NetworkxMemoryReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return {
            "networkx_memory": {
                "locator": str(_FIXTURE),
                "memory_scope_user_id": "user_42",
                "retrieval_mode": "lexical",
                "pagination_style": "none",
                "result_limit": 100,
                "threshold": 0.0,
                "mmr_lambda": 0.5,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        return {
            "networkx_memory": {
                "locator": str(_FIXTURE),
                "memory_scope_user_id": "user_42",
                "retrieval_mode": "evil",
            }
        }

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "networkx_memory__search",
            options=Options(context={"query_text": "coffee"}),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "label" in result[0]

    def test_lexical_search_finds_two_coffee_memories(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("networkx_memory", self.valid_credentials()["networkx_memory"], feat)
        labels = [r["label"] for r in rows]
        assert any("coffee" in lbl.lower() for lbl in labels)

    @pytest.mark.parametrize("mode", ["vector", "hybrid", "graph"])
    def test_unsupported_retrieval_modes_rejected_at_validate_time(self, mode: str) -> None:
        slot = dict(self.valid_credentials()["networkx_memory"])
        slot["retrieval_mode"] = mode
        creds = HashableDict({"networkx_memory": slot})
        assert NetworkxMemoryReader.is_valid_credentials(creds) is False
        with pytest.raises(InvalidCredentialShape):
            NetworkxMemoryReader._validate_shape(slot)

    def test_unknown_user_id_raises_typed_error(self) -> None:
        """A ``memory_scope_user_id`` absent from the fixture raises ``UnknownMemoryScopeError``.

        Asserted at ``connect()`` time only; ``is_valid_credentials`` cannot
        enforce store-residency by design (it is matcher-safe and swallows
        ``InvalidCredentialShape``). The typed leaf parallels
        ``UnknownTenantError`` in saas_authz so callers can distinguish
        "scope key missing" (REQUIRED_KEYS) from "scope value not provisioned".
        """
        slot = dict(self.valid_credentials()["networkx_memory"])
        slot["memory_scope_user_id"] = "user_does_not_exist"
        creds = HashableDict({"networkx_memory": slot})
        with pytest.raises(UnknownMemoryScopeError):
            NetworkxMemoryReader.connect(creds)

    def test_remote_locator_rejected(self) -> None:
        """A ``http://``/``https://`` locator must be rejected at connect time.

        Mirrors the rdflib reader's URI-scheme guard from PR #7 so a
        copy-pasted URL surfaces as a typed ``FixtureLoadError`` instead of
        a confusing ``FileNotFoundError`` against the URL-as-relative-path.
        """
        slot = dict(self.valid_credentials()["networkx_memory"])
        slot["locator"] = "http://example.invalid/memories.json"
        creds = HashableDict({"networkx_memory": slot})
        with pytest.raises(FixtureLoadError):
            NetworkxMemoryReader.connect(creds)

    def test_missing_locator_file_is_typed(self) -> None:
        """A ``locator`` pointing at a non-existent file raises ``FixtureLoadError``."""
        slot = dict(self.valid_credentials()["networkx_memory"])
        slot["locator"] = "/nonexistent/path/to/memories.json"
        creds = HashableDict({"networkx_memory": slot})
        with pytest.raises(FixtureLoadError):
            NetworkxMemoryReader.connect(creds)

    def test_malformed_json_locator_is_typed(self, tmp_path: Path) -> None:
        """Invalid JSON syntax in the locator file raises ``FixtureLoadError``."""
        bad = tmp_path / "bad.json"
        bad.write_text("this is not json {", encoding="utf-8")
        slot = dict(self.valid_credentials()["networkx_memory"])
        slot["locator"] = str(bad)
        creds = HashableDict({"networkx_memory": slot})
        with pytest.raises(FixtureLoadError):
            NetworkxMemoryReader.connect(creds)

    def test_non_object_top_level_json_is_typed(self, tmp_path: Path) -> None:
        """A locator JSON whose top level is not an object raises ``FixtureLoadError``.

        Guards against the otherwise-silent ``user_id not in store`` path
        where ``store`` is a list (sequence ``__contains__`` succeeds for
        member equality) or a scalar (raises ``TypeError`` deep inside
        ``_build_memory_graph``).
        """
        bad = tmp_path / "list.json"
        bad.write_text("[]", encoding="utf-8")
        slot = dict(self.valid_credentials()["networkx_memory"])
        slot["locator"] = str(bad)
        creds = HashableDict({"networkx_memory": slot})
        with pytest.raises(FixtureLoadError):
            NetworkxMemoryReader.connect(creds)
