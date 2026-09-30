"""Concrete tests for GraphWalkMemoryReader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.agent_memory.graph_walk_memory import GraphWalkMemoryReader
from open_kgo.feature_groups.kg.agent_memory.tests.kg_agent_memory_contract import (
    AgentMemoryContractTestBase,
)
from open_kgo.feature_groups.kg.errors import (
    FixtureLoadError,
    InvalidCredentialShape,
    MissingRequiredKeysError,
    UnknownMemoryScopeError,
)
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials


_FIXTURE = Path(__file__).parent / "fixtures" / "graph_memories.json"


class TestGraphWalkMemoryReader(AgentMemoryContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[GraphWalkMemoryReader]:
        return GraphWalkMemoryReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return make_valid_credentials(
            cls.connector_reader_class(),
            locator=str(_FIXTURE),
            memory_scope_user_id="user_42",
            retrieval_mode="graph",
            result_limit=100,
        )

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        return {
            "graph_walk_memory": {
                "locator": str(_FIXTURE),
                "memory_scope_user_id": "user_42",
                "retrieval_mode": "evil",
            }
        }

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "graph_walk_memory__from_seed",
            options=Options(context={"query_text": "m1"}),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "label" in result[0]

    def test_graph_walk_reaches_transitive_memories(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("graph_walk_memory", self.valid_credentials()["graph_walk_memory"], feat)
        assert {r["id"] for r in rows} == {"m1", "m2", "m3"}

    def test_disconnected_seed_returns_only_itself(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature("graph_walk_memory__m4", options=Options(context={"query_text": "m4"}))
        rows = run_query("graph_walk_memory", self.valid_credentials()["graph_walk_memory"], feat)
        assert {r["id"] for r in rows} == {"m4"}

    @pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
    def test_unsupported_retrieval_modes_rejected_at_validate_time(self, mode: str) -> None:
        slot = dict(self.valid_credentials()["graph_walk_memory"])
        slot["retrieval_mode"] = mode
        creds = {"graph_walk_memory": slot}
        assert GraphWalkMemoryReader.is_valid_credentials(creds) is False
        with pytest.raises(InvalidCredentialShape):
            GraphWalkMemoryReader._validate_shape(slot)

    def test_omitted_retrieval_mode_rejected_at_validate_time(self) -> None:
        """A full valid slot minus ``retrieval_mode`` fails ``is_valid_credentials``.

        ``SUPPORTED_VALUES`` only validates keys present in the slot, and the
        family default is ``lexical`` (a mode this reader does not honor), so
        the omission case must be closed by ``REQUIRED_KEYS``: without it, the
        slot would validate and silently run graph retrieval under a defaulted
        ``lexical`` label.
        """
        slot = dict(self.valid_credentials()["graph_walk_memory"])
        del slot["retrieval_mode"]
        creds = {"graph_walk_memory": slot}
        assert GraphWalkMemoryReader.is_valid_credentials(creds) is False
        with pytest.raises(MissingRequiredKeysError):
            GraphWalkMemoryReader._validate_shape(slot)

    def test_unknown_user_id_raises_typed_error(self) -> None:
        slot = dict(self.valid_credentials()["graph_walk_memory"])
        slot["memory_scope_user_id"] = "user_does_not_exist"
        creds = {"graph_walk_memory": slot}
        with pytest.raises(UnknownMemoryScopeError):
            GraphWalkMemoryReader.connect(creds)

    def test_reconvergent_node_emitted_exactly_once(self) -> None:
        """The diamond node ``m3`` (reachable via m1->m2->m3 AND m1->m3) is emitted once.

        Asserts against the raw ``load_data`` list (not a set) so a regression
        that double-emits on BFS re-convergence is caught — a set-based
        assertion would silently mask it.
        """
        creds = self.valid_credentials()
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = GraphWalkMemoryReader.load_data(creds, fs)
        ids = [r["id"] for r in rows]
        assert ids.count("m3") == 1
        assert len(ids) == len(set(ids))
        assert set(ids) == {"m1", "m2", "m3"}

    def test_cyclic_graph_bfs_terminates_and_dedups(self, tmp_path: Path) -> None:
        """A cycle (m1->m2->m1) does not loop forever; each node is emitted exactly once.

        The ``seen`` set both terminates the BFS and dedups, so a regression that
        re-enqueues an already-visited node would either spin forever or
        double-emit; pinning the exact ``[m1, m2]`` list catches both.
        """
        fixture = tmp_path / "cycle.json"
        fixture.write_text(
            json.dumps(
                {
                    "user_42": {
                        "nodes": [{"id": "m1", "label": "a"}, {"id": "m2", "label": "b"}],
                        "edges": [{"src": "m1", "tgt": "m2"}, {"src": "m2", "tgt": "m1"}],
                    }
                }
            ),
            encoding="utf-8",
        )
        slot = dict(self.valid_credentials()["graph_walk_memory"])
        slot["locator"] = str(fixture)
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = GraphWalkMemoryReader.load_data({"graph_walk_memory": slot}, fs)
        ids = [r["id"] for r in rows]
        assert ids == ["m1", "m2"]
        assert len(ids) == len(set(ids))

    def test_result_limit_caps_bfs_emission(self) -> None:
        """result_limit bounds the BFS: a 3-node reachable set capped at 2 emits exactly the first two ids.

        BFS from ``m1`` visits ``m1`` then its successors in insertion order
        (``m2`` before ``m3``), so the cap pins ``[m1, m2]`` — a regression that
        sliced a fully-expanded set instead of short-circuiting would be caught.
        """
        slot = dict(self.valid_credentials()["graph_walk_memory"])
        slot["result_limit"] = 2
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = GraphWalkMemoryReader.load_data({"graph_walk_memory": slot}, fs)
        assert [r["id"] for r in rows] == ["m1", "m2"]

    def test_malformed_user_entry_missing_edges_raises_typed_error(self, tmp_path: Path) -> None:
        """A user entry missing the required ``edges`` key raises ``FixtureLoadError``.

        Exercises the ``validate_user_data`` error branch (shared via
        ``agent_memory/shared.py``) that the linear-fixture tests never reached.
        """
        fixture = tmp_path / "broken.json"
        fixture.write_text(
            json.dumps({"user_42": {"nodes": [{"id": "m1", "label": "x"}]}}),
            encoding="utf-8",
        )
        slot = dict(self.valid_credentials()["graph_walk_memory"])
        slot["locator"] = str(fixture)
        with pytest.raises(FixtureLoadError):
            GraphWalkMemoryReader.connect({"graph_walk_memory": slot})

    def _connect_with_user_data(self, tmp_path: Path, user_data: dict[str, Any]) -> None:
        """Write ``{"user_42": user_data}`` to a fixture and run ``connect`` against it."""
        fixture = tmp_path / "shaped.json"
        fixture.write_text(json.dumps({"user_42": user_data}), encoding="utf-8")
        slot = dict(self.valid_credentials()["graph_walk_memory"])
        slot["locator"] = str(fixture)
        GraphWalkMemoryReader.connect({"graph_walk_memory": slot})

    def test_dangling_edge_target_rejected(self, tmp_path: Path) -> None:
        """An edge whose target references no declared node raises ``FixtureLoadError``.

        ``MultiDiGraph.add_edge`` auto-creates missing endpoints, so without
        the guard a dangling target would materialize an attribute-less node
        whose BFS row is missing ``label`` (a silent family row-shape
        violation).
        """
        with pytest.raises(FixtureLoadError):
            self._connect_with_user_data(
                tmp_path,
                {
                    "nodes": [{"id": "m1", "label": "a"}],
                    "edges": [{"src": "m1", "tgt": "ghost"}],
                },
            )

    def test_node_missing_id_rejected(self, tmp_path: Path) -> None:
        """A node entry without an ``id`` key raises ``FixtureLoadError`` (not a raw KeyError)."""
        with pytest.raises(FixtureLoadError):
            self._connect_with_user_data(
                tmp_path,
                {
                    "nodes": [{"label": "no id here"}],
                    "edges": [],
                },
            )

    def test_edge_missing_src_rejected(self, tmp_path: Path) -> None:
        """An edge entry without a ``src`` key raises ``FixtureLoadError`` (not a raw KeyError)."""
        with pytest.raises(FixtureLoadError):
            self._connect_with_user_data(
                tmp_path,
                {
                    "nodes": [{"id": "m1", "label": "a"}],
                    "edges": [{"tgt": "m1"}],
                },
            )

    def test_nonexistent_seed_returns_empty_list(self) -> None:
        """A seed id absent from the user's graph returns ``[]`` (pinned, not an error)."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature("graph_walk_memory__missing_seed", options=Options(context={"query_text": "not_a_node"}))
        rows = run_query("graph_walk_memory", self.valid_credentials()["graph_walk_memory"], feat)
        assert rows == []

    @pytest.mark.parametrize("context", [{}, {"query_text": ""}, {"query_text": "   "}])
    def test_build_query_missing_or_blank_query_text_raises(self, context: dict[str, Any]) -> None:
        """``build_query`` raises ``ValueError`` when ``query_text`` is absent, empty, or whitespace."""
        feat = Feature("graph_walk_memory__bad_seed", options=Options(context=context))
        fs = FeatureSet()
        fs.add(feat)
        with pytest.raises(ValueError):
            GraphWalkMemoryReader.build_query(fs)

    def test_build_query_strips_padded_seed(self) -> None:
        """``build_query`` returns the stripped seed, so a padded ``"  m1 "`` reaches the walk.

        Pins the strip-before-return behavior: an unstripped return value
        would validate but match no node, silently yielding ``[]``.
        """
        feat = Feature("graph_walk_memory__padded_seed", options=Options(context={"query_text": "  m1 "}))
        fs = FeatureSet()
        fs.add(feat)
        assert GraphWalkMemoryReader.build_query(fs) == "m1"
        rows = GraphWalkMemoryReader.load_data(self.valid_credentials(), fs)
        assert {r["id"] for r in rows} == {"m1", "m2", "m3"}
