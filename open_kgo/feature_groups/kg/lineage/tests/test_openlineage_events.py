"""Concrete tests for OpenLineageReader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.errors import FixtureLoadError
from open_kgo.feature_groups.kg.lineage.openlineage_events import (
    OpenLineageReader,
    _build_dataset_graph,
)
from open_kgo.feature_groups.kg.lineage.tests.kg_lineage_contract import (
    LineageContractTestBase,
)
from open_kgo.feature_groups.kg.tests._helpers import run_query
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials


_FIXTURE = Path(__file__).parent / "fixtures" / "openlineage_events.json"


class TestOpenLineageReader(LineageContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[OpenLineageReader]:
        return OpenLineageReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return make_valid_credentials(cls.connector_reader_class(), locator=str(_FIXTURE), result_limit=100)

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        # ``lineage_direction`` is a PARAMS key, so in the credential slot it is
        # rejected by the closed-world unknown-credential-key check. (The
        # strict-enum narrowing for it is exercised per-call by the inherited
        # ``test_strict_validation_params_enums_rejected_per_key``.)
        return {"openlineage_events": {"locator": str(_FIXTURE), "lineage_direction": "SIDEWAYS"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "openlineage_events__upstream",
            options=Options(
                context={
                    "asset_urn": "fct_orders",
                    "lineage_direction": "UPSTREAM",
                    "upstream_depth": 2,
                    "downstream_depth": 0,
                }
            ),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "name" in result[0]

    def test_upstream_walk_two_hops(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("openlineage_events", self.valid_credentials()["openlineage_events"], feat)
        names = {r["name"] for r in rows}
        assert names == {"fct_orders", "stg_orders", "raw.orders"}

    def test_result_limit_bounds_both_direction_walk(self) -> None:
        """``result_limit`` caps the BOTH walk, bounding cost rather than slicing at the end.

        From ``fct_orders`` the BOTH-direction reachable set is four datasets
        (fct_orders, stg_orders, raw.orders, report_daily); ``result_limit=2``
        must emit exactly two rows, so a regression where ``_walk`` overshoots
        the remaining budget (or load_data fails to short-circuit) is caught for
        this concrete, not only ``DbtManifestReader``.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["openlineage_events"])
        slot["result_limit"] = 2
        feat = Feature(
            "openlineage_events__both_capped",
            options=Options(
                context={
                    "asset_urn": "fct_orders",
                    "lineage_direction": "BOTH",
                    "upstream_depth": 2,
                    "downstream_depth": 1,
                }
            ),
        )
        rows = run_query("openlineage_events", slot, feat)
        assert len(rows) == 2
        assert {r["name"] for r in rows} == {"fct_orders", "stg_orders"}

    def test_downstream_walk(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "openlineage_events__downstream",
            options=Options(
                context={
                    "asset_urn": "fct_orders",
                    "lineage_direction": "DOWNSTREAM",
                    "upstream_depth": 0,
                    "downstream_depth": 1,
                }
            ),
        )
        rows = run_query("openlineage_events", self.valid_credentials()["openlineage_events"], feat)
        names = {r["name"] for r in rows}
        assert names == {"fct_orders", "report_daily"}


# --- Graph-construction unit tests (the merge/dedup behaviour that
# distinguishes this plugin from DbtManifestReader, plus the defensive
# malformed-event skip). ``_build_dataset_graph`` returns
# ``(upstream_map, downstream_map, namespace_index)``.


def test_build_graph_dedupes_duplicate_edge_across_events() -> None:
    """The same input->output edge emitted by two events is counted once.

    ``upstream``/``downstream`` are ``set``s, so a duplicate edge dedupes
    naturally; this pins that no-double-count behaviour as load-bearing.
    """
    events = [
        {"inputs": [{"name": "a", "namespace": "ns"}], "outputs": [{"name": "b", "namespace": "ns"}]},
        {"inputs": [{"name": "a", "namespace": "ns"}], "outputs": [{"name": "b", "namespace": "ns"}]},
    ]
    upstream, downstream, _ = _build_dataset_graph(events)
    assert upstream == {"b": {"a"}}
    assert downstream == {"a": {"b"}}


def test_build_graph_merges_start_no_outputs_with_complete_pair() -> None:
    """A START event with no outputs + a COMPLETE event merge into a single edge.

    eventType is not consulted (documented in the production module
    docstring): every event contributes its inputs/outputs to its group
    regardless of START/COMPLETE. Here neither event carries ``run.runId``,
    so each forms its own per-event group; the START contributes only its
    (empty) outputs and the merged dataset graph carries exactly the one
    edge the COMPLETE event produced.
    """
    events = [
        {"eventType": "START", "inputs": [{"name": "x", "namespace": "ns"}], "outputs": []},
        {
            "eventType": "COMPLETE",
            "inputs": [{"name": "x", "namespace": "ns"}],
            "outputs": [{"name": "y", "namespace": "ns"}],
        },
    ]
    upstream, downstream, namespaces = _build_dataset_graph(events)
    assert upstream == {"y": {"x"}}
    assert downstream == {"x": {"y"}}
    assert set(namespaces) == {"x", "y"}


def test_build_graph_aggregates_inputs_and_outputs_per_run_id() -> None:
    """Standard emitter shape: inputs on START, outputs on COMPLETE, same runId.

    Per-event pairing would derive zero edges here (no event carries both an
    input and an output); the per-run aggregation unions inputs and outputs
    across all events of the run and takes the cartesian, yielding the edge.
    """
    events = [
        {
            "eventType": "START",
            "run": {"runId": "r-1"},
            "inputs": [{"name": "x", "namespace": "ns"}],
            "outputs": [],
        },
        {
            "eventType": "COMPLETE",
            "run": {"runId": "r-1"},
            "inputs": [],
            "outputs": [{"name": "y", "namespace": "ns"}],
        },
    ]
    upstream, downstream, _ = _build_dataset_graph(events)
    assert upstream == {"y": {"x"}}
    assert downstream == {"x": {"y"}}


def test_build_graph_does_not_correlate_distinct_run_ids() -> None:
    """Events of DIFFERENT runs must not cross-pollinate edges.

    Run r-1 only reports an input and run r-2 only reports an output; were
    grouping keyed too loosely (or globally), a spurious a -> b edge would
    appear.
    """
    events = [
        {"run": {"runId": "r-1"}, "inputs": [{"name": "a", "namespace": "ns"}], "outputs": []},
        {"run": {"runId": "r-2"}, "inputs": [], "outputs": [{"name": "b", "namespace": "ns"}]},
    ]
    upstream, downstream, _ = _build_dataset_graph(events)
    assert upstream == {}
    assert downstream == {}


def test_build_graph_conflates_same_name_across_namespaces() -> None:
    """Dataset identity is keyed by ``name`` ONLY; namespaces do not split nodes.

    Two datasets named ``shared`` in namespaces ``ns1`` and ``ns2`` become ONE
    graph node carrying the first-seen namespace (``ns1``). This pins the
    documented conflation contract (row shape and asset_urn matching depend on
    name-only keying); a change to composite keying must update the module
    docstring and this test together.
    """
    events = [
        {"inputs": [{"name": "shared", "namespace": "ns1"}], "outputs": [{"name": "out1", "namespace": "ns1"}]},
        {"inputs": [{"name": "shared", "namespace": "ns2"}], "outputs": [{"name": "out2", "namespace": "ns2"}]},
    ]
    upstream, downstream, namespaces = _build_dataset_graph(events)
    # One conflated node fans out to both outputs.
    assert downstream == {"shared": {"out1", "out2"}}
    assert upstream == {"out1": {"shared"}, "out2": {"shared"}}
    # First-seen namespace wins for the conflated node.
    assert namespaces["shared"] == "ns1"


def test_build_graph_emits_cartesian_edges_for_multi_input_multi_output() -> None:
    """An event with N inputs and M outputs yields the full N*M edge set."""
    events = [
        {
            "inputs": [{"name": "a", "namespace": "ns"}, {"name": "b", "namespace": "ns"}],
            "outputs": [{"name": "c", "namespace": "ns"}, {"name": "d", "namespace": "ns"}],
        }
    ]
    upstream, downstream, _ = _build_dataset_graph(events)
    assert upstream == {"c": {"a", "b"}, "d": {"a", "b"}}
    assert downstream == {"a": {"c", "d"}, "b": {"c", "d"}}


def test_build_graph_skips_event_with_null_inputs() -> None:
    """A present-but-null ``inputs`` is coerced to ``[]``, not iterated as ``None``.

    Regression guard: ``event.get("inputs", [])`` only defaults on an absent
    key, so ``"inputs": null`` previously raised ``TypeError`` and aborted the
    whole build. The bad event is now skipped defensively and the valid edge
    survives.
    """
    events = [
        {"inputs": None, "outputs": [{"name": "junk", "namespace": "ns"}]},
        {"inputs": [{"name": "a", "namespace": "ns"}], "outputs": [{"name": "b", "namespace": "ns"}]},
    ]
    upstream, downstream, _ = _build_dataset_graph(events)
    assert upstream == {"b": {"a"}}
    assert downstream == {"a": {"b"}}


def test_load_data_completes_with_malformed_null_inputs_event(tmp_path: Path) -> None:
    """End-to-end: a ``"inputs": null`` event does not abort ``load_data``.

    The valid downstream edge ``a -> b`` is still walked and returned.
    """
    fixture = tmp_path / "events.json"
    fixture.write_text(
        json.dumps(
            {
                "events": [
                    {"inputs": None, "outputs": [{"name": "junk", "namespace": "ns"}]},
                    {
                        "inputs": [{"name": "a", "namespace": "ns"}],
                        "outputs": [{"name": "b", "namespace": "ns"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    feat = Feature(
        "openlineage_events__downstream",
        options=Options(
            context={
                "asset_urn": "a",
                "lineage_direction": "DOWNSTREAM",
                "upstream_depth": 0,
                "downstream_depth": 1,
            }
        ),
    )
    rows = run_query("openlineage_events", {"locator": str(fixture)}, feat)
    names = {r["name"] for r in rows}
    assert names == {"a", "b"}


def _write_events_fixture(tmp_path: Path, document: dict[str, Any]) -> str:
    fixture = tmp_path / "events.json"
    fixture.write_text(json.dumps(document), encoding="utf-8")
    return str(fixture)


def _lineage_feature(name: str, asset_urn: str, direction: str, up: int, down: int) -> Feature:
    return Feature(
        name,
        options=Options(
            context={
                "asset_urn": asset_urn,
                "lineage_direction": direction,
                "upstream_depth": up,
                "downstream_depth": down,
            }
        ),
    )


def test_both_direction_emits_cycle_node_once(tmp_path: Path) -> None:
    """A node both upstream AND downstream of the start (cycle) appears once under BOTH.

    Events form the cycle a -> b -> a, so from ``a`` the node ``b`` is
    reachable in both directions. The two directional walks share one EMITTED
    set; a regression to per-walk emission dedup would emit ``b`` twice and
    double-bill result_limit.
    """
    locator = _write_events_fixture(
        tmp_path,
        {
            "events": [
                {"inputs": [{"name": "a", "namespace": "ns"}], "outputs": [{"name": "b", "namespace": "ns"}]},
                {"inputs": [{"name": "b", "namespace": "ns"}], "outputs": [{"name": "a", "namespace": "ns"}]},
            ]
        },
    )
    feat = _lineage_feature("openlineage_events__cycle_both", "a", "BOTH", 1, 1)
    rows = run_query("openlineage_events", {"locator": locator}, feat)
    names = [r["name"] for r in rows]
    assert sorted(names) == ["a", "b"], f"expected each node once, got {names}"


def test_both_direction_expands_overlap_node_reached_by_first_walk(tmp_path: Path) -> None:
    """A dataset beyond an overlap node is still reached by the second walk under BOTH.

    Topology: edge b -> a (b is upstream of a) plus the downstream chain
    a -> c -> b -> d. The upstream walk emits ``b`` first; the downstream
    walk reaches ``b`` again via ``c`` and must still EXPAND it (without
    re-emitting it) so ``d`` is returned. A regression to one seen set shared
    for traversal would stop the downstream walk at ``b`` and silently drop
    ``d``.
    """
    locator = _write_events_fixture(
        tmp_path,
        {
            "events": [
                {"inputs": [{"name": "b", "namespace": "ns"}], "outputs": [{"name": "a", "namespace": "ns"}]},
                {"inputs": [{"name": "a", "namespace": "ns"}], "outputs": [{"name": "c", "namespace": "ns"}]},
                {"inputs": [{"name": "c", "namespace": "ns"}], "outputs": [{"name": "b", "namespace": "ns"}]},
                {"inputs": [{"name": "b", "namespace": "ns"}], "outputs": [{"name": "d", "namespace": "ns"}]},
            ]
        },
    )
    feat = _lineage_feature("openlineage_events__overlap_both", "a", "BOTH", 1, 3)
    rows = run_query("openlineage_events", {"locator": locator}, feat)
    names = [r["name"] for r in rows]
    assert sorted(names) == ["a", "b", "c", "d"], (
        f"expected every dataset exactly once (d beyond the overlap node b), got {names}"
    )


def test_non_list_events_value_raises_typed_fixture_error(tmp_path: Path) -> None:
    """``{"events": 42}`` is a malformed ARTIFACT and raises ``FixtureLoadError``.

    The defensive-skip contract covers malformed individual events; an
    uniterable top-level "events" previously leaked a raw ``TypeError`` from
    iteration. ``load_data`` is called directly (not via ``run_query``) so the
    assertion is on the reader's own raise, independent of how the mloda
    engine surfaces worker exceptions.
    """
    from mloda.provider import FeatureSet

    locator = _write_events_fixture(tmp_path, {"events": 42})
    feat = _lineage_feature("openlineage_events__bad_events", "a", "BOTH", 1, 0)
    fs = FeatureSet()
    fs.add(feat)
    creds = {"openlineage_events": {"locator": locator}}
    with pytest.raises(FixtureLoadError, match="'events' must be a list"):
        OpenLineageReader.load_data(creds, fs)


def test_unknown_asset_urn_returns_empty(tmp_path: Path) -> None:
    """An ``asset_urn`` absent from the dataset graph yields ``[]`` (no seed row, no walk hits)."""
    locator = _write_events_fixture(
        tmp_path,
        {"events": [{"inputs": [{"name": "a", "namespace": "ns"}], "outputs": [{"name": "b", "namespace": "ns"}]}]},
    )
    feat = _lineage_feature("openlineage_events__unknown_urn", "not_there", "BOTH", 2, 2)
    rows = run_query("openlineage_events", {"locator": locator}, feat)
    assert rows == []


def test_empty_events_list_returns_empty(tmp_path: Path) -> None:
    """``{"events": []}`` builds an empty graph: no seed row, no edges, ``[]``."""
    locator = _write_events_fixture(tmp_path, {"events": []})
    feat = _lineage_feature("openlineage_events__empty_events", "a", "BOTH", 2, 2)
    rows = run_query("openlineage_events", {"locator": locator}, feat)
    assert rows == []


def test_outputs_only_event_registers_dataset_without_edges(tmp_path: Path) -> None:
    """An event with outputs but no inputs is handled: dataset registered, zero edges.

    The output dataset is known to the namespace index (so the seed row is
    emitted for it) but has no upstream parents to walk.
    """
    locator = _write_events_fixture(
        tmp_path,
        {"events": [{"inputs": [], "outputs": [{"name": "only_out", "namespace": "ns"}]}]},
    )
    feat = _lineage_feature("openlineage_events__outputs_only", "only_out", "BOTH", 2, 2)
    rows = run_query("openlineage_events", {"locator": locator}, feat)
    assert rows == [{"name": "only_out", "namespace": "ns"}]
