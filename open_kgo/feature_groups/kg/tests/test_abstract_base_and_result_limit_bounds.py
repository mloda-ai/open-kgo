"""Regression tests for bounded ``result_limit`` cost.

Bounded cost: ``result_limit`` semantics are documented as *bound-output* on the
base, but readers MUST short-circuit work to bound cost. The dbt manifest reader
historically walked the entire BFS frontier and sliced at the end; tests
below assert the walk stops at the limit (observable via a counting
neighbour-iterator) and that the per-branch budget arithmetic in
``DbtManifestReader.load_data`` is correct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.errors import InvalidCredentialShape
from open_kgo.feature_groups.kg.lineage.dbt_manifest import DbtManifestReader, _walk_with_node


# ---------------------------------------------------------------------------
# Defect #12
# ---------------------------------------------------------------------------


class _CountingNeighbours:
    """Iterable that records every neighbour the walker actually consumes.

    A slice-then-trim implementation would consume every entry of this iterable
    even when ``remaining`` is small. A short-circuiting walker stops calling
    ``__next__`` as soon as the budget is hit, leaving ``consumed`` strictly
    below the source size.
    """

    def __init__(self, items: list[str]) -> None:
        self._items = items
        self.consumed: list[str] = []

    def __iter__(self) -> Iterator[str]:
        for item in self._items:
            self.consumed.append(item)
            yield item


def test_dbt_manifest_walk_consumes_only_up_to_remaining() -> None:
    """``_walk_with_node`` must abort BFS the moment the ``remaining`` budget hits.

    Builds a wide fan-out (one start, 100 children at depth 1) and asks for
    only 3 rows. Both the output size AND the underlying neighbour-iterator
    consumption must be exactly 3, proving the walker did not iterate past
    the budget.
    """
    children = [f"child_{i}" for i in range(100)]
    counter = _CountingNeighbours(children)
    edge_map: dict[str, Any] = {"start": counter}
    nodes_index: dict[str, Any] = {c: {"name": c} for c in children}

    out = _walk_with_node(edge_map, nodes_index, "start", depth=1, remaining=3)

    assert [row["urn"] for row in out] == children[:3]
    assert counter.consumed == children[:3], (
        f"walk did not short-circuit at remaining=3: consumed {len(counter.consumed)} of 100 neighbours"
    )


def test_dbt_manifest_walk_zero_remaining_emits_nothing() -> None:
    """A ``remaining=0`` call must not even start the walk (boundary case)."""
    out = _walk_with_node({"a": ["b", "c"]}, {"b": {}, "c": {}}, "a", depth=5, remaining=0)
    assert out == []


def test_dbt_manifest_walk_negative_remaining_emits_nothing() -> None:
    """``remaining < 0`` is treated identically to ``remaining == 0``.

    The guard ``remaining <= 0`` is asserted on both sides of the boundary so
    a future tightening to ``remaining < 0`` (which would silently emit one
    row when the caller passes 0) trips here.
    """
    out = _walk_with_node({"a": ["b", "c"]}, {"b": {}, "c": {}}, "a", depth=5, remaining=-1)
    assert out == []


# The expected-URN lists in the parametrize below assume these depths; binding
# them once here keeps the test data and the manifest shape in sync.
_DIAMOND_UPSTREAM_DEPTH = 1
_DIAMOND_DOWNSTREAM_DEPTH = 1


def _build_diamond_manifest() -> dict[str, Any]:
    """Manifest with 3 upstream and 2 downstream nodes around 'asset_M'.

    Used to exercise the per-branch ``result_limit - len(rows)`` arithmetic in
    ``DbtManifestReader.load_data``: with a small budget the upstream branch
    eats most of it and the downstream branch must respect the leftover.
    """
    nodes = {f"asset_U{i}": {"name": f"U{i}"} for i in range(3)}
    nodes["asset_M"] = {"name": "M"}
    nodes.update({f"asset_D{i}": {"name": f"D{i}"} for i in range(2)})
    parent_map: dict[str, list[str]] = {
        "asset_M": [f"asset_U{i}" for i in range(3)],
    }
    child_map: dict[str, list[str]] = {
        "asset_M": [f"asset_D{i}" for i in range(2)],
    }
    return {"nodes": nodes, "parent_map": parent_map, "child_map": child_map}


def _make_feature_set(direction: str) -> FeatureSet:
    feature = Feature(
        "dbt_manifest__lineage",
        options=Options(
            context={
                "asset_urn": "asset_M",
                "lineage_direction": direction,
                "upstream_depth": _DIAMOND_UPSTREAM_DEPTH,
                "downstream_depth": _DIAMOND_DOWNSTREAM_DEPTH,
            }
        ),
    )
    fs = FeatureSet()
    fs.add(feature)
    return fs


@pytest.mark.parametrize(
    "result_limit,expected",
    [
        (1, ["asset_M"]),
        (2, ["asset_M", "asset_U0"]),
        (4, ["asset_M", "asset_U0", "asset_U1", "asset_U2"]),
        (5, ["asset_M", "asset_U0", "asset_U1", "asset_U2", "asset_D0"]),
        (10, ["asset_M", "asset_U0", "asset_U1", "asset_U2", "asset_D0", "asset_D1"]),
    ],
)
def test_dbt_manifest_load_data_honours_result_limit(
    tmp_path: Path,
    result_limit: int,
    expected: list[str],
) -> None:
    """End-to-end: ``DbtManifestReader.load_data`` honours ``result_limit`` per-branch.

    Covers the budget arithmetic ``result_limit - len(rows)`` between the
    UPSTREAM and DOWNSTREAM branches: a regression where one branch overshoots
    or fails to compose the leftover would produce too many or too few rows.
    """
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_build_diamond_manifest()))

    creds = {
        DbtManifestReader.CONNECTOR_ID: {
            "locator": str(manifest_path),
            "result_limit": result_limit,
        }
    }
    rows = DbtManifestReader.load_data(creds, _make_feature_set(direction="BOTH"))
    urns = [r["urn"] for r in rows]
    assert urns == expected


def test_dbt_manifest_load_data_zero_result_limit_rejected_at_credential_surface(
    tmp_path: Path,
) -> None:
    """``result_limit=0`` is rejected at the credential surface, not silently empty.

    The policy: ``_validate_shape`` rejects
    ``result_limit < 1`` so the cross-reader divergence in append-then-check
    vs slice-at-end no longer matters. ``_prepare_load`` re-runs the check so
    direct ``load_data`` callers that bypass ``is_valid_credentials`` hit the
    same typed error.
    """
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_build_diamond_manifest()))

    creds = {
        DbtManifestReader.CONNECTOR_ID: {
            "locator": str(manifest_path),
            "result_limit": 0,
        }
    }
    with pytest.raises(InvalidCredentialShape):
        DbtManifestReader.load_data(creds, _make_feature_set(direction="BOTH"))
