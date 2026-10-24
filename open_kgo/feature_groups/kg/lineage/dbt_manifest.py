"""dbt manifest.json parser as a lineage source.

The dbt manifest is a real artifact dbt emits; ``manifest.json`` contains
``nodes`` (models, sources, tests) and ``parent_map`` (upstream edges) /
``child_map`` (downstream edges). We use these to walk lineage from a starting
node URN.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.fixtures import copy_cached_row, load_json_fixture
from open_kgo.feature_groups.kg.lineage.base import LineageFeatureGroup, LineageReader
from open_kgo.feature_groups.kg.traversal import bfs_frontier_walk


class DbtManifestReader(LineageReader):
    CONNECTOR_ID: ClassVar[str] = "dbt_manifest"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",),)
    # Strict-enum narrowings:
    #   - lineage_direction: the TraversalMixin family enum advertises
    #     Reactome-style ``ancestors`` / ``descendants`` for citation-shaped
    #     concretes; dbt manifests only carry parent_map / child_map graphs,
    #     so the dbt walker dispatches on UPSTREAM / DOWNSTREAM / BOTH only.
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "lineage_direction": frozenset({"UPSTREAM", "DOWNSTREAM", "BOTH"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> dict[str, Any]:
        """Return the parsed manifest dict; mtime-cached so a 100-feature run pays one parse.

        A real dbt manifest is 10-100MB; reparsing on every ``load_data``
        call is the largest single hit in the
        file-backed family. The returned dict is shared across calls and
        MUST be treated as read-only; ``load_data`` shallow-copies the
        nested node entry into each row so callers cannot mutate the
        cached manifest through a returned ``row["node"]`` reference
        (mirrors the citation reader; uniform with the base-class
        mutation-safety contract).
        """
        return load_json_fixture(cls.CONNECTOR_ID, slot["locator"])

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        manifest = connection
        params = cls.build_params(features, ctx.slot)

        asset_urn = params.get("asset_urn")
        if not asset_urn:
            raise ValueError(f"{cls.CONNECTOR_ID}: 'asset_urn' is required.")
        direction = params.get("lineage_direction", "BOTH")
        upstream_depth = int(params.get("upstream_depth", 1))
        downstream_depth = int(params.get("downstream_depth", 0))
        result_limit = ctx.result_limit

        parent_map = manifest.get("parent_map", {})
        child_map = manifest.get("child_map", {})
        nodes_index = manifest.get("nodes", {})

        rows: list[dict[str, Any]] = []
        if asset_urn in nodes_index and result_limit > 0:
            # copy_cached_row keeps the shared manifest cache read-only when a
            # caller holds row["node"] (see ``_connect_from_slot``).
            rows.append({"urn": asset_urn, "node": copy_cached_row(nodes_index[asset_urn])})

        # One EMITTED set shared across both directional walks: under BOTH, a
        # node reachable in both directions (cycle through the start) must be
        # emitted once and bill result_limit once. Each walk keeps its own
        # local visited set, so an already-emitted overlap node is still
        # expanded by the second walk and nodes beyond it are reached.
        emitted: set[str] = {asset_urn}
        if direction in ("UPSTREAM", "BOTH") and len(rows) < result_limit:
            rows.extend(
                _walk_with_node(parent_map, nodes_index, asset_urn, upstream_depth, result_limit - len(rows), emitted)
            )
        if direction in ("DOWNSTREAM", "BOTH") and len(rows) < result_limit:
            rows.extend(
                _walk_with_node(child_map, nodes_index, asset_urn, downstream_depth, result_limit - len(rows), emitted)
            )

        return rows


def _walk_with_node(
    edge_map: dict[str, list[str]],
    nodes_index: dict[str, Any],
    start: str,
    depth: int,
    remaining: int,
    emitted: set[str] | None = None,
) -> list[dict[str, Any]]:
    """BFS walk along edge_map; emit nodes_index[urn] entries (or {urn:...} stubs).

    Thin binding over ``bfs_frontier_walk`` (see ``kg.traversal`` for the
    budget and dedup semantics): supplies the dbt edge map and the
    ``{"urn": ..., "node": ...}`` row shape. ``nodes_index`` is a ref into
    the shared manifest cache; ``copy_cached_row`` keeps it read-only at the
    row level.
    """
    return bfs_frontier_walk(
        lambda node: edge_map.get(node, []),
        start,
        depth=depth,
        remaining=remaining,
        emit=lambda urn: {"urn": urn, "node": copy_cached_row(nodes_index.get(urn))},
        emitted=emitted,
    )


class DbtManifestFeatureGroup(LineageFeatureGroup):
    READER_CLASS: ClassVar[type[DbtManifestReader]] = DbtManifestReader
