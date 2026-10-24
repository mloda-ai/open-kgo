"""OpenLineage run-event parser as a lineage source.

Second concrete in the ``lineage`` family alongside ``DbtManifestReader``.
OpenLineage is an open standard whose run events are emitted as JSON; each
event names a ``job`` plus its ``inputs`` and ``outputs`` datasets. Inputs and
outputs are aggregated per ``run.runId`` across all events of that run, and a
dataset edge ``input -> output`` is derived per run (every input is upstream of
every output of the same run). This matters because standard emitters report
inputs on the START event and outputs on the COMPLETE event of the same run;
per-event pairing would see no edges at all. ``eventType`` is not consulted:
every event of a run contributes its inputs/outputs to the run-level union
regardless of START/RUNNING/COMPLETE/etc. Events missing a usable
``run.runId`` cannot be correlated and fall back to a per-event pairing of
their own inputs and outputs. The result is a dataset-level lineage graph we
walk from a starting ``asset_urn``. Like dbt, this is a static-artifact source
(stdlib JSON, no new dependency); it adds backend/pedigree variety on the
family's UPSTREAM / DOWNSTREAM / BOTH walk rather than a new family surface.

The fixture top level is an object ``{"events": [<run event>, ...]}`` (the
``load_json_fixture`` contract requires a dict at the top level; a bare event
array would be rejected). A present ``"events"`` value that is not a list is a
malformed artifact, not a malformed event, and raises the family's typed
``FixtureLoadError`` rather than leaking a raw ``TypeError`` from iteration.
Datasets are identified by their ``name`` field ONLY; ``namespace`` is carried
through into each row but does not participate in identity, so two datasets
with the same name in different namespaces are conflated into one graph node
that carries the first-seen namespace. This keying is load-bearing (the row
shape and ``asset_urn`` matching depend on it) and is a documented contract.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.errors import FixtureLoadError
from open_kgo.feature_groups.kg.fixtures import load_json_fixture
from open_kgo.feature_groups.kg.lineage.base import LineageFeatureGroup, LineageReader
from open_kgo.feature_groups.kg.traversal import bfs_frontier_walk


def _build_dataset_graph(
    events: list[Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """Return ``(upstream_map, downstream_map, namespace_index)`` from OpenLineage events.

    ``upstream_map[ds]`` is the set of datasets ``ds`` was produced from
    (parents); ``downstream_map[ds]`` is the set of datasets produced from
    ``ds`` (children). ``namespace_index`` records the first-seen namespace per
    dataset name. Inputs and outputs are aggregated per ``run.runId`` across
    all of that run's events (regardless of ``eventType``), then the cartesian
    ``input x output`` is taken per run: a standard emitter reports inputs on
    START and outputs on COMPLETE, so per-event pairing would derive no edges.
    Events without a usable ``run.runId`` (missing, or non-dict ``run``)
    cannot be correlated and defensively form their own per-event group.
    Malformed entries (non-dict event, non-list inputs/outputs) are skipped
    defensively: a lineage artifact assembled from many emitters routinely
    carries partial events, and one bad event should not abort the walk.
    """
    upstream: dict[str, set[str]] = {}
    downstream: dict[str, set[str]] = {}
    namespaces: dict[str, str] = {}
    # Group key is ("run", runId) for correlatable events; uncorrelatable
    # events get a unique ("event", index) key so they pair only with
    # themselves. The two tag strings keep the key spaces disjoint (a runId
    # that happens to look like an integer index cannot collide).
    group_inputs: dict[tuple[str, str], set[str]] = {}
    group_outputs: dict[tuple[str, str], set[str]] = {}

    def _register(dataset: Any) -> str | None:
        if not isinstance(dataset, dict) or "name" not in dataset:
            return None
        name = str(dataset["name"])
        if name not in namespaces:
            namespaces[name] = str(dataset.get("namespace", ""))
        return name

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        run = event.get("run")
        run_id = run.get("runId") if isinstance(run, dict) else None
        key = ("run", str(run_id)) if run_id is not None else ("event", str(index))
        # ``event.get("inputs", [])`` only defaults on an ABSENT key; a present
        # ``"inputs": null`` returns ``None`` and ``for d in None`` would raise,
        # aborting the whole walk. Coerce non-list values to ``[]`` so a single
        # malformed event is skipped defensively (per this function's contract).
        raw_inputs = event.get("inputs")
        raw_outputs = event.get("outputs")
        inputs_src = raw_inputs if isinstance(raw_inputs, list) else []
        outputs_src = raw_outputs if isinstance(raw_outputs, list) else []
        group_inputs.setdefault(key, set()).update(n for n in (_register(d) for d in inputs_src) if n is not None)
        group_outputs.setdefault(key, set()).update(n for n in (_register(d) for d in outputs_src) if n is not None)

    for key, outputs in group_outputs.items():
        inputs = group_inputs.get(key, set())
        for out in outputs:
            for inp in inputs:
                upstream.setdefault(out, set()).add(inp)
                downstream.setdefault(inp, set()).add(out)
    return upstream, downstream, namespaces


def _walk(
    edge_map: dict[str, set[str]],
    namespaces: dict[str, str],
    start: str,
    depth: int,
    remaining: int,
    emitted: set[str] | None = None,
) -> list[dict[str, Any]]:
    """BFS along ``edge_map`` from ``start`` up to ``depth`` hops; emit dataset rows.

    Thin binding over ``bfs_frontier_walk`` (see ``kg.traversal`` for the
    budget and dedup semantics): supplies the derived dataset edge map
    (neighbours sorted for a deterministic frontier order, since the edge
    sets are unordered) and the ``{"name": ..., "namespace": ...}`` row
    shape.
    """
    return bfs_frontier_walk(
        lambda node: sorted(edge_map.get(node, set())),
        start,
        depth=depth,
        remaining=remaining,
        emit=lambda name: {"name": name, "namespace": namespaces.get(name, "")},
        emitted=emitted,
    )


class OpenLineageReader(LineageReader):
    CONNECTOR_ID: ClassVar[str] = "openlineage_events"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",),)
    # The TraversalMixin enum advertises Reactome-style ancestors / descendants
    # for citation-shaped concretes; OpenLineage events describe a dataset
    # input/output graph, so this walker dispatches on UPSTREAM / DOWNSTREAM /
    # BOTH only (same disposition as DbtManifestReader).
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "lineage_direction": frozenset({"UPSTREAM", "DOWNSTREAM", "BOTH"}),
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> dict[str, Any]:
        """Return the parsed events document; mtime-cached so a 100-feature run pays one parse."""
        return load_json_fixture(cls.CONNECTOR_ID, slot["locator"])

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        document = connection
        params = cls.build_params(features, ctx.slot)

        asset_urn = params.get("asset_urn")
        if not asset_urn:
            raise ValueError(f"{cls.CONNECTOR_ID}: 'asset_urn' is required.")
        direction = params.get("lineage_direction", "BOTH")
        upstream_depth = int(params.get("upstream_depth", 1))
        downstream_depth = int(params.get("downstream_depth", 0))
        result_limit = ctx.result_limit

        events = document.get("events", [])
        if not isinstance(events, list):
            # A non-list top-level "events" is a malformed ARTIFACT (the
            # defensive-skip contract covers malformed individual events, not
            # an uniterable document); surface the family's typed error rather
            # than leaking a raw TypeError from iteration.
            raise FixtureLoadError(
                cls.CONNECTOR_ID,
                str(ctx.slot.get("locator")),
                f"top-level 'events' must be a list, got {type(events).__name__}.",
            )
        upstream_map, downstream_map, namespaces = _build_dataset_graph(events)

        rows: list[dict[str, Any]] = []
        if asset_urn in namespaces and result_limit > 0:
            rows.append({"name": asset_urn, "namespace": namespaces[asset_urn]})

        # One EMITTED set shared across both directional walks: under BOTH, a
        # node reachable in both directions (cycle through the start) must be
        # emitted once and bill result_limit once. Each walk keeps its own
        # local visited set, so an already-emitted overlap node is still
        # expanded by the second walk and nodes beyond it are reached.
        emitted: set[str] = {asset_urn}
        if direction in ("UPSTREAM", "BOTH") and len(rows) < result_limit:
            rows.extend(_walk(upstream_map, namespaces, asset_urn, upstream_depth, result_limit - len(rows), emitted))
        if direction in ("DOWNSTREAM", "BOTH") and len(rows) < result_limit:
            rows.extend(
                _walk(downstream_map, namespaces, asset_urn, downstream_depth, result_limit - len(rows), emitted)
            )

        return rows


class OpenLineageFeatureGroup(LineageFeatureGroup):
    READER_CLASS: ClassVar[type[OpenLineageReader]] = OpenLineageReader
