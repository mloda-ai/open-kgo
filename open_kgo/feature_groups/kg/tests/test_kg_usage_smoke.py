"""All-family usage smoke: every ``CASES`` recipe through the real ``mloda.run_all`` path.

One of the three holistic modules over the shared ``_family_cases`` registry (see
that module's docstring for the split). Every concrete already runs end-to-end on
its own via the inherited ``test_calculate_feature_runs_end_to_end``; this sweep is
the single readable place that shows all 9 families' usage at once.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, mloda

from open_kgo.feature_groups.kg.base import PythonDictFramework
from open_kgo.feature_groups.kg.tests._family_cases import CASE_IDS, CASES, ConnectorCase
from open_kgo.feature_groups.kg.tests._helpers import run_query

_CASES_BY_CONNECTOR = {case.connector_id: case for case in CASES}


class KgRowCount(FeatureGroup):
    """Consumer named unlike any connector: repeats the row count of the KG feature picked by the ``kg_case`` option."""

    @classmethod
    def input_features(cls, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        source = _CASES_BY_CONNECTOR[options.get("kg_case")].feature
        # A fresh copy: mloda writes the matched reader into the input feature's options.
        return {Feature(source.name, options=Options(context=dict(source.options.context)))}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature = next(iter(features.features))
        rows = data[_CASES_BY_CONNECTOR[feature.options.get("kg_case")].feature.name]
        return {feature.name: [len(rows)] * len(rows)}


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_family_usage_smoke(case: ConnectorCase, tmp_path: Path) -> None:
    """Drive each connector through the real mloda.run_all path and assert a usable row shape.

    Holistic counterpart to the per-connector ``test_calculate_feature_runs_end_to_end``:
    one sweep, all 9 families, the same DataAccessCollection -> run_all -> PythonDictFramework
    chain a caller uses. A regression in matching, validation, or the load-side wrap that
    happens to affect every family at once shows up here as a wall of red rather than one case.
    """
    slot = case.make_slot(tmp_path)
    rows = run_query(case.connector_id, slot, case.feature)
    assert isinstance(rows, list) and len(rows) >= 1, (
        f"{case.connector_id}: expected >= 1 row from {case.feature.name!r}, got {rows!r}"
    )
    bad = [row for row in rows if not case.assert_row(row)]
    assert not bad, f"{case.connector_id}: {len(bad)} row(s) failed the shape predicate; first bad row: {bad[0]!r}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_consumer_takes_kg_feature_as_input_in_one_run(case: ConnectorCase, tmp_path: Path) -> None:
    """With the connector's credentials present, the KG reader leaves the consumer's unrelated name alone."""
    dac = DataAccessCollection(credentials=[{case.connector_id: case.make_slot(tmp_path)}])
    consumer = Feature("KgRowCount", options=Options(context={"kg_case": case.connector_id}))
    partitions = mloda.run_all([consumer], compute_frameworks={PythonDictFramework}, data_access_collection=dac)
    counts = [count for partition in partitions for count in partition.get(consumer.name, [])]
    assert counts and set(counts) == {len(counts)}, f"{case.connector_id}: got {counts!r}"


def test_every_case_resolves_after_plugin_loader_all(tmp_path: Path) -> None:
    """With every stock mloda plugin loaded (``ReadDBFeature`` included), each case still resolves to its own group.

    ``<case feature>__sum_aggr`` must resolve to the stock aggregation groups, never to the KG group.
    Runs in a subprocess so ``PluginLoader.all()`` never changes matching for the rest of this pytest process.
    """
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        from mloda.steward import resolve_feature
        from mloda.user import DataAccessCollection, PluginLoader

        PluginLoader.all()

        from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase
        from open_kgo.feature_groups.kg.tests._family_cases import CASES
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        failures = []
        for index, case in enumerate(CASES):
            case_dir = Path({str(tmp_path)!r}) / str(index)
            case_dir.mkdir()
            try:
                slot = case.make_slot(case_dir)
                rows = run_query(case.connector_id, slot, case.feature)
                if not rows or not all(case.assert_row(row) for row in rows):
                    failures.append((case.connector_id, rows[:1]))
                chained = resolve_feature(
                    f"{{case.feature.name}}__sum_aggr",
                    options=case.feature.options,
                    data_access_collection=DataAccessCollection(credentials=[{{case.connector_id: slot}}]),
                )
                owners = [fg.__name__ for fg in chained.candidates]
                if not owners or any(issubclass(fg, KgConnectorFeatureGroupBase) for fg in chained.candidates):
                    failures.append((case.connector_id, "__sum_aggr candidates", owners))
            except Exception as exc:
                failures.append((case.connector_id, repr(exc)[:300]))
        sys.exit(repr(failures) if failures else 0)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
