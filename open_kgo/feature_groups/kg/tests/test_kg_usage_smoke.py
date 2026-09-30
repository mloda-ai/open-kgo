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

import pytest

from open_kgo.feature_groups.kg.tests._family_cases import CASE_IDS, CASES, ConnectorCase
from open_kgo.feature_groups.kg.tests._helpers import run_query


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


def test_every_case_resolves_after_plugin_loader_all(tmp_path: Path) -> None:
    """With every stock mloda plugin loaded (``ReadDBFeature`` included), each case still resolves to its own group.

    Runs in a subprocess so ``PluginLoader.all()`` never changes matching for the rest of this pytest process.
    """
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        from mloda.user import PluginLoader

        PluginLoader.all()

        from open_kgo.feature_groups.kg.tests._family_cases import CASES
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        failures = []
        for index, case in enumerate(CASES):
            case_dir = Path({str(tmp_path)!r}) / str(index)
            case_dir.mkdir()
            try:
                rows = run_query(case.connector_id, case.make_slot(case_dir), case.feature)
                if not rows or not all(case.assert_row(row) for row in rows):
                    failures.append((case.connector_id, rows[:1]))
            except Exception as exc:
                failures.append((case.connector_id, repr(exc)[:300]))
        sys.exit(repr(failures) if failures else 0)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
