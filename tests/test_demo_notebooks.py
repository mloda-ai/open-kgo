"""End-to-end smoke test: every demo marimo notebook executes without errored cells.

The notebooks under ``demo/`` are user-facing and excluded from ruff/mypy, so
nothing else in the suite exercises them. This test runs each one headlessly
via ``marimo export html``, which executes the notebook *in place* (so the
notebooks' ``__file__``-relative data paths resolve) and exits non-zero when
any cell raises ("Export was successful, but some cells failed to execute").
The export tool occasionally emits an unrelated msgspec serialization warning
to stderr while snapshotting the variables panel; that does not affect the
exit code, so we assert on the return code rather than on stderr being empty.

The notebooks build their sample graph offline via ``demo.data.ensure_data``
(committed fixtures, no network), so the test needs no external resources.

Marked ``notebooks`` (slow: ~5s/notebook) so the inner-loop can deselect with
``pytest -m "not notebooks"``. The default ``tox`` run *excludes* them
(``pytest -m "not notebooks"``); they run in the separate ``tox -e notebooks``
env, which CI invokes as its own job and which installs the ``demo`` extra.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# marimo ships in the ``demo`` extra; skip cleanly in an env without it
# (the ``notebooks`` tox env installs ``demo`` so this runs in CI — see tox.ini).
pytest.importorskip("marimo")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"


def _notebook_paths() -> list[Path]:
    """Every ``demo/*.py`` that is a marimo notebook (auto-includes future ones)."""
    return sorted(p for p in DEMO_DIR.glob("*.py") if "import marimo" in p.read_text(encoding="utf-8"))


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook", _notebook_paths(), ids=lambda p: p.name)
def test_demo_notebook_executes_without_errored_cells(notebook: Path, tmp_path: Path) -> None:
    """``marimo export html`` runs the whole notebook; a non-zero exit means a cell raised."""
    out_html = tmp_path / f"{notebook.stem}.html"
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html", str(notebook), "-o", str(out_html)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{notebook.name} failed to execute headlessly (exit {result.returncode}).\n"
        f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}"
    )
