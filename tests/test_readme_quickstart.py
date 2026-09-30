"""Executes the README Quickstart snippet, the first code a new user runs.

The demo notebooks are guarded end to end by test_demo_notebooks.py and the sample graph by
test_demo_data.py, but nothing ran the Quickstart, so a renamed reader, a changed option key, or a
new ``mloda.run_all`` signature could rot the one snippet every reader tries first.

The block is located rather than copied: a duplicate here would drift from README.md silently, which
is the failure this guards against. Locating it means a README restructure has to fail loudly instead
of skipping the test, hence the explicit errors in _quickstart_snippet.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

QUICKSTART_HEADING = "## Quickstart"

# A markdown heading, only ever tested outside a fence: the snippet's own '# Point at any RDF file.'
# comments match this too, so fence state has to be tracked rather than pattern-matched around.
HEADING_PATTERN = re.compile(r"^#{1,6} ")
FENCE_PATTERN = re.compile(r"^```(?P<language>\w*)")


def _quickstart_snippet() -> str:
    """Return the Quickstart's python block, raising rather than skipping if it moved.

    Scans line by line: the next heading outside a fence ends the section, so a python block further
    down the page can never stand in for a deleted one.
    """
    lines = README.read_text(encoding="utf-8").splitlines()

    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == QUICKSTART_HEADING)
    except StopIteration:
        raise AssertionError(
            f"{README.name} has no '{QUICKSTART_HEADING}' heading; this guard checks nothing."
        ) from None

    body: list[str] = []
    inside_python = False

    for line in lines[start + 1 :]:
        fence = FENCE_PATTERN.match(line)
        if fence is not None:
            if inside_python:
                break
            inside_python = fence.group("language") == "python"
            continue
        if inside_python:
            body.append(line)
            continue
        if HEADING_PATTERN.match(line):
            # Left the Quickstart section without finding the block.
            break
    else:
        if inside_python:
            raise AssertionError(f"{README.name} Quickstart ```python block is never closed.")

    if not body:
        raise AssertionError(
            f"{README.name} has a '{QUICKSTART_HEADING}' section with no ```python block. "
            "If the snippet moved, point this guard at its new home rather than deleting it."
        )
    if not "\n".join(body).strip():
        raise AssertionError(f"{README.name} Quickstart ```python block is empty.")
    return "\n".join(body) + "\n"


# Appended to the snippet so the subprocess reports what the guards below assert on.
_REPORT = """
import json as _json
_rows = [_row for _partition in partitions for _row in _partition.get(feature.name, [])]
print(_json.dumps({"feature": feature.name, "rows": len(_rows)}))
"""


@pytest.fixture(scope="module")
def snippet() -> str:
    return _quickstart_snippet()


@pytest.fixture(scope="module")
def quickstart_run(snippet: str, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """Run the snippet in a tmp-dir subprocess, keeping ``PluginLoader.all()`` out of this pytest process."""
    workdir = tmp_path_factory.mktemp("quickstart")
    result = subprocess.run(
        [sys.executable, "-c", snippet + _REPORT], cwd=workdir, capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, (
        f"Quickstart snippet failed; it must bind 'feature' and 'partitions' for this guard to read.\n{result.stderr}"
    )
    return workdir, json.loads(result.stdout.strip().splitlines()[-1])


def test_the_quickstart_block_is_locatable(snippet: str) -> None:
    """The floor under the execution test, which would pass vacuously on an empty snippet."""
    assert "mloda.run_all" in snippet, f"Quickstart snippet no longer calls mloda.run_all:\n{snippet}"


def test_the_quickstart_snippet_runs_and_returns_rows(quickstart_run: tuple[Path, dict[str, Any]]) -> None:
    """Run the snippet verbatim and assert it produces signal, not just absence of errors."""
    _, report = quickstart_run
    assert report["rows"], (
        f"Quickstart ran but returned no rows for {report['feature']}. "
        "The snippet queries a three-triple graph with two foaf:knows statements, so it should match."
    )


def test_the_quickstart_writes_nothing_into_the_repository(quickstart_run: tuple[Path, dict[str, Any]]) -> None:
    """sample.ttl belongs in the tmp dir; a cwd regression here would dirty the working tree."""
    workdir, _ = quickstart_run
    assert (workdir / "sample.ttl").is_file(), "Quickstart no longer writes sample.ttl where this guard expects it."
    assert not (REPO_ROOT / "sample.ttl").exists(), "Quickstart wrote sample.ttl into the repository working tree."
