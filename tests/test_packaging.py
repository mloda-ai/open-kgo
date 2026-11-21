"""The wheel ships no test-only code: ``pyproject.toml`` excludes every ``tests`` package, nothing else."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "open_kgo"


def test_the_package_find_rules_are_the_ones_the_scan_mirrors() -> None:
    """``_shipped_modules`` mirrors these rules; a change here must change the scan too."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert find == {"where": ["."], "include": ["open_kgo*"], "exclude": ["*.tests", "*.tests.*"]}


def _shipped_modules() -> list[Path]:
    """Every module under ``open_kgo/`` outside a ``tests`` package, which ``packages.find`` excludes."""
    return sorted(path for path in PACKAGE.rglob("*.py") if "tests" not in path.relative_to(PACKAGE).parts)


def _imports_pytest(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        if any(name.split(".")[0] in {"pytest", "_pytest"} for name in names):
            return True
    return False


def test_no_shipped_module_is_a_conftest_or_imports_pytest() -> None:
    shipped = _shipped_modules()
    assert PACKAGE / "feature_groups" / "kg" / "fixtures.py" in shipped, "the scan found no package modules"
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in shipped
        if path.name == "conftest.py" or _imports_pytest(path)
    ]
    assert not offenders, f"test-only modules would ship in the wheel: {offenders}"
