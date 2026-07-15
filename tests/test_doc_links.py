"""Tests to verify that relative markdown links in documentation exist."""

import re
import urllib.parse
from pathlib import Path
import pytest

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def get_markdown_files() -> list[Path]:
    """Find all markdown files in the repository, excluding build/cache/env directories."""
    root_dir = Path(__file__).parent.parent.resolve()
    md_files = []
    ignored_dirs = {
        ".git",
        ".venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }

    for path in root_dir.rglob("*.md"):
        if any(parent.name in ignored_dirs for parent in path.parents):
            continue
        md_files.append(path)
    return sorted(md_files)


def strip_code_blocks(content: str) -> str:
    """Remove fenced code blocks from markdown content."""
    return re.sub(r"```.*?```", "", content, flags=re.DOTALL)


@pytest.mark.parametrize("file_path", get_markdown_files())
def test_markdown_links(file_path: Path) -> None:
    """Verify that all relative markdown links in the file point to existing files or directories."""
    root_dir = Path(__file__).parent.parent.resolve()
    content = file_path.read_text(encoding="utf-8")
    content_no_code = strip_code_blocks(content)

    targets = LINK_RE.findall(content_no_code)
    broken_links = []

    for target in targets:
        # Skip absolute URLs and anchors/emails
        if target.startswith(("http://", "https://", "mailto:", "ftp:", "git://")):
            continue

        # Skip anchor-only links
        if target.startswith("#"):
            continue

        # Split fragment and query parameters
        target_clean = target.split("#")[0].split("?")[0]
        if not target_clean:
            continue

        # URL decode to handle spaces and special characters
        target_clean = urllib.parse.unquote(target_clean)

        # Resolve path
        if target_clean.startswith("/"):
            resolved_path = root_dir / target_clean.lstrip("/")
        else:
            resolved_path = (file_path.parent / target_clean).resolve()

        if not resolved_path.exists():
            broken_links.append(target)

    assert not broken_links, f"Broken relative links found in {file_path.name}: {broken_links}"
