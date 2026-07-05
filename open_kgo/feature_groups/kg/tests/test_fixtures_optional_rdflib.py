"""Regression: the shared ``kg.fixtures`` module must import without ``kg-rdf``.

``fixtures.py`` is imported by every file-backed connector family, including
non-RDF ones (``agent_memory/networkx_memory``, ``saas_authz/...``). An
unconditional top-level ``import rdflib`` therefore coupled every single-family
install (e.g. ``open-kgo[kg-memory]``) to the ``kg-rdf`` extra, breaking those
imports with ``ModuleNotFoundError: No module named 'rdflib'`` (issue #33).

The rdflib environment IS present in this dev/test env (``kg-rdf`` is one of the
installed extras), so absence is simulated in a child interpreter that blocks
any ``import rdflib`` via a meta-path finder. The subprocess mirrors the issue's
``python -c`` reproduction and keeps the blocker out of the parent test session.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_non_rdf_connector_imports_without_rdflib() -> None:
    """A non-RDF connector and the shared fixtures module import with rdflib absent."""
    code = textwrap.dedent(
        """
        import sys

        class _BlockRdflib:
            # Make any `import rdflib` / `import rdflib.<sub>` fail as if the
            # kg-rdf extra were never installed.
            def find_spec(self, name, path, target=None):
                if name == "rdflib" or name.startswith("rdflib."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.modules.pop("rdflib", None)
        sys.meta_path.insert(0, _BlockRdflib())

        # Guard: confirm the simulated absence is actually in effect, otherwise
        # the assertions below would pass trivially with rdflib importable.
        try:
            import rdflib  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            raise SystemExit("test bug: rdflib should be unimportable here")

        # The shared module and a non-RDF connector must import cleanly.
        import open_kgo.feature_groups.kg.fixtures  # noqa: F401
        import open_kgo.feature_groups.kg.agent_memory.networkx_memory  # noqa: F401
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_load_rdf_graph_still_works_with_rdflib_present(tmp_path: object) -> None:
    """The deferred import path still parses RDF when rdflib IS installed.

    Guards against a deferral that removes the top-level import but forgets to
    re-import inside the loader (which would surface as ``NameError`` on the
    first ``load_rdf_graph`` call rather than at import time).
    """
    from pathlib import Path

    from open_kgo.feature_groups.kg.fixtures import load_rdf_graph

    path = Path(str(tmp_path)) / "g.ttl"
    path.write_text('<urn:s> <urn:p> "o" .\n', encoding="utf-8")
    graph = load_rdf_graph("test", path)
    assert len(graph) == 1
