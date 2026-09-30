"""Regression: the shared ``kg.fixtures`` module must import without ``kg-rdf``.

``fixtures.py`` is imported by non-RDF connector families, so a top-level
``import rdflib`` broke single-family installs like ``open-kgo[kg-memory]`` with
``ModuleNotFoundError`` (issue #33). rdflib IS installed in this test env, so its
absence is simulated in a subprocess that blocks the import via a meta-path
finder.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_non_rdf_connector_imports_without_rdflib() -> None:
    """A non-RDF connector and the shared fixtures module import with rdflib absent."""
    code = textwrap.dedent(
        """
        import sys

        class _BlockRdflib:
            # Make any `import rdflib[...]` fail as if kg-rdf were not installed.
            def find_spec(self, name, path, target=None):
                if name == "rdflib" or name.startswith("rdflib."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.modules.pop("rdflib", None)
        sys.meta_path.insert(0, _BlockRdflib())

        # Guard: fail loudly if the blocker is a no-op (else the test passes trivially).
        try:
            import rdflib  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            raise SystemExit("test bug: rdflib should be unimportable here")

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


def test_load_rdf_graph_still_works_with_rdflib_present(tmp_path: Path) -> None:
    """The deferred import path still parses RDF when rdflib is installed.

    Guards against a deferral that drops the top-level import but forgets to
    re-import inside the loader (a ``NameError`` on first call).
    """
    from open_kgo.feature_groups.kg.fixtures import load_rdf_graph

    path = tmp_path / "g.ttl"
    path.write_text('<urn:s> <urn:p> "o" .\n', encoding="utf-8")
    graph = load_rdf_graph("test", path)
    assert len(graph) == 1


def test_plugin_loader_all_skips_only_the_rdf_family_without_rdflib() -> None:
    """``PluginLoader.all()`` skips the rdf entry point by name and still runs another family end to end."""
    sbom = Path(__file__).resolve().parent.parent / "code_build" / "tests" / "fixtures" / "sample.cdx.json"
    code = textwrap.dedent(
        f"""
        import sys

        # A None entry raises ModuleNotFoundError(name="rdflib"), the shape mloda attributes to a declared root.
        sys.modules["rdflib"] = None

        from mloda.user import Feature, Options, PluginLoader

        PluginLoader.all()
        skipped = PluginLoader.skipped_plugins()
        rdf = "open-kgo-rdf (open_kgo.feature_groups.kg.rdf.manifest:FEATURE_GROUPS)"
        assert skipped.get(rdf) == "rdflib", skipped
        assert [key for key in skipped if key.startswith("open-kgo-")] == [rdf], skipped

        from open_kgo.feature_groups.kg.tests._helpers import run_query

        rows = run_query(
            "cyclonedx_sbom",
            {{"manifest_path": {str(sbom)!r}}},
            Feature("cyclonedx_sbom__components", options=Options(context={{}})),
        )
        assert rows, rows
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
