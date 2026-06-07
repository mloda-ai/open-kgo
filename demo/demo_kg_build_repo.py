import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def title(mo):
    mo.md("""
    # Build a KG from this repo, then query it

    The first demo (`demo_kg_connectors.py`) walks all 9 KG families against
    fixtures shipped with the prototype. This second demo flips the polarity:
    pick **one** family (RDFLib / SPARQL, Group 2) and use it to ingest
    content from a **real source**, the prototype repo itself.

    The resulting graph captures two relationships:

    1. `repo:contains` between directories and the files / sub-directories
       inside them (filesystem hierarchy).
    2. `repo:imports` between Python modules, parsed from the `import` /
       `from ... import` lines of every `.py` file.

    We exclude any path component starting with `.` (so `.tox`, `.venv`,
    `.git`, `.claude`, ...) and `__pycache__` (compiled artifacts; would
    dominate the graph without adding KG value).

    Once the graph is built, we serialize it to a temp Turtle file and
    query it through the same `RdfLibSparqlReader` that demo 1 used. No
    new code in the family layer.
    """)
    return


@app.cell
def overview(mo):
    mo.md("""
    ## RDF schema

    | Term | Kind | Meaning |
    |---|---|---|
    | `repo:Directory` | class | a directory under the repo root |
    | `repo:File` | class | a regular file under the repo root |
    | `repo:Module` | class | a Python module (top-level name only) |
    | `repo:contains` | property | directory contains a file or sub-directory |
    | `repo:imports` | property | a module imports another module |
    | `repo:hasName`, `repo:hasPath`, `repo:hasExt` | properties | literal attributes |

    Modules are addressed by their dotted name (e.g.
    `open_kgo.feature_groups.kg.rdf.rdflib_sparql`); when an
    `import x.y.z` statement is parsed, we keep only the top-level name
    (`x`) on the right-hand side, so the import graph stays readable.
    """)
    return


@app.cell
def helpers():
    import os as _os
    import re as _re
    import tempfile as _tempfile
    from pathlib import Path as _Path
    from typing import Any as _Any, Iterator as _Iterator

    import rdflib as _rdflib
    from mloda.user import (
        DataAccessCollection as _DataAccessCollection,
        Feature as _Feature,
        Options as _Options,
        mloda as _mloda,
    )
    from open_kgo.compute_frameworks.python_dict_kg_framework import (
        KgPythonDictFramework as _KgPythonDictFramework,
    )

    from open_kgo.feature_groups.kg.rdf.rdflib_sparql import (  # noqa: F401
        RdfLibSparqlFeatureGroup as _RdfLibSparqlFeatureGroup,
    )

    REPO_NS = _rdflib.Namespace("http://example.org/repo/")

    _IMPORT_RE = _re.compile(
        r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
        _re.MULTILINE,
    )

    def walk_repo(root: _Path) -> _Iterator[tuple[_Path, str]]:
        """Yield (relative_path, kind) for every entry under `root`,
        skipping dot-prefixed components and `__pycache__`.
        """
        for p in root.rglob("*"):
            rel = p.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            if p.is_dir():
                yield rel, "dir"
            elif p.is_file():
                yield rel, "file"

    def parse_imports(text: str) -> _Iterator[str]:
        """Yield the top-level module name from each import line.

        Relative imports (`from . import x`, `from .X import y`) and bare
        leading dots are skipped because their top-level name is empty.
        """
        for m in _IMPORT_RE.finditer(text):
            target = m.group(1) or m.group(2) or ""
            top = target.split(".")[0]
            if top:
                yield top

    def build_graph_to_ttl(root: _Path) -> tuple[_Path, dict[str, int]]:
        """Walk `root`, build an rdflib Graph, serialize to a temp .ttl
        file, and return (ttl_path, counts).
        """
        g = _rdflib.Graph()
        g.bind("repo", REPO_NS)

        root_uri = REPO_NS["root"]
        g.add((root_uri, _rdflib.RDF.type, REPO_NS.Directory))
        g.add((root_uri, REPO_NS.hasName, _rdflib.Literal(root.name)))
        g.add((root_uri, REPO_NS.hasPath, _rdflib.Literal(".")))

        def _dir_uri(rel: _Path) -> _Any:
            if str(rel) == ".":
                return root_uri
            return REPO_NS[f"dir/{rel.as_posix()}"]

        def _file_uri(rel: _Path) -> _Any:
            return REPO_NS[f"file/{rel.as_posix()}"]

        def _module_uri(name: str) -> _Any:
            return REPO_NS[f"module/{name}"]

        files = 0
        dirs = 0
        modules: set[str] = set()
        imports = 0

        py_files: list[tuple[_Path, _Any]] = []

        for rel, kind in walk_repo(root):
            parent_uri = _dir_uri(rel.parent)
            if kind == "dir":
                dirs += 1
                u = _dir_uri(rel)
                g.add((u, _rdflib.RDF.type, REPO_NS.Directory))
                g.add((u, REPO_NS.hasName, _rdflib.Literal(rel.name)))
                g.add((u, REPO_NS.hasPath, _rdflib.Literal(rel.as_posix())))
                g.add((parent_uri, REPO_NS.contains, u))
            else:
                files += 1
                u = _file_uri(rel)
                g.add((u, _rdflib.RDF.type, REPO_NS.File))
                g.add((u, REPO_NS.hasName, _rdflib.Literal(rel.name)))
                g.add((u, REPO_NS.hasPath, _rdflib.Literal(rel.as_posix())))
                g.add((u, REPO_NS.hasExt, _rdflib.Literal(rel.suffix)))
                g.add((parent_uri, REPO_NS.contains, u))
                if rel.suffix == ".py":
                    py_files.append((rel, u))

        for rel, _ in py_files:
            parts = list(rel.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]
            if not parts:
                continue
            mod_name = ".".join(parts)
            modules.add(mod_name)
            mu = _module_uri(mod_name)
            g.add((mu, _rdflib.RDF.type, REPO_NS.Module))
            g.add((mu, REPO_NS.hasName, _rdflib.Literal(mod_name)))

            try:
                text = (root / rel).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for target in set(parse_imports(text)):
                tu = _module_uri(target)
                g.add((tu, _rdflib.RDF.type, REPO_NS.Module))
                g.add((tu, REPO_NS.hasName, _rdflib.Literal(target)))
                g.add((mu, REPO_NS.imports, tu))
                modules.add(target)
                imports += 1

        fd, ttl_path_str = _tempfile.mkstemp(prefix="kg_repo_", suffix=".ttl")
        _os.close(fd)
        ttl_path = _Path(ttl_path_str)
        g.serialize(destination=str(ttl_path), format="turtle")

        counts = {
            "dirs": dirs,
            "files": files,
            "modules": len(modules),
            "imports": imports,
            "triples": len(g),
        }
        return ttl_path, counts

    def run_sparql(ttl_path: _Path, query_text: str) -> list[dict]:
        dac = _DataAccessCollection(
            credentials=[
                {
                    "rdflib_sparql": {
                        "locator": str(ttl_path),
                        "result_format": "application/sparql-results+json",
                        "reasoning_profile": "none",
                        "result_limit": 1000,
                    }
                }
            ]
        )
        feat = _Feature(
            "repo_kg_query",
            options=_Options(context={"query_text": query_text}),
        )
        partitions = _mloda.run_all(
            [feat],
            compute_frameworks={_KgPythonDictFramework},
            data_access_collection=dac,
        )
        return [row[feat.name] for partition in partitions for row in partition if feat.name in row]

    return build_graph_to_ttl, run_sparql


@app.cell
def build(build_graph_to_ttl, mo):
    from pathlib import Path as _Path

    import open_kgo as _mp

    _root = _Path(_mp.__file__).resolve().parent.parent
    _ttl_path, _counts = build_graph_to_ttl(_root)

    ttl_path = _ttl_path

    mo.output.replace(
        mo.md(
            f"""
    ### Graph built from `{_root.name}/`

    | Metric | Value |
    |---|---|
    | Directories | {_counts["dirs"]:,} |
    | Files | {_counts["files"]:,} |
    | Python modules (defined or imported) | {_counts["modules"]:,} |
    | Import edges | {_counts["imports"]:,} |
    | Total triples | {_counts["triples"]:,} |

    Serialized to `{_ttl_path}` (Turtle).

    All five queries below run through the `RdfLibSparqlReader` from
    `feature_groups/kg/rdf/`, the same Reader demo 1 used against the
    `sample.ttl` fixture. Only the locator changed.
    """
        )
    )
    return (ttl_path,)


@app.cell
def query1(mo, run_sparql, ttl_path):
    _q = """
    PREFIX repo: <http://example.org/repo/>
    SELECT ?name WHERE {
        repo:root repo:contains ?d .
        ?d a repo:Directory ;
           repo:hasName ?name .
    }
    ORDER BY ?name
    """
    _rows = run_sparql(ttl_path, _q)
    _bullets = "\n".join(f"- `{r['name']}`" for r in _rows)
    mo.output.replace(
        mo.md(
            f"""
    ### Query 1: top-level directories

    ```sparql
    SELECT ?name WHERE {{
        repo:root repo:contains ?d .
        ?d a repo:Directory ; repo:hasName ?name .
    }}
    ```

    {_bullets}
    """
        )
    )
    return


@app.cell
def query2(mo, run_sparql, ttl_path):
    _q = """
    PREFIX repo: <http://example.org/repo/>
    SELECT ?path (COUNT(?f) AS ?n) WHERE {
        ?d a repo:Directory ;
           repo:hasPath ?path ;
           repo:contains ?f .
        ?f a repo:File .
    }
    GROUP BY ?path
    ORDER BY DESC(?n)
    LIMIT 10
    """
    _rows = run_sparql(ttl_path, _q)
    _table_rows = "\n".join(f"| `{r['path']}` | {r['n']} |" for r in _rows)
    mo.output.replace(
        mo.md(
            f"""
    ### Query 2: directories ranked by direct file count (top 10)

    ```sparql
    SELECT ?path (COUNT(?f) AS ?n) WHERE {{
        ?d a repo:Directory ; repo:hasPath ?path ; repo:contains ?f .
        ?f a repo:File .
    }} GROUP BY ?path ORDER BY DESC(?n) LIMIT 10
    ```

    | Directory | Files |
    |---|---|
    {_table_rows}

    *Counts direct children only (no recursion). The next query uses a
    SPARQL property path to traverse all descendants.*
    """
        )
    )
    return


@app.cell
def query3(mo, run_sparql, ttl_path):
    _q = """
    PREFIX repo: <http://example.org/repo/>
    SELECT DISTINCT ?source ?target WHERE {
        ?s repo:imports ?t .
        ?s repo:hasName ?source .
        ?t repo:hasName ?target .
        FILTER(STRSTARTS(?target, "mloda"))
    }
    ORDER BY ?source ?target
    LIMIT 30
    """
    _rows = run_sparql(ttl_path, _q)
    _bullets = "\n".join(f"- `{r['source']}` -> `{r['target']}`" for r in _rows)
    mo.output.replace(
        mo.md(
            f"""
    ### Query 3: which modules import anything starting with `mloda`?

    ```sparql
    SELECT DISTINCT ?source ?target WHERE {{
        ?s repo:imports ?t .
        ?s repo:hasName ?source .
        ?t repo:hasName ?target .
        FILTER(STRSTARTS(?target, "mloda"))
    }} LIMIT 30
    ```

    {_bullets if _bullets else "*(no matches)*"}
    """
        )
    )
    return


@app.cell
def query4(mo, run_sparql, ttl_path):
    _q = """
    PREFIX repo: <http://example.org/repo/>
    SELECT ?target (COUNT(?s) AS ?n) WHERE {
        ?s repo:imports ?t .
        ?t repo:hasName ?target .
    }
    GROUP BY ?target
    ORDER BY DESC(?n)
    LIMIT 10
    """
    _rows = run_sparql(ttl_path, _q)
    _table_rows = "\n".join(f"| `{r['target']}` | {r['n']} |" for r in _rows)
    mo.output.replace(
        mo.md(
            f"""
    ### Query 4: top 10 most-imported modules

    ```sparql
    SELECT ?target (COUNT(?s) AS ?n) WHERE {{
        ?s repo:imports ?t .
        ?t repo:hasName ?target .
    }} GROUP BY ?target ORDER BY DESC(?n) LIMIT 10
    ```

    | Module | Importers |
    |---|---|
    {_table_rows}

    *A blunt popularity ranking. The ontology layer Manoj is building
    next can refine this with typed predicates: `repo:imports_runtime`
    vs `repo:imports_test`, optional vs required, third-party vs
    first-party, etc.*
    """
        )
    )
    return


@app.cell
def query5(mo, run_sparql, ttl_path):
    _q = """
    PREFIX repo: <http://example.org/repo/>
    SELECT ?path WHERE {
        <http://example.org/repo/dir/open_kgo/feature_groups/kg> repo:contains+ ?f .
        ?f a repo:File ;
           repo:hasPath ?path .
    }
    ORDER BY ?path
    """
    _rows = run_sparql(ttl_path, _q)
    _bullets = "\n".join(f"- `{r['path']}`" for r in _rows)
    mo.output.replace(
        mo.md(
            f"""
    ### Query 5: every file under `feature_groups/kg/` (transitive)

    ```sparql
    SELECT ?path WHERE {{
        <http://example.org/repo/dir/open_kgo/feature_groups/kg> repo:contains+ ?f .
        ?f a repo:File ; repo:hasPath ?path .
    }} ORDER BY ?path
    ```

    The `repo:contains+` is a SPARQL property path that traverses any
    number of `repo:contains` hops. This is one of SPARQL's strongest
    features and has no clean equivalent in plain Cypher.

    **{len(_rows)} files:**

    {_bullets}
    """
        )
    )
    return


@app.cell
def summary(mo):
    mo.md("""
    ---

    ## Where this lands

    Two demos, two angles on the same prototype:

    - **`demo_kg_connectors.py`** walks all 9 families against shipped
      fixtures. It's the surface tour: "here's everything we cover".
    - **`demo_kg_build_repo.py`** (this notebook) goes deep on one
      family, builds a real graph from real content, and queries it
      with property paths and aggregations. It's the "yes, the
      connector layer is actually usable" demo.

    No new code in `feature_groups/kg/` for this demo. Everything happens
    in the notebook plus the existing `RdfLibSparqlReader`. The base
    Reader contract is what makes this composition cheap: any new
    backend (a real Wikidata endpoint, a GraphDB instance, an Oxigraph
    server) drops into the same shape and the same SPARQL flows
    through.

    Run with:

    ```
    uv sync --extra demo
    marimo edit demo/demo_kg_build_repo.py
    ```
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
