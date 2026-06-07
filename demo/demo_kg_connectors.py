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
    # KG Connector Base: 9 Families, 5 Groups

    A tour of the knowledge-graph connector prototype for mloda. The goal is a
    **connector** layer that talks to existing KG systems (Neo4j, Neptune,
    RDF stores, public REST APIs, lineage graphs, ...), not a new database.

    Each family ships:

    - A `Reader` adapter (`connect`, `build_query`, `load_data`, validation)
    - A `FeatureGroup` class wired to the Reader
    - A per-family contract test base plus a universal contract test base
    - An in-memory or file fixture, so tests run with **no Docker, no network**

    This notebook walks through all 9 families across 5 groups and runs one
    canonical operation per family. The base set is intentionally minimal.
    The ontology layer (typed relationships, edge validation, implicit
    connections) is the next layer up and lives outside this prototype.
    """)
    return


@app.cell
def overview(mo):
    mo.md("""
    ## Five groups, nine families

    | # | Group | Address shape | Query shape | Real-world examples |
    |---|---|---|---|---|
    | 1 | Network property-graph | endpoint URL + `dataset` | vendor query (Cypher / Gremlin / GSQL) | Neo4j, Neptune, Memgraph, TigerGraph |
    | 2 | RDF / SPARQL | endpoint URL + repository | SPARQL + named graphs | GraphDB, Wikidata, Fuseki, Stardog |
    | 3 | Embedded / in-memory | filesystem-path `locator` or class id | in-process API or local Cypher | KuzuDB, NetworkX, RDFLib, igraph |
    | 4 | REST non-SPARQL public | endpoint URL + `entity_type` | filter params + pagination | OpenAlex, STRING, ConceptNet |
    | 5 | Hidden / relationship-expansion | `tenant` + relation expand | `$expand` / direction+depth | DataHub, OpenFGA, SpiceDB, Notion |

    Group 5 fans out into **five sub-families**: `lineage`, `code_build`,
    `saas_authz`, `agent_memory`, `citation_rest`. That brings the total to
    **9 families**. The split mirrors the property layout each system
    actually needs (e.g. lineage walks differ from authorization tuples
    differ from citation hierarchies).
    """)
    return


@app.cell
def helpers():
    from pathlib import Path as _Path
    from typing import Any as _Any

    from mloda.user import (
        DataAccessCollection as _DataAccessCollection,
        Feature as _Feature,
        mloda as _mloda,
    )
    from open_kgo.feature_groups.kg.python_dict_kg_framework import (
        KgPythonDictFramework as _KgPythonDictFramework,
    )

    def run_query(connector_id: str, slot_creds: dict, feature: _Feature) -> _Any:
        """Run a feature through `mloda.run_all` against a single KG connector.

        mloda matches the reader by calling `is_valid_credentials` on each
        `ReadDB` subclass and selecting the one whose `CONNECTOR_ID` slot is
        present in `credentials`. The KG FG base pins `compute_framework_rule`
        to `KgPythonDictFramework`, the KG-aware adapter that wraps native rows
        as `{feature_name: row}` during column slicing; we flat-concat across
        partitions for the per-cell rendering.
        """
        dac = _DataAccessCollection(credentials=[{connector_id: slot_creds}])
        partitions = _mloda.run_all(
            [feature],
            compute_frameworks={_KgPythonDictFramework},
            data_access_collection=dac,
        )
        return [row[feature.name] for partition in partitions for row in partition if feature.name in row]

    def fixture_for(module: _Any, *parts: str) -> _Any:
        base = _Path(module.__file__).parent / "tests" / "fixtures"
        return base.joinpath(*parts) if parts else base

    return fixture_for, run_query


@app.cell
def group1_header(mo):
    mo.md("""
    ---

    ## Group 1: Network property-graph

    **Use case:** the dominant operational KG shape. Persistent service,
    vendor query language, typed nodes/edges, schema-on-write or
    schema-on-read. This is what teams reach for when "knowledge graph"
    means "a graph database with Cypher / Gremlin / GSQL".

    **Prototype implementation:** `KuzuCypherReader`. Embedded Kuzu
    database, real Cypher engine, exercises the Neo4j-shaped property
    layout against a temp directory. Real Neo4j and Neptune slot in here
    with the same property contract.
    """)
    return


@app.cell
def network_pg_demo(mo, run_query):
    import shutil as _shutil
    import tempfile as _tempfile
    from pathlib import Path as _Path

    import kuzu as _kuzu
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.network_pg.kuzu_cypher as _kuzu_mod  # noqa: F401  register reader

    _tmp = _Path(_tempfile.mkdtemp(prefix="kg_demo_kuzu_"))
    _db_dir = _tmp / "graph.kuzu"
    _db = _kuzu.Database(str(_db_dir))
    _conn = _kuzu.Connection(_db)
    _conn.execute("CREATE NODE TABLE Person(name STRING, PRIMARY KEY (name))")
    _conn.execute("CREATE (:Person {name: 'Alice'})")
    _conn.execute("CREATE (:Person {name: 'Bob'})")
    _conn.execute("CREATE (:Person {name: 'Carol'})")

    _creds = {
        "locator": str(_db_dir),
        "dataset": "default",
        "read_consistency": "read",
        "transaction_mode": "auto",
        "result_limit": 100,
    }
    _feat = _Feature(
        "kuzu_cypher__list_persons",
        options=_Options(context={"query_text": "MATCH (p:Person) RETURN p.name"}),
    )
    _rows = run_query("kuzu_cypher", _creds, _feat)
    _names = sorted(r["p.name"] for r in _rows)

    _shutil.rmtree(_tmp, ignore_errors=True)

    mo.output.replace(
        mo.md(
            f"""
    **Reader:** `KuzuCypherReader`

    **Cypher:** `MATCH (p:Person) RETURN p.name`

    **Returned `{len(_names)}` rows:** `{_names}`

    *The same property layout slots into Neo4j / Neptune / Memgraph; the
    only thing that changes is the concrete Reader subclass.*
    """
        )
    )
    return


@app.cell
def group2_header(mo):
    mo.md("""
    ---

    ## Group 2: RDF / SPARQL

    **Use case:** semantic-web stack. Triples, IRIs, named graphs,
    OWL/RDFS reasoning. Distinct enough from property-graphs to deserve
    its own family: the addressing model uses `repository`, the query
    language is SPARQL, and reasoning profiles are first-class.

    **Prototype implementation:** `RdfLibSparqlReader`. In-memory rdflib
    Graph, parses Turtle / N-Triples / RDF-XML, runs SPARQL. Wikidata,
    DBpedia, GraphDB, Stardog slot in via the same property contract.
    """)
    return


@app.cell
def rdf_demo(fixture_for, mo, run_query):
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.rdf.rdflib_sparql as _rdf_mod

    _creds = {
        "locator": str(fixture_for(_rdf_mod, "sample.ttl")),
        "result_format": "application/sparql-results+json",
        "reasoning_profile": "none",
        "result_limit": 100,
    }
    _feat = _Feature(
        "rdflib_sparql__select_knows",
        options=_Options(
            context={
                "query_text": (
                    "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?s ?o WHERE { ?s foaf:knows ?o } LIMIT 10"
                ),
            }
        ),
    )
    _rows = run_query("rdflib_sparql", _creds, _feat)

    _edges = [f"`{r['s']}` knows `{r['o']}`" for r in _rows]

    mo.output.replace(
        mo.md(
            f"""
    **Reader:** `RdfLibSparqlReader`

    **SPARQL:** `SELECT ?s ?o WHERE {{ ?s foaf:knows ?o }}`

    **`foaf:knows` triples in `sample.ttl`:**

    {chr(10).join(f"- {e}" for e in _edges)}
    """
        )
    )
    return


@app.cell
def group3_header(mo):
    mo.md("""
    ---

    ## Group 3: Embedded / in-memory

    **Use case:** no service, no auth, no network. Loaded from a file or
    pickled in process. Fastest path for analytical workloads and the
    natural "base" implementation for this prototype because it has the
    smallest surface area, so it's the family Manoj will build the
    ontology layer on top of first.

    **Prototype implementation:** `NetworkxEmbeddedReader`. Loads a
    `.gml` / `.graphml` / edgelist file into a NetworkX graph, runs
    `nodes`, `edges`, or `neighbors(start_node)` operations. KuzuDB
    (file mode), igraph, RDFLib in-memory belong here.
    """)
    return


@app.cell
def embedded_demo(fixture_for, mo, run_query):
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.embedded.networkx_embedded as _embedded_mod

    _creds = {
        "locator": str(fixture_for(_embedded_mod, "triangle.gml")),
        "graph_file_format": "gml",
        "read_only": True,
        "max_threads": 1,
        "result_limit": 100,
    }
    _feat = _Feature(
        "networkx_embedded__neighbors",
        options=_Options(context={"operation": "neighbors", "start_node": "alice"}),
    )
    _rows = run_query("networkx_embedded", _creds, _feat)

    mo.output.replace(
        mo.md(
            f"""
    **Reader:** `NetworkxEmbeddedReader`

    **Graph:** `triangle.gml` (alice, bob, carol all connected)

    **Operation:** `neighbors("alice")`

    **Result:** `{sorted(r["node"] for r in _rows)}`
    """
        )
    )
    return


@app.cell
def embedded_build_demo(mo, run_query):
    import shutil as _shutil
    import tempfile as _tempfile
    from pathlib import Path as _Path

    import networkx as _nx
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.embedded.networkx_embedded as _embedded_mod  # noqa: F401  register reader

    _g: _nx.Graph = _nx.Graph()
    _g.add_edges_from(
        [
            ("dataset", "loader"),
            ("loader", "feature_engineering"),
            ("feature_engineering", "model"),
            ("model", "evaluator"),
        ]
    )

    _tmp = _Path(_tempfile.mkdtemp(prefix="kg_demo_embedded_build_"))
    _gml_path = _tmp / "pipeline.gml"
    _nx.write_gml(_g, str(_gml_path))

    _creds = {
        "locator": str(_gml_path),
        "graph_file_format": "gml",
        "read_only": True,
        "max_threads": 1,
        "result_limit": 100,
    }
    _feat = _Feature(
        "networkx_embedded__pipeline_neighbors",
        options=_Options(context={"operation": "neighbors", "start_node": "loader"}),
    )
    _rows = run_query("networkx_embedded", _creds, _feat)

    _shutil.rmtree(_tmp, ignore_errors=True)

    mo.output.replace(
        mo.md(
            f"""
    **Reader:** `NetworkxEmbeddedReader` (build-then-query variant)

    **Build path:** `nx.Graph()` → `nx.write_gml(...)` → temp `.gml` →
    same reader as the cell above.

    **Operation:** `neighbors("loader")` against an in-process pipeline
    graph (`dataset — loader — feature_engineering — model — evaluator`)

    **Result:** `{sorted(r["node"] for r in _rows)}`

    *Mirrors what `network_pg_demo` does for Kuzu (build inline, then
    query). Closes the "fixture-only" gap for this family — the same
    workflow extends to GraphML or edge-list once the reader gains
    those formats.*
    """
        )
    )
    return


@app.cell
def group4_header(mo):
    mo.md("""
    ---

    ## Group 4: REST non-SPARQL public KGs

    **Use case:** public scientific / open-data KGs that expose REST APIs
    instead of a query language. You filter by `entity_type` plus a few
    parameters and walk pages. Reproducibility hinges on
    `dataset_version`, since the underlying graph drifts.

    **Prototype implementation:** `FileFixtureRestReader`. Simulates
    cursor pagination by reading `page_1.json`, `page_2.json` from disk.
    OpenAlex, STRING, ConceptNet, UniProt-REST slot in here.
    """)
    return


@app.cell
def rest_public_demo(fixture_for, mo, run_query):
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.rest_public.file_fixture_rest as _rest_mod

    _creds = {
        "locator": str(fixture_for(_rest_mod)),
        "pagination_style": "cursor",
        "rate_limit_pace": 100,
        "result_limit": 100,
    }
    _feat = _Feature(
        "file_fixture_rest__list_works",
        options=_Options(context={}),
    )
    _rows = run_query("file_fixture_rest", _creds, _feat)

    _bullets = "\n".join(f"- `{r['id']}`: {r.get('title', '')}" for r in _rows)

    mo.output.replace(
        mo.md(
            f"""
    **Reader:** `FileFixtureRestReader`

    **Pagination:** cursor (walker-internal page size); 3 rows total

    **Works returned:**

    {_bullets}
    """
        )
    )
    return


@app.cell
def group5_header(mo):
    mo.md("""
    ---

    ## Group 5: Hidden / relationship-expansion graphs

    **Use case:** systems that *are* graphs but don't market themselves
    as such. You don't write a query language, you call `$expand=...`,
    `direction=UPSTREAM`, or `relation=viewer`. Tenancy, consistency
    tokens, and bounded depth are first-class concerns.

    Group 5 splits into **5 sub-families**:

    | Sub-family | Domain | Examples |
    |---|---|---|
    | `lineage` | data/metadata lineage | dbt, DataHub, Atlas |
    | `code_build` | software supply chain | CycloneDX, SLSA, Maven |
    | `saas_authz` | authorization graphs | OpenFGA, SpiceDB, Zanzibar |
    | `agent_memory` | agent / GraphRAG stores | Graphiti, Cognee, mem0 |
    | `citation_rest` | scientific citation hierarchies | Reactome, PubMed |

    Each one demos below.
    """)
    return


@app.cell
def lineage_demo(fixture_for, mo, run_query):
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.lineage.dbt_manifest as _lineage_mod

    _creds = {
        "locator": str(fixture_for(_lineage_mod, "manifest.json")),
        "result_limit": 100,
    }
    _feat = _Feature(
        "dbt_manifest__upstream",
        options=_Options(
            context={
                "asset_urn": "model.shop.fct_orders",
                "lineage_direction": "UPSTREAM",
                "upstream_depth": 2,
                "downstream_depth": 0,
            }
        ),
    )
    _rows = run_query("dbt_manifest", _creds, _feat)

    _chain = " <- ".join(r["urn"] for r in _rows)

    mo.output.replace(
        mo.md(
            f"""
    ### Sub-family: `lineage`. dbt manifest walk

    **Reader:** `DbtManifestReader` (parses dbt's `manifest.json`)

    **Walk:** `model.shop.fct_orders` upstream, depth=2

    **Lineage chain:** `{_chain}`
    """
        )
    )
    return


@app.cell
def code_build_demo(fixture_for, mo, run_query):
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.code_build.cyclonedx_sbom as _cb_mod

    _creds = {
        "manifest_path": str(fixture_for(_cb_mod, "sample.cdx.json")),
        "commit_sha": "a1b2c3d4",
        "branch": "main",
        "language_code": "python",
        "result_limit": 100,
    }
    _feat = _Feature(
        "cyclonedx_sbom__components",
        options=_Options(context={}),
    )
    _rows = run_query("cyclonedx_sbom", _creds, _feat)

    _bullets = "\n".join(f"- `{r['name']}` ({r.get('version', '?')})" for r in _rows)

    mo.output.replace(
        mo.md(
            f"""
    ### Sub-family: `code_build`. Software supply chain

    **Reader:** `CycloneDxSbomReader` (parses CycloneDX SBOM)

    **Components in `sample.cdx.json`:**

    {_bullets}
    """
        )
    )
    return


@app.cell
def saas_authz_demo(mo, run_query):
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.saas_authz.in_process_tuple_store as _authz_mod  # noqa: F401  register reader

    _creds = {
        "tenant": "tenant_a",
        "api_version": "v1.0",
        "entity_type": "document",
        "relationship_type": "viewer",
        "consistency_mode": "minimize_latency",
        "pagination_style": "none",
        "result_limit": 100,
    }
    _feat = _Feature("in_process_tuple_store__viewers", options=_Options(context={}))
    _rows = run_query("in_process_tuple_store", _creds, _feat)

    _bullets = "\n".join(
        f"- `{r['user']}` is `{r['relation']}` on `{r['object_type']}:{r['object_id']}`" for r in _rows
    )

    mo.output.replace(
        mo.md(
            f"""
    ### Sub-family: `saas_authz`. Authorization tuple store

    **Reader:** `InProcessTupleStoreReader` (Zanzibar-shaped tuples)

    **Tenant `tenant_a`, relation = `viewer`:**

    {_bullets}

    *Real backends (OpenFGA, SpiceDB) plug in with the same property
    layout plus consistency tokens.*
    """
        )
    )
    return


@app.cell
def agent_memory_demo(fixture_for, mo, run_query):
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.agent_memory.networkx_memory as _memory_mod  # noqa: F401  register reader

    _creds = {
        "locator": str(fixture_for(_memory_mod, "memories.json")),
        "memory_scope_user_id": "user_42",
        "retrieval_mode": "lexical",
        "pagination_style": "none",
        "result_limit": 100,
        "threshold": 0.0,
        "mmr_lambda": 0.5,
    }
    _feat = _Feature(
        "networkx_memory__search",
        options=_Options(context={"query_text": "coffee"}),
    )
    _rows = run_query("networkx_memory", _creds, _feat)

    _bullets = "\n".join(f"- `{r['label']}`" for r in _rows)

    mo.output.replace(
        mo.md(
            f"""
    ### Sub-family: `agent_memory`. Agent / GraphRAG store

    **Reader:** `NetworkxMemoryReader` (lexical retrieval; vector and
    hybrid modes are stubbed and raise `NotImplementedError`)

    **Query:** `"coffee"` against the seeded memory graph

    **Hits:**

    {_bullets}
    """
        )
    )
    return


@app.cell
def citation_rest_demo(fixture_for, mo, run_query):
    from mloda.user import Feature as _Feature, Options as _Options

    import open_kgo.feature_groups.kg.citation_rest.file_fixture_citation as _cit_mod

    _creds = {
        "locator": str(fixture_for(_cit_mod, "reactome.json")),
        "species_prefix": "HSA",
        "dataset_version": "v90",
        "result_limit": 100,
    }
    _feat = _Feature(
        "file_fixture_citation__pathway",
        options=_Options(context={"stable_id": "R-HSA-1640170", "hierarchy_depth": 1}),
    )
    _rows = run_query("file_fixture_citation", _creds, _feat)

    _bullets = "\n".join(f"- `{r['stableId']}`: {r.get('name', '')}" for r in _rows)

    mo.output.replace(
        mo.md(
            f"""
    ### Sub-family: `citation_rest`. Scientific catalog hierarchy

    **Reader:** `FileFixtureCitationReader` (Reactome-style pathway walk)

    **Walk from `R-HSA-1640170`, depth=1:**

    {_bullets}

    *The same shape applies to PubMed citation walks; `dataset_version`
    pins the underlying snapshot for reproducibility.*
    """
        )
    )
    return


@app.cell
def auth_demo_header(mo):
    mo.md("""
    ---

    ## Opt-in env-var resolution: `_resolve_env`

    The universal base used to declare an `auth_method` + `auth_*_env`
    surface and strict-validate it at the matcher gate. Because none of the
    shipped concretes open a network socket, the surface was decorative —
    the framework loudly validated a credential surface no concrete read
    (issue #32 item 2). The auth keys were therefore removed from
    `_UNIVERSAL_PROPERTY_MAPPING`. The `_resolve_env` helper itself stayed:
    it is opt-in infrastructure that a future networked concrete (Wikidata
    SPARQL, GraphDB, OpenFGA, real Neo4j over Bolt) calls from its own
    `_connect_from_slot` after the family base / concrete re-introduces the
    matching `auth_*_env` companion keys.

    The cell below exercises `_resolve_env` standalone against a synthetic
    creds dict — no `is_valid_credentials` round-trip, since the universal
    base no longer declares an auth slot key.
    """)
    return


@app.cell
def auth_demo(mo):
    import os as _os

    from open_kgo.feature_groups.kg.base import KgConnectorReaderBase as _Base
    from open_kgo.feature_groups.kg.errors import (
        MissingEnvVarError as _MissingEnvVarError,
    )

    _ENV_TOKEN = "KG_DEMO_BEARER_TOKEN"
    _os.environ[_ENV_TOKEN] = "demo-bearer-abc123"
    try:
        _resolved = _Base._resolve_env({"auth_token_env": _ENV_TOKEN}, "auth_token_env")

        try:
            _Base._resolve_env({"auth_token_env": "KG_DEMO_UNSET_VAR"}, "auth_token_env")
            _missing_msg = "(no error — should have raised)"
        except _MissingEnvVarError as _e:
            _missing_msg = f"`MissingEnvVarError`: {_e}"

        _absent_returns_none = _Base._resolve_env({}, "auth_token_env") is None
    finally:
        _os.environ.pop(_ENV_TOKEN, None)

    mo.output.replace(
        mo.md(
            f"""
    **Helper:** `KgConnectorReaderBase._resolve_env(slot, key)` (opt-in)

    | Scenario | Result |
    |---|---|
    | Env var set: `_resolve_env({{"auth_token_env": "{_ENV_TOKEN}"}}, "auth_token_env")` | `{_resolved!r}` |
    | Env var unset (env-var name supplied, env value missing) | {_missing_msg} |
    | Credential key absent — caller opted out — returns `None` | `{_absent_returns_none}` |

    *Today no concrete plugin actually wires the resolved token into an
    HTTP client — every reader runs against in-memory libraries or file
    fixtures. A real Wikidata or GraphDB adapter would re-declare the
    `auth_*_env` companion keys on its own family base / concrete, call
    `_resolve_env` inside `_connect_from_slot`, and pass the value to
    `requests.get(..., headers={{"Authorization": f"Bearer {{token}}"}})`.*
    """
        )
    )
    return


@app.cell
def summary(mo):
    mo.md("""
    ---

    ## Where this lands

    All 9 families share one universal contract base plus one per-family
    contract base. Adding a real Neo4j adapter, a real Wikidata SPARQL
    endpoint, or a real OpenFGA backend means subclassing the relevant
    `Reader` and wiring 5 methods (`connect`, `build_query`, `load_data`,
    plus 2 validators). The contract suite then exercises them.

    **Next layer (not in this prototype):**

    - **Ontology layer:** typed relationships, edge validation,
      implicit-connection inference. Built on top of the family bases by
      Manoj after handoff.
    - **Semantic Fields** (concept 5): deferred; would require core
      mloda engine changes.
    - **Real backends:** Neo4j and AWS Neptune adapters slot into
      `network_pg`; Wikidata / GraphDB into `rdf`; DataHub into
      `lineage`; OpenFGA into `saas_authz`.

    Run this notebook with:

    ```
    uv sync --extra demo
    marimo edit demo/demo_kg_connectors.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
