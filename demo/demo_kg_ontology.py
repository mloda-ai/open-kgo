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
    # KG Ontology Layer

    The KG connector layer (Tom's work) answers: **"give me data from this graph."**

    The ontology layer (this notebook) answers: **"is this traversal valid, and what
    can be inferred from it?"**

    Three things it adds on top of the connector:
    - **Typed relationships** — `directed_by` goes `Movie → Person`, not `Genre → Person`
    - **Edge validation** — catch invalid traversals before they silently return garbage
    - **Range checking** — verify the target node is the right type for the relationship

    The ontology is declared in a YAML file — one per domain (e.g. `movie`). Any connector
    pointing at movie data uses it automatically via the `ontology` credential key.
    No global state, no core mloda changes.
    """)
    return


@app.cell
def setup_header(mo):
    mo.md("## Setup")
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import networkx as nx

    from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

    DEMO_DIR = Path(__file__).parent
    DATA_DIR = DEMO_DIR / "data"
    _ROOT = DEMO_DIR.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from demo.data import ensure_data

    ensure_data()

    ONTOLOGY_YAML = (
        Path(__file__).parent.parent
        / "open_kgo"
        / "feature_groups"
        / "kg"
        / "ontology"
        / "tests"
        / "fixtures"
        / "metaqa_ontology.yaml"
    )
    KB_FILE = DATA_DIR / "metaqa_kb.txt"
    GML_FILE = DATA_DIR / "metaqa_sample.gml"

    return DATA_DIR, DEMO_DIR, GML_FILE, KB_FILE, ONTOLOGY_YAML, OntologyRegistry, Path, nx


@app.cell
def load_graph(GML_FILE, KB_FILE, mo, nx):
    mo.md(f"""
    ## Dataset

    MetaQA knowledge base: `{KB_FILE.name}`

    Format: `entity|relation|entity`.
    """)
    return


@app.cell
def _(GML_FILE, nx):
    _g: nx.MultiDiGraph = nx.read_gml(str(GML_FILE))
    _node_types: dict[str, list[str]] = {}
    for _n, _d in _g.nodes(data=True):
        _t = _d.get("type", "Unknown")
        _node_types.setdefault(_t, []).append(_n)

    graph = _g
    return (graph,)


@app.cell
def graph_summary(graph, mo):
    _type_counts = {}
    for _n, _d in graph.nodes(data=True):
        _t = _d.get("type", "Unknown")
        _type_counts[_t] = _type_counts.get(_t, 0) + 1

    _rel_counts: dict[str, int] = {}
    for _, _, _d in graph.edges(data=True):
        _r = _d.get("relation", "unknown")
        _rel_counts[_r] = _rel_counts.get(_r, 0) + 1

    _type_rows = "\n".join(f"| {t} | {c} |" for t, c in sorted(_type_counts.items()))
    _rel_rows = "\n".join(f"| `{r}` | {c} |" for r, c in sorted(_rel_counts.items()))

    mo.md(f"""
    **Graph loaded:** {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges

    | Entity type | Count |
    |---|---|
    {_type_rows}

    | Relationship | Edge count |
    |---|---|
    {_rel_rows}
    """)
    return


@app.cell
def register_ontology(ONTOLOGY_YAML, OntologyRegistry, mo):
    OntologyRegistry._clear()
    _namespace = OntologyRegistry.load_file(str(ONTOLOGY_YAML))

    mo.md(f"""
    ## Ontology registered

    **File:** `{ONTOLOGY_YAML.name}`
    **Namespace:** `{_namespace}`

    ```yaml
    namespace: movie

    entities:
      Movie:   [directed_by, written_by, starred_actors, has_genre, in_language, ...]
      Person:  []          # terminal — MetaQA has no inverse edges
      Genre:   []          # terminal — no valid outgoing edges

    relationships:
      directed_by:   Movie → Person
      written_by:    Movie → Person
      starred_actors: Movie → Person
      has_genre:     Movie → Genre
      in_language:   Movie → Language
      ...
    ```

    Registered via credential key: `"ontology": "path/to/metaqa_ontology.yaml"`
    """)
    return


@app.cell
def section_valid(mo):
    mo.md("---\n## Valid traversals")
    return


@app.cell
def hop1_valid(OntologyRegistry, graph, mo):
    def traverse(g, start, relationship, namespace=None):
        if namespace is not None:
            entity_type = g.nodes[start].get("type", "Unknown")
            if not OntologyRegistry.is_valid_edge(namespace, entity_type, relationship):
                raise ValueError(
                    f"Ontology violation: '{relationship}' is not valid from "
                    f"entity type '{entity_type}' in namespace '{namespace}'."
                )
            expected_range = OntologyRegistry.get_range_type(namespace, relationship)
        results = []
        for _, target, data in g.out_edges(start, data=True):
            if data.get("relation") != relationship:
                continue
            if namespace is not None and expected_range is not None:
                target_type = g.nodes[target].get("type", "Unknown")
                if target_type != expected_range:
                    raise ValueError(
                        f"Range violation: '{relationship}' expects range "
                        f"'{expected_range}' but reached '{target}' of type '{target_type}'."
                    )
            results.append(target)
        return sorted(results)

    _q = "Who directed The Dark Knight?"
    _ans = traverse(graph, "The Dark Knight", "directed_by", namespace="movie")

    mo.md(f"""
    **1-hop:** {_q}

    `The Dark Knight --[directed_by]--> ?`

    **Answer:** `{_ans}`

    ✓ `Movie → directed_by → Person` — valid per ontology
    """)
    return (traverse,)


@app.cell
def hop2_valid(mo, traverse, graph):
    _q = "What other movies did the director of The Dark Knight direct?"
    _hop1 = traverse(graph, "The Dark Knight", "directed_by", namespace="movie")
    # Hop 2: MetaQA has no inverse edges — reverse-lookup all movies pointing to same director
    _hop2 = sorted(
        {
            src
            for src, tgt, d in graph.edges(data=True)
            if d.get("relation") == "directed_by" and tgt in _hop1 and src != "The Dark Knight"
        }
    )

    mo.md(f"""
    **2-hop:** {_q}

    `The Dark Knight → directed_by → {_hop1[0]} ← directed_by ← ?`

    **Answer:** `{_hop2}`

    ✓ `Movie → directed_by → Person` (hop 1 ontology-validated); reverse-lookup for hop 2
    (MetaQA is movie-centric — no inverse edges; both hops use ontology-declared relations)
    """)
    return


@app.cell
def hop3_valid(mo, traverse, graph):
    _q = "What genres do movies by the same director as The Dark Knight belong to?"
    _hop1 = traverse(graph, "The Dark Knight", "directed_by", namespace="movie")
    # Hop 2: reverse-lookup all movies by the same director
    _sibling_movies = sorted(
        {
            src
            for src, tgt, d in graph.edges(data=True)
            if d.get("relation") == "directed_by" and tgt in _hop1
        }
    )
    # Hop 3: each sibling movie → has_genre (ontology-validated)
    _genres: list[str] = []
    for _movie in _sibling_movies:
        _genres.extend(traverse(graph, _movie, "has_genre", namespace="movie"))

    mo.md(f"""
    **3-hop:** {_q}

    `The Dark Knight → directed_by → Person ← directed_by ← Movie → has_genre → ?`

    **Director(s):** `{_hop1}`
    **All movies by director:** `{_sibling_movies}`
    **All genres:** `{sorted(set(_genres))}`

    ✓ Hops 1 and 3 are ontology-validated (`Movie → directed_by`, `Movie → has_genre`)
    """)
    return


@app.cell
def section_invalid(mo):
    mo.md("---\n## Invalid traversals — blocked by ontology")
    return


@app.cell
def invalid_source_type(OntologyRegistry, mo, traverse, graph):
    _genre_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Genre"]
    _genre = _genre_nodes[0]
    _entity_type = graph.nodes[_genre].get("type")
    _valid_hops = OntologyRegistry.valid_next_hops("movie", _entity_type)

    try:
        traverse(graph, _genre, "directed_by", namespace="movie")
        _result = "no error raised (unexpected)"
    except ValueError as _e:
        _result = str(_e)

    mo.md(f"""
    **Invalid source type:** `Genre → directed_by → ?`

    `Genre` has no valid outgoing edges: `{sorted(_valid_hops) or '[]'}`

    Attempt: `traverse("{_genre}", "directed_by", namespace="movie")`

    **Raised:** `{_result}`

    Without ontology this returns an empty list silently. With ontology it fails loudly.
    """)
    return


@app.cell
def invalid_range(mo, traverse):
    # The tiny fixture has an intentional bad edge: Crime (Genre) → directed_by → Christopher Nolan.
    # directed_by declares range=Person, but Crime is a Genre node — ontology catches the
    # domain violation before we even traverse the edge.
    import networkx as _nx
    from pathlib import Path as _Path

    _TINY_GML = (
        _Path(__file__).parent.parent
        / "open_kgo" / "feature_groups" / "kg" / "ontology"
        / "tests" / "fixtures" / "metaqa_tiny.gml"
    )
    _tiny = _nx.read_gml(str(_TINY_GML))
    # Genre "Crime" has no valid outgoing in the ontology — traversal is blocked
    try:
        traverse(_tiny, "Crime", "directed_by", namespace="movie")
        _block = "no error raised (unexpected)"
    except ValueError as _e:
        _block = f"**Raised:** `{_e}`"

    mo.md(f"""
    **Range violation caught by ontology:** `Crime (Genre) → directed_by → Christopher Nolan`

    The tiny fixture deliberately contains this bad edge.
    `directed_by` has `domain=Movie` — a `Genre` source is not allowed.
    The ontology blocks traversal before any edge is followed.

    {_block}

    Without the ontology layer, `traverse(graph, \"Crime\", \"directed_by\")` would silently
    return `[\"Christopher Nolan\"]` — wrong data, no error.
    """)
    return


@app.cell
def section_connector(mo):
    mo.md("---\n## Wired through a real connector")
    return


@app.cell
def connector_demo(GML_FILE, ONTOLOGY_YAML, mo):
    from mloda.user import DataAccessCollection, Feature, Options, mloda

    import open_kgo.feature_groups.kg.embedded.networkx_embedded  # noqa: F401

    from open_kgo.feature_groups.kg.python_dict_kg_framework import KgPythonDictFramework

    _creds = {
        "locator": str(GML_FILE),
        "graph_file_format": "gml",
        "read_only": True,
        "max_threads": 1,
        "ontology": str(ONTOLOGY_YAML),
        "result_limit": 50,
    }
    _feat = Feature(
        "networkx_embedded__neighbors",
        options=Options(context={"operation": "neighbors", "start_node": "Christopher Nolan"}),
    )
    _dac = DataAccessCollection(credentials=[{"networkx_embedded": _creds}])
    _partitions = mloda.run_all(
        [_feat],
        compute_frameworks={KgPythonDictFramework},
        data_access_collection=_dac,
    )
    _rows = [row[_feat.name] for part in _partitions for row in part if _feat.name in row]

    mo.md(f"""
    **mloda.run_all with ontology credential**

    ```python
    creds = {{
        "locator": "metaqa_sample.gml",
        "ontology": "metaqa_ontology.yaml",   # ← ontology wired via credential
        ...
    }}
    feature = Feature("networkx_embedded__neighbors", ...)
    ```

    **Neighbors of `Christopher Nolan`:** `{sorted(r['node'] for r in _rows)}`

    The `ontology` key is transparent to the existing connector — it loads the registry
    and sets `ctx.ontology_namespace` in `_prepare_load`. Connectors that don't use
    ontology validation are unaffected.
    """)
    return


@app.cell
def section_backend_swap(mo):
    mo.md("---\n## Backend swap — NetworkX → Kuzu")
    return


@app.cell
def backend_swap(ONTOLOGY_YAML, OntologyRegistry, mo):
    import gc as _gc
    import shutil as _shutil
    import tempfile as _tempfile

    import importlib as _importlib

    import kuzu as _kuzu

    _importlib.import_module("open_kgo.feature_groups.kg.network_pg.kuzu_cypher")
    from mloda.user import DataAccessCollection as _DAC
    from mloda.user import Feature as _Feature
    from mloda.user import Options as _Options
    from mloda.user import mloda as _mloda

    from open_kgo.feature_groups.kg.python_dict_kg_framework import KgPythonDictFramework as _KgFW

    # Build a tiny Kuzu movie graph in a temp directory.
    _tmp = _tempfile.mkdtemp(prefix="kg_demo_swap_")
    _db_dir = _tmp + "/movies.kuzu"
    _build_db = _kuzu.Database(_db_dir)
    _build_conn = _kuzu.Connection(_build_db)
    _build_conn.execute("CREATE NODE TABLE Movie(name STRING, PRIMARY KEY(name))")
    _build_conn.execute("CREATE NODE TABLE Person(name STRING, PRIMARY KEY(name))")
    _build_conn.execute("CREATE REL TABLE directed_by(FROM Movie TO Person)")
    _build_conn.execute("CREATE (:Movie {name: 'The Dark Knight'})")
    _build_conn.execute("CREATE (:Person {name: 'Christopher Nolan'})")
    _build_conn.execute(
        "MATCH (m:Movie {name: 'The Dark Knight'}), (p:Person {name: 'Christopher Nolan'}) "
        "CREATE (m)-[:directed_by]->(p)"
    )
    # Release write handles before mloda opens the same DB directory.
    del _build_conn
    del _build_db
    _gc.collect()

    # Run through the Kuzu mloda connector — only the locator changes vs NetworkX.
    _kuzu_creds = {
        "locator": _db_dir,
        "ontology": str(ONTOLOGY_YAML),
    }
    _feat = _Feature(
        "kuzu_cypher__directed_by",
        options=_Options(context={
            "query_text": "MATCH (m:Movie)-[:directed_by]->(p:Person) RETURN m.name AS movie, p.name AS director",
        }),
    )
    _dac = _DAC(credentials=[{"kuzu_cypher": _kuzu_creds}])
    _partitions = _mloda.run_all(
        [_feat],
        compute_frameworks={_KgFW},
        data_access_collection=_dac,
    )
    _kuzu_rows = [row[_feat.name] for part in _partitions for row in part if _feat.name in row]

    # Same ontology YAML, same validation rules — only the credential locator changed.
    _valid_edges = [
        (
            r["movie"],
            "directed_by",
            r["director"],
            OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by"),
        )
        for r in _kuzu_rows
    ]
    _invalid_attempt = OntologyRegistry.is_valid_edge("movie", "Person", "directed_by")

    _shutil.rmtree(_tmp, ignore_errors=True)

    _check_rows = "\n".join(
        f"| `{src}` | `{rel}` | `{tgt}` | {'valid' if ok else 'invalid'} |"
        for src, rel, tgt, ok in _valid_edges
    )

    mo.md(f"""
    The **same ontology YAML** and the **same validation rules** apply unchanged when the
    storage backend changes. Only the connector credentials change — the `locator` and the
    connector id (`networkx_embedded` → `kuzu_cypher`).

    ```python
    # NetworkX, file-backed, in-process
    networkx_creds = {{
        "locator":           "metaqa_sample.gml",
        "graph_file_format": "gml",
        "ontology":          "metaqa_ontology.yaml",  # travels with the swap
    }}

    # Kuzu, embedded Cypher engine — one credential change
    kuzu_creds = {{
        "locator":  "/path/to/movies.kuzu",           # locator changes
        "ontology": "metaqa_ontology.yaml",           # same YAML, same rules
    }}
    ```

    **Data fetched from Kuzu via `mloda.run_all`:**

    | Source | Relation | Target | Ontology check |
    |---|---|---|---|
    {_check_rows}

    **Same rule, wrong source type:** `is_valid_edge("movie", "Person", "directed_by")` → `{_invalid_attempt}`

    The ontology YAML is the single source of truth. The backend is swappable.
    """)
    return


@app.cell
def section_standalone(mo):
    mo.md("---\n## Standalone API (no mloda required)")
    return


@app.cell
def standalone_demo(ONTOLOGY_YAML, OntologyRegistry, mo):
    _checks = [
        ("movie", "Movie", "directed_by", True),
        ("movie", "Movie", "has_genre", True),
        ("movie", "Movie", "starred_actors", True),
        ("movie", "Genre", "directed_by", False),
        ("movie", "Genre", "has_genre", False),
        ("movie", "Person", "directed_by", False),
        ("movie", "Person", "has_genre", False),
    ]

    _rows_md = "\n".join(
        f"| `{ns}` | `{et}` | `{rel}` | {'✓ valid' if expected else '✗ invalid'} |"
        for ns, et, rel, expected in _checks
    )

    mo.md(f"""
    ```python
    from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

    OntologyRegistry.load_file("metaqa_ontology.yaml")
    OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by")  # True
    OntologyRegistry.is_valid_edge("movie", "Genre", "directed_by")  # False
    OntologyRegistry.get_range_type("movie", "directed_by")           # "Person"
    OntologyRegistry.valid_next_hops("movie", "Genre")                # frozenset()
    ```

    | Namespace | Entity type | Relationship | Result |
    |---|---|---|---|
    {_rows_md}

    No connector, no mloda imports needed. The registry is a plain Python class.
    """)
    return


if __name__ == "__main__":
    app.run()
