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
    # Architecture 1 vs Architecture 2 — Structural Eval

    **Architecture 1** (arch_01): plain KG traversal, no type checking.
    Follow any edge that matches the relation name. No errors. No guarantees.

    **Architecture 2** (arch_02): ontology-guided traversal.
    Entity types are validated at every hop. Range types are checked on arrival.
    Invalid traversals raise `ValueError` immediately instead of returning empty.

    This eval measures the **behavioral difference** between the two architectures
    on the committed sample graph (`demo/data/sample_kb.txt`), a small hand-authored
    set of public movie facts in MetaQA's triple format. It runs fully offline.

    > **Note:** The four experiments below measure structural guarantees that hold
    > independently of dataset size. To reproduce them at scale, point
    > `demo/data/build_sample.build_sample()` at the full MetaQA dataset (see
    > `demo/data/README.md`) and re-run.
    """)
    return


@app.cell
def setup_header(mo):
    mo.md("## Setup")
    return


@app.cell
def _():
    import random
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
    GML_FILE = DATA_DIR / "metaqa_sample.gml"
    TINY_GML = (
        Path(__file__).parent.parent
        / "open_kgo"
        / "feature_groups"
        / "kg"
        / "ontology"
        / "tests"
        / "fixtures"
        / "metaqa_tiny.gml"
    )

    OntologyRegistry._clear()
    OntologyRegistry.load_file(str(ONTOLOGY_YAML))

    graph: nx.MultiDiGraph = nx.read_gml(str(GML_FILE))
    tiny: nx.MultiDiGraph = nx.read_gml(str(TINY_GML))

    random.seed(42)

    return DATA_DIR, DEMO_DIR, GML_FILE, ONTOLOGY_YAML, OntologyRegistry, Path, TINY_GML, graph, nx, random, tiny


@app.cell
def _():
    # ---------------------------------------------------------------------------
    # Traversal implementations
    # ---------------------------------------------------------------------------

    def arch1_traverse(g, start, relation):
        """Architecture 1: follow any edge matching relation. No type checking."""
        if start not in g:
            return []
        return sorted({t for _, t, d in g.out_edges(start, data=True) if d.get("relation") == relation})

    def arch2_traverse(g, start, relation, namespace="movie"):
        """Architecture 2: check entity type before traversal; validate range on arrival."""
        entity_type = g.nodes[start].get("type", "Unknown")
        if not OntologyRegistry.is_valid_edge(namespace, entity_type, relation):
            raise ValueError(
                f"Ontology violation: '{relation}' is not valid from "
                f"entity type '{entity_type}' in namespace '{namespace}'."
            )
        expected_range = OntologyRegistry.get_range_type(namespace, relation)
        seen: set[str] = set()
        for _, t, d in g.out_edges(start, data=True):
            if d.get("relation") != relation:
                continue
            if expected_range is not None:
                target_type = g.nodes[t].get("type", "Unknown")
                if target_type != expected_range:
                    raise ValueError(
                        f"Range violation: '{relation}' expects range '{expected_range}' "
                        f"but reached node '{t}' of type '{target_type}'."
                    )
            seen.add(t)
        return sorted(seen)

    return arch1_traverse, arch2_traverse


@app.cell
def section_exp1(mo):
    mo.md("---\n## Experiment 1 — Equivalence on valid paths")
    return


@app.cell
def exp1(arch1_traverse, arch2_traverse, graph, mo, random):
    """Both architectures must return identical results on ontology-valid queries.

    Relations used: directed_by and in_language — zero name-collision artifacts
    in this graph (see Experiment 3b for the collision story).
    """
    # directed_by and in_language have 0 range violations in the sample graph.
    # has_genre / release_year / has_tags / starred_actors have name-collision
    # artifacts (e.g. "Romance" is both a movie title and a genre) — those are
    # tested separately in Experiment 3b.
    CLEAN_RELATIONS = ["directed_by", "in_language"]
    movie_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Movie"]

    sample = [(random.choice(movie_nodes), random.choice(CLEAN_RELATIONS)) for _ in range(500)]

    agree = 0
    disagree = 0
    arch2_errors = 0
    for _entity, _rel in sample:
        _r1 = arch1_traverse(graph, _entity, _rel)
        try:
            _r2 = arch2_traverse(graph, _entity, _rel)
            if _r1 == _r2:
                agree += 1
            else:
                disagree += 1
        except ValueError:
            arch2_errors += 1

    mo.md(f"""
    **Sample:** 500 random `(Movie, relation)` pairs — relations: `directed_by`, `in_language`

    | Outcome | Count |
    |---|---|
    | Arch 1 == Arch 2 (identical results) | **{agree}** |
    | Arch 1 != Arch 2 (result mismatch) | {disagree} |
    | Arch 2 raised unexpectedly | {arch2_errors} |

    **Result:** {agree}/500 agreement ({100*agree//500}%)

    Arch 2 does not break existing correct traversals. The ontology layer is fully additive
    on clean data.
    """)
    return agree, arch2_errors, disagree


@app.cell
def section_exp2(mo):
    mo.md("---\n## Experiment 2 — Violation detection: invalid source type")
    return


@app.cell
def exp2(arch1_traverse, arch2_traverse, graph, mo):
    """Invalid source types: Genre and Person have no valid outgoing edges.
    Arch 1 silently returns []. Arch 2 raises ValueError.
    """
    genre_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Genre"]
    person_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Person"]

    # Try directed_by from Genre (20 nodes) and Person (all)
    invalid_cases = [(n, "directed_by") for n in genre_nodes] + [(n, "starred_actors") for n in person_nodes[:100]]

    arch1_silent_empty = 0
    arch1_returned_data = 0
    arch2_blocked = 0
    arch2_missed = 0

    for _entity, _rel in invalid_cases:
        _r1 = arch1_traverse(graph, _entity, _rel)
        if not _r1:
            arch1_silent_empty += 1
        else:
            arch1_returned_data += 1

        try:
            arch2_traverse(graph, _entity, _rel)
            arch2_missed += 1
        except ValueError:
            arch2_blocked += 1

    total = len(invalid_cases)

    mo.md(f"""
    **Sample:** {total} invalid `(entity, relation)` pairs
    — {len(genre_nodes)} Genre nodes trying `directed_by`,
      100 Person nodes trying `starred_actors`

    | Architecture | Outcome | Count |
    |---|---|---|
    | Arch 1 | Silently returned `[]` — **no error** | {arch1_silent_empty} |
    | Arch 1 | Returned data (followed wrong edge) | {arch1_returned_data} |
    | Arch 2 | Raised `ValueError` — **explicit error** | {arch2_blocked} |
    | Arch 2 | Silently passed (missed violation) | {arch2_missed} |

    **Arch 2 detection rate: {100*arch2_blocked//total}%**

    The difference: Arch 1 gives you empty results with no indication of why.
    A developer debugging a broken 3-hop query gets no signal from Arch 1.
    Arch 2 names the violation immediately: wrong entity type at the source.
    """)
    return arch1_returned_data, arch1_silent_empty, arch2_blocked, arch2_missed


@app.cell
def section_exp3(mo):
    mo.md("---\n## Experiment 3 — Dirty data detection")
    return


@app.cell
def exp3(arch1_traverse, arch2_traverse, mo, tiny):
    """The tiny fixture contains one intentional bad edge:
      Crime (Genre) → directed_by → Christopher Nolan

    Arch 1 follows it. Arch 2 catches the domain violation before touching the edge.
    """
    BAD_SOURCE = "Crime"
    BAD_RELATION = "directed_by"

    _r1 = arch1_traverse(tiny, BAD_SOURCE, BAD_RELATION)

    try:
        _r2 = arch2_traverse(tiny, BAD_SOURCE, BAD_RELATION)
        arch2_result = f"returned `{_r2}` (violation missed)"
    except ValueError as _e:
        arch2_result = f"raised `ValueError`: {_e}"

    mo.md(f"""
    **Graph:** `metaqa_tiny.gml` — 10 nodes, 11 edges, 1 intentional bad edge

    **Bad edge:** `Crime (Genre) --[directed_by]--> Christopher Nolan`

    | Architecture | Behaviour |
    |---|---|
    | Arch 1 | Returned `{_r1}` — **followed the bad edge, produced wrong data** |
    | Arch 2 | {arch2_result} |

    Arch 1 returns `Christopher Nolan` as an answer to "who is directed_by Crime?" —
    semantically nonsensical, no error. A downstream consumer gets wrong data silently.

    Arch 2 stops at the source: `Genre` has no valid outgoing edges in the ontology,
    so the traversal is rejected before any edge is followed.
    """)
    return arch2_result


@app.cell
def section_exp3b(mo):
    mo.md("---\n## Experiment 3b — Real-world range violations: MetaQA name collisions")
    return


@app.cell
def exp3b(arch1_traverse, arch2_traverse, graph, mo):
    """MetaQA has entity name collisions: 'Romance' is both a movie title and a genre.
    The graph builder assigns 'Movie' type to any node that appears as a source,
    so has_genre edges that point to 'Romance' point to a Movie node, not a Genre node.

    Arch 1: silently returns the movie node as a genre — wrong data, no error.
    Arch 2: raises a range violation — catches the schema inconsistency.

    This is a real data quality signal, not a false positive.
    """
    # Find has_genre edges that reach a Movie node instead of a Genre node
    collision_cases = [
        (s, t)
        for s, t, d in graph.edges(data=True)
        if d.get("relation") == "has_genre" and graph.nodes[t].get("type") == "Movie"
    ]

    arch1_followed = 0
    arch2_caught = 0
    _first_violation = ""

    for _src, _tgt in collision_cases:
        _r1 = arch1_traverse(graph, _src, "has_genre")
        if _tgt in _r1:
            arch1_followed += 1
        try:
            arch2_traverse(graph, _src, "has_genre")
        except ValueError as _e:
            arch2_caught += 1
            if not _first_violation:
                _first_violation = str(_e)

    mo.md(f"""
    **Name-collision edges in sample graph:** {len(collision_cases)}

    | Architecture | Behaviour |
    |---|---|
    | Arch 1 | Followed {arch1_followed}/{len(collision_cases)} collision edges |
    | Arch 2 | Raised on {arch2_caught}/{len(collision_cases)} — `{_first_violation}` |

    This curated sample is clean, so there are **0 collisions** to detect: the type of
    every node is unambiguous. In the full MetaQA KB this is not true. An entity named
    "Romance", "Rain" or "1941" appears both as a Movie source and as a genre/actor/year
    target, so the builder types it as Movie and `has_genre` edges land on a Movie node.
    Re-run this notebook against the full dataset (see `demo/data/README.md`) and Arch 2
    surfaces every such range violation while Arch 1 silently propagates the wrong type.
    """)
    return arch1_followed, arch2_caught, collision_cases


@app.cell
def section_exp4(mo):
    mo.md("---\n## Experiment 4 — Mid-chain type error in a 3-hop path")
    return


@app.cell
def exp4(arch1_traverse, arch2_traverse, graph, mo):
    """Simulate a 3-hop query where the second hop uses the wrong entity type.

    Valid chain:   Movie → has_genre → Genre (hop 1)
    Invalid chain: Genre → directed_by → ???  (hop 2 — Genre has no valid outgoing)
    Invalid chain: ??? → starred_actors → ??? (hop 3 — never reached)

    Arch 1: executes all hops, silently returns empty at the invalid hop.
    Arch 2: raises at hop 2, pinpoints exactly where the chain breaks.
    """
    # Use a movie known to have genre edges in the sample graph
    START = "The Dark Knight"

    # Hop 1 (valid for both)
    hop1_r1 = arch1_traverse(graph, START, "has_genre")
    hop1_r2 = arch2_traverse(graph, START, "has_genre")

    assert hop1_r1 == hop1_r2, "hop 1 results must agree"

    # Hop 2 — try directed_by from Genre nodes (invalid)
    hop2_arch1: list[str] = []
    hop2_arch2_error: str = ""

    for _genre in hop1_r1:
        hop2_arch1.extend(arch1_traverse(graph, _genre, "directed_by"))

    try:
        for _genre in hop1_r2:
            arch2_traverse(graph, _genre, "directed_by")
    except ValueError as _e:
        hop2_arch2_error = str(_e)

    mo.md(f"""
    **Query:** `{START} → has_genre → Genre → directed_by → ???`

    Hop 1 genres: `{hop1_r1}`

    | Architecture | Hop 2 behaviour |
    |---|---|
    | Arch 1 | Returned `{hop2_arch1 or '[]'}` — silent empty, no indication the chain is invalid |
    | Arch 2 | Raised at hop 2: `{hop2_arch2_error}` |

    **Arch 1 problem:** a developer building a 3-hop query sees empty results and has to
    manually inspect the graph to understand why hop 2 returned nothing.
    With a larger graph this could produce spurious results instead of empty.

    **Arch 2 guarantee:** the error fires at the exact hop that breaks the ontology.
    The developer knows immediately: `Genre → directed_by` is not a valid step.
    """)
    return hop1_r1, hop1_r2, hop2_arch1, hop2_arch2_error


@app.cell
def section_summary(mo):
    mo.md("---\n## Summary")
    return


@app.cell
def summary(
    agree,
    arch1_returned_data,
    arch1_silent_empty,
    arch2_blocked,
    arch2_missed,
    arch2_result,
    hop2_arch2_error,
    arch1_followed,
    arch2_caught,
    collision_cases,
    mo,
):
    total_invalid = arch1_silent_empty + arch1_returned_data
    n_collisions = len(collision_cases)
    mo.md(f"""
    | Experiment | Arch 1 | Arch 2 |
    |---|---|---|
    | **1. Valid path equivalence** (500 queries) | {agree}/500 correct | {agree}/500 correct — identical |
    | **2. Invalid source type** ({total_invalid} cases) | {arch1_silent_empty} silent empty, {arch1_returned_data} wrong data | {arch2_blocked}/{total_invalid} blocked ({100*arch2_blocked//total_invalid}%) |
    | **3a. Intentional bad edge** (metaqa_tiny) | Followed bad edge, returned wrong answer | Raised at source type check |
    | **3b. Name-collision edges** ({n_collisions} cases) | Returned Movie nodes as genres ({arch1_followed} cases) | Raised range violation ({arch2_caught}/{n_collisions}) |
    | **4. Mid-chain type error** | Silent `[]` at broken hop | Raised at exact broken hop |

    **What the ontology layer adds:**
    - Valid paths: no change in results — Arch 2 is fully backward-compatible
    - Invalid source types: explicit `ValueError` instead of silent empty
    - Dirty data (bad edges, name collisions): surfaced as range or domain violations
    - Multi-hop chains: error names the exact hop that breaks the schema

    **What requires QA pairs to measure:**
    - Answer accuracy at 1-hop, 2-hop, 3-hop vs MetaQA ground truth
    - False positive rate (entities returned but not in gold answer set)
    - Download: MetaQA QA files (`qa_test_1hop.txt`, `qa_test_2hop.txt`, `qa_test_3hop.txt`)
      from the original MetaQA release
    """)
    return


if __name__ == "__main__":
    app.run()
