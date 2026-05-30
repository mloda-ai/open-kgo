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
    # 1-hop QA Accuracy: Arch 1 vs Arch 2

    Measures answer accuracy on the committed sample QA set (`demo/data/sample_qa.txt`),
    a small hand-authored set of public movie facts in MetaQA's triple format.

    **Architecture 1:** plain traversal — follow any matching edge, no type checking.
    **Architecture 2:** ontology-guided — entity type validated before each hop,
    range type validated on arrival.

    Questions cover the 7 movie relations in two directions:
    - **Forward** (entity in brackets is a Movie): `Movie → relation → ?`
    - **Reverse** (entity is Person/Tag/Genre): find all Movies with that entity as target

    Metric: **hit rate** — question answered correctly if the returned set contains
    at least one gold answer.
    """)
    return


@app.cell
def _():
    import re
    import sys
    from collections import defaultdict
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
    QA_TEST = DATA_DIR / "sample_qa.txt"
    GML_FILE = DATA_DIR / "metaqa_sample.gml"

    OntologyRegistry._clear()
    OntologyRegistry.load_file(str(ONTOLOGY_YAML))

    graph: nx.MultiDiGraph = nx.read_gml(str(GML_FILE))

    return DATA_DIR, DEMO_DIR, GML_FILE, ONTOLOGY_YAML, OntologyRegistry, Path, QA_TEST, defaultdict, graph, nx, re


@app.cell
def graph_info(graph, mo):
    _types: dict[str, int] = {}
    for _, _d in graph.nodes(data=True):
        _t = _d.get("type", "Unknown")
        _types[_t] = _types.get(_t, 0) + 1
    _rows = "\n".join(f"| {t} | {c} |" for t, c in sorted(_types.items()))
    mo.md(f"""
    **QA-anchored subgraph:** {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges

    Built by `demo/data/build_sample.py` as the 1-hop induced subgraph around
    every QA topic entity. Sufficient for 1-hop accuracy: every topic and its
    1-hop neighbors (i.e. the answer space) are present.

    | Entity type | Count |
    |---|---|
    {_rows}
    """)
    return


@app.cell
def _():
    # ---------------------------------------------------------------------------
    # Traversal implementations
    # ---------------------------------------------------------------------------

    def arch1_traverse(g, start, relation):
        """Architecture 1: follow any edge matching relation. No type checking."""
        if start not in g:
            return set()
        return {t for _, t, d in g.out_edges(start, data=True) if d.get("relation") == relation}

    def arch2_traverse(g, start, relation, namespace="movie"):
        """Architecture 2: ontology-validated hop."""
        entity_type = g.nodes[start].get("type", "Unknown")
        if not OntologyRegistry.is_valid_edge(namespace, entity_type, relation):
            raise ValueError(f"Ontology violation: '{relation}' from '{entity_type}'")
        expected_range = OntologyRegistry.get_range_type(namespace, relation)
        seen: set[str] = set()
        for _, t, d in g.out_edges(start, data=True):
            if d.get("relation") != relation:
                continue
            if expected_range is not None:
                target_type = g.nodes[t].get("type", "Unknown")
                if target_type != expected_range:
                    raise ValueError(
                        f"Range violation: '{relation}' expects '{expected_range}' "
                        f"but reached '{t}' of type '{target_type}'"
                    )
            seen.add(t)
        return seen

    def reverse_traverse(g, target, relation):
        """Find all Movie nodes that point to `target` via `relation`."""
        return {s for s, t, d in g.in_edges(target, data=True) if d.get("relation") == relation}

    return arch1_traverse, arch2_traverse, reverse_traverse


@app.cell
def _():
    # ---------------------------------------------------------------------------
    # Question parser
    # ---------------------------------------------------------------------------

    def infer_relation(question: str, entity_type: str) -> tuple[str, str] | None:
        """Return (relation, direction) for a question, or None if unclassifiable.

        direction is 'forward'  (Movie → relation → answer) or
                     'reverse'  (answer Movie → relation → entity).
        """
        q = question.lower()

        if entity_type == "Movie":
            # Forward: given Movie, follow relation outward
            if any(k in q for k in ["director", "directed by", "who directed", "who is the director", "directed on", "which person directed"]):
                return ("directed_by", "forward")
            if any(k in q for k in ["who starred", "who acted", "who acts", "who are the actors", "starred which", "starred who", "who stars in"]):
                return ("starred_actors", "forward")
            if any(k in q for k in ["who wrote", "writer", "written by", "who is the author", "screenplay", "script", "who is the creator", "who in the world wrote", "which person wrote"]):
                return ("written_by", "forward")
            if any(k in q for k in ["year", "release", "when was", "date"]):
                return ("release_year", "forward")
            if any(k in q for k in ["genre", "kind of", "type of", "sort of", "what kind", "what type", "what sort", "film genre"]):
                return ("has_genre", "forward")
            if any(k in q for k in ["language"]):
                return ("in_language", "forward")
            if any(k in q for k in ["describe", "words", "topics", "terms", "about", "applicable"]):
                return ("has_tags", "forward")

        elif entity_type == "Person":
            # Reverse: given Person, find Movies pointing to them
            if any(k in q for k in ["direct", "director"]):
                return ("directed_by", "reverse")
            if any(k in q for k in ["writ", "author", "screenplay", "script", "story", "creator of the film script"]):
                return ("written_by", "reverse")
            # Default for Person is acting
            return ("starred_actors", "reverse")

        elif entity_type == "Tag":
            return ("has_tags", "reverse")

        elif entity_type == "Genre":
            return ("has_genre", "reverse")

        elif entity_type == "Language":
            return ("in_language", "reverse")

        elif entity_type == "Year":
            return ("release_year", "reverse")

        return None

    return (infer_relation,)


@app.cell
def _():
    # ---------------------------------------------------------------------------
    # Question loader
    # ---------------------------------------------------------------------------

    def load_qa(path) -> list[tuple[str, set[str]]]:
        """Load (question, gold_answer_set) pairs from a MetaQA QA file."""
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                q, raw_answers = parts
                rows.append((q, {a.strip() for a in raw_answers.split("|")}))
        return rows

    return (load_qa,)


@app.cell
def run_eval(
    QA_TEST,
    arch1_traverse,
    arch2_traverse,
    defaultdict,
    graph,
    infer_relation,
    load_qa,
    mo,
    re,
    reverse_traverse,
):
    _qa = load_qa(QA_TEST)

    # Per-relation counters: [hits, total] for each architecture
    _a1: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    _a2: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    _skipped = 0  # entity not in graph or relation not inferred
    _arch2_blocked = 0  # arch2 raised unexpectedly on a valid question
    _disagreements = 0  # arch1 hit but arch2 missed (or vice versa)

    for _q, _gold in _qa:
        _m = re.search(r"\[(.+?)\]", _q)
        if not _m:
            _skipped += 1
            continue

        _entity = _m.group(1)
        if _entity not in graph:
            _skipped += 1
            continue

        _entity_type = graph.nodes[_entity].get("type", "Unknown")
        _parsed = infer_relation(_q, _entity_type)
        if _parsed is None:
            _skipped += 1
            continue

        _rel, _direction = _parsed

        # --- Architecture 1 ---
        if _direction == "forward":
            _r1 = arch1_traverse(graph, _entity, _rel)
        else:
            _r1 = reverse_traverse(graph, _entity, _rel)

        _hit1 = bool(_r1 & _gold)
        _a1[_rel][0] += int(_hit1)
        _a1[_rel][1] += 1

        # --- Architecture 2 ---
        if _direction == "forward":
            try:
                _r2 = arch2_traverse(graph, _entity, _rel)
                _hit2 = bool(_r2 & _gold)
            except ValueError:
                _r2 = set()
                _hit2 = False
                _arch2_blocked += 1
        else:
            # Reverse traversal: no ontology checking needed (no source entity type issue)
            _r2 = reverse_traverse(graph, _entity, _rel)
            _hit2 = bool(_r2 & _gold)

        _a2[_rel][0] += int(_hit2)
        _a2[_rel][1] += 1

        if _hit1 != _hit2:
            _disagreements += 1

    # Build results table
    _all_rels = sorted(set(_a1) | set(_a2))
    _rows_md = ""
    _total_a1_hits = _total_a2_hits = _total_qs = 0
    for _rel in _all_rels:
        _h1, _n1 = _a1[_rel]
        _h2, _n2 = _a2[_rel]
        _pct1 = f"{100*_h1//_n1}%" if _n1 else "—"
        _pct2 = f"{100*_h2//_n2}%" if _n2 else "—"
        _diff = ("**DIFF**" if _h1 != _h2 else "")
        _rows_md += f"| `{_rel}` | {_n1} | {_h1} ({_pct1}) | {_h2} ({_pct2}) | {_diff} |\n"
        _total_a1_hits += _h1
        _total_a2_hits += _h2
        _total_qs += _n1

    _overall1 = f"{100*_total_a1_hits//_total_qs}%" if _total_qs else "—"
    _overall2 = f"{100*_total_a2_hits//_total_qs}%" if _total_qs else "—"

    mo.md(f"""
    ## Results

    **Test questions:** {len(_qa)} total — {_total_qs} evaluated, {_skipped} skipped
    (skipped = entity not in graph or question template not recognised)

    | Relation | Questions | Arch 1 hit rate | Arch 2 hit rate | |
    |---|---|---|---|---|
    {_rows_md}
    | **TOTAL** | **{_total_qs}** | **{_overall1}** | **{_overall2}** | |

    **Disagreements (arch1 hit ≠ arch2 hit):** {_disagreements}
    **Arch 2 unexpected blocks on valid forward queries:** {_arch2_blocked}

    > Hit rate = % of questions where at least one gold answer is in the returned set.
    > Reverse-direction questions use the same traversal for both architectures
    > (no source entity type constraint applies to reverse lookup).
    """)
    return


if __name__ == "__main__":
    app.run()
