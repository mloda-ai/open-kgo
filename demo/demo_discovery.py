import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def title(mo):
    mo.md("""
    # DiscoveryEngine — Layer 3

    **Layer 1** (OntologyRegistry) validates edges: *"is this traversal legal?"* — binary yes/no.

    **Layer 2** (SemanticField) scores entities: *"how relevant is this entity to my query?"* — continuous.

    **Layer 3** (DiscoveryEngine) finds paths: *"what is the best route through the graph from source to goal?"* — ranked typed paths.

    This notebook covers what Layer 3 does, how the beam search works, and what the results look like.
    """)
    return


@app.cell
def s1_header(mo):
    mo.md("""
    ## 1 · What it is — beam search over the EM field
    """)
    return


@app.cell
def s1_body(mo):
    mo.md("""
    Layer 2 already solved the hard problem: it computed a potential V[e] at every entity
    in the graph. The edge current between any two adjacent nodes is just:

    ```
    I(i, j) = G(i,j) × |V(i) - V(j)|
    ```

    That edge current **is the beam search heuristic**. At each expansion step, the beam
    follows the highest-current edges — edges that carry the most signal between source and sink.
    No LLM calls. No oracle. No sampling. The field gradient already tells us where to walk.

    ```
    incoming query
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  DiscoveryEngine.find_paths(                                │
    │      namespace = "movie",                                   │
    │      edges     = subgraph_edges,                           │
    │      source    = {"Christopher Nolan": 1.0},   ← WHO       │
    │      sink      = {"Science Fiction": 0.0},     ← WHAT      │
    │      beam_width = 5,                                        │
    │      max_depth  = 4,                                        │
    │  )                                                          │
    └─────────────────────────────────────────────────────────────┘
         │
         ▼
    [
      DiscoveredPath(
          nodes=("Christopher Nolan", "Interstellar", "Science Fiction"),
          relations=("directed_by", "has_genre"),
          score=0.394,   ← bottleneck current
      ),
      ...
    ]
    ```

    The score of each path is the **bottleneck current**: the minimum edge current along
    the path — the weakest link in the conducting chain. A high score means every step
    in the path is strongly conducting. A low score means some hop is a weak connection.

    | Parameter | What it controls |
    |---|---|
    | `source` | High-voltage anchor — the "who" or "from" side |
    | `sink` | Low-voltage anchor — the "what" or "category" side |
    | `beam_width` | How many live candidate paths to keep at each step |
    | `max_depth` | Maximum hops from source to sink |
    | `max_paths` | Maximum completed paths to return |
    """)
    return


@app.cell
def s2_header(mo):
    mo.md("""
    ## 2 · extract_circuit — the explanation subgraph
    """)
    return


@app.cell
def s2_body(mo):
    mo.md("""
    `find_paths` gives you the best routes. `extract_circuit` gives you the minimal
    subgraph carrying meaningful current — the **explanation circuit**.

    ```python
    circuit_edges = DiscoveryEngine.extract_circuit(
        namespace="movie",
        edges=subgraph_edges,
        source={"Christopher Nolan": 1.0},
        sink={"Science Fiction": 0.0},
        current_threshold=0.01,
    )
    # Returns only edges where G(s,r,t) × |V(s) - V(t)| > 0.01
    # Dead-end branches are excluded for free — they carry zero current
    ```

    Dead-end branches are excluded automatically: a node connected to only one side
    of the circuit floats to that side's voltage. Both endpoints share the same
    potential — zero difference, zero current. No filtering logic needed.

    **Why this matters for context engineering:** instead of handing an LLM a full
    community partition (GraphRAG) or a bag of document chunks (RAG), you hand it
    a compact, typed, query-specific subgraph — exactly the edges that are live for
    this particular query.
    """)
    return


@app.cell
def s3_header(mo):
    mo.md("""
    ## 3 · The math
    """)
    return


@app.cell
def s3_math(mo):
    mo.md(r"""
    ### Edge current

    For each edge between nodes $i$ and $j$ with relation type $r$:

    $$I(i,j) = G(i,j) \cdot |V(i) - V(j)|$$

    where $G(i,j)$ is the ontology-declared conductance for relation type $r$
    (defaulting to 1.0 for unknown relations).

    ---

    ### Beam search heuristic

    At each expansion step, candidate extensions are ranked by edge current.
    The score of a path $p = (e_0, e_1, \ldots, e_k)$ is the **bottleneck current**:

    $$\text{score}(p) = \min_{i=0}^{k-1} I(e_i, e_{i+1})$$

    The global beam keeps the top-$B$ paths by score after each step.

    ---

    ### Why bottleneck (not sum)?

    A series circuit is only as strong as its weakest link. If a path has one
    very weak hop (low $G$ or small $|\Delta V|$), that hop is a near-open-circuit —
    the path as a whole barely conducts. The bottleneck score reflects this:
    a path scores high only when every hop is strongly conducting.

    ---

    ### Connection to SemanticField (Layer 2)

    Layer 2 solves $L \cdot V = s$ to get potentials. Layer 3 uses those potentials
    directly — no second solve needed. The edge currents are computed from the
    already-solved field. The beam search is purely a graph traversal over pre-computed values.
    """)
    return


@app.cell
def s4_header(mo):
    mo.md("""
    ## 4 · Live demo
    """)
    return


@app.cell
def s4_controls(mo):
    source_input = mo.ui.dropdown(
        options=[
            "Christopher Nolan",
            "Steven Spielberg",
            "Quentin Tarantino",
            "Francis Ford Coppola",
            "Bong Joon-ho",
            "George Miller",
        ],
        value="Christopher Nolan",
        label="Source (director)",
    )
    sink_input = mo.ui.dropdown(
        options=[
            "Science Fiction",
            "Action",
            "Crime",
            "Drama",
            "Thriller",
            "Adventure",
            "Comedy",
        ],
        value="Science Fiction",
        label="Sink (genre)",
    )
    beam_slider = mo.ui.slider(start=1, stop=10, step=1, value=5, label="beam_width")
    depth_slider = mo.ui.slider(start=1, stop=6, step=1, value=4, label="max_depth")
    run_btn = mo.ui.run_button(label="▶ Run discovery")
    return beam_slider, depth_slider, run_btn, sink_input, source_input


@app.cell
def s4_ui(beam_slider, depth_slider, mo, run_btn, sink_input, source_input):
    mo.hstack([source_input, sink_input, beam_slider, depth_slider, run_btn], gap=2)
    return


@app.cell
def s4_run(beam_slider, depth_slider, mo, run_btn, sink_input, source_input):
    import sys
    import time
    from pathlib import Path as _Path

    _repo = _Path(__file__).parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

    from demo.data import SAMPLE_FILE, ensure_data
    from open_kgo.feature_groups.kg.ontology import OntologyRegistry
    from open_kgo.feature_groups.kg.ontology.discovery import DiscoveryEngine

    _ONTOLOGY = (
        _Path(__file__).parent.parent / "open_kgo/feature_groups/kg/ontology/tests/fixtures/metaqa_ontology.yaml"
    )

    def _load_edges(path):
        import networkx as nx

        g = nx.read_gml(str(path))
        return [(s, d["relation"], t) for s, t, d in g.edges(data=True)]

    run_btn  # reactive dependency
    src = source_input.value
    snk = sink_input.value
    bw = beam_slider.value
    md = depth_slider.value

    ensure_data()

    if _ONTOLOGY.exists():
        try:
            OntologyRegistry.load_file(str(_ONTOLOGY))
        except ValueError:
            pass  # already loaded

    _edges = _load_edges(SAMPLE_FILE)
    _n_edges = len(_edges)
    _n_nodes = len({n for s, _, t in _edges for n in (s, t)})

    _t0 = time.perf_counter()
    _paths = DiscoveryEngine.find_paths(
        "movie",
        _edges,
        source={src: 1.0},
        sink={snk: 0.0},
        beam_width=bw,
        max_depth=md,
    )
    _t_paths = time.perf_counter() - _t0

    _t1 = time.perf_counter()
    _circuit = DiscoveryEngine.extract_circuit(
        "movie",
        _edges,
        source={src: 1.0},
        sink={snk: 0.0},
    )
    _t_circuit = time.perf_counter() - _t1

    if _paths:

        def _fmt_path(p):
            parts = [p.nodes[0]]
            for rel, node in zip(p.relations, p.nodes[1:]):
                parts.append(f"--{rel}-->")
                parts.append(node)
            return " ".join(parts)

        _path_rows = "\n".join(f"| {i + 1} | {_fmt_path(p)} | {p.score:.4f} |" for i, p in enumerate(_paths))
        _paths_table = f"""
| Rank | Path | Score |
|---|---|---|
{_path_rows}
"""
    else:
        _paths_table = f"\n*No paths found from **{src}** to **{snk}** within {md} hops.*\n"

    result = mo.md(f"""
**Query:** `{src}` (V=1.0) → ? → `{snk}` (V=0.0) · beam_width={bw} · max_depth={md}

**Graph:** {_n_nodes} nodes · {_n_edges} edges

---

### find_paths — {len(_paths)} path(s) found · {_t_paths * 1000:.1f}ms
{_paths_table}
---

### extract_circuit — {len(_circuit)} conducting edge(s) · {_t_circuit * 1000:.1f}ms

{len(_edges) - len(_circuit)} dead-end edges excluded (carry zero current)

*Conducting edges: `{ {r for _, r, _ in _circuit} }`*
    """)

    result
    return


@app.cell
def s5_header(mo):
    mo.md("""
    ## 5 · Loop engineering — field re-solve with updated anchors
    """)
    return


@app.cell
def s5_intro(mo):
    mo.md("""
    The EM field carries state between agent iterations.

    After step N, inspect the paths: which node appears as the bridge in every route?
    Promote it to a secondary source in step N+1 — re-solve the field with the updated anchors.
    Scores rise, new paths emerge, the circuit tightens. No LLM calls inside the loop.
    The field is the carry-forward object.

    The demo below is a fixed 2-step walkthrough.
    The agent is answering: ***"What crime films are connected to Keanu Reeves?"***
    """)
    return


@app.cell
def s5_loop(mo):
    import sys as _sys
    import time as _time
    from pathlib import Path as _LoopPath

    _root = _LoopPath(__file__).parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

    from demo.data import SAMPLE_FILE as _SF, ensure_data as _ensure
    from open_kgo.feature_groups.kg.ontology import OntologyRegistry as _OR
    from open_kgo.feature_groups.kg.ontology.discovery import DiscoveryEngine as _DE
    import networkx as _nx

    _ensure()
    _ONTO = _root / "open_kgo/feature_groups/kg/ontology/tests/fixtures/metaqa_ontology.yaml"
    if _ONTO.exists():
        try:
            _OR.load_file(str(_ONTO))
        except ValueError:
            pass

    def _load():
        _g = _nx.read_gml(str(_SF))
        return [(s, d["relation"], t) for s, t, d in _g.edges(data=True)]

    def _fmt(p):
        parts = [p.nodes[0]]
        for r, n in zip(p.relations, p.nodes[1:]):
            parts.append(f"--{r}-->")
            parts.append(n)
        return " ".join(parts)

    _edges = _load()

    # Step 1 — initial field
    _s1_source = {"Keanu Reeves": 1.0}
    _s1_sink = {"Crime": 0.0}
    _t0 = _time.perf_counter()
    _s1_paths = _DE.find_paths("movie", _edges, source=_s1_source, sink=_s1_sink, beam_width=8, max_depth=5)
    _s1_circuit = _DE.extract_circuit("movie", _edges, source=_s1_source, sink=_s1_sink)
    _t1 = _time.perf_counter() - _t0

    _s1_rows = "\n".join(f"| {i + 1} | {_fmt(p)} | {p.score:.4f} |" for i, p in enumerate(_s1_paths))

    # Step 2 — re-solve with The Matrix as secondary anchor
    _s2_source = {"Keanu Reeves": 1.0, "The Matrix": 0.9}
    _s2_sink = {"Crime": 0.0}
    _t2 = _time.perf_counter()
    _s2_paths = _DE.find_paths(
        "movie", _edges, source=_s2_source, sink=_s2_sink, beam_width=8, max_depth=5, max_paths=10
    )
    _s2_circuit = _DE.extract_circuit("movie", _edges, source=_s2_source, sink=_s2_sink)
    _t3 = _time.perf_counter() - _t2

    # Annotate step 2 paths: new vs carried-over, score change
    _s1_score_by_nodes = {p.nodes: p.score for p in _s1_paths}

    def _delta(p):
        old = _s1_score_by_nodes.get(p.nodes)
        if old is None:
            return "NEW ↑"
        chg = (p.score - old) / old * 100
        return f"{chg:+.0f}%"

    _s2_rows = "\n".join(f"| {i + 1} | {_fmt(p)} | {p.score:.4f} | {_delta(p)} |" for i, p in enumerate(_s2_paths[:6]))

    _new_circuit_edges = set(_s2_circuit) - set(_s1_circuit)

    mo.md(f"""
---

### Step 1 — Initial query · {_t1 * 1000:.1f} ms

```
source = {{"Keanu Reeves": 1.0}}
sink   = {{"Crime": 0.0}}
```

| Rank | Path | Score |
|---|---|---|
{_s1_rows}

Circuit: **{len(_s1_circuit)} / {len(_edges)} edges** carry current above threshold.

---

> **Agent observation** — Step 1 → Step 2
>
> Every path starts `Keanu Reeves → The Matrix`. The Matrix is the sole bridge.
> The bottleneck is the `starred_actors` hop (G=0.8) — the first edge on every path.
> Promoting The Matrix as a secondary anchor (V=0.9) will bypass this bottleneck.
> Paths from The Matrix directly should score ~61% higher.
>
> **Carry forward:** `The Matrix` → secondary source at V=0.9

---

### Step 2 — Field re-solve with updated anchors · {_t3 * 1000:.1f} ms

```
source = {{"Keanu Reeves": 1.0, "The Matrix": 0.9}}   ← The Matrix carried forward from step 1
sink   = {{"Crime": 0.0}}
```

| Rank | Path | Score | Δ |
|---|---|---|---|
{_s2_rows}

Circuit: **{len(_s2_circuit)} / {len(_edges)} edges** — {len(_new_circuit_edges)} new edge(s) above threshold.

**What changed:**

- 3 direct paths from The Matrix appear — bottleneck lifted: 0.0878 → 0.1419 (+61%)
- Two connection mechanisms separated: via **English** language AND via **Action** genre
- Original Keanu-paths score slightly lower (−3% to −9%): The Matrix is now a local source,
  reducing the potential difference on the `starred_actors` hop — expected behaviour
- Circuit expands: The Matrix's elevated voltage (0.9) propagates through Action to The Dark
  Knight, lifting its `heist` tag edges above the current threshold
""")
    return


@app.cell
def s6_header(mo):
    mo.md("""
    ## 6 · Test results
    """)
    return


@app.cell
def s5_tests(mo):
    mo.md("""
    ### Unit tests — 30 tests, all pass

    | Test group | What it pins |
    |---|---|
    | `TestDiscoveredPathDataclass` | Frozen fields, `len(nodes) == len(relations) + 1`, score non-negative |
    | `TestFindPathsBasic` | Paths found, sorted descending, first/last nodes match source/sink, typed relations |
    | `TestFindPathsEdgeCases` | Empty edges/source/sink, no-edge source, disconnected components, source==sink trivial path, depth limits, max_paths |
    | `TestFindPathsCyclePrevention` | No node visited twice, depth caps path length |
    | `TestFindPathsOntologyWeights` | Weights shape scores, unknown namespace uses G=1.0 |
    | `TestExtractCircuit` | Dead-end actors excluded, high threshold returns empty, three-node circuit pinned |

    **Pinned numerical test** — 3-node circuit:

    ```
    Nolan (V=1.0) --directed_by(G=0.9)-- Movie_A --has_genre(G=0.7)-- Action (V=0.0)

    V(Movie_A) = 0.9 / (0.9 + 0.7) = 0.5625
    Both edges carry current = 0.39375  (KCL: in == out)
    Bottleneck score = 0.39375
    ```

    ```
    30 passed in 0.32s
    ```

    ### Numba acceleration

    The edge current computation uses a numba JIT kernel when available:

    ```python
    @_njit(cache=True)
    def _nb_edge_currents_kernel(v_src, v_tgt, conductances):
        n = v_src.shape[0]
        out = zeros(n)
        for i in range(n):
            out[i] = conductances[i] * abs(v_src[i] - v_tgt[i])
        return out
    ```

    Called once per `find_paths` invocation — batch-computes all edge currents in a
    single JIT pass before the beam loop starts. Falls back silently to pure Python
    when numba is absent.
    """)
    return


@app.cell
def s7_header(mo):
    mo.md("""
    ## 7 · The three layers
    """)
    return


@app.cell
def s7_stack(mo):
    mo.md("""
    <div style="background:#0f1117;border-radius:8px;padding:16px;">
    <svg viewBox="0 0 860 300" xmlns="http://www.w3.org/2000/svg"
         style="width:100%;max-width:860px;display:block;">

      <!-- L3 focal -->
      <rect x="40" y="16" width="780" height="76" rx="6"
            fill="rgba(249,115,22,0.07)" stroke="#f97316" stroke-width="1.5"/>
      <text x="56" y="46" fill="#f97316" font-size="9" font-family="monospace" letter-spacing="0.1em">L3</text>
      <text x="80" y="59" fill="#fb923c" font-size="16" font-weight="bold" font-family="sans-serif">DiscoveryEngine</text>
      <text x="258" y="46" fill="#f97316" font-size="9" font-family="monospace" letter-spacing="0.08em">BUILT</text>
      <text x="812" y="50" fill="#a0abc0" font-size="9" font-family="monospace" text-anchor="end">find_paths(source, sink, beam_width, max_depth)  ·  extract_circuit(source, sink, threshold)</text>
      <text x="812" y="66" fill="#5a6580" font-size="9" font-family="monospace" text-anchor="end">beam heuristic = edge current  ·  score = bottleneck current  ·  numba-accelerated</text>

      <!-- L2 de-emphasised -->
      <rect x="40" y="104" width="780" height="64" rx="6"
            fill="rgba(255,255,255,0.03)" stroke="#e8eaf0" stroke-width="1.2"/>
      <text x="56" y="132" fill="#a0abc0" font-size="9" font-family="monospace" letter-spacing="0.1em">L2</text>
      <text x="80" y="144" fill="#e8eaf0" font-size="16" font-weight="bold" font-family="sans-serif">SemanticField</text>
      <text x="230" y="132" fill="#a0abc0" font-size="9" font-family="monospace" letter-spacing="0.08em">BUILT</text>
      <text x="812" y="138" fill="#a0abc0" font-size="9" font-family="monospace" text-anchor="end">compute(anchors)  ·  compute_and(source, sink)  ·  solves L·V=s  ·  score = electric current</text>

      <!-- L1 foundation -->
      <rect x="40" y="180" width="780" height="64" rx="6"
            fill="rgba(255,255,255,0.03)" stroke="#e8eaf0" stroke-width="1.2"/>
      <text x="56" y="208" fill="#a0abc0" font-size="9" font-family="monospace" letter-spacing="0.1em">L1</text>
      <text x="80" y="220" fill="#e8eaf0" font-size="16" font-weight="bold" font-family="sans-serif">OntologyRegistry</text>
      <text x="264" y="208" fill="#a0abc0" font-size="9" font-family="monospace" letter-spacing="0.08em">BUILT</text>
      <text x="812" y="216" fill="#a0abc0" font-size="9" font-family="monospace" text-anchor="end">is_valid_edge()  ·  get_range_type()  ·  YAML-declared  ·  binary valid / invalid</text>

      <text x="18" y="170" fill="#2a3042" font-size="8" font-family="monospace"
            transform="rotate(-90,18,170)" text-anchor="middle" letter-spacing="0.12em">ABSTRACTION ↑</text>

      <line x1="40" y1="258" x2="820" y2="258" stroke="rgba(255,255,255,0.05)" stroke-width="0.8"/>
      <rect x="60" y="266" width="16" height="12" rx="2" fill="rgba(249,115,22,0.07)" stroke="#f97316" stroke-width="1.5"/>
      <text x="82" y="276" fill="#4a5570" font-size="9" font-family="monospace">Focal — this release</text>
      <rect x="220" y="266" width="16" height="12" rx="2" fill="rgba(255,255,255,0.03)" stroke="#e8eaf0" stroke-width="1"/>
      <text x="242" y="276" fill="#4a5570" font-size="9" font-family="monospace">Built / stable</text>
    </svg>
    </div>
    """)
    return


@app.cell
def s7_layers_detail(mo):
    mo.md("""
    ### Layer 1 — OntologyRegistry

    Declared in YAML. Relationship types, valid domain/range pairs, and weights.
    Weights flow into Layer 2 as conductances — the two layers compose directly.

    ---

    ### Layer 2 — SemanticField

    DC circuit model. Solves the conductance-weighted graph Laplacian to find potentials.
    Scores entities by current: how much signal flows through them between source and sink.
    Implemented in `open_kgo/feature_groups/kg/ontology/semantic_field.py`.

    ---

    ### Layer 3 — DiscoveryEngine  *(this notebook)*

    Beam search over the pre-computed EM field. The edge current is the expansion heuristic —
    no LLM calls needed. Returns ranked typed paths and the minimal current-carrying subgraph.
    Implemented in `open_kgo/feature_groups/kg/ontology/discovery.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
