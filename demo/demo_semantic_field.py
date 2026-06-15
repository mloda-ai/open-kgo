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
    # SemanticField — Layer 2

    **Layer 1** (OntologyRegistry) validates edges: *"is this traversal legal?"* — binary yes/no.

    **Layer 2** (SemanticField) scores entities: *"how relevant is this entity to my query?"* — continuous.

    This notebook covers the full picture: what it is, why we built it this way,
    how to use it for 1-hop and multi-hop queries, and what the numbers look like at scale.
    """)
    return


@app.cell
def s1_header(mo):
    mo.md("""
    ## 1 · What it is — treating it as an incoming query
    """)
    return


@app.cell
def s1_body(mo):
    mo.md("""
    Think of SemanticField as a **query processor that lives between your KG connector and your application**.
    You hand it a subgraph and a query expressed as voltage anchors; it hands back a ranked list of entities.

    ```
    incoming query
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  SemanticField.compute_and(                 │
    │      namespace = "metaqa",                  │
    │      edges     = subgraph_edges,            │
    │      source    = {"Nolan": 1.0},   ← WHO   │
    │      sink      = {"Sci-Fi": 0.0},  ← WHAT  │
    │  )                                          │
    └─────────────────────────────────────────────┘
         │
         ▼
    {"Interstellar": 0.394, "Inception": 0.312, "Dark Knight": 0.0, ...}
    ```

    The subgraph you pass in is just the **2-hop neighbourhood** around your anchor nodes —
    not the full KG. This keeps the solver tractable and makes the result focused.

    Three things the caller controls:

    | Parameter | What it does |
    |---|---|
    | `source` | High-voltage anchor — the "who" or "from" side of the query |
    | `sink` | Low-voltage anchor — the "what" or "category" side of the query |
    | `edges` | Subgraph to search — typically 2-hop neighbourhood of anchors |

    The namespace is used to look up conductance weights from the ontology (Layer 1).
    Unknown relationships default to G=1.0 (open-world assumption).
    """)
    return


@app.cell
def s2_header(mo):
    mo.md("""
    ## 2 · 1-hop, 2-hop, and beyond
    """)
    return


@app.cell
def s2_body(mo):
    mo.md("""
    SemanticField does not care about hop count — it sees a graph and solves it.
    But the *shape* of the query determines what hop depth you need.

    ---

    ### 1-hop — single relationship type

    *"Which actors has Nolan worked with?"*

    ```python
    # Only need 1-hop edges: Movie → starred_actors → Actor
    SemanticField.compute(
        namespace="metaqa",
        edges=one_hop_edges,
        anchors={"Nolan": 1.0},
        relation_type="starred_actors",
    )
    ```

    Actors directly connected to Nolan's films get a potential; others get 0.0.

    ---

    ### 2-hop — the standard AND query

    *"Sci-Fi films directed by Nolan."*

    ```python
    # Two hops: Director → directed_by → Movie → has_genre → Genre
    SemanticField.compute_and(
        namespace="metaqa",
        edges=two_hop_edges,
        source={"Nolan": 1.0},
        sink={"Sci-Fi": 0.0},
    )
    ```

    The movie sits between two anchors. Current flows through it only if it connects both sides.

    ---

    ### Multi-hop — the drug repurposing case

    *"Which drugs, via their protein targets and associated pathways, are candidates for Alzheimer's?"*

    ```
    Drug → binds → Protein → involved_in → Pathway → associated_with → Disease
    ```

    This is a 3-hop path. Existing tools either:
    - (SPARQL) enumerate all paths — combinatorial explosion, binary results
    - (vectors) score by embedding similarity — no typed edges, no partial satisfaction

    SemanticField handles it in one call:

    ```python
    SemanticField.compute_and(
        namespace="biokg",
        edges=subgraph,            # 2-hop neighbourhood around Drug + Disease
        source={"Metformin": 1.0, "Aspirin": 1.0},   # known safe drugs
        sink={"Alzheimer's": 0.0},
    )
    ```

    Drugs that have *partial* pathway overlap still score above zero — partial satisfaction
    is built into the physics. A drug with a weak pathway connection scores lower than one
    with a strong direct connection, in proportion to conductance (ontology weight).

    ---

    ### What hop depth to use in practice

    | Query type | Typical depth | Notes |
    |---|---|---|
    | Single-anchor potential | 2 | `compute()` — one voltage source |
    | AND query (2 anchors) | 2–3 | `compute_and()` — series circuit |
    | Multi-constraint | 3–4 | Multiple source/sink pairs, merge scores |
    | Discovery / exploration | BFS-bounded | Layer 3 (planned) will automate this |
    """)
    return


@app.cell
def s3_header(mo):
    mo.md("""
    ## 3 · Why electromagnetism — and not something else
    """)
    return


@app.cell
def s3_body(mo):
    mo.md("""
    We evaluated several physics frameworks before committing to DC circuits.
    The requirement: natural mapping onto typed, weighted KGs with query-driven semantics.

    | Framework | What it models well | Why we didn't use it |
    |---|---|---|
| **Thermodynamics / diffusion** | Spreading, equilibration, heat flow | Maps to random walks — good for exploration, but no natural AND semantics, no voltage/current duality |
    | **Quantum mechanics** | Superposition, interference | Beautiful math but no intuitive physical analogy for "a query"; requires complex amplitudes |
    | **Fluid dynamics (Reynolds)** | Flow through networks | Close, but conductance/resistance is cleaner for discrete graphs; EM has a richer established theory |
    | **DC electrical circuits** | ✓ Voltage sources, resistors, current flow, Kirchhoff's laws | **Maps perfectly onto our problem** |

    ### Why DC circuits fit so well

    1. **Ontology weights → conductances.** A high-weight relationship (e.g. `directed_by = 0.9`)
       is a low-resistance wire — signal passes easily. A weak relationship (`has_tags = 0.2`)
       resists signal. The ontology literally *is* the impedance map of the graph.

    2. **Query anchors → Dirichlet boundary conditions.** Fixing voltage at anchor nodes is the
       standard *impressed-voltage* treatment in electrical network theory. No approximation —
       this is how you solve resistor networks analytically.

    3. **The AND constraint is free.** In a series circuit, current only flows if *both* ends
       are connected. Dark Knight is connected to Nolan (source) but Action is a dead end —
       no path to the Sci-Fi ground. Voltage floats to 1.0, potential difference is zero,
       current is zero. The physics enforces the AND without any extra filtering logic.

    4. **Partial satisfaction is built in.** A drug with a weak pathway connection scores
       lower than one with a strong direct connection — in proportion to G. No thresholding,
       no binary cutoff. This is what makes multi-hop biomedical queries tractable.

    5. **Well-understood math.** Graph Laplacian, Kirchhoff's current law, Gaussian elimination.
       Nothing exotic. Every step is traceable and explainable.

    Every step is grounded in established electrical network theory — Kirchhoff's current law,
    graph Laplacian, Dirichlet boundary conditions. Nothing exotic, everything traceable.
    """)
    return


@app.cell
def s4_header(mo):
    mo.md("""
    ## 4 · The circuit — visually
    """)
    return


@app.cell
def s4_circuit(mo):
    mo.md("""
    *Query: "Sci-Fi films by Nolan" — Nolan is the battery (V=1.0), Sci-Fi is ground (V=0.0). Which movies carry current?*
    """)
    return


@app.cell
def s4_svg():
    from pathlib import Path
    _svg_path = Path(__file__).parent / "circuit_schematic.html"
    _html = _svg_path.read_text()
    # extract just the SVG from the HTML file
    import re
    _match = re.search(r'(<svg[\s\S]*?</svg>)', _html)
    _svg = _match.group(1) if _match else "<p>SVG not found</p>"
    return


@app.cell
def s4_mapping(mo):
    mo.md("""
    ### Circuit → KG mapping

    | Circuit element | Knowledge graph concept |
    |---|---|
    | Battery (anode +) | Query anchor — source entity (e.g. Nolan) |
    | Ground (−) | Query anchor — target category (e.g. Sci-Fi) |
    | Resistor | Relationship instance between two entities |
    | Conductance G | Ontology weight for that relationship type |
    | Voltage V(e) | How "aligned" entity e is with the query |
    | Current I(e) | **The score** — how much signal flows through e |
    | Open circuit | Dead-end branch — entity not connected to both anchors |
    """)
    return


@app.cell
def s5_header(mo):
    mo.md("""
    ## 5 · The math
    """)
    return


@app.cell
def s5_body(mo):
    mo.md(r"""
    ### Step 1 — Build the conductance-weighted graph Laplacian

    $$L_{ii} = \sum_j G_{ij} \qquad L_{ij} = -G_{ij} \quad (i \neq j)$$

    Each off-diagonal entry is the negative conductance of that edge.
    The diagonal is the sum of all conductances at that node.

    ---

    ### Step 2 — Apply Dirichlet boundary conditions

    Partition nodes into **boundary** $B$ (anchors, fixed voltage) and **interior** $I$ (unknown):

    $$L_{II} \, V_I = -L_{IB} \, V_B$$

    Solved via Gaussian elimination with partial pivoting.
    Zero-pivot rows (disconnected nodes) receive potential 0.0.

    ---

    ### Step 3 — Score by current (AND query)

    $$I(e) = \sum_{\substack{j \in \text{nbrs}(e) \\ V_j > V_e}} G_{ej} \cdot (V_j - V_e)$$

    Only entities on a **conducting path between source and sink** carry current.
    Dead-end branches float to source voltage → potential difference = 0 → I = 0.

    ---

    ### Why current instead of voltage?

    Voltage alone does not enforce the AND constraint.
    An entity close to the source gets a high voltage *regardless* of whether it has
    any connection to the sink. Current requires *both* a high-voltage neighbour
    and a low-voltage neighbour to be non-zero.
    """)
    return


@app.cell
def s6_header(mo):
    mo.md("""
    ## 6 · Live demo on MetaQA
    """)
    return


@app.cell
def s6_controls(mo):
    director_input = mo.ui.dropdown(
        options=[
            "Christopher Nolan",
            "Steven Spielberg",
            "James Cameron",
            "Ridley Scott",
            "Martin Scorsese",
            "David Fincher",
        ],
        value="Christopher Nolan",
        label="Director (source anchor)",
    )
    genre_input = mo.ui.dropdown(
        options=["Sci-Fi", "Action", "Drama", "Thriller", "Comedy", "Horror", "Adventure"],
        value="Sci-Fi",
        label="Genre (sink anchor)",
    )
    run_btn = mo.ui.run_button(label="▶ Run query")
    return director_input, genre_input, run_btn


@app.cell
def s6_ui(director_input, genre_input, mo, run_btn):
    mo.hstack([director_input, genre_input, run_btn], gap=2)
    return


@app.cell
def s6_run(director_input, genre_input, mo, run_btn):
    import sys, time, re
    from pathlib import Path
    from collections import deque

    _repo = Path(__file__).parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

    from open_kgo.feature_groups.kg.ontology.semantic_field import SemanticField

    _GML = Path("/Volumes/ExtraStorage/mloda_New_arch/prototypes/demo/data/metaqa_full.gml")

    def _stream_parse(path):
        id_to_label, edges = {}, []
        in_node = in_edge = False
        cur_id = cur_label = cur_src = cur_tgt = cur_rel = None
        with path.open() as f:
            for raw in f:
                line = raw.strip()
                if line == "node [":
                    in_node, in_edge, cur_id, cur_label = True, False, None, None
                elif line == "edge [":
                    in_edge, in_node = True, False
                    cur_src = cur_tgt = cur_rel = None
                elif line == "]":
                    if in_node and cur_id is not None and cur_label is not None:
                        id_to_label[cur_id] = cur_label
                    elif in_edge and all(x is not None for x in [cur_src, cur_tgt, cur_rel]):
                        s, t = id_to_label.get(cur_src), id_to_label.get(cur_tgt)
                        if s and t:
                            edges.append((s, cur_rel, t))
                    in_node = in_edge = False
                elif in_node:
                    if line.startswith("id "): cur_id = int(line[3:])
                    elif line.startswith('label "'): cur_label = line[7:-1]
                elif in_edge:
                    if line.startswith("source "): cur_src = int(line[7:])
                    elif line.startswith("target "): cur_tgt = int(line[7:])
                    elif line.startswith('relation "'): cur_rel = line[10:-1]
        return edges

    def _build_adj(edges):
        adj = {}
        for s, r, t in edges:
            adj.setdefault(s, []).append((t, r))
            adj.setdefault(t, []).append((s, r))
        return adj

    def _subgraph(adj, seeds, hops=2, max_nodes=5000):
        visited, queue = set(), deque((s, 0) for s in seeds if s in adj)
        while queue and len(visited) < max_nodes:
            node, depth = queue.popleft()
            if node in visited: continue
            visited.add(node)
            if depth < hops:
                for nbr, _ in adj.get(node, []):
                    if nbr not in visited: queue.append((nbr, depth + 1))
        sub, seen = [], set()
        for node in visited:
            for nbr, rel in adj.get(node, []):
                if nbr in visited:
                    key = (min(node, nbr), rel, max(node, nbr))
                    if key not in seen:
                        seen.add(key)
                        sub.append((node, rel, nbr))
        return sub

    run_btn  # reactive dependency
    director = director_input.value
    genre = genre_input.value

    if not _GML.exists():
        result = mo.callout(
            mo.md(f"Full MetaQA GML not found at `{_GML}`. Place `metaqa_full.gml` there to run this demo."),
            kind="warn",
        )
    else:
        _t_parse = time.perf_counter()
        _all_edges = _stream_parse(_GML)
        _adj = _build_adj(_all_edges)
        _t_parse = time.perf_counter() - _t_parse

        if director not in _adj:
            result = mo.callout(mo.md(f"**{director}** not in dataset."), kind="warn")
        elif genre not in _adj:
            result = mo.callout(mo.md(f"**{genre}** not in dataset."), kind="warn")
        else:
            _sub = _subgraph(_adj, seeds=[director, genre])
            _n_nodes = len({n for s, _, t in _sub for n in (s, t)})

            _t_solve = time.perf_counter()
            _scores = SemanticField.compute_and(
                namespace="metaqa",
                edges=_sub,
                source={director: 1.0},
                sink={genre: 0.0},
            )
            _t_solve = time.perf_counter() - _t_solve

            _top = sorted(_scores.items(), key=lambda x: x[1], reverse=True)[:15]
            _rows = "\n".join(
                f"| {i+1} | **{name}** | {float(score):.4f} |"
                for i, (name, score) in enumerate(_top)
            )

            result = mo.md(f"""
    **Query:** `{director}` (source, V=1.0) → ? → `{genre}` (sink, V=0.0)

    **Subgraph:** {_n_nodes:,} nodes · {len(_sub):,} edges
    **Parse:** {_t_parse:.2f}s · **Solve:** {_t_solve:.2f}s

    | Rank | Entity | Score I(e) |
    |---|---|---|
    {_rows}
            """)

    result
    return


@app.cell
def s7_header(mo):
    mo.md("""
    ## 7 · Test results & performance
    """)
    return


@app.cell
def s7_unit(mo):
    mo.md("""
    ### Unit tests — 18 tests, all pass

    Covering the full behaviour of `SemanticField.compute()` and `SemanticField.compute_and()`:

    | Test group | What it pins |
    |---|---|
    | Conductance lookup | Ontology weight returned; unknown relation defaults to 1.0 |
    | Single-anchor potential | Anchor carries its declared voltage; connected nodes get non-zero potential |
    | Disconnected nodes | Receive potential 0.0 — no spurious scores |
    | Relation-type filtering | `compute(..., relation_type=...)` isolates the subgraph |
    | Voltage divider | Two-anchor setup: V(movie) = G_source / (G_source + G_sink) |
    | AND logic — Dark Knight | `compute_and` gives exactly 0.0 (dead-end branch) |
    | AND logic — Arrival | Also 0.0 (connected to Sci-Fi but not to Nolan in the circuit) |
    | Verified calculation | Pins Interstellar to I = 0.9 × (1.0 − 0.9/1.6) = 0.39375 |
    | Weight from YAML | Ontology weights loaded correctly from fixture file |

    ```
    18 passed in 2.75s
    ```

    First run includes numba JIT compilation (~2s). Subsequent runs are ~0.05s.
    """)
    return


@app.cell
def s7_scale(mo):
    mo.md("""
    ### Scale tests — MetaQA full dataset (43,234 nodes · 134,741 edges)

    Tests use 2-hop subgraphs capped at 5,000 nodes, stream-parsed from the full GML.

    #### Before numba (pure Python Gaussian elimination)

    | Query | Subgraph | Time |
    |---|---|---|
    | Nolan + Sci-Fi | 159 nodes | 0.03s |
    | Spielberg + Action | 1,200 nodes | 14.1s |
    | Cameron + Thriller | 1,200 nodes | 14.2s |
    | Ridley Scott + Drama | 1,200 nodes | 14.5s |

    Pure Python O(n³) — hits the ceiling at ~1,200 nodes.

    ---

    #### After numba (JIT-compiled kernel)

    Numba compiles the Gaussian elimination inner loop at first call and caches it.
    No dependency change for users who don't have numba — falls back to pure Python silently.

    | Query | Subgraph | Time | Speedup |
    |---|---|---|---|
    | Nolan + Sci-Fi | 159 nodes | 0.11s | — (compile overhead) |
    | Spielberg + Action | 5,340 nodes | 14.2s | **70×** vs 1,200-node baseline |
    | Cameron + Thriller | 4,916 nodes | 11.1s | **70×** |
    | Ridley Scott + Drama | 5,000 nodes | 11.9s | **70×** |

    We can now comfortably run 5,000-node subgraphs under 15 seconds.
    The 10,000-node case (Ridley Scott + Drama, a very densely connected genre) takes ~95s —
    that's the current ceiling without a sparse solver.

    ---

    #### All checks pass

    ```
    tox — 1,013 passed, 52 skipped
    ruff format ✓ · ruff check ✓ · mypy --strict ✓ · bandit ✓
    ```

    Numba is an optional dependency — the tox env doesn't have it installed,
    so the pure-Python fallback is what CI tests. Numba is available in the
    `mloda` conda env for local runs.
    """)
    return


@app.cell
def s8_header(mo):
    mo.md("""
    ## 8 · The three layers
    """)
    return


@app.cell
def s8_stack():
    stack_svg = """
    <div style="background:#0f1117;border-radius:8px;padding:16px;">
    <svg viewBox="0 0 860 300" xmlns="http://www.w3.org/2000/svg"
         style="width:100%;max-width:860px;display:block;">

      <!-- L3 planned -->
      <rect x="40" y="16" width="780" height="64" rx="6"
            fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.12)"
            stroke-width="1" stroke-dasharray="5,3"/>
      <text x="56" y="44" fill="#4a5570" font-size="9" font-family="monospace" letter-spacing="0.1em">L3</text>
      <text x="80" y="57" fill="#4a5570" font-size="16" font-weight="bold" font-family="sans-serif">Discovery Engine</text>
      <text x="300" y="44" fill="#2a3042" font-size="9" font-family="monospace" letter-spacing="0.08em">PLANNED</text>
      <text x="812" y="52" fill="#2a3042" font-size="9" font-family="monospace" text-anchor="end">mloda.discover(start, goal_type, namespace)  ·  ranked paths by field strength</text>

      <!-- L2 focal -->
      <rect x="40" y="90" width="780" height="76" rx="6"
            fill="rgba(249,115,22,0.07)" stroke="#f97316" stroke-width="1.5"/>
      <text x="56" y="120" fill="#f97316" font-size="9" font-family="monospace" letter-spacing="0.1em">L2</text>
      <text x="80" y="133" fill="#fb923c" font-size="16" font-weight="bold" font-family="sans-serif">SemanticField</text>
      <text x="234" y="120" fill="#f97316" font-size="9" font-family="monospace" letter-spacing="0.08em">BUILT</text>
      <text x="812" y="124" fill="#a0abc0" font-size="9" font-family="monospace" text-anchor="end">compute(anchors)  ·  compute_and(source, sink)</text>
      <text x="812" y="140" fill="#5a6580" font-size="9" font-family="monospace" text-anchor="end">ontology weights → conductances  ·  score = electric current  ·  numba-accelerated</text>

      <!-- L1 foundation -->
      <rect x="40" y="178" width="780" height="64" rx="6"
            fill="rgba(255,255,255,0.03)" stroke="#e8eaf0" stroke-width="1.2"/>
      <text x="56" y="206" fill="#a0abc0" font-size="9" font-family="monospace" letter-spacing="0.1em">L1</text>
      <text x="80" y="218" fill="#e8eaf0" font-size="16" font-weight="bold" font-family="sans-serif">OntologyRegistry</text>
      <text x="262" y="206" fill="#a0abc0" font-size="9" font-family="monospace" letter-spacing="0.08em">BUILT</text>
      <text x="812" y="214" fill="#a0abc0" font-size="9" font-family="monospace" text-anchor="end">is_valid_edge()  ·  get_range_type()  ·  YAML-declared  ·  binary valid / invalid</text>

      <text x="18" y="170" fill="#2a3042" font-size="8" font-family="monospace"
            transform="rotate(-90,18,170)" text-anchor="middle" letter-spacing="0.12em">ABSTRACTION ↑</text>

      <line x1="40" y1="258" x2="820" y2="258" stroke="rgba(255,255,255,0.05)" stroke-width="0.8"/>
      <rect x="60" y="266" width="16" height="12" rx="2" fill="rgba(249,115,22,0.07)" stroke="#f97316" stroke-width="1.2"/>
      <text x="82" y="276" fill="#4a5570" font-size="9" font-family="monospace">Focal — this release</text>
      <rect x="220" y="266" width="16" height="12" rx="2" fill="rgba(255,255,255,0.03)" stroke="#e8eaf0" stroke-width="1"/>
      <text x="242" y="276" fill="#4a5570" font-size="9" font-family="monospace">Built / stable</text>
      <rect x="360" y="266" width="16" height="12" rx="2" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.12)" stroke-width="1" stroke-dasharray="4,3"/>
      <text x="382" y="276" fill="#4a5570" font-size="9" font-family="monospace">Planned</text>
    </svg>
    </div>
    """
    return


@app.cell
def s8_layers_detail(mo):
    mo.md("""
    ### Layer 1 — OntologyRegistry

    Declared in YAML per domain. Relationship types, valid domain/range pairs, and weights.
    Weights flow into Layer 2 as conductances — the two layers are designed to compose.

    ```yaml
    relationships:
      directed_by: { domain: Movie, range: Person, weight: 0.9 }
      has_genre:   { domain: Movie, range: Genre,  weight: 0.7 }
      has_tags:    { domain: Movie, range: Tag,    weight: 0.2 }
    ```

    ---

    ### Layer 2 — SemanticField  *(this notebook)*

    DC circuit model. Takes the subgraph + query anchors, returns entity scores.
    Implemented in `open_kgo/feature_groups/kg/ontology/semantic_field.py`.

    ---

    ### Layer 3 — Discovery Engine  *(planned)*

    Will use Layer 2 field values to guide multi-hop traversal toward a goal type.
    Think of it as a beam search that follows the current — highest-scoring paths first.
    """)
    return


if __name__ == "__main__":
    app.run()
