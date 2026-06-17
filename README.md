[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# open-kgo

Open Knowledge Graphs and Ontologies plugin for [mloda](https://github.com/mloda-ai/mloda): nine connector families covering the knowledge-graph landscape, from SPARQL endpoints to SBOMs to agent memory, all behind one declarative `Feature` interface. Every connector and demo runs offline against in-memory libraries or committed fixtures. No Docker, no network.

## At a glance

| Section | What you'll find |
|---|---|
| [Quickstart](#quickstart) | Run a SPARQL query against a shipped sample file in under a minute |
| [The nine connector families](#the-nine-connector-families) | The core of this repo: a 9-family KG connector taxonomy with two plugins each |
| [Semantic Fields — Layer 2](#semantic-fields--layer-2) | Continuous, schema-grounded entity scoring via DC circuit model |
| [Demos](#demos) | Marimo notebooks and evaluation harnesses, all offline |
| [Data and acknowledgments](#data-and-acknowledgments) | Where the sample data comes from |
| [Development setup](#development-setup) | uv, tox, and the individual checks |
| [Related repositories and documentation](#related-repositories-and-documentation) | mloda core, the plugin registry, and development guides |

## Quickstart

Install the connectors and run a SPARQL query against the Turtle sample shipped in this repo:

```bash
uv sync --extra kg-all
```

```python
from pathlib import Path

from mloda.user import DataAccessCollection, Feature, Options, mloda

import open_kgo.feature_groups.kg.rdf.rdflib_sparql as rdf_mod
from open_kgo.compute_frameworks.python_dict_kg_framework import KgPythonDictFramework

# Point at any RDF file. Here: the Turtle sample shipped in this repo.
ttl = Path(rdf_mod.__file__).parent / "tests" / "fixtures" / "sample.ttl"

feature = Feature(
    "rdflib_sparql__knows",
    options=Options(context={
        "query_text": "PREFIX foaf: <http://xmlns.com/foaf/0.1/> "
                      "SELECT ?s ?o WHERE { ?s foaf:knows ?o }",
    }),
)

partitions = mloda.run_all(
    [feature],
    compute_frameworks={KgPythonDictFramework},
    data_access_collection=DataAccessCollection(
        credentials=[{"rdflib_sparql": {"locator": str(ttl), "result_limit": 100}}],
    ),
)

for partition in partitions:
    for row in partition:
        print(row[feature.name])
```

Swap `rdflib_sparql` for any of the nine connector families below: same `Feature` to `mloda.run_all` shape, different reader.

## The nine connector families

`open_kgo/feature_groups/kg/` ships a connector taxonomy derived from a 103-system survey. Each family is a shared reader and feature-group base plus **two** concrete plugins running against in-memory libraries or local file fixtures:

| Family | What it connects to | Concrete plugins |
|---|---|---|
| `network_pg` | Property-graph databases with a vendor query language (Neo4j, Memgraph, Neptune, ...) | `KuzuCypherReader`, `GrandCypherReader` |
| `rdf` | RDF triple stores queried with SPARQL | `RdfLibSparqlReader`, `OxigraphSparqlReader` |
| `embedded` | In-process graph libraries with no network endpoint | `NetworkxEmbeddedReader`, `IGraphEmbeddedReader` |
| `rest_public` | Public REST (non-SPARQL) KG APIs (OpenAlex, ConceptNet, STRING, ...) | `FileFixtureRestReader`, `FileFixturePagedRestReader` |
| `lineage` | Metadata and data-lineage graphs (dbt, OpenLineage, DataHub, ...) | `DbtManifestReader`, `OpenLineageReader` |
| `code_build` | Code, build, and SBOM dependency graphs (CycloneDX, SPDX, ...) | `CycloneDxSbomReader`, `SpdxSbomReader` |
| `saas_authz` | SaaS and authorization tuple stores (OpenFGA, SpiceDB, Microsoft Graph, ...) | `InProcessTupleStoreReader`, `PaginatedTupleStoreReader` |
| `agent_memory` | LLM agent memory and GraphRAG graphs (Letta, Zep, Mem0, ...) | `NetworkxMemoryReader`, `GraphWalkMemoryReader` |
| `citation_rest` | Citation and scientific REST APIs (Reactome, OpenAlex citations, ...) | `FileFixtureCitationReader`, `PaginatedCitationReader` |

See [`open_kgo/feature_groups/kg/README.md`](open_kgo/feature_groups/kg/README.md) for the full family map, the plugin anatomy, and what the prototype does and does not validate.

Install all KG extras with: `uv sync --extra kg-all`.

> **One feature per call.** KG readers dispatch a single feature per load: every
> reader rejects a multi-feature `FeatureSet` rather than silently labelling all
> rows with one feature name. Request features individually (one `Feature` per
> `mloda.run_all` slot) rather than batching `N` of them into a single reader call.

> **No-Docker testing policy.** Every connector test runs against rdflib, networkx, kuzu (embedded), or file fixtures. No Docker, no external services, no network calls.

## Semantic Fields — Layer 2

The connector families (Layer 1 validation) answer: *"is this traversal valid?"* — binary yes/no.

**SemanticField** (Layer 2) answers: *"how relevant is this entity to my query?"* — a continuous score, grounded in DC electrical circuit theory.

The knowledge graph is modelled as a resistor network. Ontology-declared relationship weights become conductances. A query is expressed as two anchor nodes — a **source** (high voltage) and a **sink** (low voltage, ground). The solver finds the electric potential at every entity via the conductance-weighted graph Laplacian, then scores each entity by the current it carries between source and sink.

**Why current?** Current only flows through entities that bridge *both* anchors. An entity connected to only one side floats to that side's extreme voltage — zero potential difference, zero current, automatically excluded. No relation-type filtering needed; the circuit topology enforces the AND constraint.

```python
from open_kgo.feature_groups.kg.ontology import SemanticField

# AND query: Sci-Fi films by Nolan
scores = SemanticField.compute_and(
    namespace="metaqa",
    edges=subgraph_edges,          # 2-hop neighbourhood of both anchors
    source={"Nolan": 1.0},         # high-voltage anchor
    sink={"Sci-Fi": 0.0},          # ground
)
# → {"Interstellar": 0.394, "Inception": 0.312, "Dark Knight": 0.0, ...}
```

Relationship weights are declared in the ontology YAML and flow directly into the solver as conductances:

```yaml
relationships:
  directed_by: { domain: Movie, range: Person, weight: 0.9 }
  has_genre:   { domain: Movie, range: Genre,  weight: 0.7 }
  has_tags:    { domain: Movie, range: Tag,    weight: 0.2 }
```

The solver is pure Python (no numpy) with an optional numba JIT kernel that gives a **70× speedup** at ~5 000-node subgraphs; it falls back silently when numba is absent.

| Layer | What it does | Status |
|---|---|---|
| L1 — OntologyRegistry | Binary valid/invalid edge checks, YAML-declared | Built |
| L2 — SemanticField | Continuous entity scoring via DC circuit model | Built |
| L3 — Discovery Engine | Guided multi-hop traversal by field strength | Planned |

Install: `uv sync --extra kg-semantic-field`

Full visual explainer: [`demo/semantic_field_explainer.html`](demo/semantic_field_explainer.html) — open in any browser, no server needed.

## Demos

Three marimo notebooks plus two evaluation harnesses live under `demo/`:

- `demo/demo_kg_connectors.py`: surface tour of all 9 families against the shipped fixtures.
- `demo/demo_kg_build_repo.py`: builds an RDF graph from this repo (filesystem `repo:contains` + Python `repo:imports`), serializes to Turtle, and runs five SPARQL queries through `RdfLibSparqlReader` via `mloda.run_all`.
- `demo/demo_kg_ontology.py`: walks the ontology layer end to end.
- `demo/demo_semantic_field.py`: SemanticField Layer 2 — interactive director + genre query against MetaQA, with live scoring.
- `demo/semantic_field_explainer.html`: full visual explainer for SemanticField — circuit diagram, layer stack, why EM, 1/2/multi-hop examples, test results. Open in any browser.
- `demo/eval_arch1_vs_arch2.py` and `demo/eval_qa_accuracy.py`: evaluation harnesses comparing plain traversal vs. ontology-guided traversal.

Install the demo extras and open any notebook:

```bash
uv sync --extra demo
marimo edit demo/demo_kg_connectors.py
```

Every demo runs offline against a small committed sample graph: no download,
no network, no external services.

## Data and acknowledgments

The ontology demo and the two evaluation harnesses run against a small
hand-authored sample of public movie facts (`demo/data/sample_kb.txt`) written
in the triple format of the MetaQA dataset (Zhang, Yuyu et al., "Variational
Reasoning for Question Answering with Knowledge Graph", AAAI 2018,
https://github.com/yuyuz/MetaQA). The sample is committed in this repo and is
not derived from the MetaQA dataset files. The notebooks call
`demo.data.ensure_data()` at startup, which builds the sample subgraph offline.
To run against the full MetaQA benchmark (licensed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/legalcode), not
redistributed here), see [`demo/data/README.md`](demo/data/README.md).

## Development setup

**Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create virtual environment and install dependencies:**
```bash
uv venv
source .venv/bin/activate
uv sync --all-extras
```

**Run all checks with tox:**
```bash
uv tool install tox --with tox-uv
tox
```

### Run individual checks

```bash
pytest
ruff format --check --line-length 120 .
ruff check .
mypy --strict --ignore-missing-imports .
bandit -c pyproject.toml -r -q .
```

## Related repositories and documentation

- **[mloda](https://github.com/mloda-ai/mloda)**: The core library for open data access. Declaratively define what data you need, not how to get it. See [mloda.ai](https://mloda.ai) for an overview and business context and the [documentation](https://mloda-ai.github.io/mloda/) for detailed guides.
- **[mloda-registry](https://github.com/mloda-ai/mloda-registry)**: The central hub for discovering and sharing mloda plugins.
- **[Plugin development guides](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides/)**: How to build FeatureGroups, ComputeFrameworks, and Extenders.
- **[Claude Code skills](https://github.com/mloda-ai/mloda-registry/tree/main/.claude/skills/)**: Assisted plugin development for Claude Code users.
