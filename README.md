[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# open-kgo

Open Knowledge Graphs and Ontologies plugin for [mloda](https://github.com/mloda-ai/mloda). Visit [mloda.ai](https://mloda.ai) for an overview and business context, the [GitHub repository](https://github.com/mloda-ai/mloda) for technical context, or the [documentation](https://mloda-ai.github.io/mloda/) for detailed guides.

## Related Repositories

- **[mloda](https://github.com/mloda-ai/mloda)**: The core library for open data access. Declaratively define what data you need, not how to get it.
- **[mloda-registry](https://github.com/mloda-ai/mloda-registry)**: The central hub for discovering and sharing mloda plugins.

## Quickstart

Install the connectors and run a SPARQL query against the Turtle sample shipped in this repo — no Docker, no network:

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

Swap `rdflib_sparql` for any of the nine connector families — same `Feature` → `mloda.run_all` shape, different reader.

## KG connectors

`open_kgo/feature_groups/kg/` ships a 9-family knowledge-graph connector taxonomy (`network_pg`, `rdf`, `embedded`, `rest_public`, `lineage`, `code_build`, `saas_authz`, `agent_memory`, `citation_rest`), with at least one concrete plugin per family running against in-memory libraries or local file fixtures. See `open_kgo/feature_groups/kg/README.md` for the family map.

Install all KG extras with: `uv sync --extra kg-all`.

> **One feature per call.** KG readers dispatch a single feature per load: every
> reader rejects a multi-feature `FeatureSet` rather than silently labelling all
> rows with one feature name. Request features individually (one `Feature` per
> `mloda.run_all` slot) rather than batching `N` of them into a single reader call.

> **No-Docker testing policy.** Every connector test runs against rdflib, networkx, kuzu (embedded), or file fixtures. No Docker, no external services, no network calls.

## Demos

Three marimo notebooks plus two evaluation harnesses live under `demo/`:

- `demo/demo_kg_connectors.py`: surface tour of all 9 families against the shipped fixtures.
- `demo/demo_kg_build_repo.py`: builds an RDF graph from this repo (filesystem `repo:contains` + Python `repo:imports`), serializes to Turtle, and runs five SPARQL queries through `RdfLibSparqlReader` via `mloda.run_all`.
- `demo/demo_kg_ontology.py`: walks the ontology layer end to end.
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

## Development Setup with uv

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

## Related Documentation

Guides for plugin development can be found in mloda-registry:

- https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides/

Claude Code users can leverage the skills in mloda-registry for assisted plugin development:

- https://github.com/mloda-ai/mloda-registry/tree/main/.claude/skills/
