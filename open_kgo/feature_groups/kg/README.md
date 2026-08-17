# KG connector base groups (prototype)

Nine families derived from a 103-system survey (see
Version B of the family map below). Each family is a
`<Family>Reader(KgConnectorReaderBase)` + `<Family>FeatureGroup`, paired with
**two** concrete plugins that run against in-memory libraries or local fixture
files. No Docker, no external services. The second concrete per family is
either a different backend on the same contract (backend variety) or a backend
that exercises family surface the first concrete narrows / strips (see
issue #16); the **New surface** column records which.

## Family map

| Family | Concrete plugins | Pedigree | New surface vs. first plugin |
|---|---|---|---|
| `network_pg` | `KuzuCypherReader`, `GrandCypherReader` | Real Python libs (`kuzu`; `grand-cypher` over `networkx`) | None — both embedded, so `read_consistency` / `transaction_mode` stay waived. Second Cypher engine. |
| `rdf` | `RdfLibSparqlReader`, `OxigraphSparqlReader` | Real Python libs (`rdflib`; `pyoxigraph`) | None — `result_format` / `reasoning_profile` stay narrowed on both. Second SPARQL engine. |
| `embedded` | `NetworkxEmbeddedReader`, `IGraphEmbeddedReader` | Real Python libs (`networkx`; `igraph`) | None — same formats / operations. Second embedded graph lib. |
| `rest_public` | `FileFixtureRestReader`, `FileFixturePagedRestReader` | Prototype + JSON fixtures | **`pagination_style=page` + `page_size`** (cursor concrete narrows to `cursor` and drops `page_size`). |
| `lineage` | `DbtManifestReader`, `OpenLineageReader` | Real artifacts (dbt `manifest.json`; OpenLineage run-event JSON) | None — both walk UPSTREAM / DOWNSTREAM / BOTH. Second lineage artifact. |
| `code_build` | `CycloneDxSbomReader`, `SpdxSbomReader` | Real artifacts (CycloneDX; SPDX) | **TraversalMixin** (`lineage_direction` / `upstream_depth` / `downstream_depth`) — SPDX walks the `DEPENDS_ON` graph the CycloneDX concrete strips. |
| `saas_authz` | `InProcessTupleStoreReader`, `PaginatedTupleStoreReader` | Prototype | **`pagination_style=cursor` + `page_size` + `cursor_token` + `expand_paths`** (structural group expansion; still no real consistency-token semantics). |
| `agent_memory` | `NetworkxMemoryReader`, `GraphWalkMemoryReader` | Prototype on top of NetworkX | **`retrieval_mode=graph`** (lexical concrete narrows to `lexical`). |
| `citation_rest` | `FileFixtureCitationReader`, `PaginatedCitationReader` | Prototype + JSON fixtures | **`pagination_style=cursor` + `page_size` + `cursor_token` + `entity_type`** (Reactome concrete drops / strips them). |

## Layout

```
kg/
├── base.py        KgConnectorReaderBase / QueryReader / ParamReader / KgConnectorFeatureGroupBase
├── mixins.py      EntityFilterPropertyMixin / EntityFilterParamMixin / PaginationMixin / TraversalMixin / InferenceMixin
├── errors.py      InvalidCredentialShape / MissingRequiredKeysError / PropertyMappingCollision / MissingEnvVarError
├── fixtures.py    Shared resource-cache loaders (load_json_fixture / load_rdf_graph / load_oxigraph_store / load_kuzu_database)
├── conftest.py    Autouse fixture clearing the resource caches between tests
├── ontology/      Ontology registry for typed traversal (registry.py + tests/)
├── tests/
│   ├── kg_contract.py             Universal abstract test base (5 adapter methods + 26 contract tests)
│   ├── _helpers.py                Shared run_query / make_valid_credentials helpers
│   ├── _discovery.py              Reader discovery (import_all_kg_readers / walk_subclasses / ...)
│   ├── _family_cases.py           Shared all-family usage registry (CASES) + discovery helpers
│   ├── test_kg_usage_smoke.py     All-family usage smoke over CASES
│   ├── test_kg_registry_integrity.py  CASES-vs-discovery gates (coverage, two-per-family floor)
│   ├── test_kg_catalog_declarations.py  Source-slot gate + cross-family asymmetry catalog
│   ├── test_resource_cache.py     Resource-lifecycle / shared-cache contract tests
│   └── test_*.py                  Further cross-cutting suites (validation, pagination, discovery, ...)
└── <family>/
    ├── base.py             <Family>Reader + <Family>FeatureGroup (per-family PROPERTY_MAPPING)
    ├── <concrete>.py       Concrete plugin (CONNECTOR_ID + connect/build_query/load_data)
    └── tests/
        ├── kg_<family>_contract.py    Per-family contract base
        ├── test_<concrete>.py         Concrete adapter test class
        └── fixtures/                  (where applicable)
```

## How a concrete plugin works

A concrete plugin (e.g. `RdfLibSparqlReader`) is a `ReadDB` subclass with:
- `CONNECTOR_ID: ClassVar[str]` — keys the credential dict in
  `DataAccessCollection.credential_dicts`.
- `REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]]` — declares which
  credential keys are mandatory; tuple of OR-groups (all groups AND'ed,
  members within a group OR'ed). Empty tuple means no required keys.
- Inheritance from `QueryReader` (per-call input is a query string) or
  `ParamReader` (per-call input is a typed parameter dict declared on
  `PARAMS_MAPPING`).
- `connect(creds)`, `build_query(features)` or `build_params(features)`,
  `load_data(data_access, features)` — the methods mloda's
  `BaseInputData.load` calls.

Properties are declared in `PROPERTY_MAPPING` as `PropertySpec` values built
via `kg/spec.py`'s `property_spec(explanation, strict=..., allowed_values=...,
default=...)` wrapper. Strict-validation enums are checked in
`is_valid_credentials` (inherited from the universal base).

### Honest surface: two narrowing tools

A concrete plugin can narrow a family-level enum to the subset it actually
honors at runtime by setting
`SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]]`. The same hook is
consulted by both the credential validator and the per-call params
validator, so a value the family declares but this concrete does not
support is rejected at validate-time rather than silently no-oping at load.
The framework checks the invariant `SUPPORTED_VALUES[k] ⊆ family-allowed[k]`
at class-definition time, so typos and out-of-set values fail loudly rather
than silently locking the connector.

A concrete plugin that ignores an entire family-level key should override
`PROPERTY_MAPPING` / `PARAMS_MAPPING` to drop it. Both surfaces then reject
the dropped key:

- **Credential surface** (closed-world): the universal `_validate_shape`
  rejects any key not declared in this concrete's `PROPERTY_MAPPING`.
- **Per-call surface**: `ParamReader.__init_subclass__` derives
  `_STRIPPED_PARAMS = family_base.PARAMS_MAPPING - cls.PARAMS_MAPPING` at
  class creation. The base `load` hook rejects any of those keys that
  appear in `feature.options`. Unrelated keys in `feature.options.context`
  (mloda core, other plugins) still pass through; only family-declared
  keys this concrete dropped count as a surface lie.

Net effect: a concrete plugin author declares only what they keep
(`PROPERTY_MAPPING`, `PARAMS_MAPPING`, optional `SUPPORTED_VALUES`); the
framework derives the rejection contract.

## How tests work

A concrete plugin's test class subclasses the per-family `*ContractTestBase`
(which itself subclasses `KgConnectorContractBase`) and implements 5 adapter
methods (`connector_reader_class`, `valid_credentials`, `invalid_credentials`,
`feature_under_test`, `expected_row_shape`). It inherits 26 universal contract
tests + per-family assertions for free. ~20-30 lines per concrete plugin.

## No-Docker testing policy

All tests run against in-memory libraries (`rdflib`, `networkx`, `kuzu`),
file fixtures (`.ttl`, `.json`, `.gml`), and Python prototype implementations.
**Do not** introduce Docker, real DB clients, or network calls. If a contract
assertion needs semantics the in-memory substrate cannot reproduce (e.g. true
Zanzibar consistency tokens, OpenMetadata API versioning, real OAuth2 flows),
document the gap in the family `__init__.py` and skip the test — do not add
infrastructure.

## What this prototype does NOT validate

- **`network_pg`**: both concretes (KuzuDB, GrandCypher) are embedded;
  `read_consistency`, `transaction_mode`, routing, bookmarks are no-ops. Real
  Neo4j / Memgraph / Neptune behavior is out of scope.
- **`saas_authz`**: neither tuple-store concrete implements real Zanzibar
  consistency tokens, model-id versioning, or namespaced check evaluation.
  `PaginatedTupleStoreReader` adds cursor pagination and *structural*
  `expand_paths` group expansion, but the expansion is a single in-process
  pass, not a real userset-rewrite evaluator.
- **`citation_rest`** / **`rest_public`**: file fixtures exercise both
  pagination shapes (cursor and page-number, across the two concretes per
  family) but not real `rate_limit_pace` enforcement.
- **`rdf`**: both SPARQL engines (rdflib, oxigraph) return JSON-binding-shaped
  rows and have no inference, so `result_format` stays narrowed to JSON and
  `reasoning_profile` to `none`.
- **`agent_memory`**: `retrieval_mode=lexical` (NetworkxMemoryReader) and
  `retrieval_mode=graph` (GraphWalkMemoryReader) are implemented; `vector` /
  `hybrid` remain rejected at `is_valid_credentials` time via
  `SUPPORTED_VALUES` so neither concrete lies about what it honors.
