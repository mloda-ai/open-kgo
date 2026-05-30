# KG connector base groups (prototype)

Nine families derived from a 103-system survey (see
`docs/kg-connector-base-classes.md` Version B). Each family is a
`<Family>Reader(KgConnectorReaderBase)` + `<Family>FeatureGroup`, paired with
at least one concrete plugin that runs against in-memory libraries or local
fixture files. No Docker, no external services.

## Family map

| Family | Concrete plugin | Pedigree | Notes |
|---|---|---|---|
| `network_pg` | `KuzuCypherReader` | Real Python lib (`kuzu`) | Embedded but Cypher-shaped. Does NOT exercise `read_consistency` / `transaction_mode`. |
| `rdf` | `RdfLibSparqlReader` | Real Python lib (`rdflib`) | Canonical fit — in-memory Graph + SPARQL. |
| `embedded` | `NetworkxEmbeddedReader` | Real Python lib (`networkx`) | Loads `.gml` from `locator` (filesystem path; no network). |
| `rest_public` | `FileFixtureRestReader` | Prototype + JSON fixtures | Pagination/cursor against static `page_<N>.json`; `pagination_style` narrowed to `cursor` via `SUPPORTED_VALUES`; `page_size` rejected at credential time, `cursor_token` / `entity_type` rejected at per-call time (`_STRIPPED_PARAMS`). No real `rate_limit_pace`. |
| `lineage` | `DbtManifestReader` | Real artifact (dbt manifest.json) | Walks `parent_map` / `child_map`. |
| `code_build` | `CycloneDxSbomReader` | Real artifact (CycloneDX) | Parses fixture SBOM components; does not walk the `dependencies` graph; all TraversalMixin / EntityFilter keys rejected at per-call time (`_STRIPPED_PARAMS`). |
| `saas_authz` | `InProcessTupleStoreReader` | Prototype | Zanzibar-shaped tuple list. Validates property shape only — no real consistency-token semantics. |
| `agent_memory` | `NetworkxMemoryReader` | Prototype on top of NetworkX | Lexical search only; vector / hybrid / graph rejected at validate-time via `SUPPORTED_VALUES`. |
| `citation_rest` | `FileFixtureCitationReader` | Prototype + JSON fixture | Reactome-shaped catalog with ancestor walk; `pagination_style` / `page_size` rejected at credential time, `cursor_token` / `entity_type` rejected at per-call time (`_STRIPPED_PARAMS`). |

## Layout

```
kg/
├── base.py        KgConnectorReaderBase / QueryReader / ParamReader / KgConnectorFeatureGroupBase
├── mixins.py      EntityFilterPropertyMixin / EntityFilterParamMixin / PaginationMixin / TraversalMixin / InferenceMixin
├── errors.py      InvalidCredentialShape / MissingRequiredKeysError / PropertyMappingCollision / MissingEnvVarError
├── tests/
│   └── kg_contract.py   Universal abstract test base (5 adapter methods + 9 contract tests)
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

Properties are declared in `PROPERTY_MAPPING` per the data-operations
discipline (`DefaultOptionKeys.context: True`,
`DefaultOptionKeys.strict_validation: True/False`). Strict-validation enums are
checked in `is_valid_credentials` (inherited from the universal base).

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
`feature_under_test`, `expected_row_shape`). It inherits ~10 universal contract
assertions + per-family assertions for free. ~20-30 lines per concrete plugin.

## No-Docker testing policy

All tests run against in-memory libraries (`rdflib`, `networkx`, `kuzu`),
file fixtures (`.ttl`, `.json`, `.gml`), and Python prototype implementations.
**Do not** introduce Docker, real DB clients, or network calls. If a contract
assertion needs semantics the in-memory substrate cannot reproduce (e.g. true
Zanzibar consistency tokens, OpenMetadata API versioning, real OAuth2 flows),
document the gap in the family `__init__.py` and skip the test — do not add
infrastructure.

## What this prototype does NOT validate

- **`network_pg`**: KuzuDB is embedded; `read_consistency`, `transaction_mode`,
  routing, bookmarks are no-ops. Real Neo4j / Memgraph / Neptune behavior is
  out of scope.
- **`saas_authz`**: in-process tuple store has no Zanzibar consistency tokens,
  no model-id versioning, no namespaced check evaluation.
- **`citation_rest`** / **`rest_public`**: file fixtures exercise pagination
  shape but not real `rate_limit_pace` enforcement.
- **`agent_memory`**: only `retrieval_mode=lexical` is implemented; the
  other family-level values (`vector`, `hybrid`, `graph`) are rejected at
  `is_valid_credentials` time via `SUPPORTED_VALUES` so the surface does
  not lie about what this concrete honors.
