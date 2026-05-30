"""In-process Zanzibar-shaped tuple store fake.

Validates the saas_authz contract (tenant, entity_type, relationship_type,
expand_paths, consistency_mode) without standing up SpiceDB or OpenFGA. The
tuples are loaded from a canonical JSON fixture co-located with the connector
(shape: ``{tenant: [[object_type, object_id, relation, user], ...]}``); check
/ list / expand are simple dict ops.

PROTOTYPE NOTE: this fake exercises the property *shape*. It does NOT provide
real consistency-token semantics, model-id versioning, or namespaced check
evaluation. For real semantics, use the OpenFGA / SpiceDB Python clients.

Pinned fixture (issue #5 item 16):

The connector is pinned to a single canonical fixture (``_FIXTURE_PATH``); the
family-level ``locator`` slot is dropped from ``PROPERTY_MAPPING`` and
``REQUIRED_KEYS`` because accepting a configurable locator on a connector
whose tenant enum is closed at class load would create a silent scope
mismatch (user-supplied fixtures with tenants outside ``allowed_values``
would be rejected at the matcher with no diagnostic). This mirrors the
``page_size`` drop on ``FileFixtureRestReader``: a credential slot that the
concrete cannot honor is a surface lie, so we drop it.

Tenant validation has two layers:

1. Closed-world enum at ``is_valid_credentials`` / ``connect()`` shape gate
   (issue #18 item A5). ``PROPERTY_MAPPING`` is overridden to flip
   ``tenant`` to ``strict_validation=True`` with
   ``allowed_values={"tenant_a"}``, so an unknown tenant is rejected at the
   matcher surface and at ``connect()`` before any fixture I/O.
2. Typed ``UnknownTenantError`` at ``_connect_from_slot`` for callers that
   bypass the shape gate (and as a defense-in-depth check against a
   future fixture/enum drift). Mirrors ``UnknownMemoryScopeError`` on the
   agent_memory family as a typed-error precedent.

Asymmetry note: agent_memory's ``networkx_memory`` keeps ``memory_scope_user_id``
non-strict and relies on connect-time only. The validate-time gate here is
appropriate because the saas_authz family enumerates ``tenant`` per concrete
(one tenant per fake), whereas ``memory_scope_user_id`` is open across the
agent_memory family (graphs hold many users). Closed/enumerable warrants the
matcher-safe gate; open does not.

The fixture file is the source of truth for which tenants the connector
serves; the strict-validation enum mirrors that set so the credential
surface is honest. ``test_allowed_tenants_mirror_fixture`` enforces the
alignment so a drift fails the build rather than yielding a silent
mismatch. The override pattern (rather than ``SUPPORTED_VALUES``) is forced
by the family base keeping ``tenant`` open: tenant shape varies across SaaS
systems (subdomain, instance_url, store_id, ...), so the family base cannot
declare a closed ``allowed_values`` for ``SUPPORTED_VALUES`` to narrow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import narrow_property_mapping
from open_kgo.feature_groups.kg.errors import FixtureLoadError, UnknownTenantError
from open_kgo.feature_groups.kg.fixtures import load_json_fixture
from open_kgo.feature_groups.kg.saas_authz.base import (
    SaasAuthzFeatureGroup,
    SaasAuthzReader,
)


def _validate_tuples(connector_id: str, locator: str, tenant: str, raw: Any) -> list[tuple[str, str, str, str]]:
    """Raise ``FixtureLoadError`` unless ``raw`` is a list of 4-string tuples; return as ``list[tuple]``."""
    if not isinstance(raw, list):
        raise FixtureLoadError(
            connector_id,
            locator,
            f"tenant {tenant!r} entry must be a list, got {type(raw).__name__}.",
        )
    out: list[tuple[str, str, str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 4 or not all(isinstance(s, str) for s in item):
            raise FixtureLoadError(
                connector_id,
                locator,
                f"tenant {tenant!r} index {index}: each tuple must be a list of 4 strings, got {item!r}.",
            )
        out.append((item[0], item[1], item[2], item[3]))
    return out


class InProcessTupleStoreReader(SaasAuthzReader):
    CONNECTOR_ID: ClassVar[str] = "in_process_tuple_store"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("tenant",),)
    # Strict-enum narrowings:
    #   - pagination_style: load_data does not paginate; only "none" honored.
    #   - tenant: the spec-override above already pins allowed_values to
    #     {"tenant_a"}; mirroring it in SUPPORTED_VALUES makes the contract
    #     uniform (avoids exempting it from the strict-enum honesty check).
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "pagination_style": frozenset({"none"}),
        "tenant": frozenset({"tenant_a"}),
    }
    # Waive consistency_mode: the in-process fake accepts every value at the
    # property layer but does not implement consistency semantics; real
    # saas_authz backends (SpiceDB, OpenFGA, OData) WILL dispatch on these
    # values, so narrowing here would lock the family contract to the fake's
    # single honored value and force future concretes to widen.
    _WAIVED_ENUM_KEYS: ClassVar[frozenset[str]] = frozenset({"consistency_mode"})

    # The canonical fixture this concrete serves. Pinned at class load so the
    # closed ``allowed_values`` enum on ``tenant`` (below) is a truthful
    # statement about the connector's tenant space rather than a constraint
    # the caller has to keep in sync with whatever fixture they happen to
    # point ``locator`` at.
    _FIXTURE_PATH: ClassVar[Path] = Path(__file__).parent / "fixtures" / "tuples.json"

    # Drop ``locator`` from the family-level PROPERTY_MAPPING: the fixture is
    # baked in via ``_FIXTURE_PATH``, so accepting a ``locator`` credential
    # slot would be a surface lie (mirrors ``FileFixtureRestReader``'s drop
    # of ``page_size``). Override the family-level open ``tenant`` spec with
    # a strict-validation enum scoped to the tenants this in-process store
    # actually serves (issue #5 item 16). The closed-world enum check in
    # ``_validate_mapping`` then rejects unknown tenants at
    # ``is_valid_credentials`` time and at ``connect()`` (issue #18 item A5)
    # before any fixture I/O. ``_connect_from_slot`` retains its
    # ``UnknownTenantError`` defense-in-depth check so a direct call
    # bypassing the shape gate (or a future fixture/enum drift) still
    # surfaces a typed error. ``SUPPORTED_VALUES`` does not apply here
    # because the family base sets ``strict_validation=False`` for
    # ``tenant``; see module docstring.
    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
        **narrow_property_mapping(SaasAuthzReader.PROPERTY_MAPPING, "locator"),
        "tenant": {
            **SaasAuthzReader.PROPERTY_MAPPING["tenant"],
            "explanation": "Sole tenant served by the in-process tuple store fixture.",
            "allowed_values": {"tenant_a": "Sample tenant pre-loaded in the canonical demo fixture."},
            DefaultOptionKeys.strict_validation: True,
        },
    }

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
        fixture = str(cls._FIXTURE_PATH)
        stores = load_json_fixture(cls.CONNECTOR_ID, fixture)
        tenant = str(slot["tenant"])
        if tenant not in stores:
            raise UnknownTenantError(cls.CONNECTOR_ID, tenant)
        return _validate_tuples(cls.CONNECTOR_ID, fixture, tenant, stores[tenant])

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> list[dict[str, Any]]:
        ctx = cls._prepare_load(data_access)
        tuples = cls._connect_from_slot(ctx.slot)
        # Engages PaginationMixin._validate_cross_layer: this concrete keeps
        # cursor_token in PARAMS_MAPPING (no _STRIPPED_PARAMS narrowing), so
        # it's the one composer where the cursor_token / pagination_style
        # consistency check actually fires against a real misconfiguration.
        # The fake's load_data does not paginate, so the returned dict is
        # discarded; the call exists for its validation side effect.
        cls.build_params(features, ctx.slot)

        # entity_type / relationship_type are connector defaults for SaasAuthz
        # (EntityFilterPropertyMixin) — they live in the credential slot.
        entity_type = ctx.slot.get("entity_type")
        relationship_type = ctx.slot.get("relationship_type")

        rows: list[dict[str, Any]] = []
        for ot, oid, rel, user in tuples:
            if entity_type is not None and ot != entity_type:
                continue
            if relationship_type is not None and rel != relationship_type:
                continue
            rows.append({"object_type": ot, "object_id": oid, "relation": rel, "user": user})
            if len(rows) >= ctx.result_limit:
                break
        return rows


class InProcessTupleStoreFeatureGroup(SaasAuthzFeatureGroup):
    READER_CLASS: ClassVar[type[InProcessTupleStoreReader]] = InProcessTupleStoreReader  # type: ignore[assignment]


# Regression guard: ``SUPPORTED_VALUES["tenant"]`` mirrors the spec-override's
# ``allowed_values`` for the tenant key. Pinned at module-load so a future
# change that adds a tenant to one side without the other surfaces at import
# time rather than as a confused contract-test failure later. Uses an
# explicit ``raise`` instead of ``assert`` so the check survives ``python -O``
# and doesn't trip bandit B101.
_supported_tenants = InProcessTupleStoreReader.SUPPORTED_VALUES["tenant"]
_spec_tenants = frozenset(InProcessTupleStoreReader.PROPERTY_MAPPING["tenant"]["allowed_values"].keys())
if _supported_tenants != _spec_tenants:
    raise RuntimeError(
        f"InProcessTupleStoreReader: SUPPORTED_VALUES['tenant']={sorted(_supported_tenants)} drifted from "
        f"PROPERTY_MAPPING['tenant']['allowed_values']={sorted(_spec_tenants)}; update both sides in sync."
    )
