"""In-process tuple store that paginates results and expands group usersets.

Second concrete in the ``saas_authz`` family alongside
``InProcessTupleStoreReader`` (single-page, no expansion). This reader *honors*
the family surface the first concrete pins off: cursor pagination
(``pagination_style=cursor`` + ``page_size`` + per-call ``cursor_token``) and
``expand_paths`` (single-pass structural group expansion). It is the family's
proof that the pagination + expand contract is real.

HONESTY NOTE: like the first concrete, this is a shape-only fake. It does NOT
implement real Zanzibar consistency tokens, model-id versioning, or namespaced
check evaluation; ``consistency_mode`` is accepted but not honored (waived). The
expansion is a single structural pass: a tuple whose user is a userset
reference (``group:<id>#<rel>``) is replaced by the members of that group when
``<rel>`` is listed in ``expand_paths``.

CURSOR HONESTY NOTE: ``pagination_style=cursor`` is documented family-wide
(``PaginationMixin``) as an opaque server-returned cursor sent back on the
next request. This reader never returns a cursor: ``load_data`` returns only
rows, and callers fabricate positional ``offset:<N>`` tokens themselves
(decoded by ``parse_offset_cursor``). The tokens are positional offsets over
the re-sorted expanded tuple list, so if the fixture (or its expansion)
changes between pages, a saved token silently shifts to different rows; there
is no snapshot consistency behind the token.

The tuples are loaded from a JSON fixture at ``locator`` (shape:
``{tenant: [[object_type, object_id, relation, user], ...]}``). An unknown
tenant raises ``UnknownTenantError`` at connect time (the agent_memory
``UnknownMemoryScopeError`` precedent), so ``tenant`` stays open (non-strict)
rather than pinned to a closed enum.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
from open_kgo.feature_groups.kg.errors import InvalidCredentialShape, UnknownTenantError
from open_kgo.feature_groups.kg.fixtures import load_json_fixture
from open_kgo.feature_groups.kg.mixins import cursor_page_slice
from open_kgo.feature_groups.kg.saas_authz.base import (
    SaasAuthzFeatureGroup,
    SaasAuthzReader,
)
from open_kgo.feature_groups.kg.saas_authz.shared import AuthzTuple as _Tuple
from open_kgo.feature_groups.kg.saas_authz.shared import validate_tuples


def _validate_expand_paths(connector_id: str, raw: Any) -> tuple[str, ...]:
    """Coerce the ``expand_paths`` slot value into a tuple of relation strings.

    ``expand_paths`` is ``strict_validation=False`` (no closed enum), so
    ``_validate_shape`` never inspects it; this reader is the first to honor it
    and must guard the shape itself (page_size has ``parse_page_size``; this is
    its sibling guard). The expansion opt-out is deliberately narrow: only
    ``None`` (absent) and an empty ``list``/``tuple`` map to an empty tuple.
    Everything else that is not a list/tuple of non-empty strings raises a
    typed ``InvalidCredentialShape``: a bare ``str``/``bytes`` would iterate
    character-wise and silently disable expansion, falsey scalars (``0``,
    ``False``, ``b""``) would silently opt out, and a ``dict`` would otherwise
    pass an element check via key iteration. All of those are caller mistakes
    worth a loud error rather than a silent no-expansion run.
    """
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        raise InvalidCredentialShape(
            f"{connector_id}: expand_paths must be a list of relation strings, not a bare "
            f"{type(raw).__name__} ({raw!r}); wrap a single path in a list (e.g. ['member'])."
        )
    if not isinstance(raw, (list, tuple)):
        raise InvalidCredentialShape(
            f"{connector_id}: expand_paths must be a list or tuple of relation strings, "
            f"got {type(raw).__name__} ({raw!r})."
        )
    for item in raw:
        if not isinstance(item, str) or not item:
            raise InvalidCredentialShape(
                f"{connector_id}: expand_paths entries must be non-empty strings, got {type(item).__name__} ({item!r})."
            )
    return tuple(raw)


def _expand_usersets(rows: list[_Tuple], all_tuples: list[_Tuple], expand_paths: tuple[str, ...]) -> list[_Tuple]:
    """Replace ``group:<id>#<rel>`` users in ``rows`` with that group's members.

    The membership index is built from ``all_tuples`` (not the filtered
    ``rows``) so group-membership tuples removed by an ``entity_type`` /
    ``relationship_type`` filter are still available to resolve a userset.
    Only usersets whose ``<rel>`` is listed in ``expand_paths`` are expanded;
    everything else passes through unchanged.

    Structural-expansion semantics: an expanded userset that resolves to no
    members (an empty or absent group) yields **zero** grants — the original
    ``group:<id>#<rel>`` row is replaced by its (empty) membership, so it drops
    out of the result. A userset whose ``<rel>`` is NOT in ``expand_paths`` is
    left untouched and passes through as-is.

    Single-pass semantics (a deliberate divergence from Zanzibar's recursive
    Expand): the membership index is consulted exactly once per input row, so
    expansion *output* is never re-expanded. Two consequences callers must
    know about:

    - A nested userset passes through unexpanded EVEN IF its ``<rel>`` is in
      ``expand_paths``: when ``group:eng#member`` contains
      ``group:sub#member`` as a member, expanding the eng row emits the
      ``group:sub#member`` reference itself, not sub's members.
    - A self-referential userset (``group:eng#member`` listing itself as a
      member) terminates trivially by emitting itself; there is no recursion
      to cycle through.
    """
    if not expand_paths:
        return rows
    members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for object_type, object_id, relation, user in all_tuples:
        if object_type == "group":
            members[(object_id, relation)].append(user)

    expanded: list[_Tuple] = []
    for object_type, object_id, relation, user in rows:
        if user.startswith("group:") and "#" in user:
            group_ref = user[len("group:") :]
            group_id, _, group_rel = group_ref.partition("#")
            if group_rel in expand_paths:
                # dict.fromkeys dedupes members defined by repeated membership
                # tuples while preserving order, so a single userset's member
                # list can't emit the same member twice. This is scoped to one
                # userset's expansion only: cross-path / global dedup across
                # different usersets or pass-through rows is NOT done (and not
                # intended) here.
                for member_user in dict.fromkeys(members.get((group_id, group_rel), [])):
                    expanded.append((object_type, object_id, relation, member_user))
                continue
        expanded.append((object_type, object_id, relation, user))
    return expanded


class PaginatedTupleStoreReader(SaasAuthzReader):
    CONNECTOR_ID: ClassVar[str] = "paginated_tuple_store"
    # pagination_style is REQUIRED (not just narrowed): the family default is
    # "none", which this reader does not honor, and SUPPORTED_VALUES only
    # validates keys present in the slot. An omitted pagination_style would
    # otherwise pass is_valid_credentials, serve a cursor-paginated page 1,
    # and then reject its own continuation token (cursor_token + the
    # defaulted "none" fails PaginationMixin._validate_cross_layer).
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",), ("tenant",), ("pagination_style",))

    # pagination_style is narrowed to the cursor style this reader implements;
    # page_size, cursor_token, and expand_paths are RETAINED (the new surface).
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {
        "pagination_style": frozenset({"cursor"}),
    }
    # Waive consistency_mode: accepted for forward-compat but this fake
    # implements no consistency semantics (same disposition as the first
    # concrete; real SpiceDB / OpenFGA backends will dispatch on it).
    _WAIVED_ENUM_KEYS: ClassVar[frozenset[str]] = frozenset({"consistency_mode"})

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> list[_Tuple]:
        locator = str(slot["locator"])
        stores = load_json_fixture(cls.CONNECTOR_ID, locator)
        tenant = str(slot["tenant"])
        if tenant not in stores:
            raise UnknownTenantError(cls.CONNECTOR_ID, tenant)
        return validate_tuples(cls.CONNECTOR_ID, locator, tenant, stores[tenant])

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        all_tuples = connection
        # Engages PaginationMixin._validate_cross_layer (cursor_token requires a
        # cursor-family pagination_style, which this concrete pins).
        params = cls.build_params(features, ctx.slot)

        # entity_type / relationship_type are connector defaults (slot-level).
        entity_type = ctx.slot.get("entity_type")
        relationship_type = ctx.slot.get("relationship_type")
        expand_paths = _validate_expand_paths(cls.CONNECTOR_ID, ctx.slot.get("expand_paths"))

        filtered = [
            t
            for t in all_tuples
            if (entity_type is None or t[0] == entity_type) and (relationship_type is None or t[2] == relationship_type)
        ]
        expanded = _expand_usersets(filtered, all_tuples, expand_paths)
        expanded.sort()

        page = cursor_page_slice(
            cls.CONNECTOR_ID,
            expanded,
            cursor_token=params.get("cursor_token"),
            page_size_value=ctx.slot.get("page_size"),
            result_limit=ctx.result_limit,
        )
        return [{"object_type": ot, "object_id": oid, "relation": rel, "user": user} for ot, oid, rel, user in page]


class PaginatedTupleStoreFeatureGroup(SaasAuthzFeatureGroup):
    READER_CLASS: ClassVar[type[PaginatedTupleStoreReader]] = PaginatedTupleStoreReader
