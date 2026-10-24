"""Cross-cutting mixins for KG connector readers.

Each mixin contributes a ``PROPERTY_MAPPING_DELTA`` dict (for connector-level
credential keys) and / or a ``PARAMS_MAPPING_DELTA`` dict (for per-call
parameter keys); family bases merge these into their ``PROPERTY_MAPPING`` /
``PARAMS_MAPPING`` via ``compose_property_mapping`` (which raises on collision).

Mixins are plain ``object`` subclasses so MRO stays trivial. A family base
declared as ``class FooReader(SomeMixin, KgConnectorReaderBase)`` ends up with
``[FooReader, SomeMixin, KgConnectorReaderBase, CredentialRules, object]``.

The Property/Param split mirrors the ``QueryReader`` / ``ParamReader`` split
in ``base.py``: the same key may legitimately appear on either side depending
on whether the family treats it as a connector default or as per-call input.
The ``EntityFilter*`` pair is the canonical example (saas_authz uses property
defaults; lineage / code_build use per-call params).
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Mapping, Sequence, TypeVar

from open_kgo.feature_groups.kg.errors import InvalidCredentialShape
from open_kgo.feature_groups.kg.spec import property_spec
from open_kgo.feature_groups.kg.validation import parse_bounded_int

# Single source of truth for pagination styles: each value carries a family
# tag and a docstring. The PaginationMixin spec dict and the cursor-family set
# are both derived from this mapping so adding a new style requires classifying
# it; silent drift (style added to the spec but absent from the cursor-family
# membership check) is structurally impossible.
_PAGINATION_STYLES: dict[str, tuple[str, str]] = {
    "cursor": ("cursor", "Opaque cursor returned in response, sent on next request."),
    "page": ("counter", "Integer page number, increment to advance."),
    "offset": ("counter", "Integer offset + limit (start_rows style)."),
    "odata-nextLink": ("cursor", "Full URL embedded in @odata.nextLink response field."),
    "cursorMark": ("cursor", "Solr-style cursorMark token."),
    "start_rows": ("counter", "start + rows pair (Solr legacy)."),
    "none": ("none", "No pagination (single response)."),
}

_CURSOR_FAMILY_STYLES: frozenset[str] = frozenset(
    style for style, (family, _) in _PAGINATION_STYLES.items() if family == "cursor"
)


# Exact token grammar for parse_offset_cursor: "offset:" followed by one or
# more ASCII decimal digits. fullmatch() anchors both ends.
_OFFSET_CURSOR_RE = re.compile(r"offset:[0-9]+")


def parse_offset_cursor(connector_id: str, cursor_token: Any) -> int:
    """Parse an ``"offset:<N>"`` cursor token into a non-negative integer offset.

    Shared by the cursor-paginating concretes (``PaginatedCitationReader``,
    ``PaginatedTupleStoreReader``) so the token convention lives in one place
    alongside ``PaginationMixin``. ``None`` (the first-page case) maps to
    offset 0. The grammar is exactly ``^offset:[0-9]+$``: after the
    ``offset:`` prefix the remainder must be one or more ASCII decimal
    digits. Leading zeros are plain decimal (``"offset:007"`` is 7), but
    signs (``"offset:+5"``), whitespace (``"offset: 5"``), underscore
    separators (``"offset:5_0"``, which a bare ``int()`` would silently parse
    as 50), and non-ASCII digits are all malformed and raise
    ``InvalidCredentialShape``, so a hand-typed cursor surfaces a typed error
    rather than silently restarting from the first page or landing on the
    wrong offset.
    """
    if cursor_token is None:
        return 0
    token = str(cursor_token)
    if _OFFSET_CURSOR_RE.fullmatch(token):
        return int(token.removeprefix("offset:"))
    raise InvalidCredentialShape(
        f"{connector_id}: cursor_token {cursor_token!r} is malformed; expected 'offset:<N>' "
        f"with N one or more ASCII decimal digits."
    )


def parse_page_size(connector_id: str, value: Any, default: int) -> int:
    """Validate and return a positive-int ``page_size``, falling back to ``default`` when unset.

    Shared by the cursor/page-paginating concretes (``PaginatedCitationReader``,
    ``FileFixturePagedRestReader``, ``PaginatedTupleStoreReader``) so the
    runtime guard lives in one place. ``page_size`` is ``strict_validation=False``
    by family design (no closed enum), so ``_validate_shape`` never sees it;
    these are the first concretes to honor the value and must guard it
    themselves. ``None`` (absent / opt-out) maps to ``default``; otherwise the
    value must be a non-bool ``int >= 1``, mirroring ``_validate_result_limit``,
    bool is rejected explicitly (it is an ``int`` subclass, but a page size of
    ``True``/``False`` is a caller mistake), and strings / floats / ``< 1``
    raise ``InvalidCredentialShape`` so a typo surfaces a typed error rather
    than a raw ``ValueError`` mid-load or a silently wrong slice.
    """
    return parse_bounded_int(connector_id, "page_size", value, min_value=1, default=default)


_ItemT = TypeVar("_ItemT")


def cursor_page_slice(
    connector_id: str,
    items: Sequence[_ItemT],
    *,
    cursor_token: Any,
    page_size_value: Any,
    result_limit: int,
    default_page_size: int = 100,
) -> list[_ItemT]:
    """Shared cursor-pagination epilogue: decode the token, guard the size, slice the page.

    The two cursor-paginating concretes (``PaginatedCitationReader``,
    ``PaginatedTupleStoreReader``) used to spell the same three lines each:
    ``parse_offset_cursor`` + ``parse_page_size`` + the
    ``[offset : offset + page_size][:result_limit]`` double slice. Centralising
    the epilogue keeps the token convention, the size guard, and the
    result-limit cap in one place next to ``PaginationMixin``.

    ``items`` must already be deterministically ordered (both call sites sort
    first); the positional ``offset:<N>`` token indexes into that order, so an
    unsorted input would make a saved token return arbitrary rows. The caller
    passes the raw ``cursor_token`` param and ``page_size`` slot values; the
    typed-error guards in the two parse helpers fire from here unchanged.
    """
    offset = parse_offset_cursor(connector_id, cursor_token)
    page_size = parse_page_size(connector_id, page_size_value, default_page_size)
    return list(items[offset : offset + page_size][:result_limit])


_ENTITY_FILTER_KEYS: dict[str, Any] = {
    "entity_type": property_spec(
        "Object type for the request (e.g. 'document', 'group').",
    ),
    "relationship_type": property_spec(
        "Relation/permission type (e.g. 'viewer', 'transitiveMembers').",
    ),
    "expand_paths": property_spec(
        "Relationship/property paths to expand (e.g. OData $expand or Zanzibar Expand).",
        default=(),
    ),
}


class EntityFilterPropertyMixin:
    """Connector-default entity/relation filter keys.

    Carries ``entity_type``, ``relationship_type``, ``expand_paths`` as
    connector-level defaults (i.e. credentials). Used by ``saas_authz`` where
    the bound tenant fixes the filter shape. Lineage / code_build use the
    ``EntityFilterParamMixin`` variant because there those keys are per-call
    walk constraints.
    """

    PROPERTY_MAPPING_DELTA: ClassVar[dict[str, Any]] = dict(_ENTITY_FILTER_KEYS)


class EntityFilterParamMixin:
    """Per-call entity/relation filter keys.

    Mirror of ``EntityFilterPropertyMixin`` for connectors that interpret
    ``entity_type``/``relationship_type``/``expand_paths`` as per-call walk
    constraints rather than connector defaults. Used by ``lineage`` and
    ``code_build``.
    """

    PARAMS_MAPPING_DELTA: ClassVar[dict[str, Any]] = dict(_ENTITY_FILTER_KEYS)


class PaginationMixin:
    """Pagination properties for connectors that page through results.

    Splits into:
    - ``PROPERTY_MAPPING_DELTA`` (``pagination_style``, ``page_size``):
      connector-level defaults.
    - ``PARAMS_MAPPING_DELTA`` (``cursor_token``): per-call continuation
      token, supplied by the caller when resuming a paged scan.

    Used by ``rest_public``, ``citation_rest``, ``saas_authz``, ``agent_memory``.

    Note: the ``_validate_cross_layer`` override is reachable only for
    ``ParamReader`` composers via ``ParamReader.build_params(features, creds)``.
    ``QueryReader`` composers (``agent_memory``) inherit the property keys but
    have no ``build_params`` to invoke the cross-layer hook; the override is
    inert for them. Concrete plugins that strip ``cursor_token`` from their
    ``PARAMS_MAPPING`` also short-circuit the check (params dict won't carry
    the key), but the ``_STRIPPED_PARAMS`` rejection fires earlier in
    ``build_params`` regardless.
    """

    PROPERTY_MAPPING_DELTA: ClassVar[dict[str, Any]] = {
        "pagination_style": property_spec(
            "Pagination strategy used by the remote endpoint.",
            strict=True,
            allowed_values={style: explanation for style, (_, explanation) in _PAGINATION_STYLES.items()},
            default="none",
        ),
        "page_size": property_spec(
            "Records per page; bounded by remote per-system maximum.",
            default=100,
        ),
    }

    PARAMS_MAPPING_DELTA: ClassVar[dict[str, Any]] = {
        "cursor_token": property_spec(
            "Opaque cursor for pagination_style=cursor; supplied by caller on continuation.",
        ),
    }

    @classmethod
    def _validate_cross_layer(cls, creds: Mapping[str, Any], params: dict[str, Any]) -> None:
        """Reject ``cursor_token`` paired with a non-cursor-family ``pagination_style``.

        ``pagination_style`` is a connector default (PROPERTY_MAPPING) and
        ``cursor_token`` is a per-call param (PARAMS_MAPPING). Only a
        post-merge check sees both. The literal "require cursor_token on
        continuation calls" rule from the design notes isn't enforceable from
        a single call (there's no first-vs-continuation signal), but the
        symmetric "if cursor_token is supplied, pagination_style must accept
        it" rule is, and it catches the same misconfiguration class.

        Deliberately under-covers the original intent: a cursor-family
        ``pagination_style`` with no ``cursor_token`` passes here, even
        though a user who never supplies one cannot advance past the first
        page. Detecting that requires a first-call signal the framework
        does not provide, so it stays out of scope.
        """
        super()._validate_cross_layer(creds, params)  # type: ignore[misc]
        if params.get("cursor_token") is None:
            return
        # Default to "none" so a missing pagination_style + cursor_token still raises
        # rather than silently bypassing the membership check.
        style = creds.get("pagination_style", "none")
        if style not in _CURSOR_FAMILY_STYLES:
            connector_id = getattr(cls, "CONNECTOR_ID", "") or cls.__name__
            raise InvalidCredentialShape(
                f"{connector_id}: cursor_token is set but pagination_style={style!r} is not a "
                f"cursor-family style (allowed: {sorted(_CURSOR_FAMILY_STYLES)}). Either drop "
                f"cursor_token or set pagination_style to a cursor-family value."
            )


class TraversalMixin:
    """Traversal direction/depth properties for connectors that walk relationships.

    The keys are per-call (a call can vary direction or depth without
    rebinding credentials), so this mixin contributes only to
    ``PARAMS_MAPPING``. Composes with ``EntityFilterParamMixin`` when an
    entity/relation filter is also needed.

    Used by ``lineage`` and ``code_build``. The split ``upstream_depth`` /
    ``downstream_depth`` (vs single ``hops``) follows OpenMetadata's API which
    exposes them as independent integers.
    """

    PARAMS_MAPPING_DELTA: ClassVar[dict[str, Any]] = {
        "lineage_direction": property_spec(
            "Direction of the lineage walk relative to the start asset.",
            strict=True,
            allowed_values={
                "UPSTREAM": "Walk towards sources/dependencies.",
                "DOWNSTREAM": "Walk towards dependents/consumers.",
                "BOTH": "Walk both directions.",
                "ancestors": "Reactome-style ancestors traversal.",
                "descendants": "Reactome-style descendants traversal.",
            },
            default="BOTH",
        ),
        "upstream_depth": property_spec(
            "Depth limit for upstream traversal (independent of downstream).",
            default=1,
        ),
        "downstream_depth": property_spec(
            "Depth limit for downstream traversal (independent of upstream).",
            default=0,
        ),
    }


class InferenceMixin:
    """Reasoning/inference profile property.

    Used by ``rdf`` (always) and ``network_pg`` (optional, when the engine
    supports inference). Replaces the older boolean ``infer`` flag with a
    profile enum because real engines (Stardog, GraphDB, Oracle) expose named
    rule sets, not a single on/off switch.
    """

    PROPERTY_MAPPING_DELTA: ClassVar[dict[str, Any]] = {
        "reasoning_profile": property_spec(
            "Inference profile / rule set the engine should apply.",
            strict=True,
            allowed_values={
                "none": "No inference; raw triples/edges.",
                "rdfs": "RDFS entailment.",
                "owl-rl": "OWL 2 RL profile.",
                "owl-dl": "OWL 2 DL profile.",
                "custom": "Vendor-specific named ruleset (concrete plugin should validate further).",
            },
            default="none",
        ),
    }
