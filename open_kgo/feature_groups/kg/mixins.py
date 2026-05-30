"""Cross-cutting mixins for KG connector readers.

Each mixin contributes a ``PROPERTY_MAPPING_DELTA`` dict (for connector-level
credential keys) and / or a ``PARAMS_MAPPING_DELTA`` dict (for per-call
parameter keys); family bases merge these into their ``PROPERTY_MAPPING`` /
``PARAMS_MAPPING`` via ``compose_property_mapping`` (which raises on collision).

Mixins are plain ``object`` subclasses so MRO stays trivial. A family base
declared as ``class FooReader(SomeMixin, KgConnectorReaderBase)`` ends up with
``[FooReader, SomeMixin, KgConnectorReaderBase, ReadDB, BaseInputData, object]``.

The Property/Param split mirrors the ``QueryReader`` / ``ParamReader`` split
in ``base.py``: the same key may legitimately appear on either side depending
on whether the family treats it as a connector default or as per-call input.
The ``EntityFilter*`` pair is the canonical example (saas_authz uses property
defaults; lineage / code_build use per-call params).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.errors import InvalidCredentialShape

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


_ENTITY_FILTER_KEYS: dict[str, Any] = {
    "entity_type": {
        "explanation": "Object type for the request (e.g. 'document', 'group').",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: None,
    },
    "relationship_type": {
        "explanation": "Relation/permission type (e.g. 'viewer', 'transitiveMembers').",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: None,
    },
    "expand_paths": {
        "explanation": "Relationship/property paths to expand (e.g. OData $expand or Zanzibar Expand).",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: (),
    },
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
        "pagination_style": {
            "explanation": "Pagination strategy used by the remote endpoint.",
            "allowed_values": {style: explanation for style, (_, explanation) in _PAGINATION_STYLES.items()},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "none",
        },
        "page_size": {
            "explanation": "Records per page; bounded by remote per-system maximum.",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: 100,
        },
    }

    PARAMS_MAPPING_DELTA: ClassVar[dict[str, Any]] = {
        "cursor_token": {
            "explanation": "Opaque cursor for pagination_style=cursor; supplied by caller on continuation.",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
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
        "lineage_direction": {
            "explanation": "Direction of the lineage walk relative to the start asset.",
            "allowed_values": {
                "UPSTREAM": "Walk towards sources/dependencies.",
                "DOWNSTREAM": "Walk towards dependents/consumers.",
                "BOTH": "Walk both directions.",
                "ancestors": "Reactome-style ancestors traversal.",
                "descendants": "Reactome-style descendants traversal.",
            },
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "BOTH",
        },
        "upstream_depth": {
            "explanation": "Depth limit for upstream traversal (independent of downstream).",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: 1,
        },
        "downstream_depth": {
            "explanation": "Depth limit for downstream traversal (independent of upstream).",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: 0,
        },
    }


class InferenceMixin:
    """Reasoning/inference profile property.

    Used by ``rdf`` (always) and ``network_pg`` (optional, when the engine
    supports inference). Replaces the older boolean ``infer`` flag with a
    profile enum because real engines (Stardog, GraphDB, Oracle) expose named
    rule sets, not a single on/off switch.
    """

    PROPERTY_MAPPING_DELTA: ClassVar[dict[str, Any]] = {
        "reasoning_profile": {
            "explanation": "Inference profile / rule set the engine should apply.",
            "allowed_values": {
                "none": "No inference; raw triples/edges.",
                "rdfs": "RDFS entailment.",
                "owl-rl": "OWL 2 RL profile.",
                "owl-dl": "OWL 2 DL profile.",
                "custom": "Vendor-specific named ruleset (concrete plugin should validate further).",
            },
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "none",
        },
    }
