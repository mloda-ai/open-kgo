"""Per-call input flavors for KG connector readers: ``QueryReader`` and ``ParamReader``.

Split out of the ``kg.base`` module (which re-exports both, so import sites
are unaffected). A family base picks exactly one flavor: ``QueryReader``
families take a query string per call (``query_text`` / ``operation``);
``ParamReader`` families take a typed parameter dict declared on
``PARAMS_MAPPING`` and validated by ``_validate_params``. The conceptual
rules both flavors enforce (the "Honest credential surface" rule, the
``_STRIPPED_PARAMS`` rejection) are documented on the ``kg.reader_base``
module docstring.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.errors import InvalidCredentialShape, MissingRequiredParamsError
from open_kgo.feature_groups.kg.reader_base import KgConnectorReaderBase, _collect_kg_known_keys


class QueryReader(KgConnectorReaderBase):
    """Reader for connectors whose per-call input is a query string.

    Concrete plugins (Cypher, SPARQL, embedded operations, lexical search) read
    the query text from ``feature.options.context["query_text"]`` (or
    ``"operation"`` for embedded). The credential slot carries connection and
    authentication state; per-call values do not leak into credentials.
    """

    @classmethod
    def build_query(cls, features: FeatureSet) -> str:
        """Default: read ``query_text`` from the first feature's options context.

        Subclasses may override (e.g. ``networkx_embedded`` reads ``operation``
        instead). Raises ``ValueError`` if missing or empty.
        """
        feature = next(iter(features.features))
        text = feature.options.get("query_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{cls.CONNECTOR_ID}: feature {feature.name!r} options missing non-empty 'query_text'.")
        return text


class ParamReader(KgConnectorReaderBase):
    """Reader for connectors whose per-call input is a typed parameter dict.

    Concrete plugins (lineage walks, REST list/expand, citation lookups,
    authorization tuple filters, SBOM components) read per-call parameters
    from ``feature.options.context``. The shape is declared on
    ``PARAMS_MAPPING`` (parallel to ``PROPERTY_MAPPING`` for credentials) and
    enforced by ``_validate_params``. Per-call params never appear in the
    credential slot.
    """

    PARAMS_MAPPING: ClassVar[dict[str, Any]] = {}
    REQUIRED_PARAMS: ClassVar[tuple[tuple[str, ...], ...]] = ()
    # Family-base PARAMS keys this concrete dropped from ``PARAMS_MAPPING``.
    # Computed at class-definition time by ``__init_subclass__``; consulted by
    # ``build_params`` so a key the family declared but this concrete does not
    # honor is rejected at validate-time rather than silently no-oping.
    # Single source of truth: the concrete narrows ``PARAMS_MAPPING``; the
    # framework derives the rejection set.
    _STRIPPED_PARAMS: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._compute_stripped_params()

    @classmethod
    def _compute_stripped_params(cls) -> None:
        """Compute ``_STRIPPED_PARAMS`` as ``family_base.PARAMS_MAPPING - cls.PARAMS_MAPPING``.

        The family base is the **top-most** MRO ancestor (furthest from ``cls``)
        that is a non-``cls``, non-``ParamReader`` ``ParamReader`` subclass with
        a non-empty ``PARAMS_MAPPING``. Picking the top-most (not the nearest)
        is the contract: a multi-level chain like ``FamilyP{α,β}`` →
        ``FirstC(FamilyP){α}`` (strips β) → ``SecondC(FirstC){}`` (strips α)
        otherwise loses β from ``SecondC._STRIPPED_PARAMS`` because ``FirstC``
        is nearest but no longer declares β. Comparing against the top-most
        family base preserves every strip the chain accumulated. Mixins (which
        sit above family bases in MRO) are not ``ParamReader`` subclasses and
        so are skipped. If no qualifying family base exists (i.e. ``cls`` is
        itself an immediate ``ParamReader`` subclass with empty PARAMS_MAPPING),
        ``_STRIPPED_PARAMS`` stays empty.
        """
        family_base: type[ParamReader] | None = None
        # Last-write-wins: ``cls.__mro__`` is ordered most-derived → least-derived,
        # so the *last* qualifying assignment lands on the top-most ancestor
        # (furthest from ``cls``). The walk deliberately does NOT break on the
        # first qualifying ancestor (that would re-introduce the nearest-only
        # bug). The break on ``ParamReader`` bounds the walk; everything beyond
        # it (``KgConnectorReaderBase``, ``ReadDB``, ``object``) cannot be a
        # ``ParamReader`` subclass anyway.
        for ancestor in cls.__mro__[1:]:
            if ancestor is ParamReader:
                break
            if isinstance(ancestor, type) and issubclass(ancestor, ParamReader) and ancestor.PARAMS_MAPPING:
                family_base = ancestor
        if family_base is None:
            cls._STRIPPED_PARAMS = frozenset()
            return
        cls._STRIPPED_PARAMS = frozenset(family_base.PARAMS_MAPPING) - frozenset(cls.PARAMS_MAPPING)

    @classmethod
    def build_query(cls, features: FeatureSet) -> str:
        """ParamReader has no query language; per-call inputs live in build_params."""
        return ""

    def load(self, features: FeatureSet) -> dict[str, list[Any]]:
        """Run the per-call ``_STRIPPED_PARAMS`` check before delegating to the framework.

        Concrete ``load_data`` implementations are not required to call
        ``build_params`` (some only consume credentials), so the stripped-key
        rejection cannot live solely in ``build_params``. Hooking it into
        ``load`` ensures every ParamReader concrete enforces the contract
        regardless of its ``load_data`` shape. The single-feature guard runs
        first so a multi-feature FeatureSet does not silently feed the
        first-iterated feature into ``_reject_stripped_params``.
        """
        self._assert_single_feature(features)
        self._reject_stripped_params(features)
        return super().load(features)

    @classmethod
    def _reject_stripped_params(cls, features: FeatureSet) -> None:
        """Raise if any family-declared but concrete-stripped param is set on the feature.

        This is the per-call enforcement of option 1 of the "Honest credential
        surface" rule (see the ``kg.reader_base`` module docstring): a param the concrete
        dropped from ``PARAMS_MAPPING`` must not be silently accepted. Scope is
        intentionally narrow: only checks ``feature.options.context`` (where
        per-call params live), not ``feature.options.group`` (mloda's
        feature-grouping concept, out of scope for the per-call surface lie
        check). Unrelated keys in ``feature.options.context`` pass through;
        only family-declared keys this concrete dropped count as surface lies.
        """
        if not cls._STRIPPED_PARAMS:
            return
        feature = next(iter(features.features))
        for stripped in cls._STRIPPED_PARAMS:
            if stripped in feature.options.context:
                raise InvalidCredentialShape(
                    f"{cls.CONNECTOR_ID}.params.{stripped}: this concrete plugin does not honor the "
                    f"family-level {stripped!r} parameter; it was stripped from this concrete's "
                    f"PARAMS_MAPPING and may not appear in feature.options.context."
                )

    @classmethod
    def build_params(cls, features: FeatureSet, creds: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Read PARAMS_MAPPING keys from the first feature's options context, validate, return.

        Default impl extracts only keys declared in ``cls.PARAMS_MAPPING``.
        ``feature.options`` is shared with mloda core and other plugins, so
        unknown keys are not rejected outright. Three cross-checks layer on top:

        - ``_assert_single_feature`` runs first: ``build_params`` is reachable
          from concrete ``load_data`` implementations directly, not only
          through ``load``, so the multi-feature guard cannot rely on the
          ``load`` path having already rejected the FeatureSet.
        - Keys declared by the family base but stripped from this concrete's
          ``PARAMS_MAPPING`` (computed in ``_STRIPPED_PARAMS``) are rejected
          via ``_reject_stripped_params`` so the concrete cannot silently
          no-op a family-declared key.
        - Keys reserved by another KG reader's ``PROPERTY_MAPPING`` /
          ``PARAMS_MAPPING`` but not declared on this reader emit a warning
          (almost always a typo for a reserved name or an attempt to use
          this reader as another, e.g. passing ``lineage_direction`` to a
          citation reader). The keys are still dropped from the params dict;
          the warning surfaces the misuse without breaking callers.

        When ``creds`` (the already-extracted credential slot) is supplied,
        ``_validate_cross_layer`` is also invoked so cross-layer rules
        (e.g. ``cursor_token`` requires a cursor-family ``pagination_style``)
        can fire. ``creds`` is typed ``Mapping[str, Any]`` so concrete
        ``load_data`` sites can pass ``ctx.slot`` (a ``MappingProxyType``)
        directly. Concrete ``load_data`` sites that compose
        ``PaginationMixin`` and keep ``cursor_token`` in ``PARAMS_MAPPING``
        must thread the slot through to engage the check; sites that strip
        ``cursor_token`` are short-circuited at ``_reject_stripped_params``
        before this hook runs.
        """
        cls._assert_single_feature(features)
        cls._reject_stripped_params(features)
        feature = next(iter(features.features))
        cls._warn_on_reserved_unknown_keys(feature.options)
        params: dict[str, Any] = {}
        for key in cls.PARAMS_MAPPING:
            value = feature.options.get(key)
            if value is not None:
                params[key] = value
        cls._validate_params(params)
        if creds is not None:
            cls._validate_cross_layer(creds, params)
        return params

    @classmethod
    def _warn_on_reserved_unknown_keys(cls, options: Any) -> None:
        """Emit a warning if options carry KG-reserved keys this reader does not declare.

        Walks the ``KgConnectorReaderBase`` subclass tree on each call to
        union ``PROPERTY_MAPPING`` and ``PARAMS_MAPPING`` keys (the "reserved"
        set), then flags any options key that is reserved-elsewhere-but-not-
        here. Generic mloda keys and concrete-plugin-local keys (``query_text``
        for QueryReader, ``operation`` for embedded) are unaffected because
        they are not declared in any mapping.
        """
        user_keys = set(options.keys())
        declared_here = set(cls.PARAMS_MAPPING.keys()) | set(cls.PROPERTY_MAPPING.keys())
        leaked = user_keys & (_collect_kg_known_keys() - declared_here)
        if leaked:
            # stacklevel=3 surfaces the caller of build_params (e.g. a concrete
            # load_data, or a test). Chain is: caller -> build_params ->
            # _warn_on_reserved_unknown_keys -> warnings.warn.
            warnings.warn(
                f"{cls.CONNECTOR_ID}: dropping options key(s) reserved by other KG readers: "
                f"{sorted(leaked)}. Move them to the matching reader or remove the typo.",
                stacklevel=3,
            )

    @classmethod
    def _validate_params(cls, params: dict[str, Any]) -> None:
        """Mirror of ``_validate_shape`` for per-call params.

        - Open-world key check + strict-validation enums via ``_validate_mapping``
          (open-world because ``feature.options.context`` is shared with mloda
          core and other plugins). When ``SUPPORTED_VALUES`` narrows a
          strict-enum key, the narrowed set is authoritative for this concrete.
        - Required params: at least one key per OR-group must be set to a
          non-``None`` value. Presence is tested with ``is not None`` rather
          than truthiness so a legitimately falsey param value (``0``, ``""``,
          ``False``) is not misread as absent — consistent with the
          ``kg_contract`` REQUIRED_KEYS presence convention. Collect every
          unsatisfied
          group before raising ``MissingRequiredParamsError`` (an
          ``InvalidCredentialShape`` subclass) so callers can scope handlers
          to the leaf class while a generic ``InvalidCredentialShape``
          handler still catches both credential and per-call-param errors.
        """
        cls._validate_mapping(params, cls.PARAMS_MAPPING, kind="params", closed_world=False)
        unsatisfied: list[tuple[str, ...]] = []
        for group in cls.REQUIRED_PARAMS:
            if not group:
                raise InvalidCredentialShape(
                    f"{cls.CONNECTOR_ID}: REQUIRED_PARAMS contains an empty group; misconfigured."
                )
            if not any(params.get(k) is not None for k in group):
                unsatisfied.append(group)
        if unsatisfied:
            raise MissingRequiredParamsError(cls.CONNECTOR_ID, tuple(unsatisfied))
