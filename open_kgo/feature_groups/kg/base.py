"""Universal base classes for KG connector readers and feature groups.

The split mirrors mloda core's ``ReadDB`` + ``ReadDBFeature`` pattern (see
``mloda_plugins/feature_group/input_data/read_db.py`` and
``read_db_feature.py``). Each KG family supplies a ``<Family>Reader`` that
extends ``KgConnectorReaderBase`` (which extends ``ReadDB``) and a
``<Family>FeatureGroup`` that extends ``KgConnectorFeatureGroupBase``.

Concrete plugins set ``CONNECTOR_ID`` and implement ``connect``,
``build_query``, ``load_data``. mloda's ``BaseInputData.match_data_access``
walks the ``ReadDB`` subclass tree and finds the right reader by calling
``is_valid_credentials`` on each candidate, which the universal base
implements once against ``CONNECTOR_ID``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import BaseInputData, ComputeFramework, FeatureGroup, HashableDict
from mloda_plugins.feature_group.input_data.read_db import ReadDB

from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingEnvVarError,
    MissingRequiredKeysError,
    MissingRequiredParamsError,
    NonDictSpecError,
    PropertyMappingCollision,
)
from open_kgo.feature_groups.kg.python_dict_kg_framework import KgPythonDictFramework


@dataclass(frozen=True)
class LoadContext:
    """Bundle of values every concrete ``load_data`` needs at the top.

    ``wrapped`` is the normalised ``HashableDict``; ``slot`` is the
    already-extracted credential dict for this connector, exposed as a
    read-only ``Mapping`` (``MappingProxyType``) so concrete code can't
    accidentally mutate it through the frozen dataclass; ``result_limit``
    is the per-call row cap pulled from the slot (default 1000). Built
    once via ``_prepare_load(data_access)``.

    ``ontology_namespace`` is the namespace string from the loaded ontology
    file (e.g. ``"movie"``), or ``None`` when no ``ontology`` key was
    supplied in credentials. Connectors that support typed traversal pass
    this to ``OntologyRegistry.is_valid_edge(ctx.ontology_namespace, ...)``.
    """

    wrapped: HashableDict
    slot: Mapping[str, Any]
    result_limit: int
    ontology_namespace: str | None = None


def compose_property_mapping(*sources: dict[str, Any], context: str = "") -> dict[str, Any]:
    """Merge property-mapping dicts and raise on duplicate keys.

    Replaces the spread-merge ``{**A, **B, **C}`` idiom used by family bases.
    The spread-merge silently overwrites duplicates; this helper raises
    ``PropertyMappingCollision`` so structural collisions surface at import
    time. ``context`` is the family/mixin name carried into the error message.

    Also rejects non-dict spec values at composition time via
    ``NonDictSpecError``. A ``None`` spec would otherwise pass downstream
    ``mapping.get(key) is None`` lookups and surface as a self-contradicting
    "unknown credential key" error: the key is listed in ``allowed`` but
    ``_validate_mapping`` cannot distinguish a missing key from a ``None``
    spec via ``.get``. Catching it here keeps the closed-world check honest.
    ``NonDictSpecError`` is a sibling of ``PropertyMappingCollision`` under
    ``InvalidCredentialShape`` so callers can scope a single typed handler
    across the whole structural-error family.
    """
    merged: dict[str, Any] = {}
    for source in sources:
        for key, spec in source.items():
            if key in merged:
                raise PropertyMappingCollision(key, context=context)
            if not isinstance(spec, dict):
                raise NonDictSpecError(key, spec, context=context)
            merged[key] = spec
    return merged


def narrow_property_mapping(source: dict[str, Any], *exclude: str) -> dict[str, Any]:
    """Return ``source`` minus the ``exclude`` keys — the narrowing companion to ``compose_property_mapping``.

    Concrete plugins drop family-level keys they do not honor (advertising
    a key the reader ignores would be a surface lie that the closed-world
    credential check then rejects). Several concretes spelled this as an
    inline ``{k: v for k, v in Parent.PROPERTY_MAPPING.items() if k not in {...}}``
    comprehension; centralising it names the intent and keeps the narrowing
    rule in one place. Keys in ``exclude`` that are absent from ``source``
    are silently ignored (narrowing is idempotent).
    """
    excluded = set(exclude)
    return {k: v for k, v in source.items() if k not in excluded}


# History: the universal property mapping previously declared
# ``auth_method`` + the three ``auth_*_env`` companion keys, and a paired
# ``_UNIVERSAL_CONDITIONAL_REQUIRED_KEYS`` tuple tied them together. No
# concrete plugin in the shipped 9 families ever called ``_resolve_env`` from
# ``_connect_from_slot``; every concrete narrowed ``auth_method`` to
# ``frozenset({"none"})`` via ``SUPPORTED_VALUES`` because none of them
# actually opens a network socket. The auth surface was therefore decorative:
# the framework loudly validated a credential surface no concrete read.
#
# The fix follows the same narrowing approach used elsewhere in this base:
# drop the surface from the universal base until at least one networked
# concrete honors it, and re-introduce per-concrete (or per-family) when that
# lands. The ``_resolve_env`` helper below is kept as opt-in infrastructure so
# a future networked concrete has something to call without re-implementing the
# env-var-resolution contract.


_UNIVERSAL_PROPERTY_MAPPING: dict[str, Any] = {
    "locator": {
        "explanation": "Endpoint URL or filesystem path. May be None for purely in-process backends.",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: None,
    },
    "ontology": {
        "explanation": (
            "Path to a YAML ontology definition file. Declares entity types, valid outgoing "
            "relationship types per entity, and domain/range constraints per relationship. "
            "When supplied, the file is loaded into OntologyRegistry under the namespace "
            "declared in the file. Connectors access typed-traversal lookups via "
            "OntologyRegistry using ctx.ontology_namespace. Optional: connectors without "
            "an ontology file behave exactly as before (no validation applied)."
        ),
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: None,
    },
    "result_limit": {
        "explanation": (
            "Maximum rows/records returned per query (bound-output semantics). "
            "Concrete readers MUST short-circuit work when the limit is reached "
            "rather than walking the full source then slicing. Slicing-at-end "
            "leaks unbounded cost on wide inputs (e.g. dbt manifests, paginated "
            "REST, large in-memory graphs). Use itertools.islice or an explicit "
            "early-return loop; only slice at the end if the source is already "
            "a fully-materialized list of bounded size (i.e. the JSON parser "
            "produced it for you and walking it is O(result_limit) anyway)."
        ),
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.default: 1000,
    },
}


def _collect_kg_known_keys() -> set[str]:
    """Return the union of declared keys across every ``KgConnectorReaderBase`` subclass.

    Walks ``KgConnectorReaderBase.__subclasses__()`` transitively and unions
    each subclass's ``PROPERTY_MAPPING`` and ``PARAMS_MAPPING`` (when present).
    Used by ``ParamReader.build_params`` to detect cross-reader leakage of
    reserved-but-not-declared keys passed via ``feature.options``. Walks the
    tree on each call rather than caching: the count is small (under 20),
    and a cache would shadow families imported after first call.
    """
    known: set[str] = set()
    pending: list[type[KgConnectorReaderBase]] = [KgConnectorReaderBase]
    while pending:
        klass = pending.pop()
        # PROPERTY_MAPPING is always present (declared on the base class).
        known.update(klass.PROPERTY_MAPPING.keys())
        # PARAMS_MAPPING is only on ParamReader and its descendants.
        params_mapping = getattr(klass, "PARAMS_MAPPING", None)
        if params_mapping is not None:
            known.update(params_mapping.keys())
        pending.extend(klass.__subclasses__())
    return known


class KgConnectorReaderBase(ReadDB):
    """Universal base for KG connector readers.

    Subclasses (per family) extend this and add family-specific properties to
    ``PROPERTY_MAPPING``. Concrete plugins (per system) further subclass the
    family base, set ``CONNECTOR_ID``, and implement ``connect``,
    ``build_query``, ``load_data``.

    The class attribute ``CONNECTOR_ID`` keys the credential dict inside
    ``DataAccessCollection.credentials``. ``REQUIRED_KEYS`` declares
    which credential keys are mandatory: a tuple of OR-groups (all groups
    AND'ed, members within a group OR'ed). Empty tuple means "no required
    keys". E.g. ``(("locator",),)`` requires ``locator``;
    ``(("manifest_path", "locator"),)`` requires either; multiple groups
    require one from each.
    """

    CONNECTOR_ID: ClassVar[str] = ""
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = ()

    # Per-property "requires" rules resolved against sibling values. Each entry
    # is ``(prop_name, prop_value, OR-groups)``: when ``creds.get(prop_name) ==
    # prop_value``, the OR-groups are enforced just like ``REQUIRED_KEYS`` (each
    # group needs one truthy member). Empty by default: the universal base no
    # longer declares any conditional rule (the prior ``auth_method`` rules were
    # paired with a credential surface no concrete honored). Subclasses that
    # introduce conditional rules add them directly without an ``EXTEND`` step.
    CONDITIONAL_REQUIRED_KEYS: ClassVar[tuple[tuple[str, Any, tuple[tuple[str, ...], ...]], ...]] = ()

    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = dict(_UNIVERSAL_PROPERTY_MAPPING)

    # Narrow a family-level strict-validation enum down to the subset this
    # concrete plugin actually honors at runtime. Keys not present here pass
    # through with the family-base allowed set; keys present here must
    # additionally appear in the supplied frozenset to validate.
    # Used by ``_validate_shape`` and ``_validate_params`` (the latter via
    # ``ParamReader``) so the same hook covers credential and per-call surfaces.
    # Invariant: each ``SUPPORTED_VALUES[key]`` is a non-empty subset of the
    # spec's allowed set, and the key has ``strict_validation=True``. Enforced
    # at class-definition time by ``__init_subclass__``.
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {}

    # Strict-validation enum keys this concrete deliberately accepts at the
    # family-wide allowed set without honoring every value at runtime. Waivers
    # are documentation, not code: they make "we accept all values for
    # forward-compat" explicit and reviewable. The honest alternative is
    # ``SUPPORTED_VALUES``; use a waiver only when narrowing would lock out a
    # forward-compatible value the family base legitimately advertises (e.g.
    # ``read_consistency`` is a Kuzu no-op today but real network_pg backends
    # will honor it). The ``test_strict_enum_honored_or_waived`` contract test
    # rejects any strict-validation enum that is neither in
    # ``SUPPORTED_VALUES`` nor in ``_WAIVED_ENUM_KEYS``; each waived key carries
    # a one-line comment on the concrete class explaining the waiver. The
    # contract test enforces membership only; the comment is review-time
    # discipline.
    _WAIVED_ENUM_KEYS: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._validate_mapping_spec_shapes()
        cls._validate_supported_values_invariant()

    @classmethod
    def _validate_mapping_spec_shapes(cls) -> None:
        """Reject non-dict spec values in ``PROPERTY_MAPPING`` / ``PARAMS_MAPPING`` at class-definition time.

        ``compose_property_mapping`` already enforces this for mappings built
        through the helper, but concretes that hand-assemble their mapping
        via a dict-comprehension off an already-composed parent (e.g.
        ``FileFixtureRestReader``, ``FileFixtureCitationReader``,
        ``InProcessTupleStoreReader``) bypass the compose-time check. Running
        the same guard once more here closes that hole so the rule is
        "any spec in any mapping must be a dict" regardless of how the
        mapping was assembled. Raises ``NonDictSpecError`` (the same typed
        error ``compose_property_mapping`` uses) so callers can catch both
        compose-time and class-definition-time bypasses with one handler.
        """
        for layer_name in ("PROPERTY_MAPPING", "PARAMS_MAPPING"):
            mapping = getattr(cls, layer_name, None)
            if not mapping:
                continue
            for key, spec in mapping.items():
                if not isinstance(spec, dict):
                    raise NonDictSpecError(key, spec, context=f"{cls.__name__}.{layer_name}")

    @classmethod
    def _validate_supported_values_invariant(cls) -> None:
        """Raise ``ValueError`` at class-definition time on ill-formed ``SUPPORTED_VALUES``.

        Catches typos and out-of-set values that would otherwise silently lock
        a connector at runtime (every value rejected). For each narrowed key:

        - The key must be declared in ``PROPERTY_MAPPING`` or ``PARAMS_MAPPING``.
        - The spec must have ``strict_validation=True`` (narrowing a non-strict
          key is meaningless: the family already accepts anything).
        - The narrowed frozenset must be non-empty and a subset of the spec's
          allowed set.
        """
        if not cls.SUPPORTED_VALUES:
            return
        params_mapping: Mapping[str, Any] = getattr(cls, "PARAMS_MAPPING", {})
        for key, narrowed in cls.SUPPORTED_VALUES.items():
            spec = cls.PROPERTY_MAPPING.get(key) or params_mapping.get(key)
            if spec is None:
                raise ValueError(
                    f"{cls.__name__}.SUPPORTED_VALUES[{key!r}] names a key not present in "
                    f"PROPERTY_MAPPING or PARAMS_MAPPING."
                )
            if spec.get(DefaultOptionKeys.strict_validation) is not True:
                raise ValueError(
                    f"{cls.__name__}.SUPPORTED_VALUES[{key!r}] requires the spec to set "
                    f"strict_validation=True; narrowing a non-strict key is meaningless."
                )
            if not narrowed:
                raise ValueError(
                    f"{cls.__name__}.SUPPORTED_VALUES[{key!r}] is empty; an empty narrowed "
                    f"set rejects every value. Use ``del SUPPORTED_VALUES[key]`` and strip "
                    f"the key from the mapping instead if the concrete cannot honor any value."
                )
            allowed = cls._spec_allowed_values(key, spec)
            if not narrowed <= allowed:
                raise ValueError(
                    f"{cls.__name__}.SUPPORTED_VALUES[{key!r}]={sorted(narrowed)} is not a "
                    f"subset of the family-allowed set {sorted(allowed)}."
                )

    def load(self, features: FeatureSet) -> Any:
        """Reject multi-feature FeatureSets, then return native rows from ``load_data``.

        Concrete ``load_data`` implementations across every family read a single
        feature via ``next(iter(features.features))``; passing more than one
        feature would silently use whichever one the iterator yields first and
        label every row with that single name. Reject the multi-feature shape
        loudly so the contract violation surfaces immediately.

        The ``{feature_name: row}`` wrap that mloda's ``PythonDictFramework``
        column-matcher expects no longer lives here; it has moved to
        ``KgPythonDictFramework`` (the FG base pins that adapter via
        ``compute_framework_rule``). Native KG rows therefore flow out of
        ``load_data`` unchanged, so any other compute framework or direct
        consumer sees what the concrete plugin actually produced.

        ``load_data`` is contractually required to return ``list[dict[str, Any]]``
        (every concrete in this package satisfies that). The shape check below
        turns a future drift (a concrete returning a single dict, ``None``, or
        a generator) into a typed error here rather than an indirect failure
        downstream in ``select_data_by_column_names``.
        """
        self._assert_single_feature(features)
        result = super().load(features)
        cls_name = type(self).__name__
        if not isinstance(result, list):
            raise TypeError(f"{cls_name}.load_data must return list[dict[str, Any]], got {type(result).__name__}.")
        for index, row in enumerate(result):
            if not isinstance(row, dict):
                raise TypeError(
                    f"{cls_name}.load_data must return list[dict[str, Any]]; "
                    f"row at index {index} is {type(row).__name__}."
                )
        return result

    @classmethod
    def _assert_single_feature(cls, features: FeatureSet) -> None:
        """Universal precondition: ``load_data`` dispatches one feature at a time.

        Extracted so subclasses (e.g. ``ParamReader``) can run the guard at the
        very top of their own ``load`` override, before any per-call validation
        that iterates ``features.features`` and would otherwise pick one
        feature silently.
        """
        if len(features.features) != 1:
            raise ValueError(
                f"{cls.__name__}.load expects exactly one feature per call "
                f"(KG concrete load_data implementations all dispatch one feature at a time), "
                f"got {len(features.features)}: {sorted(f.name for f in features.features)}."
            )

    @classmethod
    def supports_scoped_data_access(cls) -> bool:
        # Abstract bases (CONNECTOR_ID == "") are filtered out at discovery time
        # so they never appear in the scoped-access subclass list. Concrete
        # subclasses (non-empty CONNECTOR_ID) return True directly; this override
        # replaces mloda's default load_data(None, None) probe path, which would
        # otherwise hit _wrap_credentials(None), raise NotImplementedError, and
        # be interpreted by the default as "not scoped" (wrong for our plugins).
        return bool(cls.CONNECTOR_ID)

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        """Return True if credentials carry a valid dict slot for this CONNECTOR_ID.

        Matcher-safe: returns False on any ``Exception`` raised while probing
        or shape-checking the credentials object, not only ``InvalidCredentialShape``.
        (``BaseException`` subclasses such as ``KeyboardInterrupt``,
        ``SystemExit``, and ``GeneratorExit`` still propagate by design, so a
        Ctrl-C inside the matcher loop is not swallowed.) mloda's
        ``ReadDB.match_read_db_data_access`` only catches ``NotImplementedError``
        from the matcher loop, so any other propagating ``Exception`` (e.g. a
        misbehaving ``Mapping`` whose ``.get`` raises ``RuntimeError``, an
        ``AttributeError`` from a custom dict-alike, or a framework hiccup deep
        inside ``_validate_shape``) would abort iteration over unrelated
        readers in the same ``DataAccessCollection``. The broad ``except`` is
        therefore part of the matcher-safety contract, not a swallowed bug.
        Loud-failure diagnostics for malformed slots and content errors live
        in ``_extract_slot`` and ``_validate_shape``, which direct callers
        invoke separately.
        """
        if not cls.CONNECTOR_ID:
            return False
        try:
            creds = cls._extract_slot(credentials)
            if creds is None:
                return False
            cls._validate_shape(creds)
        except Exception:
            return False
        return True

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        """Default: any feature name is acceptable once credentials match.

        Concrete plugins may override (e.g. by parsing a query name prefix
        against ``CONNECTOR_ID``).
        """
        return True

    @classmethod
    def _extract_slot(cls, credentials: Any) -> dict[str, Any] | None:
        """Return the dict at credentials[CONNECTOR_ID], or None if absent.

        A slot value of ``None`` is treated as opt-out (absent). Any other
        non-dict value (e.g. a bare string path like ``"/data/x.ttl"``) is a
        misuse: the slot key is present but malformed, which would otherwise
        be indistinguishable from "this connector's slot is absent" and
        silently mismatch. Raise ``InvalidCredentialShape`` so the typo
        surfaces loudly.
        """
        _ABSENT = object()
        slot: Any = _ABSENT
        if isinstance(credentials, HashableDict):
            slot = credentials.data.get(cls.CONNECTOR_ID, _ABSENT)
        elif isinstance(credentials, dict):
            slot = credentials.get(cls.CONNECTOR_ID, _ABSENT)

        if slot is _ABSENT or slot is None:
            return None
        if isinstance(slot, HashableDict):
            return dict(slot.data)
        if isinstance(slot, dict):
            return dict(slot)
        raise InvalidCredentialShape(
            f"{cls.CONNECTOR_ID}: credential slot must be a dict mapping property names to values, "
            f"got {type(slot).__name__} ({slot!r})."
        )

    @classmethod
    def _validate_shape(cls, creds: dict[str, Any]) -> None:
        """Validate a single connector's credential dict against PROPERTY_MAPPING.

        Order:
        1. ``REQUIRED_KEYS`` — at least one key per OR-group must be set+truthy.
           Reported first so missing keys surface a clear "you forgot X" error.
        2. ``CONDITIONAL_REQUIRED_KEYS`` — for each ``(prop, value, groups)``
           rule whose trigger matches ``creds.get(prop) == value``, enforce the
           OR-groups. Reported before the closed-world / enum loop so semantic
           "you picked X but didn't supply its companion keys" errors land
           before incidental typos.
        3. ``result_limit`` boundary check — must be a non-bool ``int >= 1``.
           Pinned at the credential surface so the cross-reader divergence in
           append-then-check vs slice-at-end behavior at ``result_limit ∈
           {0, -1, False, ...}`` ceases to matter; every reader sees a
           validated positive int by the time ``_prepare_load`` returns. Bool
           is rejected explicitly: ``True``/``False`` are int subclasses in
           Python, but a row count expressed as a truth-value almost always
           reflects a caller mistake.
        4. Closed-world key check + strict-validation enums via
           ``_validate_mapping`` (the latter consults
           ``SUPPORTED_VALUES`` for per-concrete narrowing, falling back
           to the spec's ``allowed_values``).
        """
        cls._validate_required_keys(creds)
        cls._validate_conditional_required_keys(creds)
        cls._validate_result_limit(creds)
        cls._validate_mapping(creds, cls.PROPERTY_MAPPING, kind="credential key", closed_world=True)

    @classmethod
    def _validate_result_limit(cls, creds: dict[str, Any]) -> None:
        """Reject ``result_limit`` values that aren't positive ints.

        ``result_limit`` is universal (in ``_UNIVERSAL_PROPERTY_MAPPING``) and
        the spec defaults to 1000, so the key only reaches this check when the
        caller set it. Bool is rejected explicitly: it is an ``int`` subclass
        in Python, but a row cap of ``True`` or ``False`` is almost always a
        caller mistake. Strings, floats, and negative integers fail likewise.
        """
        if "result_limit" not in creds:
            return
        value = creds["result_limit"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidCredentialShape(
                f"{cls.CONNECTOR_ID}: result_limit must be a positive int (>= 1, bool not accepted), "
                f"got {type(value).__name__} {value!r}."
            )

    @classmethod
    def _validate_mapping(
        cls,
        values: dict[str, Any],
        mapping: dict[str, Any],
        *,
        kind: str,
        closed_world: bool,
    ) -> None:
        """Shared shape + strict-enum validation loop.

        Used by ``_validate_shape`` (closed-world over PROPERTY_MAPPING) and
        ``_validate_params`` (open-world over PARAMS_MAPPING; params share
        ``feature.options.context`` with mloda core and other plugins, so
        unknown keys must pass through). The ``kind`` label appears in the
        error message ("credential key" vs "params") so the source of the
        bad key is obvious in diagnostics.
        """
        for key, value in values.items():
            spec = mapping.get(key)
            if spec is None:
                if closed_world:
                    raise InvalidCredentialShape(
                        f"{cls.CONNECTOR_ID}: unknown {kind} {key!r}; allowed: {sorted(mapping.keys())}"
                    )
                continue
            if spec.get(DefaultOptionKeys.strict_validation) is True:
                narrowed = cls.SUPPORTED_VALUES.get(key)
                if narrowed is not None:
                    if value not in narrowed:
                        raise InvalidCredentialShape(
                            f"{cls.CONNECTOR_ID}.{key}={value!r} is not supported by this connector "
                            f"(supported: {sorted(narrowed)})"
                        )
                else:
                    allowed = cls._spec_allowed_values(key, spec)
                    if value not in allowed:
                        raise InvalidCredentialShape(
                            f"{cls.CONNECTOR_ID}: {kind} {key!r}={value!r} is not in allowed set {sorted(allowed)}"
                        )

    @staticmethod
    def _spec_allowed_values(key: str, spec: dict[str, Any]) -> set[Any]:
        """Return the explicit ``allowed_values`` set declared on a strict-validation spec.

        Strict-validation specs must declare their value space explicitly via
        an ``allowed_values`` field (a dict mapping value to its docstring, or
        any iterable of values). Deriving the set from the spec's plain string
        keys would silently expand the allowed set whenever a future doc-only
        key like ``"see_also"`` is added; the explicit field separates docs
        from validation data.
        """
        raw = spec.get("allowed_values")
        if raw is None:
            raise InvalidCredentialShape(
                f"spec for {key!r} declares strict_validation=True but is missing 'allowed_values'."
            )
        if isinstance(raw, dict):
            return set(raw.keys())
        return set(raw)

    @classmethod
    def _validate_required_keys(cls, creds: dict[str, Any]) -> None:
        """Enforce ``REQUIRED_KEYS``: each OR-group must have a present member.

        Presence is tested with ``is not None`` rather than truthiness so a
        legitimately falsey credential value (``0``, ``""``, ``False``) is not
        misread as absent — matching the ``REQUIRED_PARAMS`` presence
        convention (``_validate_required_params``) and the ``kg_contract``
        presence rule (``key in ... and value is not None``).
        """
        unsatisfied: list[tuple[str, ...]] = []
        for group in cls.REQUIRED_KEYS:
            if not group:
                raise InvalidCredentialShape(
                    f"{cls.CONNECTOR_ID}: REQUIRED_KEYS contains an empty group; misconfigured."
                )
            if not any(creds.get(k) is not None for k in group):
                unsatisfied.append(group)
        if unsatisfied:
            raise MissingRequiredKeysError(cls.CONNECTOR_ID, tuple(unsatisfied))

    @classmethod
    def _validate_conditional_required_keys(cls, creds: dict[str, Any]) -> None:
        """Enforce ``CONDITIONAL_REQUIRED_KEYS``: rules triggered by sibling values.

        Each rule is ``(prop, value, OR-groups)``. If ``creds.get(prop)`` equals
        ``value``, every OR-group must have at least one present (non-``None``)
        member in ``creds`` (same presence convention as
        ``_validate_required_keys``). Aggregates all unsatisfied groups across
        all triggered rules into a single ``MissingRequiredKeysError`` so the
        caller sees the full picture in one error message.
        """
        unsatisfied: list[tuple[str, ...]] = []
        for prop, trigger_value, groups in cls.CONDITIONAL_REQUIRED_KEYS:
            if creds.get(prop) != trigger_value:
                continue
            for group in groups:
                if not group:
                    raise InvalidCredentialShape(
                        f"{cls.CONNECTOR_ID}: CONDITIONAL_REQUIRED_KEYS for "
                        f"{prop}={trigger_value!r} contains an empty group; misconfigured."
                    )
                if not any(creds.get(k) is not None for k in group):
                    unsatisfied.append(group)
        if unsatisfied:
            raise MissingRequiredKeysError(cls.CONNECTOR_ID, tuple(unsatisfied))

    @classmethod
    def _require_slot(cls, credentials: Any) -> dict[str, Any]:
        """Extract the credential slot or raise. ``connect()`` and ``_prepare_load`` both call this.

        By the time ``_connect_from_slot`` runs, the slot has been
        shape-validated by one of two upstream gates: the matcher path
        validates via ``is_valid_credentials`` (matcher-safe ``False`` on
        error), and the direct-call path validates via ``connect()``
        (loud ``InvalidCredentialShape`` / ``MissingRequiredKeysError`` on
        error). This helper only unpacks the slot; concrete readers no longer
        need to defensively re-check for ``None``.

        ``_prepare_load`` calls this *without* running ``_validate_shape``
        because the matcher already validated before dispatch; a direct call
        to ``load_data`` that bypasses both gates relies on the slot being
        well-formed, which is the documented contract for that direct path.
        """
        slot = cls._extract_slot(credentials)
        if slot is None:
            raise InvalidCredentialShape(f"{cls.CONNECTOR_ID}: credentials missing the {cls.CONNECTOR_ID!r} slot.")
        return slot

    @classmethod
    def _wrap_credentials(cls, data_access: Any) -> HashableDict:
        """Normalise the data_access mloda hands us into a HashableDict({CONNECTOR_ID: dict}).

        mloda's BaseInputData passes the matched data_access through. Concrete
        plugins receive either the full credentials dict (with our slot inside)
        or just our slot. This helper unifies both shapes so concrete code can
        always call ``cls._extract_slot(cls._wrap_credentials(data_access))``.

        ``data_access=None`` raises ``NotImplementedError``: mloda's
        scoped-access discovery probes ``load_data(None, None)`` and expects
        that error class (not ``TypeError``) to mean "this reader needs real
        credentials". A real caller passing ``None`` by mistake also lands here.
        """
        if data_access is None:
            raise NotImplementedError(
                f"{cls.__name__}.load_data requires a credentials dict; received None. "
                "mloda's scoped-access discovery probe also reaches this path."
            )
        if isinstance(data_access, HashableDict):
            if cls.CONNECTOR_ID in data_access.data:
                return data_access
            return HashableDict({cls.CONNECTOR_ID: dict(data_access.data)})
        if isinstance(data_access, dict):
            if cls.CONNECTOR_ID in data_access:
                return HashableDict(dict(data_access))
            return HashableDict({cls.CONNECTOR_ID: dict(data_access)})
        raise TypeError(f"data_access must be a dict or HashableDict, got {type(data_access).__name__}")

    @classmethod
    def _prepare_load(cls, data_access: Any) -> LoadContext:
        """Wrap credentials, extract the slot, and parse the result_limit default.

        Uses ``_require_slot`` (not ``_extract_slot or {}``): a missing slot
        here means the caller bypassed mloda discovery and should hit a typed
        error rather than a silent default. ``ctx.slot`` is a
        ``MappingProxyType`` so the frozen dataclass behaves immutably.

        ``_validate_shape`` (run by ``is_valid_credentials``) has already
        pinned ``result_limit`` to a positive ``int`` via
        ``_validate_result_limit``; the dict lookup default of 1000 here
        covers the no-key path only. Callers that bypass ``is_valid_credentials``
        (direct ``connect``/``load_data`` calls) hit the same check on the
        next line: re-running it makes ``_prepare_load`` independent of
        whether the matcher path validated.

        The ``LoadContext.result_limit: int`` annotation is load-bearing: it
        relies on ``_validate_result_limit`` running before the dataclass is
        constructed. A future refactor that reorders these two lines (or
        skips the second validation) silently turns the annotation into a
        lie, since ``slot.get`` returns ``Any``.
        """
        wrapped = cls._wrap_credentials(data_access)
        slot = cls._require_slot(wrapped)
        cls._validate_result_limit(slot)
        ontology_namespace: str | None = None
        ontology_path = slot.get("ontology")
        if ontology_path is not None:
            from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

            ontology_namespace = OntologyRegistry.load_file(str(ontology_path))
        return LoadContext(
            wrapped=wrapped,
            slot=MappingProxyType(slot),
            result_limit=slot.get("result_limit", 1000),
            ontology_namespace=ontology_namespace,
        )

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        """Public ReadDB hook: extract + validate the slot, then dispatch to ``_connect_from_slot``.

        Concrete plugins override ``_connect_from_slot(slot)``, not this
        method. Keeping ``connect(credentials)`` aligned with ``ReadDB``'s
        signature means ``ReadDB.read_db`` and ``ReadDB.get_connection`` still
        work against KG readers without any further override; concrete plugin
        code never sees the full credentials object.

        Calls ``_validate_shape`` before dispatching so direct callers (tests,
        demos, programmatic users that bypass mloda's matcher) hit the same
        typed errors the matcher path enforces via ``is_valid_credentials``.
        Without this, a partial slot like ``{CONNECTOR_ID: {}}`` would surface
        downstream as ``FileNotFoundError`` or ``KeyError`` from
        ``_connect_from_slot`` instead of the typed
        ``MissingRequiredKeysError`` / ``InvalidCredentialShape`` the matcher
        produces. Concretes that want to bypass shape validation (e.g. to
        probe the typed-leaf error in ``_connect_from_slot``) call
        ``_connect_from_slot`` directly.

        On the matcher path, ``_validate_shape`` runs twice for the same slot:
        once inside ``is_valid_credentials`` during reader selection, and once
        here when mloda dispatches into ``ReadDB.read_db`` /
        ``ReadDB.get_connection``. The cost is negligible and the second run
        is what gives direct callers parity with the matcher. A future refactor
        that wants to dedupe must preserve the direct-call shape gate (e.g.
        via a "validated" sentinel threaded through the call), not just remove
        one of the two ``_validate_shape`` calls.
        """
        slot = cls._require_slot(credentials)
        cls._validate_shape(slot)
        return cls._connect_from_slot(slot)

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> Any:
        """Open a backend connection from this connector's already-extracted slot.

        Concrete plugins implement this hook; ``connect(credentials)`` and
        ``load_data`` both call into it so ``_extract_slot`` runs at most once
        per ``load_data``.

        Connection-lifecycle contract: every concrete
        plugin's returned object falls into one of three categories. The
        category determines what direct callers may do with the return
        value and whether two ``_connect_from_slot`` calls for the same
        slot are permitted to share the same object.

        - **Cached, shared, read-only.** The shared resource cache in
          ``kg.fixtures`` (``load_json_fixture``, ``load_rdf_graph``,
          ``load_kuzu_database``) memoises parses and FDs across calls so
          a 100-feature ``mloda.run_all`` pays one parse / one open
          instead of N. Concretes routing through these helpers return
          the same dict / ``rdflib.Graph`` / ``kuzu.Database`` to every
          caller; downstream code MUST NOT mutate the object, MUST NOT
          ``close()`` it, and MUST shallow-copy any row dict that escapes
          into a returned row list. The shallow copy is **shell-only**:
          it isolates top-level field assignments (``row["name"] = ...``)
          but nested mutable refs (e.g. ``row["ancestors"].append(...)``
          on the citation reader, ``row["dependencies"].append(...)`` on
          an SBOM component) are NOT isolated and would still poison the
          cache. Callers therefore MUST treat cached returns as read-only
          all the way down; the shallow shell copy is a best-effort guard
          against the most common surface-level mutation and is not a
          deep-copy substitute. Today: dbt manifest, dbt citation
          catalog, CycloneDX SBOM, REST page bodies, agent memory store,
          in-process tuple store, rdflib SPARQL graph, kuzu database.
        - **Caller-owned, transient.** Created fresh on every call and
          handed to the caller; closing it does not poison any cache.
          Today: ``kuzu.Connection`` (the contract-test path that
          closes ``connect()``'s return depends on this category).
        - **Transient, no resource.** A small object (e.g. ``Path``) is
          rebuilt per call and holds no backing FD. Today: REST
          page-directory ``Path`` returned by
          ``FileFixtureRestReader._connect_from_slot``.

        Two ``_connect_from_slot`` calls with the same slot MAY return
        the same object (category 1) or distinct objects (categories
        2 / 3); contract tests that close the return value rely
        exclusively on the category-2 contract. Concretes that want to
        opt out of caching for a future per-tenant or per-request
        substrate need to fall back to a fresh build (categories 2 / 3)
        AND update this docstring + their per-family note so the
        category is reviewable rather than implicit.
        """
        raise NotImplementedError

    @classmethod
    def _validate_cross_layer(cls, creds: Mapping[str, Any], params: dict[str, Any]) -> None:
        """Cross-layer validation hook seeing both credentials and per-call params.

        Default no-op. Mixins that span both layers (e.g. ``PaginationMixin``
        contributes ``pagination_style`` to ``PROPERTY_MAPPING`` *and*
        ``cursor_token`` to ``PARAMS_MAPPING``) override this cooperatively
        with ``super()._validate_cross_layer(creds, params)`` and add their
        own checks. ``ParamReader.build_params`` calls this once both inputs
        are known.

        ``creds`` is typed ``Mapping[str, Any]`` (read-only) so concrete
        ``load_data`` sites can pass ``ctx.slot`` (a ``MappingProxyType``)
        directly without a defensive ``dict(...)`` round-trip.
        """

    @classmethod
    def _resolve_env(cls, creds: dict[str, Any], key: str) -> str | None:
        """Read an env-var NAME from creds[key], return the stripped env-var value.

        Opt-in helper for concretes that consume credentials from an env var
        (a bearer token, a username/password pair, etc.). The universal base
        does NOT call this hook: no shipped concrete authenticated against a
        network, so a universally-required
        env-var surface would be a contract the framework could not enforce.
        Concretes that introduce a real auth surface declare the matching
        ``auth_*_env`` keys themselves (on a family base or the concrete) and
        call ``_resolve_env`` from their own ``_connect_from_slot``.

        Returns None if creds[key] itself is unset (caller is opting out, e.g.
        an absent ``auth_token_env`` for a method that does not need one).
        Raises MissingEnvVarError if creds[key] names an env var that is not
        set in the environment, or if it is set to a value whose ``strip()``
        is empty (i.e. any value with no non-whitespace character: ``""``,
        ``"   "``, ``"\t"``, ``"\n"``, or any mix). Downstream auth would
        otherwise fail opaquely with no diagnostic on such values.

        The contract is "value must contain at least one
        non-whitespace character." The returned value is ``value.strip()`` so
        stray surrounding whitespace (a common ``.env``/copy-paste artifact)
        does not leak through to downstream auth either — if the rejection
        rationale is "whitespace breaks downstream tokens," partial-whitespace
        tokens deserve the same treatment as fully-whitespace ones.
        """
        env_name = creds.get(key)
        if env_name is None:
            return None
        if not isinstance(env_name, str):
            raise InvalidCredentialShape(
                f"{cls.CONNECTOR_ID}.{key} must be a str env-var name, got {type(env_name).__name__}"
            )
        value = os.environ.get(env_name)
        if value is None:
            raise MissingEnvVarError(env_name, key)
        stripped = value.strip()
        if not stripped:
            raise MissingEnvVarError(env_name, key)
        return stripped


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

    def load(self, features: FeatureSet) -> Any:
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

        Scope is intentionally narrow: only checks ``feature.options.context``
        (where per-call params live), not ``feature.options.group`` (mloda's
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


class KgConnectorFeatureGroupBase(FeatureGroup):
    """Universal thin FeatureGroup that delegates to a KG reader.

    Subclasses (per family) set ``READER_CLASS`` to the matching reader. The
    body is identical to ``ReadDBFeature``: ``input_data()`` returns an
    instance of ``READER_CLASS``; ``calculate_feature`` calls
    ``reader.load(features)``.

    ``compute_framework_rule`` is pinned to ``{KgPythonDictFramework}`` so the
    feature-name wrap that mloda's column-matcher needs lives in a
    framework-specific adapter rather than every reader's ``load``.
    Subclasses MUST NOT override this hook to a framework that does
    not perform an equivalent wrap: native KG rows have keys like ``s``/``p``/``o``
    that never match the user-defined feature name, so a non-wrapping framework
    silently loses every row in column slicing. If a different framework is
    truly needed, the override must point at another wrap-equivalent adapter.
    """

    READER_CLASS: ClassVar[type[KgConnectorReaderBase] | None] = None

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        if cls.READER_CLASS is None:
            return None
        return cls.READER_CLASS()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {KgPythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        reader = cls.input_data()
        if reader is None:
            raise ValueError(f"{cls.__name__}.READER_CLASS is None; concrete subclasses must pin a reader.")
        return reader.load(features)
