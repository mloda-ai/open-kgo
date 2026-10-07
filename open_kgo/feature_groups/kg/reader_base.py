"""Universal reader base for KG connectors, plus the conceptual rules it enforces.

The split mirrors mloda core's ``ReadDB`` + ``ReadDBFeature`` pattern (see
``mloda_plugins/feature_group/input_data/read_db.py`` and
``read_db_feature.py``). Each KG family supplies a ``<Family>Reader`` that
extends ``KgConnectorReaderBase`` (which extends ``ReadDB``) and a
``<Family>FeatureGroup`` that extends ``KgConnectorFeatureGroupBase``.

Module layout: this module owns ``LoadContext`` and ``KgConnectorReaderBase``;
the two per-call reader flavors (``QueryReader`` / ``ParamReader``) live in
``kg.readers`` and the FeatureGroup base in ``kg.feature_group``. All of it is
re-exported through ``kg.base``, the documented front door. Runtime credential
rules come from the ``kg.credentials.CredentialRules`` mixin; class-definition
guards and composition bodies live in ``kg.class_guards`` and ``kg.composition``.

Concrete plugins set ``CONNECTOR_ID`` and implement ``connect``,
``build_query``, ``load_data``. Each KG FeatureGroup probes only its own
reader, which accepts the credentials via ``is_valid_credentials``,
implemented once in the universal base against ``CONNECTOR_ID``.

Honest credential surface
-------------------------
The rule, enforced rather than conventional: a connector must not advertise a
slot (``PROPERTY_MAPPING``) or param (``PARAMS_MAPPING``) it silently ignores.
Advertising a key a reader never reads lets a caller set it, see no error, and
get the wrong behavior. A connector reconciles its surface one of three ways:

1. **Drop it** via ``narrow_property_mapping`` (slot) or by narrowing
   ``PARAMS_MAPPING`` (param; dropped keys land in ``_STRIPPED_PARAMS`` and are
   rejected at call time).
2. **Narrow/waive an enum**: pin ``SUPPORTED_VALUES[key]`` to the honored
   subset, or list it in ``_WAIVED_ENUM_KEYS`` for forward-compat. Enforced by
   ``test_strict_enum_honored_or_waived``.
3. **Waive a non-enum key** kept for a future concrete: list it in
   ``_WAIVED_UNCONSUMED_KEYS`` with a one-line reason. Enforced by
   ``test_no_unconsumed_advertised_keys``, which treats an exact string-literal
   reference in a reader method as proof of consumption.

The two tests partition the advertised surface (strict enums vs. the rest), so
every key has an explicit disposition and a new connector that forgets to trim
goes red instead of misleading callers.

Source-slot convention
----------------------
A connector identifies "where is the data" via the credential key named in
``SOURCE_SLOT`` (default ``"locator"``). A family or concrete that renames the
address slot declares the new name (``SOURCE_SLOT = "manifest_path"`` on
``CodeBuildReader``, issue #18), and one that bakes its source into the reader
declares ``SOURCE_SLOT = None`` (``InProcessTupleStoreReader``, issue #19).
The declaration is enforced two ways (issue #21): ``_validate_source_slot``
rejects an undeclared rename or drop at class-definition time, and the
cross-family catalog test reads the declaration as data and fails on any
spelling outside its known vocabulary
(``test_source_slot_declaration_matches_catalog`` in
``tests/test_kg_catalog_declarations.py``), so a fourth spelling cannot creep in
silently. The guarantee is scoped: it stops a silent drop, rename, or
unconsumed-waiver of the *declared* slot (a declared ``SOURCE_SLOT`` may not
appear in ``_WAIVED_UNCONSUMED_KEYS``); it cannot decide that some *other*
advertised key has become the de-facto address, which stays a review judgement
the asymmetry catalog makes visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    match_rejection_owners,
    restamp_match_rejections_since,
)
from mloda.provider import CHAIN_SEPARATOR, COLUMN_SEPARATOR, INPUT_DATA_STAGE, ComputeFramework, record_match_rejection
from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_db import ReadDB

from open_kgo.feature_groups.kg import class_guards, composition
from open_kgo.feature_groups.kg.composition import (  # noqa: F401 -- back-compat re-export, moved to composition.py
    compose_property_mapping,
    narrow_property_mapping,
)
from open_kgo.feature_groups.kg.credentials import CredentialRules
from open_kgo.feature_groups.kg.errors import InvalidCredentialShape
from open_kgo.feature_groups.kg.spec import property_spec


@dataclass(frozen=True)
class LoadContext:
    """Bundle of values every concrete ``load_data`` needs at the top.

    ``slot`` is the already-extracted credential dict for this connector, exposed as a
    read-only ``Mapping`` (``MappingProxyType``) so concrete code can't
    accidentally mutate it through the frozen dataclass; ``result_limit``
    is the per-call row cap pulled from the slot (default 1000). Built
    once via ``_prepare_load(data_access)``.

    ``ontology_namespace`` is the namespace string from the loaded ontology
    file (e.g. ``"movie"``), or ``None`` when no ``ontology`` key was
    supplied in credentials. Connectors that support typed traversal pass
    this to ``OntologyRegistry.is_valid_edge(ctx.ontology_namespace, ...)``.
    """

    slot: Mapping[str, Any]
    result_limit: int
    ontology_namespace: str | None = None


_UNIVERSAL_PROPERTY_MAPPING: dict[str, Any] = {
    "locator": property_spec(
        "Endpoint URL or filesystem path. May be None for purely in-process backends.",
    ),
    "ontology": property_spec(
        (
            "Path to a YAML ontology definition file. Declares entity types, valid outgoing "
            "relationship types per entity, and domain/range constraints per relationship. "
            "When supplied, the file is loaded into OntologyRegistry under the namespace "
            "declared in the file. Connectors access typed-traversal lookups via "
            "OntologyRegistry using ctx.ontology_namespace. Optional: connectors without "
            "an ontology file behave exactly as before (no validation applied)."
        ),
    ),
    "result_limit": property_spec(
        (
            "Maximum rows/records returned per query (bound-output semantics). "
            "Concrete readers MUST short-circuit work when the limit is reached "
            "rather than walking the full source then slicing. Slicing-at-end "
            "leaks unbounded cost on wide inputs (e.g. dbt manifests, paginated "
            "REST, large in-memory graphs). Use itertools.islice or an explicit "
            "early-return loop; only slice at the end if the source is already "
            "a fully-materialized list of bounded size (i.e. the JSON parser "
            "produced it for you and walking it is O(result_limit) anyway)."
        ),
        default=1000,
    ),
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


class KgConnectorReaderBase(CredentialRules, ReadDB):
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

    # The credential key a caller sets to point this connector at its data
    # (the "Source-slot convention" in this module's docstring). Default is
    # the shared ``locator``. A family/concrete that renames the address slot
    # declares the new name here; one that bakes the source into the reader
    # declares ``None``. ``_validate_source_slot`` enforces at class-definition
    # time that the declaration matches ``PROPERTY_MAPPING``, so a rename or
    # drop without a matching declaration fails the import instead of becoming
    # a silent fourth spelling.
    SOURCE_SLOT: ClassVar[str | None] = "locator"

    # True when the credential slot alone decides the rows (no query or per-call params), so
    # ``count_rows`` can answer before a run. Counting runs the load through the shared parse cache.
    ROWS_FROM_SLOT: ClassVar[bool] = False

    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = dict(_UNIVERSAL_PROPERTY_MAPPING)

    # Narrow a family-level strict-validation enum down to the subset this
    # concrete plugin actually honors at runtime. Keys not present here pass
    # through with the family-base allowed set; keys present here must
    # additionally appear in the supplied frozenset to validate.
    # Used by ``_validate_shape`` and ``_validate_params`` (the latter via
    # ``ParamReader``) so the same hook covers credential and per-call surfaces.
    # Invariant: each ``SUPPORTED_VALUES[key]`` is a non-empty subset of the
    # spec's allowed set, and the key has ``strict_validation=True``. For
    # property-layer (credential slot) keys, if the narrowed set excludes the
    # spec's non-None default, the key must also appear in ``REQUIRED_KEYS``
    # (omission would otherwise bypass the narrowing; see
    # ``_validate_supported_values_invariant``). Enforced at class-definition
    # time by ``__init_subclass__``.
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]] = {}

    # Strict-validation enum keys this concrete deliberately accepts at the
    # family-wide allowed set without honoring every value at runtime. Waivers
    # are documentation, not code: they make "we accept all values for
    # forward-compat" explicit and reviewable. The honest alternative is
    # ``SUPPORTED_VALUES``; use a waiver only when narrowing would lock out a
    # forward-compatible value the family base legitimately advertises (e.g.
    # ``read_consistency`` is a Kuzu no-op today but real network_pg backends
    # will honor it). This is option 2 of the "Honest credential surface" rule
    # in this module's docstring. The ``test_strict_enum_honored_or_waived``
    # contract test rejects any strict-validation enum that is neither in
    # ``SUPPORTED_VALUES`` nor in ``_WAIVED_ENUM_KEYS``; each waived key carries
    # a one-line comment on the concrete class explaining the waiver. The
    # contract test enforces membership only; the comment is review-time
    # discipline.
    _WAIVED_ENUM_KEYS: ClassVar[frozenset[str]] = frozenset()

    # Option 3 of the "Honest credential surface" rule (module docstring): the
    # non-enum counterpart to ``_WAIVED_ENUM_KEYS``. Non-strict keys advertised
    # but not consumed at runtime, kept as forward-compat surface; each needs a
    # one-line reason. ``test_no_unconsumed_advertised_keys`` enforces it and
    # unions this set across the MRO (see ``effective_unconsumed_waivers``), so a
    # family base may waive family-wide keys and a concrete add its own.
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(
        cls,
        family_properties: dict[str, Any] | None = None,
        family_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if family_properties is not None or family_params is not None:
            cls._compose_family_surface(family_properties or {}, family_params or {})
        cls._validate_mapping_spec_shapes()
        cls._validate_supported_values_invariant()
        cls._validate_unconsumed_waivers()
        cls._validate_source_slot()

    @classmethod
    def _compose_family_surface(cls, family_properties: dict[str, Any], family_params: dict[str, Any]) -> None:
        """Auto-compose ``PROPERTY_MAPPING`` / ``PARAMS_MAPPING`` for a family base; see ``composition.compose_family_surface``."""
        composition.compose_family_surface(cls, family_properties, family_params)

    @classmethod
    def _validate_mapping_spec_shapes(cls) -> None:
        """Reject non-dict spec values in the mappings; see ``class_guards.validate_mapping_spec_shapes``."""
        class_guards.validate_mapping_spec_shapes(cls)

    @classmethod
    def _validate_supported_values_invariant(cls) -> None:
        """Reject ill-formed ``SUPPORTED_VALUES``; see ``class_guards.validate_supported_values_invariant``."""
        class_guards.validate_supported_values_invariant(cls)

    @classmethod
    def _validate_unconsumed_waivers(cls) -> None:
        """Reject stale unconsumed-key waivers; see ``class_guards.validate_unconsumed_waivers``."""
        class_guards.validate_unconsumed_waivers(cls)

    @classmethod
    def _validate_source_slot(cls) -> None:
        """Reject a contradictory ``SOURCE_SLOT`` declaration; see ``class_guards.validate_source_slot``."""
        class_guards.validate_source_slot(cls)

    def load(self, features: FeatureSet) -> dict[str, list[Any]]:
        """Reject multi-feature FeatureSets, then wrap native rows (or one row key's values) under each feature name.

        Concrete ``load_data`` implementations across every family read a single
        feature via ``next(iter(features.features))``; passing more than one
        feature would silently use whichever one the iterator yields first and
        label every row with that single name. Reject the multi-feature shape
        loudly so the contract violation surfaces immediately.

        Native KG rows (SPARQL bindings, Cypher rows, BFS hops, ...) carry their
        own keys (``s``, ``p``, ``o``, ``ancestor``, ...), none of which match
        the user-defined feature name, so mloda's columnar column selection
        would drop every row. The wrap returns the columnar single-column frame
        ``{feature_name: [row, ...]}`` that the stock ``PythonDictFramework``
        (mloda >= 0.9.0) accepts as-is: each cell holds the whole native row
        dict and is unwrapped by feature name downstream. The wrap is
        unconditional, even when the feature name collides with a native row
        key. An empty result yields the schema-bearing zero-row frame
        ``{feature_name: []}`` (not the schema-less ``{}`` mloda rejects), so a
        query with no matches returns zero rows without raising. A
        ``<id>__<name>~<row_key>`` feature gets that row key's values instead,
        a scalar column stock chained groups can consume.

        ``load_data`` is contractually required to return ``list[dict[str, Any]]``
        (every concrete in this package satisfies that). The shape check below
        turns a future drift (a concrete returning a single dict, ``None``, or
        a generator) into a typed error here rather than an indirect failure
        downstream in ``PythonDictFramework.transform``.
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
        return {str(feature.name): self._project(str(feature.name), result) for feature in features.features}

    @classmethod
    def _project(cls, feature_name: str, rows: list[dict[str, Any]]) -> list[Any]:
        """Whole rows for ``<id>__<name>``; one row key's values for ``<id>__<name>~<row_key>``."""
        _, separator, key = feature_name.partition(COLUMN_SEPARATOR)
        if not separator:
            return rows
        # A row lacking the key (e.g. an unbound SPARQL OPTIONAL) yields None; a key no row has is a typo.
        if not key or (rows and not any(key in row for row in rows)):
            raise ValueError(
                f"{cls.__name__}: no row carries key {key!r} requested by {feature_name!r}; "
                f"row keys: {sorted({k for row in rows for k in row})}."
            )
        return [cls._project_value(row.get(key)) for row in rows]

    @classmethod
    def _project_value(cls, value: Any) -> Any:
        """Hook for a projected cell; whole rows never pass through it. Identity by default."""
        return value

    @classmethod
    def _assert_single_feature(cls, features: FeatureSet) -> None:
        """Universal precondition: ``load_data`` dispatches one feature at a time.

        Sibling ``~<row_key>`` projections of one feature with equal options share
        that load. Extracted so subclasses (e.g. ``ParamReader``) can run the guard
        at the very top of their own ``load`` override, before any per-call
        validation that iterates ``features.features`` and would otherwise pick one
        feature silently.
        """
        batch = list(features.features)
        first = batch[0] if batch else None
        if first is None or any(
            str(f.name).partition(COLUMN_SEPARATOR)[0] != str(first.name).partition(COLUMN_SEPARATOR)[0]
            or dict(f.options.group) != dict(first.options.group)
            or dict(f.options.context) != dict(first.options.context)
            for f in batch
        ):
            raise ValueError(
                f"{cls.__name__}.load expects exactly one feature per call "
                f"(KG concrete load_data implementations all dispatch one feature at a time; only "
                f"'~<row_key>' projections of one feature with equal options share a load), "
                f"got {len(batch)}: {sorted(f.name for f in batch)}."
            )

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
        invoke separately. A present but malformed own slot records a match
        rejection naming the error type only, since the message may echo a secret.
        """
        if not cls.CONNECTOR_ID:
            return False
        try:
            creds = cls._extract_slot(credentials)
            if creds is None:
                return False
            cls._validate_shape(creds)
        except InvalidCredentialShape as exc:
            record_match_rejection(
                cls.get_class_name(),
                f"{cls.get_class_name()}: the {cls.CONNECTOR_ID!r} credential slot is present but invalid "
                f"({type(exc).__name__}); call {cls.get_class_name()}.connect() with the same dict for details",
                stage=INPUT_DATA_STAGE,
            )
            return False
        except Exception:
            return False
        return True

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        """Claim only ``<CONNECTOR_ID>__<name>``: chained (``...__sum_aggr``) and unrelated names go to other groups."""
        prefix = f"{cls.CONNECTOR_ID}{CHAIN_SEPARATOR}"
        if not feature_name.startswith(prefix):
            return False
        name = feature_name[len(prefix) :]
        # A leading "_" would let "<id>___op" read as a chain on "<id>_".
        return bool(name) and not name.startswith("_") and CHAIN_SEPARATOR not in name

    @classmethod
    def claims_feature_name(cls, feature_name: str) -> bool:
        """Only ``<CONNECTOR_ID>__...`` names probe this reader's credentials."""
        return bool(cls.CONNECTOR_ID) and str(feature_name).startswith(f"{cls.CONNECTOR_ID}{CHAIN_SEPARATOR}")

    @classmethod
    def is_final_reader(cls) -> bool:
        # Hides KG readers from the stock ReadDBFeature's subclass walk (a final ReadDB would also match it and stop
        # the stock read_dbs auto-load); each KG FeatureGroup matches its own reader in the two overrides below.
        return False

    @classmethod
    def feature_scope_data_access(cls, options: Options, feature_name: str) -> bool:
        """Core's pinned-reader path for this reader alone: options keyed by its name carry its data access."""
        for key in options.keys():
            if cls.deal_with_base_input_data_name_as_cls_or_str(key) != cls.data_access_name():
                continue
            if not cls.claims_feature_name(feature_name):
                return False
            before_checks = match_rejection_owners()
            if cls._reader_options_admit(options, record_absence=True):
                # As core does: the pinned slot reaches the plan as a redacting credential, not a plain dict.
                value = options.get(key)
                wrapped = cls.wrap_feature_scoped_access(value)
                if wrapped is not value:
                    options.set(key, wrapped)
                known_owners = match_rejection_owners()
                matched = cls.match_subclass_data_access(wrapped, [feature_name], options=options)
                if matched:
                    unmet = cls._unmet_current_declaration()
                    if unmet is None:
                        cls.add_base_input_data_to_options(cls, matched, options)
                        return True
                    record_match_rejection(cls.get_class_name(), unmet, stage=INPUT_DATA_OWNED_STAGE)
                else:
                    restamp_match_rejections_since(known_owners, INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)
            if match_rejection_owners() <= before_checks:
                name = cls.get_class_name()
                record_match_rejection(
                    name,
                    f"{name} is pinned for feature '{feature_name}' but matched nothing; a pinned reader is final",
                    stage=INPUT_DATA_OWNED_STAGE,
                )
            return False
        return False

    @classmethod
    def match_data_access(
        cls,
        feature_names: list[str],
        data_access_collection: DataAccessCollection,
        options: Options | None = None,
    ) -> tuple[Any, Any]:
        """Match only this reader, never a sibling found by walking subclasses."""
        if not all(cls.claims_feature_name(name) for name in feature_names):
            return None, None
        if not cls._reader_options_admit(options, record_absence=False):
            return None, None
        matched = cls.match_subclass_data_access(data_access_collection, feature_names, options=options)  # type: ignore[arg-type]
        if not matched:
            return None, None
        unmet = cls._unmet_current_declaration()
        if unmet is not None:
            record_match_rejection(cls.get_class_name(), unmet, stage=INPUT_DATA_OWNED_STAGE)
            return None, None
        return cls, matched

    @classmethod
    def data_access_identity(cls, data_access: Any) -> str:
        """The ``SOURCE_SLOT`` value without userinfo, query or fragment; mloda's default if unusable. Never raises."""
        try:
            slot = cls._extract_slot(data_access)
        except Exception:  # A malformed slot fails the load elsewhere; identity stays best-effort.
            slot = None
        for key in cls._source_keys():
            source = slot.get(key) if slot is not None else None
            if isinstance(source, (str, PurePath)) and str(source):
                projected = super().data_access_identity(str(source))
                if projected != str.__name__:
                    return projected
        return super().data_access_identity(data_access)

    @classmethod
    def _source_keys(cls) -> tuple[str, ...]:
        """``SOURCE_SLOT`` first, then its ``REQUIRED_KEYS`` alternatives (e.g. ``manifest_path`` or ``locator``)."""
        if cls.SOURCE_SLOT is None:
            return ()
        group = next((g for g in cls.REQUIRED_KEYS if cls.SOURCE_SLOT in g), ())
        return (cls.SOURCE_SLOT, *(key for key in group if key != cls.SOURCE_SLOT))

    @classmethod
    def count_rows(cls, data_access: Any, compute_framework: type[ComputeFramework]) -> int | None:
        """Rows a load returns for ``ROWS_FROM_SLOT`` readers; None for the rest."""
        if not cls.ROWS_FROM_SLOT:
            return None
        cls._validate_shape(cls._require_slot(data_access))
        return len(cls.load_data(data_access, FeatureSet()))

    @classmethod
    def _prepare_load(cls, data_access: Any) -> LoadContext:
        """Extract the slot and parse the result_limit default.

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
        slot = cls._require_slot(data_access)
        cls._validate_result_limit(slot)
        ontology_namespace: str | None = None
        ontology_path = slot.get("ontology")
        if ontology_path is not None:
            from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

            ontology_namespace = OntologyRegistry.load_file(str(ontology_path))
        return LoadContext(
            slot=MappingProxyType(slot),
            result_limit=slot.get("result_limit", 1000),
            ontology_namespace=ontology_namespace,
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> list[dict[str, Any]]:
        """Template method: prepare the ``LoadContext``, open the connection, delegate to ``_load_rows``.

        Every concrete used to open its ``load_data`` with the same two lines
        (``ctx = cls._prepare_load(data_access)`` then
        ``cls._connect_from_slot(ctx.slot)``); the prologue now lives here
        once and concretes implement ``_load_rows`` only. ``ReadDB.load``
        dispatches into this hook unchanged, so direct callers and the
        matcher path see identical behavior. ``data_access`` must be the
        ``{CONNECTOR_ID: slot}`` dict mloda matched; anything else raises
        ``InvalidCredentialShape``.
        """
        ctx = cls._prepare_load(data_access)
        connection = cls._connect_from_slot(ctx.slot)
        return cls._load_rows(ctx, connection, features)

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        """Produce native rows from the prepared context and opened connection.

        The concrete hook behind the ``load_data`` template. ``connection``
        is whatever this concrete's ``_connect_from_slot`` returned (a parsed
        manifest dict, an ``rdflib.Graph``, a ``kuzu.Connection``, a fixture
        ``Path``, ...); concretes alias it to a domain name on the first
        line. Per-call inputs are read here via ``build_query(features)``
        (QueryReader concretes) or ``build_params(features, ctx.slot)``
        (ParamReader concretes). Must return ``list[dict[str, Any]]``; the
        base ``load`` enforces that shape after ``ReadDB.load`` returns.
        """
        raise NotImplementedError

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
