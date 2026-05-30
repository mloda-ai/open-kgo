"""Class-definition-time integrity checks for KG connector readers.

Two families of invariant:

1. Composed ``PROPERTY_MAPPING`` is collision-free across family bases and
   mixins. Importing this module exercises composition; the helper raises
   ``PropertyMappingCollision`` on any duplicate, so a passing import means
   composition stayed honest.
2. ``SUPPORTED_VALUES`` declarations on concrete plugins satisfy the
   framework invariants enforced by
   ``KgConnectorReaderBase.__init_subclass__``: declared key, strict spec,
   non-empty narrowed set, narrowed set is a subset of family-allowed.

Plugin modules are loaded via ``_discovery.import_all_kg_readers`` rather than
a hand-maintained ``# noqa: F401`` block.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.agent_memory.base import (
    AgentMemoryReader,
    _MEMORY_SCOPE_SPECS,
)
from open_kgo.feature_groups.kg.agent_memory.networkx_memory import NetworkxMemoryReader
from open_kgo.feature_groups.kg.base import (
    KgConnectorReaderBase,
    ParamReader,
    compose_property_mapping,
    narrow_property_mapping,
)
from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    NonDictSpecError,
    PropertyMappingCollision,
)
from open_kgo.feature_groups.kg.tests._discovery import (
    clean_kg_subclass_registry,
    import_all_kg_readers,
    walk_subclasses,
)

import_all_kg_readers()


def test_every_family_property_mapping_composes_without_collision() -> None:
    """Importing the kg package already exercises composition at class-creation time.

    If any family base or mixin combination duplicated a key, the
    ``compose_property_mapping`` call in the family base body would raise
    ``PropertyMappingCollision`` at import — and ``import_all_kg_readers`` above
    would have failed. Reaching this assertion means every family composed
    cleanly.
    """
    subclasses = walk_subclasses(KgConnectorReaderBase)
    assert subclasses, "expected at least one concrete subclass to be registered"
    for sub in subclasses:
        mapping = sub.PROPERTY_MAPPING
        assert isinstance(mapping, dict)
        assert mapping, f"{sub.__name__} has empty PROPERTY_MAPPING"


# -- SUPPORTED_VALUES invariant tests -----------------------------------------
#
# Each test defines a class inline that violates one of the four invariants
# enforced by ``KgConnectorReaderBase._validate_supported_values_invariant``,
# and asserts the violation is caught at class-definition time. Inline
# definition is required because ``__init_subclass__`` runs synchronously at
# class creation; defining the class outside ``pytest.raises`` would fail at
# module import. The narrowed-key target is ``retrieval_mode`` (declared with
# ``strict_validation=True`` on ``AgentMemoryReader``), unless the test
# specifically targets a non-strict key.


def test_supported_values_invariant_rejects_unknown_key() -> None:
    """An entry in SUPPORTED_VALUES naming a key absent from PROPERTY_MAPPING /
    PARAMS_MAPPING fails at class definition."""
    with clean_kg_subclass_registry():
        with pytest.raises(ValueError, match="not present in PROPERTY_MAPPING or PARAMS_MAPPING"):

            class _Bad(NetworkxMemoryReader):
                CONNECTOR_ID = "_invariant_test_unknown_key"
                SUPPORTED_VALUES = {"definitely_not_a_real_kg_key": frozenset({"x"})}


def test_supported_values_invariant_rejects_non_strict_spec() -> None:
    """Narrowing a key whose spec has strict_validation=False is meaningless and rejected.

    ``locator`` is declared with ``strict_validation=False`` on the universal
    base, so narrowing it would silently lock the key without the family
    contract treating it as an enum. The framework rejects at class
    definition.
    """
    with clean_kg_subclass_registry():
        with pytest.raises(ValueError, match="strict_validation=True"):

            class _Bad(NetworkxMemoryReader):
                CONNECTOR_ID = "_invariant_test_non_strict"
                SUPPORTED_VALUES = {"locator": frozenset({"/tmp/x"})}


def test_supported_values_invariant_rejects_empty_narrowed_set() -> None:
    """An empty narrowed frozenset would reject every value of the key — caught early."""
    with clean_kg_subclass_registry():
        with pytest.raises(ValueError, match="empty"):

            class _Bad(NetworkxMemoryReader):
                CONNECTOR_ID = "_invariant_test_empty"
                SUPPORTED_VALUES = {"retrieval_mode": frozenset()}


def test_supported_values_invariant_rejects_value_outside_family_allowed() -> None:
    """Narrowing to a value the family contract never declared is caught.

    ``retrieval_mode`` allows ``{lexical, vector, hybrid, graph}``; narrowing
    to ``{telepathic}`` (not in family-allowed) would silently lock the key
    because no caller could ever supply a valid value. Rejected at class
    definition.
    """
    with clean_kg_subclass_registry():
        with pytest.raises(ValueError, match="not a subset"):

            class _Bad(NetworkxMemoryReader):
                CONNECTOR_ID = "_invariant_test_not_subset"
                SUPPORTED_VALUES = {"retrieval_mode": frozenset({"telepathic"})}


# -- __init_subclass__ chain test ---------------------------------------------
#
# Every ParamReader concrete depends on TWO ``__init_subclass__`` methods
# firing in sequence: ``KgConnectorReaderBase.__init_subclass__`` (validates
# the SUPPORTED_VALUES invariant) and ``ParamReader.__init_subclass__``
# (computes ``_STRIPPED_PARAMS``). The latter chains the former via
# ``super().__init_subclass__(**kwargs)``. A refactor that drops the super()
# call would silently disable the invariant on ParamReader subclasses; this
# test pins the chain so the regression surfaces immediately.


def test_init_subclass_chain_runs_both_invariants_for_param_reader() -> None:
    """Defining a ParamReader concrete with narrowed PARAMS_MAPPING and valid
    SUPPORTED_VALUES must (a) succeed without error (invariant ran via the
    super() chain) and (b) populate ``_STRIPPED_PARAMS`` correctly
    (ParamReader hook ran). Both observations together prove the chain.

    The class is built inside a factory so its local binding is reclaimed at
    factory return; only the primitive ``_STRIPPED_PARAMS`` frozenset escapes.
    Wrapping the factory call in ``clean_kg_subclass_registry`` pins the
    contract that this test does not leak into the global registry.
    """
    from open_kgo.feature_groups.kg.code_build.base import CodeBuildReader

    def _build_and_extract_stripped() -> frozenset[str]:
        class _Chain(CodeBuildReader):
            # CodeBuildReader.PARAMS_MAPPING includes lineage_direction etc. via
            # TraversalMixin / EntityFilterParamMixin. Narrow to a single key so
            # _STRIPPED_PARAMS is non-empty; lineage_direction is strict, so we
            # can safely also exercise SUPPORTED_VALUES against it.
            PARAMS_MAPPING = {k: v for k, v in CodeBuildReader.PARAMS_MAPPING.items() if k == "lineage_direction"}
            SUPPORTED_VALUES = {"lineage_direction": frozenset({"UPSTREAM"})}

        return _Chain._STRIPPED_PARAMS

    with clean_kg_subclass_registry():
        observed_stripped = _build_and_extract_stripped()
    # (a) Class created without error → KgConnectorReaderBase.__init_subclass__ ran.
    # (b) _STRIPPED_PARAMS computed correctly → ParamReader.__init_subclass__ ran.
    expected_stripped = frozenset(CodeBuildReader.PARAMS_MAPPING) - {"lineage_direction"}
    assert observed_stripped == expected_stripped


# -- MEMORY_SCOPE_KEYS single source of truth ---------------------------------
#
# Issue #5 item 18: ``MEMORY_SCOPE_KEYS`` and the AgentMemoryReader property
# mapping used to be independently maintained copies of the same list, with a
# third copy in ``NetworkxMemoryReader.REQUIRED_KEYS``. The third copy is now
# moot — that concrete reader narrows to ``(("locator",), ("memory_scope_user_id",))``
# because its JSON fixture is keyed by user_id only — so the constant exists
# for *future* concretes (Mem0, Letta, Zep+Graphiti) that consume the full
# scope. The cross-check below pins the *shape of the derivation*: that each
# spec materialised inside ``AgentMemoryReader.PROPERTY_MAPPING`` matches the
# explanation/default declared in ``_MEMORY_SCOPE_SPECS``, and that
# ``context``/``strict_validation`` are wired the way the comprehension says.
# Breaking the comprehension shape (e.g. swapping ``explanation`` and
# ``default``, dropping ``context``, or hand-rolling the mapping again
# without using the tuple) lights this test up; an inert pass means the
# comprehension still maps the same fields the same way.


def test_memory_scope_specs_propagate_into_agent_memory_property_mapping() -> None:
    """Each ``_MEMORY_SCOPE_SPECS`` row must materialise as the canonical
    four-key spec inside ``AgentMemoryReader.PROPERTY_MAPPING``: the
    ``explanation`` and ``default`` come from the tuple, ``context`` is
    pinned to ``True`` and ``strict_validation`` to ``False``. This is
    the cross-check that catches a future refactor of
    ``_MEMORY_SCOPE_PROPERTY_MAPPING`` that silently drops or rewires
    fields while leaving ``MEMORY_SCOPE_KEYS`` (and therefore the trivial
    set-containment check) intact.
    """
    declared = AgentMemoryReader.PROPERTY_MAPPING
    for name, explanation, default in _MEMORY_SCOPE_SPECS:
        assert name in declared, f"{name!r} declared in _MEMORY_SCOPE_SPECS but absent from PROPERTY_MAPPING"
        spec = declared[name]
        assert spec["explanation"] == explanation
        assert spec[DefaultOptionKeys.default] == default
        assert spec[DefaultOptionKeys.context] is True
        assert spec[DefaultOptionKeys.strict_validation] is False


# -- compose_property_mapping duplicate handling ------------------------------
#
# The helper raises ``PropertyMappingCollision`` on any duplicate key across
# composed sources (including value-equal duplicates). The reasoning is
# YAGNI: no current callsite produces duplicates (the ``EntityFilter*``
# mixin pair splits across ``PROPERTY_MAPPING_DELTA`` /
# ``PARAMS_MAPPING_DELTA``, so it never collides), and a stricter "any
# duplicate raises" rule is easier to reason about than a "duplicates are
# OK iff value-equal" rule. If a future composition genuinely needs to
# share a key across two sources, the right move is the
# ``_MEMORY_SCOPE_SPECS`` extraction pattern in ``agent_memory/base.py``,
# not loosening the helper.


def test_compose_property_mapping_rejects_duplicate_keys() -> None:
    """Any duplicate key across composed sources raises ``PropertyMappingCollision``."""
    source_a = {"k": {"explanation": "first"}}
    source_b = {"k": {"explanation": "second"}}

    with pytest.raises(PropertyMappingCollision):
        compose_property_mapping(source_a, source_b, context="test")


# -- compose_property_mapping non-dict spec handling --------------------------
#
# A ``None`` spec value (or any non-dict) would otherwise propagate to
# ``_validate_mapping`` and surface as a self-contradicting error: the key
# appears in ``allowed: [...]`` but ``mapping.get(key) is None`` cannot
# distinguish a missing key from a ``None`` spec. Catching the misconfiguration
# at composition time keeps the closed-world check honest. Mirrors the same
# pattern as the duplicate-key rejection above: structural mistakes loud at
# import time rather than at runtime. The error type is ``NonDictSpecError``
# (a sibling of ``PropertyMappingCollision`` under ``InvalidCredentialShape``)
# so callers can catch the whole compose-time structural-error family with a
# single typed handler.


def test_compose_property_mapping_rejects_none_spec_value() -> None:
    """A ``None`` spec value is misconfiguration and must raise at compose time."""
    with pytest.raises(NonDictSpecError, match=r"must be a dict"):
        compose_property_mapping({"extra": None}, context="test")


def test_compose_property_mapping_rejects_non_dict_spec_value() -> None:
    """Any non-dict spec value (string, number, list, ...) is rejected at compose time."""
    with pytest.raises(NonDictSpecError, match=r"must be a dict"):
        compose_property_mapping({"extra": "not_a_dict"}, context="test")


def test_compose_property_mapping_rejects_none_spec_value_across_sources() -> None:
    """A ``None`` spec value in a later source is caught alongside valid earlier sources."""
    with pytest.raises(NonDictSpecError, match=r"must be a dict"):
        compose_property_mapping({"first": {"explanation": "ok"}}, {"second": None}, context="test")


def test_compose_property_mapping_non_dict_spec_error_is_invalid_credential_shape() -> None:
    """Single-rooted tree: ``NonDictSpecError`` is also catchable as ``InvalidCredentialShape``.

    Lets callers scope one handler across the whole compose-time
    structural-error family (duplicate keys + non-dict specs) without
    enumerating each sibling typed error.
    """
    with pytest.raises(InvalidCredentialShape):
        compose_property_mapping({"extra": None}, context="test")


def test_compose_property_mapping_non_dict_spec_error_carries_context_prefix() -> None:
    """The ``context`` argument is prefixed onto the error message and exposed as an attribute.

    Pins the context-prefix code path: without this assertion, the ``if context``
    branch in the error formatter would be silently disable-able.
    """
    with pytest.raises(NonDictSpecError) as info:
        compose_property_mapping({"extra": None}, context="MyFamily.PROPERTY_MAPPING")
    assert "MyFamily.PROPERTY_MAPPING: " in str(info.value)
    assert info.value.context == "MyFamily.PROPERTY_MAPPING"
    assert info.value.key == "extra"
    assert info.value.spec is None


def test_compose_property_mapping_non_dict_spec_error_omits_prefix_without_context() -> None:
    """Without a context argument, no leading ``": "`` separator appears in the message."""
    with pytest.raises(NonDictSpecError) as info:
        compose_property_mapping({"extra": None})
    msg = str(info.value)
    assert msg.startswith("spec for key 'extra'")
    assert ": spec for key" not in msg


# -- narrow_property_mapping --------------------------------------------------
#
# The narrowing companion to ``compose_property_mapping``: concrete plugins
# drop family-level keys they do not honor. Centralised so the intent is named
# rather than spelled as an inline ``{k: v ... if k not in {...}}`` comprehension
# at each concrete.


def test_narrow_property_mapping_drops_excluded_keys() -> None:
    """Excluded keys are removed; every other key and its spec is preserved by identity."""
    source = {"a": {"explanation": "1"}, "b": {"explanation": "2"}, "c": {"explanation": "3"}}
    narrowed = narrow_property_mapping(source, "b")
    assert set(narrowed) == {"a", "c"}
    assert narrowed["a"] is source["a"]
    assert narrowed["c"] is source["c"]


def test_narrow_property_mapping_excludes_multiple_keys() -> None:
    """Multiple exclude args are all dropped in one call."""
    source: dict[str, dict[str, Any]] = {"a": {}, "b": {}, "c": {}, "d": {}}
    assert set(narrow_property_mapping(source, "b", "d")) == {"a", "c"}


def test_narrow_property_mapping_ignores_absent_excludes() -> None:
    """Excluding a key not present in the source is a no-op (idempotent narrowing)."""
    source: dict[str, dict[str, Any]] = {"a": {}, "b": {}}
    assert narrow_property_mapping(source, "missing") == source


def test_narrow_property_mapping_returns_fresh_dict() -> None:
    """The result is a new dict; mutating it does not touch the source."""
    source: dict[str, dict[str, Any]] = {"a": {}, "b": {}}
    narrowed = narrow_property_mapping(source)
    narrowed["c"] = {}
    assert "c" not in source


# -- Class-definition-time spec-shape guard -----------------------------------
#
# ``compose_property_mapping`` only catches non-dict specs that arrive through
# the helper. Concretes that build their mapping via dict-comprehension off
# an already-composed parent bypass that check. ``__init_subclass__`` runs the
# same guard once more so a direct-assignment regression also surfaces loudly,
# at class-creation time rather than at first runtime use.


def test_init_subclass_rejects_direct_assignment_of_non_dict_spec_in_property_mapping() -> None:
    """A concrete that assigns ``PROPERTY_MAPPING`` with a non-dict value fails at class definition."""
    from open_kgo.feature_groups.kg.agent_memory.networkx_memory import NetworkxMemoryReader

    with pytest.raises(NonDictSpecError, match=r"must be a dict"):

        class _BadDirectAssign(NetworkxMemoryReader):
            CONNECTOR_ID = "_invariant_test_non_dict_property_spec"
            PROPERTY_MAPPING = {
                **{k: v for k, v in NetworkxMemoryReader.PROPERTY_MAPPING.items()},
                "_injected_bad_key": None,
            }


def test_init_subclass_rejects_direct_assignment_of_non_dict_spec_in_params_mapping() -> None:
    """A concrete that assigns ``PARAMS_MAPPING`` with a non-dict value fails at class definition."""
    from open_kgo.feature_groups.kg.code_build.base import CodeBuildReader

    with pytest.raises(NonDictSpecError, match=r"must be a dict"):

        class _BadDirectAssignParams(CodeBuildReader):
            CONNECTOR_ID = "_invariant_test_non_dict_params_spec"
            PARAMS_MAPPING = {
                **{k: v for k, v in CodeBuildReader.PARAMS_MAPPING.items()},
                "_injected_bad_key": None,
            }


# -- Default-value legality on strict-validation specs ------------------------
#
# Every strict-validation spec carries a ``default`` paired with an
# ``allowed_values`` set. A typo in a future spec (or a refactor that drifts
# the default off the allowed set) would silently install a default that fails
# the first time the framework or a caller stamps it onto a credential dict.
# Today every default is legal across all readers — this test pins that fact
# as a regression guard. Walked per-subclass to catch concrete-level spec
# overrides (today the specs are uniformly inherited from family bases, but
# nothing structurally prevents a concrete from re-declaring a spec).


def test_strict_validation_defaults_are_legal() -> None:
    """For every ``strict_validation=True`` spec across PROPERTY_MAPPING/PARAMS_MAPPING
    on every concrete reader, ``spec[default]`` must be in
    ``_spec_allowed_values(spec)`` (or be ``None``).

    Aggregates failures so a future spec drift surfaces every violation at
    once rather than the first one only.
    """
    subclasses = walk_subclasses(KgConnectorReaderBase)
    assert subclasses, "expected at least one concrete subclass to be registered"
    failures: list[str] = []
    for sub in subclasses:
        for layer_name in ("PROPERTY_MAPPING", "PARAMS_MAPPING"):
            mapping: dict[str, object] = getattr(sub, layer_name, {}) or {}
            for key, spec in mapping.items():
                if not isinstance(spec, dict):
                    continue
                if spec.get(DefaultOptionKeys.strict_validation) is not True:
                    continue
                default = spec.get(DefaultOptionKeys.default)
                if default is None:
                    continue
                allowed = KgConnectorReaderBase._spec_allowed_values(key, spec)
                if default not in allowed:
                    failures.append(
                        f"{sub.__name__}.{layer_name}[{key!r}] default={default!r} not in allowed set {sorted(allowed)}"
                    )
    assert not failures, "strict-validation defaults out of allowed set:\n" + "\n".join(failures)


def test_init_subclass_chain_propagates_invariant_failure_through_param_reader() -> None:
    """If ``ParamReader.__init_subclass__`` ever drops the ``super()`` call, a
    bad ``SUPPORTED_VALUES`` on a ParamReader subclass would silently slip
    through. This test pins the propagation: defining a ParamReader concrete
    with bad SUPPORTED_VALUES must raise ``ValueError`` from the inherited
    invariant check.
    """
    from open_kgo.feature_groups.kg.code_build.base import CodeBuildReader

    with clean_kg_subclass_registry():
        with pytest.raises(ValueError, match="not a subset"):

            class _BadChain(CodeBuildReader):
                SUPPORTED_VALUES = {"lineage_direction": frozenset({"sideways"})}


# -- _STRIPPED_PARAMS chain inheritance (issue #18 item A6) -------------------
#
# ``_compute_stripped_params`` must compare ``cls.PARAMS_MAPPING`` against the
# **top-most** ParamReader-subclass ancestor (with a non-empty PARAMS_MAPPING),
# not the nearest. A multi-level chain like ``FamilyP{α,β}`` →
# ``FirstC{α}`` (strips β) → ``SecondC{}`` (strips α) would otherwise drop β
# from ``SecondC._STRIPPED_PARAMS`` because ``FirstC.PARAMS_MAPPING`` no
# longer contains it. The bug was latent: today there is no concrete-of-
# concrete plugin in the registry, but the regression-guard pins the
# inheritance shape so a future sub-concrete inherits the full strip set.
#
# Synthetic classes are built per-test via ``_build_three_level_chain`` and
# held only by the test frame; ``__subclasses__`` keeps weak references so
# they are collected with the frame and do not surface in the cross-group
# registry walk.


_A6_NON_STRICT_SPEC: dict[str, Any] = {
    DefaultOptionKeys.context: True,
    DefaultOptionKeys.strict_validation: False,
}


def _build_three_level_chain(
    *,
    top_keys: tuple[str, ...],
    mid_keys: tuple[str, ...],
    bottom_keys: tuple[str, ...],
    prefix: str,
) -> tuple[type[ParamReader], type[ParamReader], type[ParamReader]]:
    """Build a ``Top → Mid → Bottom`` ParamReader chain with the given PARAMS_MAPPING keys.

    Each level's ``PARAMS_MAPPING`` is built from ``_A6_NON_STRICT_SPEC`` so
    callers only have to express *which keys* the level declares, not the spec
    body. ``prefix`` is used to disambiguate ``CONNECTOR_ID`` across tests
    (avoids the cross-group ``test_no_duplicate_connector_ids`` invariant
    catching synthetic-class collisions if a test happens to leak a reference).
    Classes are returned to the caller and otherwise unreferenced; they fall
    out of ``__subclasses__`` once the test frame is collected.
    """

    def _mapping(keys: tuple[str, ...]) -> dict[str, Any]:
        return {k: _A6_NON_STRICT_SPEC for k in keys}

    class _Top(ParamReader):
        CONNECTOR_ID: ClassVar[str] = f"_test_a6_{prefix}_top"
        PARAMS_MAPPING: ClassVar[dict[str, Any]] = _mapping(top_keys)

    class _Mid(_Top):
        CONNECTOR_ID: ClassVar[str] = f"_test_a6_{prefix}_mid"
        PARAMS_MAPPING: ClassVar[dict[str, Any]] = _mapping(mid_keys)

    class _Bottom(_Mid):
        CONNECTOR_ID: ClassVar[str] = f"_test_a6_{prefix}_bottom"
        PARAMS_MAPPING: ClassVar[dict[str, Any]] = _mapping(bottom_keys)

    return _Top, _Mid, _Bottom


def test_stripped_params_inherits_from_top_most_family_ancestor() -> None:
    """3-level chain: ``Bottom._STRIPPED_PARAMS`` must include both α and β.

    Reproduces the issue #18 A6 scenario:
    - ``Top`` declares ``{α, β}``.
    - ``Mid`` extends ``Top`` and strips β (``PARAMS_MAPPING = {α}``).
    - ``Bottom`` extends ``Mid`` and strips α (``PARAMS_MAPPING = {}``).

    Expected: ``Bottom._STRIPPED_PARAMS == {α, β}``. Before the fix the MRO
    walk broke at the nearest ParamReader ancestor (``Mid``) and computed
    ``{α} - {} == {α}``, silently accepting β in ``feature.options.context``.
    """
    top, mid, bottom = _build_three_level_chain(
        top_keys=("alpha", "beta"),
        mid_keys=("alpha",),
        bottom_keys=(),
        prefix="classtime",
    )
    # Family base itself has no strips: it IS the top-most.
    assert top._STRIPPED_PARAMS == frozenset()
    # 2-level case (existing concrete pattern): nearest == top-most == Top.
    assert mid._STRIPPED_PARAMS == frozenset({"beta"})
    # 3-level case (the bug): both α and β must be in the strip set.
    assert bottom._STRIPPED_PARAMS == frozenset({"alpha", "beta"})


def test_stripped_params_rejection_engages_for_inherited_strips() -> None:
    """End-to-end: a stripped-param key set in ``feature.options.context``
    must be rejected even when the strip was inherited from a grand-ancestor.

    Mirrors ``test_stripped_params_inherits_from_top_most_family_ancestor``
    but exercises the runtime guard (``_reject_stripped_params``) rather than
    the class-time computation, so a future refactor that recomputes the
    field correctly but breaks the rejection hook still trips this test.
    """
    from mloda.core.abstract_plugins.components.feature_set import FeatureSet
    from mloda.user import Feature, Options

    _, _, bottom = _build_three_level_chain(
        top_keys=("alpha", "beta"),
        mid_keys=("alpha",),
        bottom_keys=(),
        prefix="runtime",
    )
    fs = FeatureSet()
    # beta was stripped at the mid level and inherited through bottom; setting it
    # in a bottom feature's options.context must raise.
    fs.add(Feature("probe__beta", options=Options(context={"beta": "x"})))
    with pytest.raises(InvalidCredentialShape):
        bottom._reject_stripped_params(fs)


def test_stripped_params_skips_param_reader_ancestors_with_empty_mapping() -> None:
    """An intermediate ParamReader ancestor with empty ``PARAMS_MAPPING`` is
    skipped when picking the family base.

    Edge case for the top-most-with-non-empty-mapping rule: if the top-most
    ParamReader ancestor happens to declare an empty ``PARAMS_MAPPING`` (e.g.
    an intermediate abstract base), the walk should keep going and pick the
    next ancestor down that has a non-empty mapping. This is the natural
    consequence of the "non-empty" qualifier in the rule.
    """
    _, _, leaf = _build_three_level_chain(
        top_keys=(),
        mid_keys=("alpha", "beta"),
        bottom_keys=("alpha",),
        prefix="emptytop",
    )
    # Leaf compares against the mid level (the only non-empty ancestor), not
    # the abstract top; the strip set is therefore {β}.
    assert leaf._STRIPPED_PARAMS == frozenset({"beta"})
