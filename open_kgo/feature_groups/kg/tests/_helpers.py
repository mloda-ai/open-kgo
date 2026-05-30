"""Shared test helpers for KG connector contract suites.

``run_query`` exercises the same path the demo uses:
``DataAccessCollection.credentials`` + ``mloda.run_all`` +
``KgPythonDictFramework`` (the KG-aware ``PythonDictFramework`` adapter).
This means contract tests verify the real reader matching, validation, and
load chain, not a pre-bound shortcut. If ``CONNECTOR_ID`` matching,
``is_valid_credentials`` discovery, the subclass-walk wiring, or the
framework adapter regresses, the failure surfaces here rather than silently
passing.

Zero-result paths (citation ``stable_id=NOT_THERE``, an empty fixture dir,
an ``agent_memory`` query with no matches) are returned as ``[]`` by
``run_query`` itself: ``KgPythonDictFramework`` now relaxes the parent
``PythonDictFramework``'s "empty is fatal" guard for KG semantics (issue
#30). The earlier ``run_query_allowing_empty`` workaround, which bypassed
``mloda.run_all`` to surface ``[]`` to callers, is therefore retired; all
contract tests go through ``run_query``.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.user import DataAccessCollection, Feature, mloda

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase
from open_kgo.feature_groups.kg.python_dict_kg_framework import KgPythonDictFramework


def run_query(connector_id: str, slot_creds: dict[str, Any], feature: Feature) -> list[Any]:
    """Run a feature through ``mloda.run_all`` against a single KG connector.

    mloda walks ``KgConnectorReaderBase`` subclasses and selects the one whose
    ``CONNECTOR_ID`` slot is present in ``credentials``. The slot is wrapped in
    a one-element list (``credentials=[{CONNECTOR_ID: slot}]``) so mloda 0.7.0
    resolves it back to the ``{CONNECTOR_ID: slot}`` bundle the reader matcher
    expects; passing the bare dict makes mloda treat ``CONNECTOR_ID`` as a
    handle and unwrap to the inner slot, which fails ``is_valid_credentials``.
    The selected
    reader's ``load`` returns native KG rows; ``KgPythonDictFramework``
    (pinned by ``KgConnectorFeatureGroupBase.compute_framework_rule``) wraps
    each row as ``{feature_name: row}`` during column slicing so this helper
    can unwrap by feature name and return the underlying row values flattened
    across partitions.

    Zero-result queries return ``[]`` (issue #30): the adapter passes empty
    data through unchanged, so the comprehension below contributes no rows
    and the helper yields ``[]`` to the caller.
    """
    dac = DataAccessCollection(credentials=[{connector_id: slot_creds}])
    partitions = mloda.run_all(
        [feature],
        compute_frameworks={KgPythonDictFramework},
        data_access_collection=dac,
    )
    return [row[feature.name] for partition in partitions for row in partition if feature.name in row]


def make_valid_credentials(
    reader_class: type[KgConnectorReaderBase],
    *,
    validate: bool = True,
    **overrides: Any,
) -> dict[str, dict[str, Any]]:
    """Build a contract-conformant ``{CONNECTOR_ID: slot}`` credentials dict.

    Pre-populates every ``PROPERTY_MAPPING`` key whose spec declares an
    explicit, non-``None`` ``default``; callers supply the rest via kwargs
    (e.g. ``locator=str(tmp_path)``). The result is drop-in compatible with
    ``DataAccessCollection.credentials`` and with the
    ``valid_credentials()`` adapter return value, so a concrete test can
    write ``return make_valid_credentials(KuzuCypherReader, locator=path)``
    instead of re-spelling the whole slot for each scenario.

    Scope note: the architectural ``cls._tmp`` class-mutable-state pattern
    in concretes that need a setup_method-built resource (e.g.
    ``test_kuzu_cypher.py``) is **not** removed by this helper. The pattern
    exists because the contract base calls ``valid_credentials()`` as a
    classmethod with no access to a per-instance ``tmp_path``; lifting that
    requires a contract-base signature change which belongs in a follow-up.
    What this helper does fix is the hand-rolled slot-dict duplication
    across scenarios within a single concrete test file.

    Defaults are read via ``DefaultOptionKeys.default`` (not the bare
    string), matching the spec authoring convention in
    ``open_kgo/feature_groups/kg/base.py``. ``None`` defaults are
    treated as "no default to apply" — the spec uses ``None`` to make
    "absence is intentional" explicit for optional keys like ``locator``
    or family-level pin keys (``species_prefix``, ``dataset_version``,
    ``stable_id``), and pre-filling them with ``None`` would only invite
    a redundant ``slot.pop(key)`` at every call site. The same branch
    also covers specs that omit the ``default`` field entirely
    (``spec.get`` returns ``None``); since every KG spec under
    ``base.py`` declares ``default`` explicitly, that case currently
    never fires, but the behaviour is intentional and not accidental.

    When ``validate=True`` (default), the resulting slot runs through
    ``reader_class._validate_shape`` so a missing-required-key or
    strict-enum error surfaces at construction time, not later inside a
    contract assertion. Pass ``validate=False`` when deliberately
    constructing a partial slot to exercise a negative scenario.
    """
    slot: dict[str, Any] = {}
    for key, spec in reader_class.PROPERTY_MAPPING.items():
        if not isinstance(spec, dict):
            continue
        default = spec.get(DefaultOptionKeys.default)
        if default is None:
            continue
        slot[key] = default
    slot.update(overrides)
    if validate:
        reader_class._validate_shape(slot)
    return {reader_class.CONNECTOR_ID: slot}


def bogus_value_for_strict_spec(spec: dict[str, Any]) -> Any:
    """Return a value guaranteed not to be in this strict spec's allowed set.

    Type-coheres to the existing allowed values when possible — prefers a
    same-type "not in" value (int max+1, or a fresh string) over a
    sentinel — so the bogus value mirrors a realistic typo (wrong member
    of the right type) rather than a wrong-type misuse. The validator at
    ``KgConnectorReaderBase._validate_mapping`` only checks set
    membership, so any non-member would technically work; matching the
    member type keeps future type-aware refactors from accidentally
    masking the rejection path.

    Type selection inspects the union of types present in ``allowed``,
    not just the first iterated element, so behaviour is deterministic
    across Python invocations with different ``PYTHONHASHSEED`` even
    for mixed-type sets (no current KG spec hits this; the determinism
    pin matters as a future-proofing contract). Branch order:

    - int (excluding bool, since ``bool`` is an ``int`` subclass): pick
      ``max(...) + 1``. The filter is what protects an all-bool
      ``allowed`` from landing here and returning the semantically
      confusing ``max({True, False}) + 1 == 2``.
    - str: sweep an integer-suffixed sentinel string. The sweep is
      load-bearing only when an ``allowed_values`` set happens to
      already contain ``"__bogus_strict_spec_value_0__"``; ordinary
      specs hit the first candidate.
    - fallthrough: ``object()``. Covers every remaining shape —
      missing ``allowed_values``, empty allowed set, all-bool set
      ({True}/{False}/{True, False}), and any exotic value type. The
      sentinel is unequal to every concrete value so the validator
      rejects it for the right reason. An empty *non-missing*
      ``allowed_values`` survives ``_validate_mapping``'s upstream
      check and lands here too.

    The helper accepts only the spec dict — not a separately-supplied
    "narrowed" set — because ``SUPPORTED_VALUES`` is enforced at class
    definition time to be a subset of the family-allowed set (see
    ``KgConnectorReaderBase._validate_supported_values_invariant`` in
    ``base.py``). A value outside the family set is therefore also
    outside any narrowing of it, so the simpler signature is honest
    about the invariant.

    ``allowed_values`` is parsed inline (dict → keys, iterable → set)
    rather than calling ``KgConnectorReaderBase._spec_allowed_values``
    so the helper does not bind to a private static method whose
    signature may evolve to use its currently-message-only ``key``
    argument for behaviour.
    """
    raw = spec.get("allowed_values")
    if raw is None:
        return object()
    allowed: set[Any] = set(raw.keys()) if isinstance(raw, dict) else set(raw)
    ints = [v for v in allowed if isinstance(v, int) and not isinstance(v, bool)]
    if ints:
        return max(ints) + 1
    strings = [v for v in allowed if isinstance(v, str)]
    if strings:
        counter = 0
        while True:
            candidate = f"__bogus_strict_spec_value_{counter}__"
            if candidate not in allowed:
                return candidate
            counter += 1
    return object()
