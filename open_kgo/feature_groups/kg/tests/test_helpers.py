"""Unit tests for ``_helpers.py``.

Covers the test-library helpers exposed by ``_helpers``:

- ``bogus_value_for_strict_spec``: type-coherent bogus value for any
  strict-validation spec, replacing the hand-rolled
  ``__obviously_not_a_valid_enum_value__`` retry idiom. The future-proofing
  claim ("generalises to non-string-valued strict specs") is only honest if
  a test actually exercises int/bool/exotic specs, that's what
  ``test_bogus_value_for_strict_spec_*`` does.
- ``make_valid_credentials``: pre-populate the spec's ``default`` keys
  and let callers override the rest. The class-mutable-state pattern
  (``cls._tmp``) is intentionally out of scope here; lifting it requires a
  contract-base signature change.
- ``run_query`` zero-result paths: the prior
  ``run_query_allowing_empty`` workaround is retired now that the framework
  adapter passes ``[]`` through unchanged. The end-to-end assertions live
  in ``test_run_query_returns_empty_list_for_unknown_stable_id`` and
  ``test_run_query_returns_native_rows_when_present`` below; they exercise
  the full ``mloda.run_all`` path instead of the direct-load shortcut.

Fixture coupling: the ``make_valid_credentials`` and ``run_query`` tests use
``FileFixtureCitationReader`` and its bundled
``citation_rest/tests/fixtures/reactome.json`` because it is the only
no-network, no-tempdir-setup concrete in the package (every other concrete
needs either a backend or a per-test resource). If the ``citation_rest``
family is reorganised, this module needs the new path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.citation_rest.file_fixture_citation import (
    FileFixtureCitationReader,
)
from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingRequiredKeysError,
)
from open_kgo.feature_groups.kg.tests._helpers import (
    bogus_value_for_strict_spec,
    make_valid_credentials,
    run_query,
)

_CITATION_FIXTURE = Path(__file__).parent.parent / "citation_rest" / "tests" / "fixtures" / "reactome.json"


# --- bogus_value_for_strict_spec ---------------------------------------------


def test_bogus_value_for_strict_spec_string_allowed_values_returns_string_outside_set() -> None:
    """Mirrors the existing kg_contract.py call shape: spec with a string set."""
    spec: dict[str, Any] = {
        "allowed_values": {"alpha", "beta", "gamma"},
        DefaultOptionKeys.strict_validation: True,
    }
    bogus = bogus_value_for_strict_spec(spec)
    assert isinstance(bogus, str)
    assert bogus not in {"alpha", "beta", "gamma"}


def test_bogus_value_for_strict_spec_string_dict_allowed_values_uses_keys() -> None:
    """``allowed_values`` may be a dict mapping value → docstring; we extract keys."""
    spec: dict[str, Any] = {
        "allowed_values": {"read": "read consistency", "linearisable": "strong"},
        DefaultOptionKeys.strict_validation: True,
    }
    bogus = bogus_value_for_strict_spec(spec)
    assert isinstance(bogus, str)
    assert bogus not in {"read", "linearisable"}


def test_bogus_value_for_strict_spec_avoids_canonical_candidate_when_collision() -> None:
    """If the canonical ``__bogus_strict_spec_value_0__`` is in the set, the helper sweeps."""
    spec: dict[str, Any] = {
        "allowed_values": {"alpha", "__bogus_strict_spec_value_0__"},
        DefaultOptionKeys.strict_validation: True,
    }
    bogus = bogus_value_for_strict_spec(spec)
    assert isinstance(bogus, str)
    assert bogus not in {"alpha", "__bogus_strict_spec_value_0__"}


def test_bogus_value_for_strict_spec_int_allowed_values_returns_int_above_max() -> None:
    """Future-proofing: an int-valued strict spec gets a same-type bogus value.

    Today no KG strict spec uses ints; this test pins the helper's behavior
    so the contract test in ``kg_contract.py`` keeps working without
    refactor when a future spec adds an int enum (e.g. an HTTP status code
    set, a max-hops bound).
    """
    spec: dict[str, Any] = {
        "allowed_values": {1, 2, 5, 10},
        DefaultOptionKeys.strict_validation: True,
    }
    bogus = bogus_value_for_strict_spec(spec)
    assert isinstance(bogus, int) and not isinstance(bogus, bool)
    assert bogus not in {1, 2, 5, 10}
    assert bogus > 10


def test_bogus_value_for_strict_spec_bool_allowed_values_returns_sentinel() -> None:
    """``{True, False}`` is the universe of bool; no same-type bogus value can exist.

    A fresh ``object()`` sentinel is unequal to every bool (set
    membership at the validator returns False) so the rejection path
    engages for the right reason. The implementation reaches the
    sentinel via fallthrough: the int branch filters out bools
    (``isinstance(v, int) and not isinstance(v, bool)``), so a bool-only
    ``allowed`` is naturally an empty int list; the str branch is also
    empty; the final ``object()`` covers it. Without the int filter, a
    bool spec would receive ``max({True, False}) + 1 == 2``, which is
    still not in the set but is a semantically confusing answer.
    """
    spec: dict[str, Any] = {
        "allowed_values": {True, False},
        DefaultOptionKeys.strict_validation: True,
    }
    bogus = bogus_value_for_strict_spec(spec)
    # ``object()`` is unequal to True/False; the membership check at the
    # validator's `in` returns False.
    assert bogus not in {True, False}
    assert not isinstance(bogus, bool)


def test_bogus_value_for_strict_spec_mixed_bool_int_picks_int_branch() -> None:
    """Mixed ``{bool, int}`` is deterministic: the int branch wins via the bool filter.

    No current KG spec mixes types in ``allowed_values``, but the helper
    advertises deterministic behaviour for the heterogeneous case. This
    test pins it: for ``{True, 5}``, the int filter excludes ``True``
    and yields ``[5]``, so the helper returns ``6`` rather than the
    order-dependent answer the first implementation would have returned
    (``object()`` if ``True`` was sampled first, ``6`` if ``5`` was).
    """
    spec: dict[str, Any] = {
        "allowed_values": {True, 5},
        DefaultOptionKeys.strict_validation: True,
    }
    bogus = bogus_value_for_strict_spec(spec)
    assert bogus == 6
    assert not isinstance(bogus, bool)


def test_bogus_value_for_strict_spec_mixed_bool_str_picks_str_branch() -> None:
    """Mixed ``{bool, str}`` is deterministic: the str branch wins via fallthrough.

    The int branch finds no qualifying values (the only int-like member
    is the bool, which the filter excludes), so the helper sweeps for
    a fresh string sentinel. Pins the symmetric case to its int sibling
    so a future refactor that re-orders the branches surfaces here.
    """
    spec: dict[str, Any] = {
        "allowed_values": {True, "alpha"},
        DefaultOptionKeys.strict_validation: True,
    }
    bogus = bogus_value_for_strict_spec(spec)
    assert isinstance(bogus, str)
    assert bogus not in {True, "alpha"}
    assert bogus.startswith("__bogus_strict_spec_value_")


def test_bogus_value_for_strict_spec_empty_allowed_returns_sentinel() -> None:
    """An empty iterable allowed_values falls through to the object() sentinel.

    ``_spec_allowed_values`` rejects a *missing* ``allowed_values``, but an
    empty *non-missing* iterable survives that check. The helper covers
    the latter for defense-in-depth — a non-member exists trivially.
    """
    spec: dict[str, Any] = {
        "allowed_values": frozenset(),
        DefaultOptionKeys.strict_validation: True,
    }
    bogus = bogus_value_for_strict_spec(spec)
    # The sentinel is by construction outside any set.
    assert bogus not in frozenset()


# --- make_valid_credentials --------------------------------------------------


def test_make_valid_credentials_returns_wrapped_form_with_connector_id() -> None:
    """The helper returns ``{CONNECTOR_ID: slot}`` so callers can drop it into DAC."""
    creds = make_valid_credentials(
        FileFixtureCitationReader,
        locator=str(_CITATION_FIXTURE),
    )
    assert FileFixtureCitationReader.CONNECTOR_ID in creds
    slot = creds[FileFixtureCitationReader.CONNECTOR_ID]
    assert isinstance(slot, dict)
    # The override key landed:
    assert slot["locator"] == str(_CITATION_FIXTURE)


def test_make_valid_credentials_prefills_spec_defaults() -> None:
    """Spec defaults are pre-populated; the caller only supplies the rest.

    ``result_limit`` carries an explicit non-``None`` default on the universal
    base; the helper must lift it into the slot so the caller doesn't re-spell
    family-level scaffolding.
    """
    creds = make_valid_credentials(
        FileFixtureCitationReader,
        locator=str(_CITATION_FIXTURE),
    )
    slot = creds[FileFixtureCitationReader.CONNECTOR_ID]
    assert slot["result_limit"] == 1000


def test_make_valid_credentials_override_wins_over_default() -> None:
    """A kwarg override replaces the spec default (last-write-wins)."""
    creds = make_valid_credentials(
        FileFixtureCitationReader,
        locator=str(_CITATION_FIXTURE),
        result_limit=42,
    )
    slot = creds[FileFixtureCitationReader.CONNECTOR_ID]
    assert slot["result_limit"] == 42


def test_make_valid_credentials_skips_none_defaults() -> None:
    """Spec defaults of ``None`` are *not* pre-populated.

    ``species_prefix`` (declared on the citation_rest family base) carries
    ``default: None`` to express "no value by default". Pre-filling it with
    ``None`` invites a redundant ``slot.pop(key)`` at every call site that
    wants the key absent.
    """
    creds = make_valid_credentials(
        FileFixtureCitationReader,
        locator=str(_CITATION_FIXTURE),
    )
    slot = creds[FileFixtureCitationReader.CONNECTOR_ID]
    assert "species_prefix" not in slot


def test_make_valid_credentials_validates_by_default() -> None:
    """``validate=True`` (default) runs ``_validate_shape``; missing required keys raise."""
    with pytest.raises(MissingRequiredKeysError):
        make_valid_credentials(FileFixtureCitationReader)  # locator missing


def test_make_valid_credentials_validate_false_does_not_raise_on_missing_required() -> None:
    """``validate=False`` lets the caller construct a deliberately partial slot."""
    creds = make_valid_credentials(FileFixtureCitationReader, validate=False)
    # locator is required by REQUIRED_KEYS but we opted out of validation.
    assert "locator" not in creds[FileFixtureCitationReader.CONNECTOR_ID]


def test_make_valid_credentials_validate_true_rejects_closed_world_violation() -> None:
    """Closed-world unknown-key violations surface at construction (default validate=True).

    The earlier seed was ``auth_method="evil"`` against the universal
    strict-enum gate; the auth surface was removed, and
    ``FileFixtureCitationReader`` no longer narrows any strict enum at this
    layer. The closed-world check still triggers on any key not declared in
    ``PROPERTY_MAPPING``, which exercises the same ``validate=True`` hook.
    """
    with pytest.raises(InvalidCredentialShape):
        make_valid_credentials(
            FileFixtureCitationReader,
            locator=str(_CITATION_FIXTURE),
            definitely_not_a_kg_key="x",
        )


# --- run_query zero-result paths ---------------------------------------------


def test_run_query_returns_empty_list_for_unknown_stable_id() -> None:
    """An unknown ``stable_id`` is a legitimate zero-result path; we get ``[]``, not a raise.

    Before the empty-result relaxation, ``run_query`` flowed through
    ``KgPythonDictFramework.select_data_by_column_names`` which rejected
    ``[]`` with the parent ``PythonDictFramework``'s "Data cannot be empty"
    guard. The adapter now passes empty data through unchanged so the
    full ``mloda.run_all`` path is usable for legitimate zero-result
    queries.

    Dogfoods ``make_valid_credentials`` (B5) to build the slot so both
    helpers exercise each other in the same test.
    """
    slot = make_valid_credentials(FileFixtureCitationReader, locator=str(_CITATION_FIXTURE))[
        FileFixtureCitationReader.CONNECTOR_ID
    ]
    feat = Feature(
        "file_fixture_citation__missing_id",
        options=Options(context={"stable_id": "R-HSA-NOT-A-REAL-ID", "hierarchy_depth": 1}),
    )
    rows = run_query(FileFixtureCitationReader.CONNECTOR_ID, slot, feat)
    assert rows == []


def test_run_query_returns_native_rows_when_present() -> None:
    """A known ``stable_id`` round-trips: ``run_query`` unwraps the framework
    wrap and yields the reader's native row dicts.

    Pairs with the zero-result sibling so a regression that re-tightens
    the "empty is fatal" guard would be caught either by the empty path
    raising (the sibling) or by this path's non-empty assertion (the
    framework wrap regressing both directions at once would show up here
    instead of silently passing the empty case).
    """
    slot = make_valid_credentials(FileFixtureCitationReader, locator=str(_CITATION_FIXTURE))[
        FileFixtureCitationReader.CONNECTOR_ID
    ]
    feat = Feature(
        "file_fixture_citation__known_id",
        options=Options(context={"stable_id": "R-HSA-1640170", "hierarchy_depth": 0}),
    )
    rows = run_query(FileFixtureCitationReader.CONNECTOR_ID, slot, feat)
    assert len(rows) == 1
    assert rows[0]["stableId"] == "R-HSA-1640170"
