"""Collect-all semantics for ``ParamReader._validate_params``.

When a reader declares multiple ``REQUIRED_PARAMS`` OR-groups and more than
one is unsatisfied, ``MissingRequiredParamsError`` should carry **every**
unsatisfied group, mirroring ``MissingRequiredKeysError``. The cross-group
contract suite only covers single-group enforcement on real plugins (today
only ``NetworkxEmbeddedReader`` declares non-empty ``REQUIRED_PARAMS``, with
one group), so the multi-group collect-all behavior is exercised here against
a synthetic subclass.

The synthetic ``ParamReader`` is built inside each test's inner exercise
function so its only reference is the function frame's local, which is
reclaimed when the function returns. Wrapping the exercise call in
``clean_kg_subclass_registry`` pins the contract that the synthetic class
does not persist past the test.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import ParamReader
from open_kgo.feature_groups.kg.errors import MissingRequiredParamsError
from open_kgo.feature_groups.kg.tests._discovery import clean_kg_subclass_registry

_CONNECTOR_ID = "_test_multi_group_param_reader"


def _build_multi_group_reader() -> type[ParamReader]:
    """Return a fresh ParamReader subclass with two REQUIRED_PARAMS groups.

    Defined as a factory so the class is bound to the caller's local scope
    only; no module-level subclass leaks into the global registry.
    """

    class _MultiGroupParamReader(ParamReader):
        CONNECTOR_ID: ClassVar[str] = _CONNECTOR_ID
        PARAMS_MAPPING: ClassVar[dict[str, Any]] = {
            "alpha": {DefaultOptionKeys.context: True, DefaultOptionKeys.strict_validation: False},
            "bravo": {DefaultOptionKeys.context: True, DefaultOptionKeys.strict_validation: False},
            "charlie": {DefaultOptionKeys.context: True, DefaultOptionKeys.strict_validation: False},
        }
        REQUIRED_PARAMS: ClassVar[tuple[tuple[str, ...], ...]] = (
            ("alpha",),
            ("bravo", "charlie"),
        )

    return _MultiGroupParamReader


def test_validate_params_collects_all_unsatisfied_groups() -> None:
    """Both OR-groups missing should surface in ``unsatisfied_groups`` in declared order."""

    def _exercise() -> tuple[tuple[tuple[str, ...], ...], str]:
        reader = _build_multi_group_reader()
        with pytest.raises(MissingRequiredParamsError) as excinfo:
            reader._validate_params({})
        return excinfo.value.unsatisfied_groups, excinfo.value.connector_id

    with clean_kg_subclass_registry():
        unsatisfied, connector_id = _exercise()
    assert unsatisfied == (("alpha",), ("bravo", "charlie"))
    assert connector_id == _CONNECTOR_ID


def test_validate_params_reports_only_unsatisfied_groups() -> None:
    """A single satisfied group is dropped; the rest still surface in order."""

    def _exercise() -> tuple[tuple[str, ...], ...]:
        reader = _build_multi_group_reader()
        with pytest.raises(MissingRequiredParamsError) as excinfo:
            reader._validate_params({"alpha": "x"})
        return excinfo.value.unsatisfied_groups

    with clean_kg_subclass_registry():
        unsatisfied = _exercise()
    assert unsatisfied == (("bravo", "charlie"),)


def test_validate_params_accepts_falsey_but_present_values() -> None:
    """A required param set to a falsey-but-non-None value (``0``, ``""``) satisfies its group.

    Presence is tested with ``is not None``, not truthiness, so a
    legitimately falsey value is not misread as absent (mirrors the
    ``kg_contract`` REQUIRED_KEYS presence convention).
    """

    def _exercise() -> None:
        reader = _build_multi_group_reader()
        # alpha=0 satisfies ("alpha",); bravo="" satisfies ("bravo", "charlie").
        # No MissingRequiredParamsError should be raised.
        reader._validate_params({"alpha": 0, "bravo": ""})

    with clean_kg_subclass_registry():
        _exercise()


def test_validate_params_treats_none_value_as_absent() -> None:
    """An explicit ``None`` value does not satisfy a required group."""

    def _exercise() -> tuple[tuple[str, ...], ...]:
        reader = _build_multi_group_reader()
        with pytest.raises(MissingRequiredParamsError) as excinfo:
            reader._validate_params({"alpha": None, "bravo": "y"})
        return excinfo.value.unsatisfied_groups

    with clean_kg_subclass_registry():
        unsatisfied = _exercise()
    assert unsatisfied == (("alpha",),)
