"""Unit tests for ``_discovery`` helpers.

``walk_subclasses`` / ``family_subpackages`` / ``family_of`` /
``import_all_kg_readers`` are exercised implicitly by every sibling test that
uses them. ``iter_strict_specs`` is also exercised implicitly via the
strict-validation contract tests in ``kg_contract.py`` and
``test_validation_contract.py``. This file covers the corners that the
consumer tests don't pin:

- ``clean_kg_subclass_registry`` positive / negative / invariant-failure paths.
- ``iter_strict_specs`` skips non-dict spec entries, walks ``PARAMS_MAPPING``
  only for ``ParamReader`` subclasses, and skips non-strict keys.

Every synthetic class is built inside a factory so its local binding is
reclaimed at factory return; primitive observations escape so the cm wrapping
the factory call can assert "no class persisted past the block."
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase, ParamReader, QueryReader
from open_kgo.feature_groups.kg.tests._discovery import (
    clean_kg_subclass_registry,
    iter_strict_specs,
)


# -- clean_kg_subclass_registry ----------------------------------------------


def test_clean_kg_subclass_registry_passes_when_class_is_locally_scoped() -> None:
    """A synthetic class defined inside a factory leaves no trace after factory return.

    Python releases the factory frame's locals on return, the class's
    refcount drops to zero, and the weakref in ``__subclasses__`` is cleared
    eagerly. The cm sees no persistent leak.
    """

    def _create_and_discard() -> None:
        class _CleanProbeReader(KgConnectorReaderBase):
            CONNECTOR_ID = "_b2_clean_probe"

    with clean_kg_subclass_registry():
        _create_and_discard()


def test_clean_kg_subclass_registry_detects_persistent_leak() -> None:
    """A class held alive past the block is flagged with its fully-qualified name."""
    pinner: list[type[KgConnectorReaderBase]] = []
    try:

        def _create_and_pin() -> None:
            class _LeakyProbeReader(KgConnectorReaderBase):
                CONNECTOR_ID = "_b2_leaky_probe"

            pinner.append(_LeakyProbeReader)

        with pytest.raises(AssertionError, match=r"_LeakyProbeReader") as excinfo:
            with clean_kg_subclass_registry():
                _create_and_pin()
        assert "leaked past the block" in str(excinfo.value)
    finally:
        # ``finally`` clear so the leaked class is reclaimed even if the
        # assertion above fires; a missed cleanup would pollute downstream
        # tests with a synthetic subclass. ``.clear()`` (not rebind) so the
        # list referenced by the traceback frame is the one we empty.
        pinner.clear()


def test_clean_kg_subclass_registry_passes_when_init_subclass_raises() -> None:
    """Synthetic classes whose ``__init_subclass__`` raises do not leak.

    When class creation fails, Python never binds the local name, so the
    weakref in ``__subclasses__`` is cleared immediately. The cm sees no
    persistent leak even though the class statement was inside the cm.
    """
    with clean_kg_subclass_registry():
        with pytest.raises(ValueError):

            class _BadProbe(KgConnectorReaderBase):
                CONNECTOR_ID = "_b2_bad_probe"
                # Unknown key in SUPPORTED_VALUES → ``__init_subclass__`` raises.
                SUPPORTED_VALUES = {"definitely_not_a_real_kg_key": frozenset({"x"})}


def test_clean_kg_subclass_registry_propagates_body_exception_unmasked() -> None:
    """A body that raises while also leaking surfaces the body's exception, not the leak.

    Without the ``sys.exc_info`` skip in the cm's ``finally``, a test that
    already failed for an unrelated reason would be reported as a leak
    instead — pytest's primary header would mask the actual test failure.
    The body here both raises ``RuntimeError`` and pins a class; the
    assertion is that ``RuntimeError`` propagates (not ``AssertionError``)
    and the leak is silent.
    """
    pinner: list[type[KgConnectorReaderBase]] = []
    try:
        with pytest.raises(RuntimeError, match="synthetic test failure"):
            with clean_kg_subclass_registry():

                class _MaskedProbeReader(KgConnectorReaderBase):
                    CONNECTOR_ID = "_b2_masked_probe"

                pinner.append(_MaskedProbeReader)
                raise RuntimeError("synthetic test failure")
    finally:
        pinner.clear()


# -- iter_strict_specs --------------------------------------------------------


def test_iter_strict_specs_skips_non_strict_specs() -> None:
    """Specs without ``strict_validation=True`` are skipped."""

    def _exercise() -> list[str]:
        class _MixedStrictness(KgConnectorReaderBase):
            CONNECTOR_ID = "_b3_mixed"
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "loose": {DefaultOptionKeys.strict_validation: False, "allowed_values": ["a"]},
                "strict": {DefaultOptionKeys.strict_validation: True, "allowed_values": ["b"]},
                "unset": {"allowed_values": ["c"]},
            }

        return [key for key, _spec, _layer in iter_strict_specs(_MixedStrictness)]

    with clean_kg_subclass_registry():
        keys = _exercise()
    assert keys == ["strict"]


def test_iter_strict_specs_includes_params_mapping_for_param_reader() -> None:
    """``PARAMS_MAPPING`` is walked for ``ParamReader`` subclasses."""

    def _exercise() -> list[tuple[str, str]]:
        class _PR(ParamReader):
            CONNECTOR_ID = "_b3_param_reader"
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "prop_strict": {DefaultOptionKeys.strict_validation: True, "allowed_values": ["a"]},
            }
            PARAMS_MAPPING: ClassVar[dict[str, Any]] = {
                "param_strict": {DefaultOptionKeys.strict_validation: True, "allowed_values": ["b"]},
            }

        return [(key, layer) for key, _spec, layer in iter_strict_specs(_PR)]

    with clean_kg_subclass_registry():
        observed = _exercise()
    assert sorted(observed) == [("param_strict", "PARAMS_MAPPING"), ("prop_strict", "PROPERTY_MAPPING")]


def test_iter_strict_specs_excludes_params_mapping_for_query_reader() -> None:
    """``PARAMS_MAPPING`` is not walked for ``QueryReader`` subclasses.

    The helper gates the layer addition on ``issubclass(cls, ParamReader)``
    rather than ``hasattr``. Pins the drift scenario explicitly: the
    synthetic ``_QR`` bolts on a ``PARAMS_MAPPING`` attribute (which the
    domain model forbids for ``QueryReader``) and the assertion verifies
    that its key is NOT yielded. A future refactor that swapped the
    ``issubclass`` check for ``hasattr`` would silently broaden the walk
    and surface here.
    """

    def _exercise() -> list[tuple[str, str]]:
        class _QR(QueryReader):
            CONNECTOR_ID = "_b3_query_reader"
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "qr_strict": {DefaultOptionKeys.strict_validation: True, "allowed_values": ["x"]},
            }
            # Drift: QueryReader subclass with a PARAMS_MAPPING attribute.
            # The helper must not yield its keys regardless of presence —
            # only ParamReader subclasses are supposed to declare per-call
            # params, and the helper enforces that via the issubclass gate.
            PARAMS_MAPPING: ClassVar[dict[str, Any]] = {
                "should_not_appear": {DefaultOptionKeys.strict_validation: True, "allowed_values": ["y"]},
            }

        return [(key, layer) for key, _spec, layer in iter_strict_specs(_QR)]

    with clean_kg_subclass_registry():
        observed = _exercise()
    assert observed == [("qr_strict", "PROPERTY_MAPPING")]
    assert all(key != "should_not_appear" for key, _layer in observed)
