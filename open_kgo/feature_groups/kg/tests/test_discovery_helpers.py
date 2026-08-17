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
- ``iter_nonstrict_specs`` / ``reader_string_literals`` /
  ``effective_unconsumed_waivers``: the surface-honesty building blocks,
  including an end-to-end disposition that flags a synthetic surface lie and
  shows a waiver clearing it (the negative test for
  ``test_no_unconsumed_advertised_keys``).

Every synthetic class is built inside a factory so its local binding is
reclaimed at factory return; primitive observations escape so the cm wrapping
the factory call can assert "no class persisted past the block."
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.provider import property_spec

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase, ParamReader, QueryReader
from open_kgo.feature_groups.kg.tests._discovery import (
    clean_kg_subclass_registry,
    effective_unconsumed_waivers,
    iter_nonstrict_specs,
    iter_strict_specs,
    reader_string_literals,
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
            # Synthetic mapping drops 'locator', so the source-slot convention
            # requires declaring the baked shape (same below for every
            # synthetic reader that replaces PROPERTY_MAPPING wholesale).
            SOURCE_SLOT = None
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "loose": property_spec("loose", default=None, allowed_values=["a"]),
                "strict": property_spec("strict", default=None, strict=True, allowed_values=["b"]),
                "unset": property_spec("unset", default=None, allowed_values=["c"]),
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
            SOURCE_SLOT = None
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "prop_strict": property_spec("prop_strict", default=None, strict=True, allowed_values=["a"]),
            }
            PARAMS_MAPPING: ClassVar[dict[str, Any]] = {
                "param_strict": property_spec("param_strict", default=None, strict=True, allowed_values=["b"]),
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
            SOURCE_SLOT = None
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "qr_strict": property_spec("qr_strict", default=None, strict=True, allowed_values=["x"]),
            }
            # Drift: QueryReader subclass with a PARAMS_MAPPING attribute.
            # The helper must not yield its keys regardless of presence —
            # only ParamReader subclasses are supposed to declare per-call
            # params, and the helper enforces that via the issubclass gate.
            PARAMS_MAPPING: ClassVar[dict[str, Any]] = {
                "should_not_appear": property_spec(
                    "should_not_appear", default=None, strict=True, allowed_values=["y"]
                ),
            }

        return [(key, layer) for key, _spec, layer in iter_strict_specs(_QR)]

    with clean_kg_subclass_registry():
        observed = _exercise()
    assert observed == [("qr_strict", "PROPERTY_MAPPING")]
    assert all(key != "should_not_appear" for key, _layer in observed)


# -- iter_nonstrict_specs -----------------------------------------------------


def test_iter_nonstrict_specs_is_the_complement_of_iter_strict_specs() -> None:
    """``iter_nonstrict_specs`` yields exactly the non-strict keys over the same layers.

    The two helpers partition the advertised surface so the strict-enum and
    surface-honesty contract tests never double-cover or skip a key.
    """

    def _exercise() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        class _PR(ParamReader):
            CONNECTOR_ID = "_b4_nonstrict"
            SOURCE_SLOT = None
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "prop_strict": property_spec("prop_strict", default=None, strict=True, allowed_values=["a"]),
                "prop_loose": property_spec("prop_loose", default=None),
            }
            PARAMS_MAPPING: ClassVar[dict[str, Any]] = {
                "param_loose": property_spec("param_loose", default=None),
            }

        strict = [(k, str(layer)) for k, _s, layer in iter_strict_specs(_PR)]
        nonstrict = [(k, str(layer)) for k, _s, layer in iter_nonstrict_specs(_PR)]
        return strict, nonstrict

    with clean_kg_subclass_registry():
        strict, nonstrict = _exercise()
    assert strict == [("prop_strict", "PROPERTY_MAPPING")]
    assert sorted(nonstrict) == [("param_loose", "PARAMS_MAPPING"), ("prop_loose", "PROPERTY_MAPPING")]


# -- reader_string_literals ---------------------------------------------------


def test_reader_string_literals_separates_read_keys_from_declared_only_keys() -> None:
    """A key read by literal in a method appears; a key only declared in a mapping does not."""

    def _exercise() -> set[str]:
        class _ProbeReader(KgConnectorReaderBase):
            CONNECTOR_ID = "_b4_literals"
            SOURCE_SLOT = None
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                # Declared but never read in any method body below.
                "zzz_advertised_only": property_spec("zzz_advertised_only", default=None),
                "zzz_consumed_key": property_spec("zzz_consumed_key", default=None),
            }

            @classmethod
            def _connect_from_slot(cls, slot: Any) -> Any:
                # Reads one key by literal; the docstring mentions the other key
                # name in prose, which must NOT count as consumption.
                """Mentions zzz_advertised_only in prose only."""
                return slot.get("zzz_consumed_key")

        return reader_string_literals(_ProbeReader)

    with clean_kg_subclass_registry():
        literals = _exercise()
    assert "zzz_consumed_key" in literals
    assert "zzz_advertised_only" not in literals


def test_reader_string_literals_includes_inherited_base_consumption() -> None:
    """Universal keys read in the base (``ontology`` / ``result_limit``) surface for any reader."""

    def _exercise() -> set[str]:
        class _BareReader(KgConnectorReaderBase):
            CONNECTOR_ID = "_b4_inherited"

        return reader_string_literals(_BareReader)

    with clean_kg_subclass_registry():
        literals = _exercise()
    assert {"ontology", "result_limit"} <= literals


# -- effective_unconsumed_waivers ---------------------------------------------


def test_effective_unconsumed_waivers_unions_across_mro() -> None:
    """A concrete's waiver set unions with every ancestor's, rather than shadowing it."""

    def _exercise() -> set[str]:
        class _FamilyBase(KgConnectorReaderBase):
            SOURCE_SLOT = None
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "family_key": property_spec("family_key", default=None),
                "concrete_key": property_spec("concrete_key", default=None),
            }
            _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"family_key"})

        class _Concrete(_FamilyBase):
            CONNECTOR_ID = "_b4_waiver_union"
            _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"concrete_key"})

        return effective_unconsumed_waivers(_Concrete)

    with clean_kg_subclass_registry():
        waived = _exercise()
    assert {"family_key", "concrete_key"} <= waived


def test_unconsumed_waiver_naming_unknown_key_is_rejected_at_class_definition() -> None:
    """``_WAIVED_UNCONSUMED_KEYS`` naming a key absent from the mappings raises at definition."""
    with clean_kg_subclass_registry():
        with pytest.raises(ValueError, match="not present in PROPERTY_MAPPING or PARAMS_MAPPING"):

            class _StaleWaiver(KgConnectorReaderBase):
                CONNECTOR_ID = "_b4_stale_waiver"
                _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"definitely_not_advertised"})


def test_surface_honesty_disposition_flags_a_lie_and_a_waiver_clears_it() -> None:
    """End-to-end: an advertised-but-unread non-strict key is flagged; waiving it clears it.

    Composes the three helpers as ``test_no_unconsumed_advertised_keys`` does,
    proving the contract catches a real lie rather than passing vacuously.
    """

    def _disposition(reader_cls: type[KgConnectorReaderBase]) -> list[str]:
        consumed = reader_string_literals(reader_cls)
        waived = effective_unconsumed_waivers(reader_cls)
        return [
            key for key, _spec, _layer in iter_nonstrict_specs(reader_cls) if key not in consumed and key not in waived
        ]

    def _exercise() -> tuple[list[str], list[str]]:
        class _LyingReader(KgConnectorReaderBase):
            CONNECTOR_ID = "_b4_lie"
            SOURCE_SLOT = None
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "zzz_never_read": property_spec("zzz_never_read", default=None),
            }

        class _WaivedReader(KgConnectorReaderBase):
            CONNECTOR_ID = "_b4_waived"
            SOURCE_SLOT = None
            PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {
                "zzz_never_read": property_spec("zzz_never_read", default=None),
            }
            _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"zzz_never_read"})

        return _disposition(_LyingReader), _disposition(_WaivedReader)

    with clean_kg_subclass_registry():
        lying, waived = _exercise()
    assert lying == ["zzz_never_read"]
    assert waived == []
