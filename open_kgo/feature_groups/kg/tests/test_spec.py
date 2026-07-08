"""Tests for the KG ``property_spec`` wrapper (``kg/spec.py``).

The wrapper delegates construction and invariant validation to mloda core
(``mloda.provider.property_spec``); these tests pin the one open-kgo-specific
behavior it adds (``default`` always emitted) and confirm core's validation is
in force through the wrapper.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

from open_kgo.feature_groups.kg.spec import property_spec


class TestPropertySpecEmission:
    def test_non_strict_spec_emits_conventional_dict(self) -> None:
        """The emitted dict matches the hand-written literal shape byte-for-byte."""
        emitted = property_spec("Endpoint URL or filesystem path.", default=1000)
        assert emitted == {
            "explanation": "Endpoint URL or filesystem path.",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: 1000,
        }

    def test_strict_spec_emits_allowed_values_unchanged(self) -> None:
        """``allowed_values`` passes through by reference so doc-dicts survive intact."""
        allowed = {"a": "Doc for a.", "b": "Doc for b."}
        emitted = property_spec("A strict enum.", strict=True, allowed_values=allowed, default="a")
        assert emitted["allowed_values"] is allowed
        assert emitted[DefaultOptionKeys.strict_validation] is True
        assert emitted[DefaultOptionKeys.default] == "a"

    def test_omitted_default_emits_none(self) -> None:
        """The wrapper always emits ``default`` (explicit ``None``); core omits it when None."""
        emitted = property_spec("An optional key.")
        assert emitted[DefaultOptionKeys.default] is None


class TestPropertySpecInvariants:
    def test_strict_without_allowed_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty allowed_values"):
            property_spec("Strict but empty.", strict=True)

    def test_strict_with_empty_allowed_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty allowed_values would reject"):
            property_spec("Strict but empty.", strict=True, allowed_values={})

    def test_allowed_values_without_strict_rejected(self) -> None:
        with pytest.raises(ValueError, match="never enforced"):
            property_spec("Decorative enum.", allowed_values={"a": "Doc."})

    def test_strict_default_outside_allowed_set_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in the accepted set"):
            property_spec("Bad default.", strict=True, allowed_values={"a": "Doc."}, default="z")

    def test_strict_none_default_permitted(self) -> None:
        """A strict enum may leave ``default=None``; an absent key is simply not validated."""
        emitted = property_spec("Strict, no default.", strict=True, allowed_values={"a": "Doc."})
        assert emitted[DefaultOptionKeys.default] is None

    def test_iterable_allowed_values_accepted(self) -> None:
        """Plain iterables mirror ``spec_allowed_values``; the default check still applies."""
        emitted = property_spec("Tuple enum.", strict=True, allowed_values=("a", "b"), default="b")
        assert emitted["allowed_values"] == ("a", "b")
        with pytest.raises(ValueError, match="not in the accepted set"):
            property_spec("Tuple enum.", strict=True, allowed_values=("a", "b"), default="z")

    def test_generator_allowed_values_materialized(self) -> None:
        """A one-shot iterable is materialized, not emitted exhausted."""
        emitted = property_spec("Generator enum.", strict=True, allowed_values=(v for v in ("a", "b")), default="a")
        assert emitted["allowed_values"] == ("a", "b")
        no_values: tuple[str, ...] = ()
        with pytest.raises(ValueError, match="empty allowed_values would reject"):
            property_spec("Empty generator.", strict=True, allowed_values=(v for v in no_values))
