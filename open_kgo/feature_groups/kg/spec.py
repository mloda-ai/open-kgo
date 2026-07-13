"""Thin KG wrapper over mloda core's ``property_spec`` PROPERTY_MAPPING builder.

Core ships ``mloda.provider.property_spec`` plus the first-class
``DefaultOptionKeys.allowed_values`` field (since 0.9.0), so open-kgo no longer
maintains its own copy of the builder/validation (see mloda-ai/open-kgo#29).
The floor is 0.10.0 because core renamed ``RESERVED_PROPERTY_KEYS`` to
``PROPERTY_SPEC_KEYS`` with no alias and relaxed the non-strict
``allowed_values`` rule this wrapper still enforces.

Core validates the spec's invariants; this wrapper preserves two open-kgo
conventions core does not:

- ``default`` is always emitted (explicit ``None`` when unset) so KG specs read
  uniformly via subscript across the validation/discovery/contract layer. The
  extra key is in core's ``PROPERTY_SPEC_KEYS`` and so is ignored by core's own
  spec parser.
- ``allowed_values`` requires ``strict=True``. Core permits a non-strict value
  space because it still maps a name-parsed value back onto its key; KG specs
  are never name-parsed, and ``_validate_mapping`` enforces the allowed set only
  under ``strict_validation``, so a non-strict enum here would be decorative.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.provider import property_spec as _core_property_spec


def property_spec(
    explanation: str,
    *,
    strict: bool = False,
    allowed_values: Mapping[Any, str] | Iterable[Any] | None = None,
    default: Any = None,
    context: bool = True,
) -> dict[str, Any]:
    """Build a PROPERTY_MAPPING spec dict via core, keeping ``default`` always present."""
    if allowed_values is not None and not strict:
        raise ValueError(
            f"property_spec({explanation!r}): allowed_values is never enforced without strict=True. "
            f"KG specs validate credentials, not feature names, so a non-strict value space is "
            f"decorative. Pass strict=True, or drop allowed_values."
        )
    spec = _core_property_spec(
        explanation,
        strict=strict,
        allowed_values=allowed_values,
        default=default,
        context=context,
    )
    spec.setdefault(DefaultOptionKeys.default, None)
    return spec
