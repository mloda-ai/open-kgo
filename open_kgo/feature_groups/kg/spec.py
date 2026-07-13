"""Thin KG wrapper over mloda core's ``property_spec`` PROPERTY_MAPPING builder.

Core owns the builder and its invariants (see mloda-ai/open-kgo#29); this wrapper
adds two open-kgo conventions:

- ``default`` is always emitted (explicit ``None`` when unset) so KG specs read
  uniformly via subscript. The key is in core's ``PROPERTY_SPEC_KEYS``, so core's
  own parser ignores it.
- ``allowed_values`` requires ``strict=True``. Core permits a non-strict value
  space (it still maps a name-parsed value back onto its key), but KG specs are
  never name-parsed and ``_validate_mapping`` enforces the set only under
  ``strict_validation``, so a non-strict enum here would be decorative.
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
            f"Pass strict=True, or drop allowed_values."
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
