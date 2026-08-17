"""Thin KG wrapper over mloda core's ``property_spec`` PROPERTY_MAPPING builder.

Core owns the builder and its invariants (see mloda-ai/open-kgo#29); this wrapper
adds two open-kgo conventions:

- ``default`` defaults to ``None`` rather than core's ``NO_DEFAULT``, so a KG spec
  is optional unless the caller explicitly opts into a different default. Core's
  ``PropertySpec`` treats an omitted ``default`` as required (mloda>=0.11.0).
- ``allowed_values`` requires ``strict=True``. Core permits a non-strict value
  space (it still maps a name-parsed value back onto its key), but KG specs are
  never name-parsed and ``_validate_mapping`` enforces the set only under
  ``strict_validation``, so a non-strict enum here would be decorative.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mloda.provider import PropertySpec, property_spec as _core_property_spec

_AllowedValues = Mapping[Any, str] | tuple[Any, ...] | list[Any] | set[Any] | frozenset[Any]


def property_spec(
    explanation: str,
    *,
    strict: bool = False,
    allowed_values: Mapping[Any, str] | Iterable[Any] | None = None,
    default: Any = None,
    context: bool = True,
) -> PropertySpec:
    """Build a PROPERTY_MAPPING ``PropertySpec`` via core, defaulting to optional.

    ``allowed_values`` accepts any iterable (a generator included) for caller
    convenience; anything that isn't already one of core's accepted concrete
    shapes is materialized into a tuple before reaching core, whose own type
    is narrower.
    """
    if allowed_values is not None and not strict:
        raise ValueError(
            f"property_spec({explanation!r}): allowed_values is never enforced without strict=True. "
            f"Pass strict=True, or drop allowed_values."
        )
    materialized: _AllowedValues | None
    if allowed_values is None or isinstance(allowed_values, Mapping):
        materialized = allowed_values
    else:
        materialized = tuple(allowed_values)
    return _core_property_spec(
        explanation,
        strict=strict,
        allowed_values=materialized,
        default=default,
        context=context,
    )
