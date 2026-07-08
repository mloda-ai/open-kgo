"""Thin KG wrapper over mloda core's ``property_spec`` PROPERTY_MAPPING builder.

mloda>=0.9.0 ships ``mloda.provider.property_spec`` plus the first-class
``DefaultOptionKeys.allowed_values`` field, so open-kgo no longer maintains its
own copy of the builder/validation (see mloda-ai/open-kgo#29). Core validates
the spec's invariants; this wrapper only preserves one open-kgo convention core
does not: ``default`` is always emitted (explicit ``None`` when unset) so KG
specs read uniformly via subscript across the validation/discovery/contract
layer. The extra key is in core's ``RESERVED_PROPERTY_KEYS`` and so is ignored
by core's own spec parser.
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
    spec = _core_property_spec(
        explanation,
        strict=strict,
        allowed_values=allowed_values,
        default=default,
        context=context,
    )
    spec.setdefault(DefaultOptionKeys.default, None)
    return spec
