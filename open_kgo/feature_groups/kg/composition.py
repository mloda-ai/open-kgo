"""Family-surface composition helpers for KG connector readers.

Extracted from ``reader_base.py``; sibling concern modules are
``class_guards.py`` and ``credentials.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import PropertySpec

from open_kgo.feature_groups.kg.errors import NonDictSpecError, PropertyMappingCollision

if TYPE_CHECKING:
    from open_kgo.feature_groups.kg.reader_base import KgConnectorReaderBase


def compose_property_mapping(*sources: dict[str, Any], context: str = "") -> dict[str, Any]:
    """Merge property-mapping dicts, raising on duplicate keys or non-PropertySpec spec values."""
    merged: dict[str, Any] = {}
    for source in sources:
        for key, spec in source.items():
            if key in merged:
                raise PropertyMappingCollision(key, context=context)
            if not isinstance(spec, PropertySpec):
                raise NonDictSpecError(key, spec, context=context)
            merged[key] = spec
    return merged


def narrow_property_mapping(source: dict[str, Any], *exclude: str) -> dict[str, Any]:
    """Return ``source`` minus the ``exclude`` keys."""
    excluded = set(exclude)
    return {k: v for k, v in source.items() if k not in excluded}


def compose_family_surface(
    cls: type["KgConnectorReaderBase"],
    family_properties: dict[str, Any],
    family_params: dict[str, Any],
) -> None:
    """Compose ``PROPERTY_MAPPING``/``PARAMS_MAPPING`` for a family base from parent, mixin deltas, and family keys."""
    for layer in ("PROPERTY_MAPPING", "PARAMS_MAPPING"):
        if layer in cls.__dict__:
            raise ValueError(
                f"{cls.__name__} passes family_properties/family_params but also assigns {layer} "
                f"in its class body; declare the surface one way only."
            )
    property_deltas = [
        klass.__dict__["PROPERTY_MAPPING_DELTA"] for klass in cls.__mro__ if "PROPERTY_MAPPING_DELTA" in klass.__dict__
    ]
    cls.PROPERTY_MAPPING = compose_property_mapping(
        cls.PROPERTY_MAPPING, *property_deltas, family_properties, context=cls.__name__
    )
    params_mapping = getattr(cls, "PARAMS_MAPPING", None)
    if params_mapping is None:
        if family_params:
            raise ValueError(
                f"{cls.__name__} passes family_params but is not a ParamReader descendant; "
                f"query-flavored families have no PARAMS_MAPPING layer."
            )
        return
    params_deltas = [
        klass.__dict__["PARAMS_MAPPING_DELTA"] for klass in cls.__mro__ if "PARAMS_MAPPING_DELTA" in klass.__dict__
    ]
    if params_deltas or family_params:
        setattr(  # noqa: B010 -- PARAMS_MAPPING is declared on ParamReader, not this base
            cls,
            "PARAMS_MAPPING",
            compose_property_mapping(
                params_mapping, *params_deltas, family_params, context=f"{cls.__name__}.PARAMS_MAPPING"
            ),
        )
