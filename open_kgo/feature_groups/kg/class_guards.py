"""Class-definition-time guards for KG connector reader subclasses.

The import-time half of the validation machinery behind
``KgConnectorReaderBase``: ``__init_subclass__`` runs these four guards for
every family base and concrete plugin, so a contradictory declaration fails
the import instead of surfacing later as a confusing runtime error. Every
function takes the reader class being defined and reads its declarative
surface (``PROPERTY_MAPPING``, ``PARAMS_MAPPING``, ``SUPPORTED_VALUES``,
``REQUIRED_KEYS``, ``SOURCE_SLOT``, ``_WAIVED_UNCONSUMED_KEYS``). The base
class keeps thin delegating classmethods with the same names (single leading
underscore) so a subclass can still override an individual guard.

The runtime credential validation lives in the sibling ``kg.credentials``
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from mloda.provider import PropertySpec

from open_kgo.feature_groups.kg.credentials import spec_allowed_values
from open_kgo.feature_groups.kg.errors import NonDictSpecError

if TYPE_CHECKING:
    from open_kgo.feature_groups.kg.reader_base import KgConnectorReaderBase


def validate_mapping_spec_shapes(cls: type[KgConnectorReaderBase]) -> None:
    """Reject non-``PropertySpec`` spec values in ``PROPERTY_MAPPING`` / ``PARAMS_MAPPING`` at class-definition time.

    ``compose_property_mapping`` already enforces this for mappings built
    through the helper, but concretes that hand-assemble their mapping
    via a dict-comprehension off an already-composed parent (e.g.
    ``FileFixtureRestReader``, ``FileFixtureCitationReader``,
    ``InProcessTupleStoreReader``) bypass the compose-time check. Running
    the same guard once more here closes that hole so the rule is
    "any spec in any mapping must be a PropertySpec" regardless of how the
    mapping was assembled. Raises ``NonDictSpecError`` (the same typed
    error ``compose_property_mapping`` uses) so callers can catch both
    compose-time and class-definition-time bypasses with one handler.
    """
    for layer_name in ("PROPERTY_MAPPING", "PARAMS_MAPPING"):
        mapping = getattr(cls, layer_name, None)
        if not mapping:
            continue
        for key, spec in mapping.items():
            if not isinstance(spec, PropertySpec):
                raise NonDictSpecError(key, spec, context=f"{cls.__name__}.{layer_name}")


def validate_supported_values_invariant(cls: type[KgConnectorReaderBase]) -> None:
    """Raise ``ValueError`` at class-definition time on ill-formed ``SUPPORTED_VALUES``.

    Catches typos and out-of-set values that would otherwise silently lock
    a connector at runtime (every value rejected). For each narrowed key:

    - The key must be declared in ``PROPERTY_MAPPING`` or ``PARAMS_MAPPING``.
    - The spec must have ``strict_validation=True`` (narrowing a non-strict
      key is meaningless: the family already accepts anything).
    - The narrowed frozenset must be non-empty and a subset of the spec's
      allowed set.
    - Omission-bypass guard (property-layer keys only): when the narrowed
      set excludes the spec's non-None default, the key must appear in
      some ``REQUIRED_KEYS`` group. ``_validate_mapping`` only checks keys
      PRESENT in the slot, so without the requirement an omitted key would
      validate and the reader would run under a defaulted value it does
      not honor (e.g. serve a cursor-paginated page 1, then reject its own
      continuation token because the cross-layer guard defaults the style
      to ``"none"``). Per-call params are out of scope: there is no
      slot-omission concept on that layer (``build_params`` simply leaves
      absent keys out of the params dict).
    """
    if not cls.SUPPORTED_VALUES:
        return
    params_mapping: Mapping[str, Any] = getattr(cls, "PARAMS_MAPPING", {})
    for key, narrowed in cls.SUPPORTED_VALUES.items():
        spec = cls.PROPERTY_MAPPING.get(key) or params_mapping.get(key)
        if spec is None:
            raise ValueError(
                f"{cls.__name__}.SUPPORTED_VALUES[{key!r}] names a key not present in "
                f"PROPERTY_MAPPING or PARAMS_MAPPING."
            )
        if spec.strict_validation is not True:
            raise ValueError(
                f"{cls.__name__}.SUPPORTED_VALUES[{key!r}] requires the spec to set "
                f"strict_validation=True; narrowing a non-strict key is meaningless."
            )
        if not narrowed:
            raise ValueError(
                f"{cls.__name__}.SUPPORTED_VALUES[{key!r}] is empty; an empty narrowed "
                f"set rejects every value. Use ``del SUPPORTED_VALUES[key]`` and strip "
                f"the key from the mapping instead if the concrete cannot honor any value."
            )
        allowed = spec_allowed_values(key, spec)
        if not narrowed <= allowed:
            raise ValueError(
                f"{cls.__name__}.SUPPORTED_VALUES[{key!r}]={sorted(narrowed)} is not a "
                f"subset of the family-allowed set {sorted(allowed)}."
            )
        default = spec.default
        if (
            key in cls.PROPERTY_MAPPING
            and default is not None
            and default not in narrowed
            and not any(key in group for group in cls.REQUIRED_KEYS)
        ):
            raise ValueError(
                f"{cls.__name__}.SUPPORTED_VALUES[{key!r}]={sorted(narrowed)} excludes the "
                f"spec default {default!r} but {key!r} is missing from REQUIRED_KEYS. "
                f"_validate_mapping only checks keys present in the credential slot, so an "
                f"omitted {key!r} would bypass the narrowing and the reader would run under "
                f"a defaulted value it does not honor. Add ({key!r},) to REQUIRED_KEYS, or "
                f"widen the narrowed set to include the default."
            )


def validate_unconsumed_waivers(cls: type[KgConnectorReaderBase]) -> None:
    """Reject locally-declared ``_WAIVED_UNCONSUMED_KEYS`` entries that name no advertised key.

    Catches a typo or stale waiver (key later stripped) that would silently
    no-op. Only the class's own declaration (``cls.__dict__``) is checked
    against the resolved mappings: an inherited waiver naming a key a
    subclass strips is inert (the contract test iterates advertised keys
    only), not an error.
    """
    local = cls.__dict__.get("_WAIVED_UNCONSUMED_KEYS")
    if not local:
        return
    advertised = set(cls.PROPERTY_MAPPING)
    params_mapping: Mapping[str, Any] = getattr(cls, "PARAMS_MAPPING", {})
    advertised |= set(params_mapping)
    unknown = set(local) - advertised
    if unknown:
        raise ValueError(
            f"{cls.__name__}._WAIVED_UNCONSUMED_KEYS names key(s) not present in "
            f"PROPERTY_MAPPING or PARAMS_MAPPING: {sorted(unknown)}. Remove the stale "
            f"waiver, or fix the key name."
        )


def validate_source_slot(cls: type[KgConnectorReaderBase]) -> None:
    """Reject a ``SOURCE_SLOT`` declaration that contradicts ``PROPERTY_MAPPING`` at class-definition time.

    The enforcement half of the "Source-slot convention" (``reader_base``
    module docstring). Two contradictions are possible and both are rejected:

    - ``SOURCE_SLOT`` names a key the class does not advertise. This is
      either a typo, or the class renamed/dropped the address slot
      (``narrow_property_mapping(..., "locator")``) while inheriting the
      default declaration. Both mean the declaration lies about how a
      caller points the connector at its data.
    - ``SOURCE_SLOT is None`` (source baked into the reader) while
      ``locator`` is still advertised. A baked connector accepting a
      ``locator`` credential would be a surface lie (the honest-credential
      rule), so the declaration and the mapping must drop it together.

    A declared slot listed in ``_WAIVED_UNCONSUMED_KEYS`` (anywhere in the
    MRO) is also rejected: waiving the source slot as unconsumed means the
    reader does not actually read it, so the declaration would lie while
    some other key serves as the de-facto address. That is the one
    concrete way a fourth spelling could otherwise creep in behind a green
    gate. A non-``None``, non-``str`` value is a type error caught here
    rather than later in a confusing membership check.
    """
    slot_name = cls.SOURCE_SLOT
    if slot_name is None:
        if "locator" in cls.PROPERTY_MAPPING:
            raise ValueError(
                f"{cls.__name__}.SOURCE_SLOT is None (source baked into the reader) but 'locator' "
                f"is still advertised in PROPERTY_MAPPING. Drop 'locator' via "
                f"narrow_property_mapping, or declare SOURCE_SLOT = 'locator'."
            )
        return
    if not isinstance(slot_name, str):
        raise ValueError(
            f"{cls.__name__}.SOURCE_SLOT must be a str credential key or None, "
            f"got {type(slot_name).__name__} ({slot_name!r})."
        )
    if slot_name not in cls.PROPERTY_MAPPING:
        raise ValueError(
            f"{cls.__name__}.SOURCE_SLOT={slot_name!r} is not a key in PROPERTY_MAPPING. "
            f"Every connector must declare how a caller points it at its data: keep the "
            f"declared slot in PROPERTY_MAPPING, override SOURCE_SLOT to the renamed key, "
            f"or declare SOURCE_SLOT = None if the source is baked into the reader."
        )
    waived = {key for klass in cls.__mro__ for key in klass.__dict__.get("_WAIVED_UNCONSUMED_KEYS", ())}
    if slot_name in waived:
        raise ValueError(
            f"{cls.__name__}.SOURCE_SLOT={slot_name!r} is waived as unconsumed in "
            f"_WAIVED_UNCONSUMED_KEYS: the reader does not read its own declared source slot, "
            f"so the declaration lies about how a caller points it at its data. Override "
            f"SOURCE_SLOT to the key actually consumed, or declare SOURCE_SLOT = None if the "
            f"source is baked into the reader."
        )
