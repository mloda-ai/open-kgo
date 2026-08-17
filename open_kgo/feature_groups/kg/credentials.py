"""Credential slot extraction and shape validation for KG connector readers.

The runtime half of the validation machinery behind ``KgConnectorReaderBase``.
Every function takes the reader class as its first argument and reads the
declarative surface from it (``CONNECTOR_ID``, ``REQUIRED_KEYS``,
``CONDITIONAL_REQUIRED_KEYS``, ``PROPERTY_MAPPING``, ``SUPPORTED_VALUES``).
The base class keeps thin delegating classmethods with the same names (single
leading underscore), so subclasses and tests keep their call surface and a
subclass can still override an individual step; this module owns the bodies
so ``reader_base`` stays focused on the reader lifecycle.

The class-definition-time guards (mapping shapes, ``SUPPORTED_VALUES``
invariants, waiver hygiene, source-slot convention) live in the sibling
``kg.class_guards`` module.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from mloda.provider import HashableDict, PropertySpec

from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingEnvVarError,
    MissingRequiredKeysError,
)
from open_kgo.feature_groups.kg.validation import parse_bounded_int

if TYPE_CHECKING:
    from open_kgo.feature_groups.kg.reader_base import KgConnectorReaderBase


def extract_slot(cls: type[KgConnectorReaderBase], credentials: Any) -> dict[str, Any] | None:
    """Return the dict at credentials[CONNECTOR_ID], or None if absent.

    A slot value of ``None`` is treated as opt-out (absent). Any other
    non-dict value (e.g. a bare string path like ``"/data/x.ttl"``) is a
    misuse: the slot key is present but malformed, which would otherwise
    be indistinguishable from "this connector's slot is absent" and
    silently mismatch. Raise ``InvalidCredentialShape`` so the typo
    surfaces loudly.
    """
    _ABSENT = object()
    slot: Any = _ABSENT
    if isinstance(credentials, HashableDict):
        slot = credentials.data.get(cls.CONNECTOR_ID, _ABSENT)
    elif isinstance(credentials, dict):
        slot = credentials.get(cls.CONNECTOR_ID, _ABSENT)

    if slot is _ABSENT or slot is None:
        return None
    if isinstance(slot, HashableDict):
        return dict(slot.data)
    if isinstance(slot, dict):
        return dict(slot)
    raise InvalidCredentialShape(
        f"{cls.CONNECTOR_ID}: credential slot must be a dict mapping property names to values, "
        f"got {type(slot).__name__} ({slot!r})."
    )


def wrap_credentials(cls: type[KgConnectorReaderBase], data_access: Any) -> HashableDict:
    """Normalise the data_access mloda hands us into a HashableDict({CONNECTOR_ID: dict}).

    mloda's BaseInputData passes the matched data_access through. Concrete
    plugins receive either the full credentials dict (with our slot inside)
    or just our slot. This helper unifies both shapes so concrete code can
    always call ``cls._extract_slot(cls._wrap_credentials(data_access))``.

    ``data_access=None`` raises ``NotImplementedError``: mloda's
    scoped-access discovery probes ``load_data(None, None)`` and expects
    that error class (not ``TypeError``) to mean "this reader needs real
    credentials". A real caller passing ``None`` by mistake also lands here.
    """
    if data_access is None:
        raise NotImplementedError(
            f"{cls.__name__}.load_data requires a credentials dict; received None. "
            "mloda's scoped-access discovery probe also reaches this path."
        )
    if isinstance(data_access, HashableDict):
        if cls.CONNECTOR_ID in data_access.data:
            return data_access
        return HashableDict({cls.CONNECTOR_ID: dict(data_access.data)})
    if isinstance(data_access, dict):
        if cls.CONNECTOR_ID in data_access:
            return HashableDict(dict(data_access))
        return HashableDict({cls.CONNECTOR_ID: dict(data_access)})
    raise TypeError(f"data_access must be a dict or HashableDict, got {type(data_access).__name__}")


def validate_result_limit(cls: type[KgConnectorReaderBase], creds: dict[str, Any]) -> None:
    """Reject ``result_limit`` values that aren't positive ints.

    ``result_limit`` is universal (in the base ``PROPERTY_MAPPING``) and
    the spec defaults to 1000, so the key only reaches this check when the
    caller set it. Bool is rejected explicitly: it is an ``int`` subclass
    in Python, but a row cap of ``True`` or ``False`` is almost always a
    caller mistake. Strings, floats, and negative integers fail likewise.
    Delegates to ``parse_bounded_int`` with no default: the key is only
    checked when present, and a present-but-``None`` value is rejected
    like any other non-int.
    """
    if "result_limit" not in creds:
        return
    parse_bounded_int(cls.CONNECTOR_ID, "result_limit", creds["result_limit"], min_value=1)


def validate_mapping(
    cls: type[KgConnectorReaderBase],
    values: dict[str, Any],
    mapping: dict[str, Any],
    *,
    kind: str,
    closed_world: bool,
) -> None:
    """Shared shape + strict-enum validation loop.

    Used by ``_validate_shape`` (closed-world over PROPERTY_MAPPING) and
    ``_validate_params`` (open-world over PARAMS_MAPPING; params share
    ``feature.options.context`` with mloda core and other plugins, so
    unknown keys must pass through). The ``kind`` label appears in the
    error message ("credential key" vs "params") so the source of the
    bad key is obvious in diagnostics.
    """
    for key, value in values.items():
        spec = mapping.get(key)
        if spec is None:
            if closed_world:
                raise InvalidCredentialShape(
                    f"{cls.CONNECTOR_ID}: unknown {kind} {key!r}; allowed: {sorted(mapping.keys())}"
                )
            continue
        if spec.strict_validation is True:
            narrowed = cls.SUPPORTED_VALUES.get(key)
            if narrowed is not None:
                if value not in narrowed:
                    raise InvalidCredentialShape(
                        f"{cls.CONNECTOR_ID}.{key}={value!r} is not supported by this connector "
                        f"(supported: {sorted(narrowed)})"
                    )
            else:
                allowed = spec_allowed_values(key, spec)
                if value not in allowed:
                    raise InvalidCredentialShape(
                        f"{cls.CONNECTOR_ID}: {kind} {key!r}={value!r} is not in allowed set {sorted(allowed)}"
                    )


def spec_allowed_values(key: str, spec: PropertySpec) -> set[Any]:
    """Return the explicit ``allowed_values`` set declared on a strict-validation spec.

    Strict-validation specs must declare their value space explicitly via
    the core ``PropertySpec.allowed_values`` field (a dict mapping value
    to its docstring, or any iterable of values). Deriving the set from the
    spec's plain string keys would silently expand the allowed set whenever a
    future doc-only key like ``"see_also"`` is added; the explicit field
    separates docs from validation data.
    """
    raw = spec.allowed_values
    if raw is None:
        raise InvalidCredentialShape(
            f"spec for {key!r} declares strict_validation=True but is missing 'allowed_values'."
        )
    if isinstance(raw, dict):
        return set(raw.keys())
    return set(raw)


def validate_required_keys(cls: type[KgConnectorReaderBase], creds: dict[str, Any]) -> None:
    """Enforce ``REQUIRED_KEYS``: each OR-group must have a present member.

    Presence is tested with ``is not None`` rather than truthiness so a
    legitimately falsey credential value (``0``, ``""``, ``False``) is not
    misread as absent, matching the ``REQUIRED_PARAMS`` presence
    convention (``_validate_required_params``) and the ``kg_contract``
    presence rule (``key in ... and value is not None``).
    """
    unsatisfied: list[tuple[str, ...]] = []
    for group in cls.REQUIRED_KEYS:
        if not group:
            raise InvalidCredentialShape(f"{cls.CONNECTOR_ID}: REQUIRED_KEYS contains an empty group; misconfigured.")
        if not any(creds.get(k) is not None for k in group):
            unsatisfied.append(group)
    if unsatisfied:
        raise MissingRequiredKeysError(cls.CONNECTOR_ID, tuple(unsatisfied))


def validate_conditional_required_keys(cls: type[KgConnectorReaderBase], creds: dict[str, Any]) -> None:
    """Enforce ``CONDITIONAL_REQUIRED_KEYS``: rules triggered by sibling values.

    Each rule is ``(prop, value, OR-groups)``. If ``creds.get(prop)`` equals
    ``value``, every OR-group must have at least one present (non-``None``)
    member in ``creds`` (same presence convention as
    ``validate_required_keys``). Aggregates all unsatisfied groups across
    all triggered rules into a single ``MissingRequiredKeysError`` so the
    caller sees the full picture in one error message.
    """
    unsatisfied: list[tuple[str, ...]] = []
    for prop, trigger_value, groups in cls.CONDITIONAL_REQUIRED_KEYS:
        if creds.get(prop) != trigger_value:
            continue
        for group in groups:
            if not group:
                raise InvalidCredentialShape(
                    f"{cls.CONNECTOR_ID}: CONDITIONAL_REQUIRED_KEYS for "
                    f"{prop}={trigger_value!r} contains an empty group; misconfigured."
                )
            if not any(creds.get(k) is not None for k in group):
                unsatisfied.append(group)
    if unsatisfied:
        raise MissingRequiredKeysError(cls.CONNECTOR_ID, tuple(unsatisfied))


def resolve_env(cls: type[KgConnectorReaderBase], creds: dict[str, Any], key: str) -> str | None:
    """Read an env-var NAME from creds[key], return the stripped env-var value.

    Opt-in helper for concretes that consume credentials from an env var
    (a bearer token, a username/password pair, etc.). The universal base
    does NOT call this hook: no shipped concrete authenticated against a
    network, so a universally-required
    env-var surface would be a contract the framework could not enforce.
    Concretes that introduce a real auth surface declare the matching
    ``auth_*_env`` keys themselves (on a family base or the concrete) and
    call ``_resolve_env`` from their own ``_connect_from_slot``.

    Returns None if creds[key] itself is unset (caller is opting out, e.g.
    an absent ``auth_token_env`` for a method that does not need one).
    Raises MissingEnvVarError if creds[key] names an env var that is not
    set in the environment, or if it is set to a value whose ``strip()``
    is empty (i.e. any value with no non-whitespace character: ``""``,
    ``"   "``, ``"\t"``, ``"\n"``, or any mix). Downstream auth would
    otherwise fail opaquely with no diagnostic on such values.

    The contract is "value must contain at least one
    non-whitespace character." The returned value is ``value.strip()`` so
    stray surrounding whitespace (a common ``.env``/copy-paste artifact)
    does not leak through to downstream auth either: if the rejection
    rationale is "whitespace breaks downstream tokens," partial-whitespace
    tokens deserve the same treatment as fully-whitespace ones.
    """
    env_name = creds.get(key)
    if env_name is None:
        return None
    if not isinstance(env_name, str):
        raise InvalidCredentialShape(
            f"{cls.CONNECTOR_ID}.{key} must be a str env-var name, got {type(env_name).__name__}"
        )
    value = os.environ.get(env_name)
    if value is None:
        raise MissingEnvVarError(env_name, key)
    stripped = value.strip()
    if not stripped:
        raise MissingEnvVarError(env_name, key)
    return stripped
