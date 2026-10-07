"""Runtime credential slot extraction and shape validation, mixed into ``KgConnectorReaderBase``.

The class-definition-time guards live in the sibling ``kg.class_guards`` module.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from typing import Any, ClassVar

from mloda.provider import PropertySpec

from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingEnvVarError,
    MissingRequiredKeysError,
)
from open_kgo.feature_groups.kg.validation import parse_bounded_int



def _is_hashable(value: Any) -> bool:
    """Return whether ``value`` can be tested for membership in a set.

    Strict enums are stored as sets/frozensets. An unhashable value (a list,
    dict, or set) cannot be a member; ``value in allowed`` would raise
    ``TypeError`` instead of the typed credential error.
    """
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _is_member(value: Any, allowed: Collection[Any]) -> bool:
    """Return whether ``value`` is in ``allowed``, treating unhashable values as absent."""
    if not _is_hashable(value):
        return False
    return value in allowed


def _value_for_message(value: Any) -> str:
    """Render a rejected enum value without echoing unhashable payloads.

    Hashable mismatches keep the existing ``repr`` so diagnostics stay stable.
    Unhashable values may be large or incidental; the type name is enough.
    """
    if _is_hashable(value):
        return repr(value)
    return f"<{type(value).__name__}>"


class CredentialRules:
    """Credential rules read from the reader's declarative surface; values are set on ``KgConnectorReaderBase``."""

    CONNECTOR_ID: ClassVar[str]
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]]
    PROPERTY_MAPPING: ClassVar[dict[str, Any]]
    SUPPORTED_VALUES: ClassVar[Mapping[str, frozenset[Any]]]

    @classmethod
    def _extract_slot(cls, credentials: Any) -> dict[str, Any] | None:
        """Return the dict at credentials[CONNECTOR_ID], or None if absent.

        A slot value of ``None`` is treated as opt-out (absent). Any other
        non-dict value (e.g. a bare string path like ``"/data/x.ttl"``) is a
        misuse: the slot key is present but malformed, which would otherwise
        be indistinguishable from "this connector's slot is absent" and
        silently mismatch. Raise ``InvalidCredentialShape`` so the typo
        surfaces loudly. mloda hands readers dicts only (``Credential`` and a
        pinned slot become a redacting dict subclass), so a non-dict
        ``credentials`` is not ours.
        """
        if not isinstance(credentials, dict):
            return None
        slot = credentials.get(cls.CONNECTOR_ID)
        if slot is None:
            return None
        if isinstance(slot, dict):
            return dict(slot)
        # Type name only: the value may carry a secret.
        raise InvalidCredentialShape(
            f"{cls.CONNECTOR_ID}: credential slot must be a plain dict mapping property names to values, "
            f"got {type(slot).__name__}. Register it as "
            f"DataAccessCollection(credentials=Credential({{{cls.CONNECTOR_ID!r}: {{...}}}}))."
        )

    @classmethod
    def _require_slot(cls, credentials: Any) -> dict[str, Any]:
        """Extract the credential slot or raise. ``connect()`` and ``_prepare_load`` both call this.

        By the time ``_connect_from_slot`` runs, the slot has been
        shape-validated by one of two upstream gates: the matcher path
        validates via ``is_valid_credentials`` (matcher-safe ``False`` on
        error), and the direct-call path validates via ``connect()``
        (loud ``InvalidCredentialShape`` / ``MissingRequiredKeysError`` on
        error). This helper only unpacks the slot; concrete readers need not
        re-check for ``None``.

        ``_prepare_load`` calls this *without* running ``_validate_shape``
        because the matcher already validated before dispatch; a direct call
        to ``load_data`` that bypasses both gates relies on the slot being
        well-formed, which is the documented contract for that direct path.
        """
        if not isinstance(credentials, dict):
            raise InvalidCredentialShape(
                f"{cls.CONNECTOR_ID}: credentials must be a plain dict {{{cls.CONNECTOR_ID!r}: {{...}}}}, "
                f"got {type(credentials).__name__}."
            )
        slot = cls._extract_slot(credentials)
        if slot is None:
            raise InvalidCredentialShape(f"{cls.CONNECTOR_ID}: credentials missing the {cls.CONNECTOR_ID!r} slot.")
        return slot

    @classmethod
    def _validate_shape(cls, creds: dict[str, Any]) -> None:
        """Validate a single connector's credential dict against PROPERTY_MAPPING.

        Order:
        1. ``REQUIRED_KEYS``: at least one key per OR-group must be present
           (not ``None``). Reported first so missing keys surface a clear
           "you forgot X" error.
        2. ``result_limit`` boundary check: must be a non-bool ``int >= 1``.
           Pinned at the credential surface so the cross-reader divergence in
           append-then-check vs slice-at-end behavior at ``result_limit ∈
           {0, -1, False, ...}`` ceases to matter; every reader sees a
           validated positive int by the time ``_prepare_load`` returns. Bool
           is rejected explicitly: ``True``/``False`` are int subclasses in
           Python, but a row count expressed as a truth-value almost always
           reflects a caller mistake.
        3. Closed-world key check + strict-validation enums via
           ``_validate_mapping`` (the latter consults
           ``SUPPORTED_VALUES`` for per-concrete narrowing, falling back
           to the spec's ``allowed_values``).
        """
        cls._validate_required_keys(creds)
        cls._validate_result_limit(creds)
        cls._validate_mapping(creds, cls.PROPERTY_MAPPING, kind="credential key", closed_world=True)

    @classmethod
    def _validate_result_limit(cls, creds: dict[str, Any]) -> None:
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

    @classmethod
    def _validate_mapping(
        cls,
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
                    if not _is_member(value, narrowed):
                        raise InvalidCredentialShape(
                            f"{cls.CONNECTOR_ID}.{key}={_value_for_message(value)} is not supported by this connector "
                            f"(supported: {sorted(narrowed)})"
                        )
                else:
                    allowed = cls._spec_allowed_values(key, spec)
                    if not _is_member(value, allowed):
                        raise InvalidCredentialShape(
                            f"{cls.CONNECTOR_ID}: {kind} {key!r}={_value_for_message(value)} is not in allowed set {sorted(allowed)}"
                        )

    @staticmethod
    def _spec_allowed_values(key: str, spec: PropertySpec) -> set[Any]:
        """Return the explicit ``allowed_values`` set declared on a strict-validation spec.

        A dict form (value to docstring) contributes its keys. A strict spec whose
        value space is an ``element_validator`` has no ``allowed_values`` and is a
        shape error here, since the KG surface validates by membership only.
        """
        if spec.allowed_values is None:
            raise InvalidCredentialShape(
                f"spec for {key!r} declares strict_validation=True but is missing 'allowed_values'."
            )
        return set(spec.allowed_values)

    @classmethod
    def _validate_required_keys(cls, creds: dict[str, Any]) -> None:
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
                raise InvalidCredentialShape(
                    f"{cls.CONNECTOR_ID}: REQUIRED_KEYS contains an empty group; misconfigured."
                )
            if not any(creds.get(k) is not None for k in group):
                unsatisfied.append(group)
        if unsatisfied:
            raise MissingRequiredKeysError(cls.CONNECTOR_ID, tuple(unsatisfied))

    @classmethod
    def _resolve_env(cls, creds: dict[str, Any], key: str) -> str | None:
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
        ``"   "``, ``"\\t"``, ``"\\n"``, or any mix). Downstream auth would
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
