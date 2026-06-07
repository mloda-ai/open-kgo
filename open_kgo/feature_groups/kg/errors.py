"""Error types raised by the KG connector bases."""

from __future__ import annotations


class InvalidCredentialShape(ValueError):
    """Raised when a credential dict fails PROPERTY_MAPPING shape validation."""


class MissingRequiredKeysError(InvalidCredentialShape):
    """Raised when a connector's REQUIRED_KEYS rule is unsatisfied.

    ``REQUIRED_KEYS`` is a tuple of OR-groups; every group must have at least
    one present (non-``None``) key in the credential dict. Carries the
    unsatisfied group(s) for diagnostics.
    """

    def __init__(self, connector_id: str, unsatisfied_groups: tuple[tuple[str, ...], ...]) -> None:
        rendered = " AND ".join("(" + " OR ".join(g) + ")" for g in unsatisfied_groups)
        super().__init__(f"{connector_id}: missing required credential keys: {rendered}")
        self.connector_id = connector_id
        self.unsatisfied_groups = unsatisfied_groups


class MissingRequiredParamsError(InvalidCredentialShape):
    """Raised when a ParamReader's REQUIRED_PARAMS rule is unsatisfied.

    Mirror of ``MissingRequiredKeysError`` for per-call params: ``REQUIRED_PARAMS``
    is a tuple of OR-groups; every group must have at least one present
    (non-``None``) key in the params dict (assembled from
    ``feature.options.context``). Carries the
    unsatisfied group(s) so the caller can pinpoint which per-call inputs are
    missing rather than getting a generic ``InvalidCredentialShape`` with a
    hand-rolled message.

    Inherits from ``InvalidCredentialShape`` rather than a separate root so
    the error tree stays single-rooted; callers that want to scope a handler
    catch the leaf class explicitly.
    """

    def __init__(self, connector_id: str, unsatisfied_groups: tuple[tuple[str, ...], ...]) -> None:
        rendered = " AND ".join("(" + " OR ".join(g) + ")" for g in unsatisfied_groups)
        super().__init__(f"{connector_id}: missing required params: {rendered}")
        self.connector_id = connector_id
        self.unsatisfied_groups = unsatisfied_groups


class PropertyMappingCollision(InvalidCredentialShape):
    """Raised at composition time when two PROPERTY_MAPPING sources define the same key.

    Indicates a structural mistake in mixin/family-base composition: two sources
    contributed the same property key, so the merge would silently overwrite. The
    helper ``compose_property_mapping`` raises this so the collision is loud at
    import time rather than at runtime.
    """

    def __init__(self, key: str, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        super().__init__(f"{prefix}duplicate property key {key!r} across composed sources.")
        self.key = key
        self.context = context


class NonDictSpecError(InvalidCredentialShape):
    """Raised when a PROPERTY_MAPPING / PARAMS_MAPPING entry's spec value is not a dict.

    Sibling to ``PropertyMappingCollision``: both signal compose-time /
    class-definition-time misconfiguration of a property-mapping source, and
    both inherit from ``InvalidCredentialShape`` so callers can scope a
    single handler across the whole structural-error family. A non-dict
    spec (typically ``None`` from a stale stub) would otherwise propagate
    to ``_validate_mapping`` and surface as a self-contradicting "unknown
    key" error because ``mapping.get(key) is None`` cannot distinguish a
    missing key from a ``None`` spec.
    """

    def __init__(self, key: str, spec: object, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        super().__init__(
            f"{prefix}spec for key {key!r} must be a dict, got {type(spec).__name__} ({spec!r}). "
            f"Non-dict specs would surface downstream as self-contradicting 'unknown key' errors."
        )
        self.key = key
        self.spec = spec
        self.context = context


class MissingEnvVarError(RuntimeError):
    """Raised when a credential references an env var that is unset or empty.

    Concrete plugins that consume credentials from the environment carry env
    var *names* on the slot (e.g. an ``auth_token_env="MY_TOKEN"`` companion
    declared by a networked family base or by the concrete itself); the
    actual token is read from ``os.environ`` at connect time via
    ``KgConnectorReaderBase._resolve_env``. If the env var is unset
    (``os.environ.get`` returns ``None``) or set to a whitespace-only value,
    ``_resolve_env`` raises this error so the failure is loud rather than
    degrading into a silent downstream auth error. The universal base does not
    currently declare any auth surface; this error type exists as opt-in
    infrastructure for future networked concretes.
    """

    def __init__(self, env_var_name: str, credential_key: str) -> None:
        super().__init__(
            f"Credential key {credential_key!r} pointed to env var {env_var_name!r}, "
            f"but {env_var_name!r} is unset or empty in the environment."
        )
        self.env_var_name = env_var_name
        self.credential_key = credential_key


class UnknownTenantError(ValueError):
    """Raised by saas_authz connectors when ``tenant`` is not in the configured store.

    The closed-world ``REQUIRED_KEYS`` check enforces that ``tenant`` is
    *present*, but cannot enforce that the value is *known* to a given
    backend. ``connect()`` resolves the tenant against the live store and
    raises this typed error so callers can distinguish "tenant key missing"
    from "tenant value is not provisioned" without parsing a generic
    ``ValueError``.
    """

    def __init__(self, connector_id: str, tenant: str) -> None:
        super().__init__(f"{connector_id}: tenant {tenant!r} is not provisioned in the configured store.")
        self.connector_id = connector_id
        self.tenant = tenant


class UnknownMemoryScopeError(ValueError):
    """Raised by agent_memory connectors when ``memory_scope_user_id`` is not in the configured store.

    Symmetric to ``UnknownTenantError`` for the agent_memory family: the
    closed-world ``REQUIRED_KEYS`` check enforces that one of the
    ``memory_scope_*`` keys is *present*, but cannot enforce that the value
    is *provisioned* in a given backend. ``connect()`` resolves the user_id
    against the live store and raises this typed error so callers can
    distinguish "scope key missing" from "scope value not provisioned".
    """

    def __init__(self, connector_id: str, user_id: str) -> None:
        super().__init__(
            f"{connector_id}: memory_scope_user_id {user_id!r} is not provisioned in the configured store."
        )
        self.connector_id = connector_id
        self.user_id = user_id


class FixtureLoadError(InvalidCredentialShape):
    """Raised when a connector's ``locator`` cannot be loaded as a JSON fixture.

    Wraps the underlying ``OSError`` / ``json.JSONDecodeError`` / shape
    mismatch with a typed credential-shape error so callers chasing
    "validation errors at connect time" don't have to enumerate the ad-hoc
    IO/JSON/shape exceptions per concrete reader. Inherits from
    ``InvalidCredentialShape`` because a malformed locator is, semantically,
    a credential-shape problem (the slot points at something the connector
    can't use).
    """

    def __init__(self, connector_id: str, locator: str, reason: str) -> None:
        super().__init__(f"{connector_id}: cannot load fixture from locator {locator!r}: {reason}")
        self.connector_id = connector_id
        self.locator = locator
        self.reason = reason
