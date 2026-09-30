"""Credential-slot contract tests: matching, shape validation, required keys, env vars.

One of the four concern mixins aggregated by ``KgConnectorContractBase``
(see ``kg_contract.py``). Everything here exercises the credential layer:
``is_valid_credentials`` matcher behavior, ``_validate_shape`` loud-failure
behavior, ``REQUIRED_KEYS`` enforcement, and ``_resolve_env``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.steward import resolve_feature
from mloda.user import Credential, DataAccessCollection

from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingEnvVarError,
    MissingRequiredKeysError,
)
from open_kgo.feature_groups.kg.tests._discovery import iter_strict_specs
from open_kgo.feature_groups.kg.tests._helpers import bogus_value_for_strict_spec
from open_kgo.feature_groups.kg.tests.contract_adapters import KgContractAdapterBase


class CredentialContract(KgContractAdapterBase):
    """Contract tests for the credential-slot surface of a concrete KG plugin."""

    def assert_strict_enum_value_rejected(self, key: str, invalid_value: Any) -> None:
        """Shared body for the per-family "invalid <enum key> rejected" tests.

        Builds a valid slot, plants ``invalid_value`` under ``key``, and
        asserts the matcher rejects it (``False``) or raises
        ``InvalidCredentialShape``. Five family contract bases used to spell
        this 10-line block verbatim per enum key; they now declare one-line
        tests over this helper, so a new strict-enum key costs one line and
        cannot drift from the canonical rejection shape.
        """
        cls = self.connector_reader_class()
        slot = dict(next(iter(self.valid_credentials().values())))
        slot[key] = invalid_value
        creds = {cls.CONNECTOR_ID: slot}
        try:
            ok = cls.is_valid_credentials(creds)
        except InvalidCredentialShape:
            return
        assert ok is False, f"{cls.__name__}.is_valid_credentials accepted {key}={invalid_value!r}"

    def assert_remote_locator_rejected(self, url: str = "http://example.invalid/remote.fixture") -> None:
        """Shared body for the per-family "http/https locator rejected" tests.

        File-backed connectors must refuse remote schemes at ``connect()``
        time (no network IO at fetch time; PR #7's rdflib guard, since
        centralised in ``kg.fixtures``). A copy-pasted URL then surfaces as a
        typed error naming the scheme instead of a confusing
        ``FileNotFoundError`` against the URL-as-relative-path. Six concrete
        test classes used to carry this body; the file-backed family contract
        bases now declare it once and every current and future concrete in
        those families inherits the check. The valid slot is used as the
        starting point so ``REQUIRED_KEYS`` validation passes and the scheme
        guard is the check that actually fires.
        """
        cls = self.connector_reader_class()
        slot = dict(next(iter(self.valid_credentials().values())))
        assert "locator" in slot, f"{cls.__name__}: assert_remote_locator_rejected needs a locator-backed slot."
        slot["locator"] = url
        with pytest.raises(ValueError, match="scheme"):
            cls.connect({cls.CONNECTOR_ID: slot})

    def test_credentials_match_connector_id(self) -> None:
        """is_valid_credentials returns True when CONNECTOR_ID slot is present and valid."""
        creds = self.valid_credentials()
        assert self.connector_reader_class().is_valid_credentials(creds) is True

    def test_empty_credentials_do_not_match(self) -> None:
        """is_valid_credentials returns False when credential_dicts is empty."""
        empty: dict[str, Any] = {}
        assert self.connector_reader_class().is_valid_credentials(empty) is False

    def test_other_connector_id_does_not_match(self) -> None:
        """is_valid_credentials returns False when only an unrelated connector slot is present."""
        unrelated = {"some_other_connector_xyz": {"locator": "irrelevant"}}
        assert self.connector_reader_class().is_valid_credentials(unrelated) is False

    def test_invalid_credentials_rejected(self) -> None:
        """invalid_credentials() should be rejected (False) or raise InvalidCredentialShape."""
        creds = self.invalid_credentials()
        try:
            result = self.connector_reader_class().is_valid_credentials(creds)
        except InvalidCredentialShape:
            return
        assert result is False, (
            f"{self.connector_reader_class().__name__}.is_valid_credentials accepted a dict that should fail: "
            f"{self.invalid_credentials()}"
        )

    def test_strict_validation_enums_rejected_per_key(self) -> None:
        """Auto-parametrised: every ``strict_validation=True`` PROPERTY_MAPPING key on this
        concrete must reject a value outside its effective allowed set.

        The effective allowed set is ``SUPPORTED_VALUES[key]`` if the concrete
        narrows the key, else the family-base allowed values from the spec.
        Coverage is uniform across families: when a family adds a strict enum,
        every concrete inherits rejection coverage for free, and there is no
        per-concrete drift when ``invalid_credentials`` happens to probe a
        different key.

        ``is_valid_credentials`` is matcher-safe (catches and returns False);
        the loud entry point is ``_validate_shape``, which is what we exercise
        here so a silent acceptance surfaces as a typed error instead of a
        rejection that looks like "no plugin matched".

        Scoped to ``PROPERTY_MAPPING`` only: fanning ``PARAMS_MAPPING`` keys
        through ``_validate_shape`` would surface as the closed-world
        "unknown credential key" rejection rather than the surface lie this
        test exists to catch. The ``PARAMS_MAPPING`` equivalent lives in the
        sibling ``test_strict_validation_params_enums_rejected_per_key``,
        which dispatches through ``_validate_params`` per-layer.
        """
        cls = self.connector_reader_class()
        base_slot = dict(next(iter(self.valid_credentials().values())))

        accepted: list[str] = []
        for key, spec, layer_name in iter_strict_specs(cls):
            if layer_name != "PROPERTY_MAPPING":
                continue
            # ``bogus_value_for_strict_spec(spec)`` is guaranteed to be outside
            # the family-allowed set. ``SUPPORTED_VALUES`` is a subset of that
            # set (enforced at class definition time by
            # ``_validate_supported_values_invariant``), so the bogus value is
            # also outside any narrowed set — no per-call retry against
            # ``effective_allowed`` needed.
            slot = dict(base_slot)
            slot[key] = bogus_value_for_strict_spec(spec)
            try:
                cls._validate_shape(slot)
            except InvalidCredentialShape:
                continue
            accepted.append(key)
        if accepted:
            raise AssertionError(
                f"{cls.__name__}: strict_validation enums silently accepted bogus values for keys: {accepted}"
            )

    def test_result_limit_boundary_behavior(self) -> None:
        """``_validate_shape`` AND ``_prepare_load`` reject every ``result_limit`` outside ``int >= 1``.

        Pins the policy once at the credential surface so the cross-reader
        divergence in append-then-check vs slice-at-end behavior at
        ``result_limit ∈ {0, -1, False, ...}`` ceases to matter. Bool values
        are rejected explicitly: ``True``/``False`` are int subclasses in
        Python, but a row cap of ``True`` is almost always a caller mistake.

        Both validation paths are exercised universally: ``_validate_shape``
        for matcher-path callers (``is_valid_credentials``), and
        ``_prepare_load`` for direct ``connect``/``load_data`` callers that
        bypass the matcher. Defense-in-depth at the credential surface only
        matters if both paths actually reject; pinning both here means a
        future refactor that removes either call surfaces here.

        Mixed inside the contract test so a future regression that relaxes
        the check fails universally rather than at one arbitrary concrete.
        ``pytest.mark.parametrize`` is not used here because the surrounding
        contract tests collect failures rather than decorate (``cls`` is not
        resolvable at decoration time on this abstract base).
        """
        cls = self.connector_reader_class()
        canonical = dict(next(iter(self.valid_credentials().values())))
        rejected_values: list[Any] = [0, -1, False, True, "abc", 1.5, 10.0]
        for value in rejected_values:
            slot = dict(canonical)
            slot["result_limit"] = value
            with pytest.raises(InvalidCredentialShape):
                cls._validate_shape(slot)
            assert cls.is_valid_credentials({cls.CONNECTOR_ID: slot}) is False
            # _prepare_load bypasses is_valid_credentials but must also reject.
            with pytest.raises(InvalidCredentialShape):
                cls._prepare_load({cls.CONNECTOR_ID: slot})
        # Positive ints (including very large values) pass validation.
        for value in (1, 100, 10**9):
            slot = dict(canonical)
            slot["result_limit"] = value
            cls._validate_shape(slot)
            assert cls.is_valid_credentials({cls.CONNECTOR_ID: slot}) is True

    def test_result_limit_validation_order(self) -> None:
        """``_validate_required_keys`` fires before ``_validate_result_limit``.

        Pins the documented order in ``_validate_shape``: a slot missing a
        required key AND carrying a bad ``result_limit`` surfaces
        ``MissingRequiredKeysError`` first, not the result-limit error. A
        future re-shuffle that flips this order silently changes which
        typed error callers see, which has the kind of "it's still typed
        so the test still passes" failure mode that's worth pinning.

        Skipped for concretes with empty ``REQUIRED_KEYS`` (no required key
        to drop).
        """
        cls = self.connector_reader_class()
        if not cls.REQUIRED_KEYS:
            pytest.skip(f"{cls.__name__} has empty REQUIRED_KEYS; ordering check does not apply.")
        canonical = dict(next(iter(self.valid_credentials().values())))
        slot = dict(canonical)
        for k in cls.REQUIRED_KEYS[0]:
            slot.pop(k, None)
        slot["result_limit"] = 0
        with pytest.raises(MissingRequiredKeysError):
            cls._validate_shape(slot)

    def test_unknown_credential_key_rejected(self) -> None:
        """Closed-world: a key not in PROPERTY_MAPPING is rejected.

        ``is_valid_credentials`` is matcher-safe (returns False on any shape
        error so mloda's matcher loop can keep iterating other readers).
        ``_validate_shape`` is the loud-failure entry point and raises
        ``InvalidCredentialShape``.
        """
        cls = self.connector_reader_class()
        slot = dict(next(iter(self.valid_credentials().values())))
        slot["definitely_not_a_kg_key_xyz"] = "x"
        creds = {cls.CONNECTOR_ID: slot}
        assert cls.is_valid_credentials(creds) is False, f"{cls.__name__}.is_valid_credentials accepted an unknown key."
        with pytest.raises(InvalidCredentialShape):
            cls._validate_shape(slot)

    def test_malformed_slot_rejected(self) -> None:
        """A slot value that isn't a plain dict must raise from ``_extract_slot``.

        ``is_valid_credentials`` is matcher-safe (catches and returns False so
        mloda's matcher loop can keep iterating). The loud-failure entry point
        for direct callers is ``_extract_slot`` itself: a slot key with a
        non-dict value (e.g. a bare path string) raises
        ``InvalidCredentialShape`` so the typo surfaces instead of silently
        masquerading as "no plugin matched".

        On the ``run_all`` path, the decline is recorded as a match rejection naming the reader and
        error type, never the slot values; absent and foreign slots record nothing.
        """
        cls = self.connector_reader_class()
        creds = {cls.CONNECTOR_ID: "this-should-be-a-dict-but-isnt"}
        assert cls.is_valid_credentials(creds) is False, (
            f"{cls.__name__}.is_valid_credentials should swallow the malformed-slot error and return False."
        )
        with pytest.raises(InvalidCredentialShape):
            cls._extract_slot(creds)
        # Credential belongs in DataAccessCollection; a direct call takes the plain dict.
        with pytest.raises(InvalidCredentialShape, match="got Credential"):
            cls.connect(Credential(self.valid_credentials()))

        secret = "s3cr3t-slot-value"
        reason = f"{cls.__name__}: the {cls.CONNECTOR_ID!r} credential slot is present but invalid"
        bad_limit = {**next(iter(self.valid_credentials().values())), "result_limit": secret}
        for slot in (secret, bad_limit):
            error = self._resolution_error({cls.CONNECTOR_ID: slot})
            assert reason in error and secret not in error, f"{cls.__name__}: {error}"
        assert "(InvalidCredentialShape)" in self._resolution_error({cls.CONNECTOR_ID: secret})
        for silent in ({}, {"some_other_connector_xyz": secret}):
            assert reason not in self._resolution_error(silent), f"{cls.__name__}: {silent} recorded a rejection."

    def _resolution_error(self, credentials: dict[str, Any]) -> str:
        """Return the resolution error text for ``feature_under_test`` under ``credentials``."""
        feature = self.feature_under_test()
        dac = DataAccessCollection(credentials=Credential(credentials)) if credentials else None
        return resolve_feature(feature.name, options=feature.options, data_access_collection=dac).error or ""

    def test_is_valid_credentials_is_matcher_safe_against_misbehaving_mapping(self) -> None:
        """``is_valid_credentials`` must not propagate non-``InvalidCredentialShape`` exceptions.

        mloda's ``ReadDB.match_read_db_data_access`` only catches
        ``NotImplementedError`` from the matcher loop. Any other propagating
        exception from a misbehaving credentials object aborts iteration over
        unrelated readers sharing the same ``DataAccessCollection``.
        A dict subclass whose ``.get`` raises ``RuntimeError`` pins the broad
        ``except Exception`` guard so a future narrowing surfaces immediately.
        """

        class _MisbehavingDict(dict[str, Any]):
            def get(self, key: Any, default: Any = None) -> Any:
                raise RuntimeError("synthetic probe failure to prove matcher-safety")

        cls = self.connector_reader_class()
        assert cls.is_valid_credentials(_MisbehavingDict()) is False, (
            f"{cls.__name__}.is_valid_credentials must swallow probe-time exceptions "
            f"raised by a misbehaving Mapping and return False (matcher-safety)."
        )

    def test_env_var_resolution_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_resolve_env returns the env var value when set."""
        monkeypatch.setenv("KG_CONTRACT_TEST_TOKEN", "abc123")
        cls = self.connector_reader_class()
        creds: dict[str, Any] = {"auth_token_env": "KG_CONTRACT_TEST_TOKEN"}
        assert cls._resolve_env(creds, "auth_token_env") == "abc123"

    def test_env_var_resolution_typed_error_on_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_resolve_env raises MissingEnvVarError when the env var name is set but the env var is unset."""
        monkeypatch.delenv("KG_CONTRACT_TEST_MISSING", raising=False)
        cls = self.connector_reader_class()
        creds: dict[str, Any] = {"auth_token_env": "KG_CONTRACT_TEST_MISSING"}
        with pytest.raises(MissingEnvVarError):
            cls._resolve_env(creds, "auth_token_env")

    def test_env_var_resolution_returns_none_when_unset(self) -> None:
        """_resolve_env returns None when the credential key itself is absent."""
        cls = self.connector_reader_class()
        assert cls._resolve_env({}, "auth_token_env") is None

    def test_required_keys_enforced(self) -> None:
        """If REQUIRED_KEYS is non-empty, dropping every OR-group's keys must reject.

        Iterates every OR-group, not just ``REQUIRED_KEYS[0]``: a concrete
        with multiple OR-groups (e.g. agent_memory's
        ``(("locator",), ("memory_scope_user_id",))``) would otherwise have
        only the first group exercised, and a regression that drops the
        second group from validation would not surface here.

        Empty ``REQUIRED_KEYS`` means there's no required-key rule; in that
        case the test verifies that a slot still validates. Otherwise:
        ``is_valid_credentials`` returns False (matcher-safe) and
        ``_validate_shape`` raises ``MissingRequiredKeysError``
        (loud-failure entry point). Tests do not attempt to construct a
        slot that satisfies the strict-validation enums, so they tolerate
        any ``InvalidCredentialShape`` subclass from downstream checks.
        """
        cls = self.connector_reader_class()
        canonical = dict(next(iter(self.valid_credentials().values())))
        if not cls.REQUIRED_KEYS:
            assert cls.is_valid_credentials({cls.CONNECTOR_ID: canonical}) is True
            return
        for group_idx, group in enumerate(cls.REQUIRED_KEYS):
            slot = dict(canonical)
            for k in group:
                slot.pop(k, None)
            creds = {cls.CONNECTOR_ID: slot}
            assert cls.is_valid_credentials(creds) is False, (
                f"{cls.__name__}: dropping REQUIRED_KEYS group {group_idx} ({group}) did not "
                f"flip is_valid_credentials to False."
            )
            with pytest.raises(MissingRequiredKeysError):
                cls._validate_shape(slot)

    def test_required_keys_each_alternative_is_coherent(self) -> None:
        """Every ``REQUIRED_KEYS`` alternative is coherent: validator True → ``connect()`` succeeds.

        For each OR-group, for each alternative key within the group: build
        a slot that satisfies that group via only that alternative (every
        other alt in the same group is dropped, other groups stay intact
        from ``valid_credentials()``). Assert ``is_valid_credentials`` is
        True AND ``connect()`` does not raise. This forces the design
        question that the original single-alternative test side-stepped:
        if the validator accepts an alternative the runtime can't honor
        (agent_memory's original motivating scenario), one of the two checks
        will diverge.

        The forward direction (validator True → connect ok) is what this
        test pins. The reverse direction (connect raises → validator
        False) is structurally outside the suite's reach today because
        ``connect()`` does not run ``_validate_shape`` itself; if that
        validation parity is ever added, this docstring should drop
        to a bi-conditional.

        ``valid_credentials()`` MUST supply every alternative in every
        OR-group; an absent alternative is itself a contract gap (either
        fixture the alternative or narrow ``REQUIRED_KEYS``). The test
        fails loudly rather than skipping so the gap surfaces in CI. The
        presence check uses ``key in ... and value is not None`` rather
        than a truthy test so a legitimately falsey alternative value
        (e.g. ``0``) would not be misread as absent.

        ``connect()`` may return a backend resource (e.g. a
        ``kuzu.Connection`` that holds an open file descriptor); the
        returned handle is best-effort closed in a ``finally`` so this
        test doesn't accrete leaks across the parametrised loop. The
        ``connect()`` call itself is inside the ``try`` so a partially
        initialised backend that opened sub-resources before raising
        still gets the closer hook invoked (no-op when the call raised
        before binding ``handle``).
        """
        cls = self.connector_reader_class()
        canonical = dict(next(iter(self.valid_credentials().values())))
        if not cls.REQUIRED_KEYS:
            assert cls.is_valid_credentials({cls.CONNECTOR_ID: canonical}) is True, (
                f"{cls.__name__}: REQUIRED_KEYS is empty but the canonical valid_credentials() slot "
                f"does not validate; one of the two is wrong."
            )
            return
        for group_idx, group in enumerate(cls.REQUIRED_KEYS):
            for alt in group:
                assert alt in canonical and canonical[alt] is not None, (
                    f"{cls.__name__}: REQUIRED_KEYS group {group_idx} lists alternative {alt!r} but "
                    f"valid_credentials() does not supply a non-None value for it. Either fixture {alt!r} "
                    f"or narrow REQUIRED_KEYS so the contract reflects what the concrete actually honors."
                )
                slot = dict(canonical)
                for other in group:
                    if other != alt:
                        slot.pop(other, None)
                creds = {cls.CONNECTOR_ID: slot}
                assert cls.is_valid_credentials(creds) is True, (
                    f"{cls.__name__}: slot using only REQUIRED_KEYS alt {alt!r} (group {group_idx}) "
                    f"failed is_valid_credentials."
                )
                handle: Any = None
                try:
                    handle = cls.connect(creds)
                finally:
                    closer = getattr(handle, "close", None)
                    if callable(closer):
                        closer()

    def test_connect_raises_typed_on_missing_required_keys(self) -> None:
        """``connect()`` validates shape (parity with ``is_valid_credentials``).

        Direct callers that bypass mloda's matcher (tests, demos, programmatic
        users) used to surface partial slots as downstream IO/key errors
        (``FileNotFoundError``, ``KeyError``) thrown from ``_connect_from_slot``.
        ``connect()`` now runs ``_validate_shape`` after extracting the slot,
        so a partial slot like ``{CONNECTOR_ID: {}}`` raises the typed
        ``MissingRequiredKeysError`` instead. Skips cleanly for readers
        declaring empty ``REQUIRED_KEYS`` (no required-key rule to enforce).
        """
        cls = self.connector_reader_class()
        if not cls.REQUIRED_KEYS:
            pytest.skip(f"{cls.__name__} declares no REQUIRED_KEYS; nothing to enforce.")
        creds: dict[str, Any] = {cls.CONNECTOR_ID: {}}
        with pytest.raises(MissingRequiredKeysError):
            cls.connect(creds)
