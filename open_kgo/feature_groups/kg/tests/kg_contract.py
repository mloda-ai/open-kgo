"""Shared abstract test base for KG connector contract suites.

Mirrors ``DataOpsTestBase`` from
``mloda-registry/mloda/testing/feature_groups/data_operations/base.py``: a
small set of abstract adapter methods that concrete plugin tests implement
once, plus a body of contract tests that those plugins inherit for free.

Per-family contract bases (e.g. ``RdfContractTestBase`` in
``rdf/tests/kg_rdf_contract.py``) extend this base with family-specific
assertions; concrete plugin tests subclass the per-family base and just wire
up the 5 adapters.

The cross-group contract suite (``test_cross_group_contract.py``) walks every
``KgConnectorContractBase`` subclass and runs the universal contract
assertions against each.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import HashableDict
from mloda.user import DataAccessCollection, Feature, Options

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase, ParamReader
from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingEnvVarError,
    MissingRequiredKeysError,
    MissingRequiredParamsError,
)
from open_kgo.feature_groups.kg.tests._discovery import iter_strict_specs
from open_kgo.feature_groups.kg.tests._helpers import bogus_value_for_strict_spec


class KgConnectorContractBase(ABC):
    """Abstract test base every concrete KG plugin's tests inherit from.

    Subclasses implement 5 adapter methods (or inherit them from a per-family
    intermediate base). Concrete contract tests are inherited for free.
    """

    @classmethod
    @abstractmethod
    def connector_reader_class(cls) -> type[KgConnectorReaderBase]:
        """Return the concrete ``KgConnectorReaderBase`` subclass under test."""

    @classmethod
    @abstractmethod
    def valid_credentials(cls) -> dict[str, Any]:
        """Return a credentials dict that should match this connector.

        Shape: ``{CONNECTOR_ID: {locator: ..., ...}}``. The outer key matches
        the reader's CONNECTOR_ID; the inner dict carries all per-family +
        universal properties this concrete honors.
        """

    @classmethod
    @abstractmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        """Return a credentials dict that should be rejected.

        Should violate at least one strict-validation enum (e.g. a
        ``lineage_direction="SIDEWAYS"`` for the dbt concrete, or any value
        outside the concrete's ``SUPPORTED_VALUES`` for a family-narrowed
        key), introduce an unknown closed-world key, or omit a required
        property. The earlier canonical seed was ``auth_method="evil"``;
        the auth surface was removed from the universal base, so concretes
        whose only family-level strict enum was
        ``auth_method`` now reach this contract via a closed-world
        unknown-key violation instead.
        """

    @classmethod
    @abstractmethod
    def feature_under_test(cls) -> Feature:
        """Return the canonical ``Feature`` instance for end-to-end load tests."""

    @classmethod
    @abstractmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        """Return a predicate that asserts the load result has the right shape."""

    # -- Universal contract tests (inherited by every concrete plugin) --------

    def test_credentials_match_connector_id(self) -> None:
        """is_valid_credentials returns True when CONNECTOR_ID slot is present and valid."""
        creds = HashableDict(self.valid_credentials())
        assert self.connector_reader_class().is_valid_credentials(creds) is True

    def test_empty_credentials_do_not_match(self) -> None:
        """is_valid_credentials returns False when credential_dicts is empty."""
        empty = HashableDict({})
        assert self.connector_reader_class().is_valid_credentials(empty) is False

    def test_other_connector_id_does_not_match(self) -> None:
        """is_valid_credentials returns False when only an unrelated connector slot is present."""
        unrelated = HashableDict({"some_other_connector_xyz": {"locator": "irrelevant"}})
        assert self.connector_reader_class().is_valid_credentials(unrelated) is False

    def test_invalid_credentials_rejected(self) -> None:
        """invalid_credentials() should be rejected (False) or raise InvalidCredentialShape."""
        creds = HashableDict(self.invalid_credentials())
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

    def test_strict_validation_params_enums_rejected_per_key(self) -> None:
        """Mirror of ``test_strict_validation_enums_rejected_per_key`` for ``PARAMS_MAPPING``.

        Walks every ``strict_validation=True`` key on ``PARAMS_MAPPING`` and
        asserts that ``_validate_params`` rejects a value outside its effective
        allowed set. The single hand-rolled dbt case in
        ``test_validation_contract.py`` is the seed; this generalises
        the assertion so every ParamReader concrete inherits coverage for free.

        Skips explicitly for ``QueryReader`` concretes (no ``PARAMS_MAPPING``).
        For ParamReader concretes whose ``PARAMS_MAPPING`` has no
        strict-enum keys the inner loop is empty and the test passes
        vacuously — symmetric with ``test_strict_validation_enums_rejected_per_key``
        on the credential layer. The baseline params dict is extracted from
        ``feature_under_test().options.context`` and filtered to declared
        ``PARAMS_MAPPING`` keys so ``REQUIRED_PARAMS`` stays satisfied and a
        silent-acceptance regression cannot masquerade as a
        ``MissingRequiredParamsError`` from the post-mapping check.
        """
        cls = self.connector_reader_class()
        if not issubclass(cls, ParamReader):
            pytest.skip(f"{cls.__name__} is not a ParamReader; PARAMS_MAPPING does not apply.")

        feat = self.feature_under_test()
        raw = feat.options.context
        full_ctx = dict(raw.data) if isinstance(raw, HashableDict) else dict(raw)
        base_params = {k: v for k, v in full_ctx.items() if k in cls.PARAMS_MAPPING}

        accepted: list[str] = []
        for key, spec, layer_name in iter_strict_specs(cls):
            if layer_name != "PARAMS_MAPPING":
                continue
            # Same ``bogus_value_for_strict_spec`` simplification as the
            # credential-layer sibling: the
            # bogus value is outside the family-allowed set, and
            # ``SUPPORTED_VALUES`` is a subset of that set (enforced at
            # class definition by ``_validate_supported_values_invariant``),
            # so the value is automatically outside any narrowing too.
            params = dict(base_params)
            params[key] = bogus_value_for_strict_spec(spec)
            try:
                cls._validate_params(params)
            except InvalidCredentialShape:
                continue
            accepted.append(key)
        if accepted:
            raise AssertionError(
                f"{cls.__name__}: strict_validation params enums silently accepted bogus values for keys: {accepted}"
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
            assert cls.is_valid_credentials(HashableDict({cls.CONNECTOR_ID: slot})) is False
            # _prepare_load bypasses is_valid_credentials but must also reject.
            with pytest.raises(InvalidCredentialShape):
                cls._prepare_load(HashableDict({cls.CONNECTOR_ID: slot}))
        # Positive ints (including very large values) pass validation.
        for value in (1, 100, 10**9):
            slot = dict(canonical)
            slot["result_limit"] = value
            cls._validate_shape(slot)
            assert cls.is_valid_credentials(HashableDict({cls.CONNECTOR_ID: slot})) is True

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

    def test_strict_enum_honored_or_waived(self) -> None:
        """Every strict-validation enum the concrete inherits is narrowed OR waived.

        Walks every strict-validation key across ``PROPERTY_MAPPING`` and
        ``PARAMS_MAPPING`` (the latter only on ``ParamReader`` concretes).
        For each key, asserts one of:

        - ``key in cls.SUPPORTED_VALUES`` — narrowed to the subset honored
          (mirroring the full family set is also acceptable; it makes
          "every advertised value is dispatched" explicit and surfaces
          future family-level additions as a contract failure here);
        - ``key in cls._WAIVED_ENUM_KEYS`` — deliberately forward-compat:
          the concrete accepts the family-advertised values without
          dispatch today because a future concrete in the same family
          (real backend) will honor them.

        A surface lie is a strict enum the validator accepts but the
        runtime ignores. The contract closes the gap: every strict enum
        must have an explicit disposition.

        ``_spec_allowed_values`` is only invoked when building the failure
        message: it can itself raise on a malformed spec, and we don't want
        an integrity error to mask the disposition gap this test exists to
        surface.
        """
        cls = self.connector_reader_class()
        lies: list[tuple[str, str, dict[str, Any]]] = []
        for key, spec, layer_name in iter_strict_specs(cls):
            if key in cls.SUPPORTED_VALUES:
                continue
            if key in cls._WAIVED_ENUM_KEYS:
                continue
            lies.append((layer_name, key, spec))
        if lies:
            rendered = [
                f"{layer}.{key} (allowed={sorted(cls._spec_allowed_values(key, spec))})" for layer, key, spec in lies
            ]
            raise AssertionError(
                f"{cls.__name__}: strict-validation enum(s) neither narrowed via SUPPORTED_VALUES nor "
                f"waived via _WAIVED_ENUM_KEYS: {rendered}. Either pin SUPPORTED_VALUES[key] to the "
                f"subset the runtime honors, or add the key to _WAIVED_ENUM_KEYS with a one-line waiver "
                f"comment on the concrete class."
            )

    def test_stripped_params_rejected_at_per_call(self) -> None:
        """Per-call surface is honest: family-declared params this concrete dropped must reject.

        Auto-applies to ParamReader concretes that narrow ``PARAMS_MAPPING``
        (``_STRIPPED_PARAMS`` non-empty). No-op for QueryReader concretes and
        ParamReader concretes that honor the full family contract. Closes the
        per-call counterpart of the credential-surface closed-world check: a
        family-declared key dropped by this concrete may not appear in
        ``feature.options.context`` even though the dict is shared with other
        plugins.
        """
        from open_kgo.feature_groups.kg.base import ParamReader

        cls = self.connector_reader_class()
        if not issubclass(cls, ParamReader):
            pytest.skip(f"{cls.__name__} is not a ParamReader concrete")
        if not cls._STRIPPED_PARAMS:
            pytest.skip(f"{cls.__name__} has no stripped per-call params")

        accepted: list[str] = []
        for stripped in sorted(cls._STRIPPED_PARAMS):
            feat = Feature(f"{cls.CONNECTOR_ID}__probe_{stripped}", options=Options(context={stripped: "x"}))
            fs = FeatureSet()
            fs.add(feat)
            try:
                cls._reject_stripped_params(fs)
            except InvalidCredentialShape:
                continue
            accepted.append(stripped)
        if accepted:
            raise AssertionError(
                f"{cls.__name__}: stripped per-call params silently accepted in feature.options.context: {accepted}"
            )

    def test_stripped_params_in_options_group_not_policed(self) -> None:
        """``_reject_stripped_params`` polices ``feature.options.context`` only, not ``group``.

        ``Options.group`` is mloda's feature-grouping concept; a stripped
        per-call key landing there is mloda's domain, not a KG surface lie.
        Auto-applies to every ParamReader concrete with stripped params, so a
        regression that re-broadens the rejection scope fails universally
        rather than at one arbitrary concrete.
        """
        from open_kgo.feature_groups.kg.base import ParamReader

        cls = self.connector_reader_class()
        if not issubclass(cls, ParamReader):
            pytest.skip(f"{cls.__name__} is not a ParamReader concrete")
        if not cls._STRIPPED_PARAMS:
            pytest.skip(f"{cls.__name__} has no stripped per-call params")

        for stripped in sorted(cls._STRIPPED_PARAMS):
            feat = Feature(
                f"{cls.CONNECTOR_ID}__group_probe_{stripped}",
                options=Options(group={stripped: "x"}),
            )
            fs = FeatureSet()
            fs.add(feat)
            # Must not raise: stripped key in group is out of scope for the per-call check.
            cls._reject_stripped_params(fs)

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
        creds = HashableDict({cls.CONNECTOR_ID: slot})
        assert cls.is_valid_credentials(creds) is False, f"{cls.__name__}.is_valid_credentials accepted an unknown key."
        with pytest.raises(InvalidCredentialShape):
            cls._validate_shape(slot)

    def test_malformed_slot_rejected(self) -> None:
        """A slot value that isn't a dict/HashableDict must raise from ``_extract_slot``.

        ``is_valid_credentials`` is matcher-safe (catches and returns False so
        mloda's matcher loop can keep iterating). The loud-failure entry point
        for direct callers is ``_extract_slot`` itself: a slot key with a
        non-dict value (e.g. a bare path string) raises
        ``InvalidCredentialShape`` so the typo surfaces instead of silently
        masquerading as "no plugin matched".
        """
        cls = self.connector_reader_class()
        creds = HashableDict({cls.CONNECTOR_ID: "this-should-be-a-dict-but-isnt"})
        assert cls.is_valid_credentials(creds) is False, (
            f"{cls.__name__}.is_valid_credentials should swallow the malformed-slot error and return False."
        )
        with pytest.raises(InvalidCredentialShape):
            cls._extract_slot(creds)

    def test_is_valid_credentials_is_matcher_safe_against_misbehaving_mapping(self) -> None:
        """``is_valid_credentials`` must not propagate non-``InvalidCredentialShape`` exceptions.

        mloda's ``ReadDB.match_read_db_data_access`` only catches
        ``NotImplementedError`` from the matcher loop. Any other propagating
        exception from a misbehaving credentials object aborts iteration over
        unrelated readers sharing the same ``DataAccessCollection``.
        ``_extract_slot`` has two probe sites (``credentials.data.get(...)``
        for ``HashableDict`` and ``credentials.get(...)`` for plain dicts), so
        both are exercised here against a dict subclass whose ``.get`` raises
        ``RuntimeError``. The fix broadens the matcher-safety guard to
        ``Exception``; this test pins that contract so a future narrowing
        surfaces immediately.
        """

        class _MisbehavingDict(dict[str, Any]):
            def get(self, key: Any, default: Any = None) -> Any:
                raise RuntimeError("synthetic probe failure to prove matcher-safety")

        cls = self.connector_reader_class()
        # Plain dict branch: ``_extract_slot`` calls ``credentials.get`` directly.
        bogus_plain = _MisbehavingDict()
        assert cls.is_valid_credentials(bogus_plain) is False, (
            f"{cls.__name__}.is_valid_credentials must swallow probe-time exceptions "
            f"raised by a misbehaving plain-dict Mapping and return False (matcher-safety)."
        )
        # HashableDict branch: ``_extract_slot`` calls ``credentials.data.get``. A
        # plain ``HashableDict`` with the bogus dict as its ``data`` exercises the
        # second probe path that the plain-dict case never reaches.
        bogus_wrapped = HashableDict(_MisbehavingDict())
        assert cls.is_valid_credentials(bogus_wrapped) is False, (
            f"{cls.__name__}.is_valid_credentials must swallow probe-time exceptions "
            f"raised by a misbehaving HashableDict.data and return False (matcher-safety)."
        )

    def test_required_params_enforced(self) -> None:
        """If REQUIRED_PARAMS is non-empty, stripping every OR-group's keys must raise.

        Mirrors ``test_required_keys_enforced`` for ParamReader plugins. Skips
        cleanly for ``QueryReader`` plugins (no ``REQUIRED_PARAMS`` attribute)
        and for ``ParamReader`` plugins that declare an empty
        ``REQUIRED_PARAMS``.

        Opens with a positive-control assertion: ``build_params`` on the
        unmodified feature must succeed AND every OR-group must already be
        satisfied via the returned params dict. Without that, an adapter whose
        ``feature_under_test()`` happens to omit the required keys turns the
        subsequent strip into a no-op and the test passes for the wrong reason
        (``build_params`` raises because the keys were never there in the
        first place, not because the strip removed them). The strip step then
        removes the keys for *every* OR-group so the positive control and the
        negative case stay symmetric: any group whose keys are not stripped
        leaves the contract trivially satisfied and the expected raise would
        not surface.
        """
        cls = self.connector_reader_class()
        if not issubclass(cls, ParamReader):
            pytest.skip(f"{cls.__name__} is not a ParamReader; REQUIRED_PARAMS does not apply.")
        if not cls.REQUIRED_PARAMS:
            pytest.skip(f"{cls.__name__} declares no REQUIRED_PARAMS; nothing to enforce.")

        feat = self.feature_under_test()

        positive_fs = FeatureSet()
        positive_fs.add(feat)
        params = cls.build_params(positive_fs)
        for group in cls.REQUIRED_PARAMS:
            assert any(params.get(k) for k in group), (
                f"{cls.__name__}: feature_under_test() does not satisfy REQUIRED_PARAMS group {group!r}; "
                f"build_params returned {params!r}. The strip-and-expect-raise test below would pass "
                f"for the wrong reason; supply a feature whose options satisfy every REQUIRED_PARAMS group."
            )

        raw = feat.options.context
        ctx = dict(raw.data) if isinstance(raw, HashableDict) else dict(raw)
        for group in cls.REQUIRED_PARAMS:
            for k in group:
                ctx.pop(k, None)
        stripped = Feature(feat.name, options=Options(context=ctx))

        fs = FeatureSet()
        fs.add(stripped)
        with pytest.raises(MissingRequiredParamsError):
            cls.build_params(fs)

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
            assert cls.is_valid_credentials(HashableDict({cls.CONNECTOR_ID: canonical})) is True
            return
        for group_idx, group in enumerate(cls.REQUIRED_KEYS):
            slot = dict(canonical)
            for k in group:
                slot.pop(k, None)
            creds = HashableDict({cls.CONNECTOR_ID: slot})
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
            assert cls.is_valid_credentials(HashableDict({cls.CONNECTOR_ID: canonical})) is True, (
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
                creds = HashableDict({cls.CONNECTOR_ID: slot})
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
        creds = HashableDict({cls.CONNECTOR_ID: {}})
        with pytest.raises(MissingRequiredKeysError):
            cls.connect(creds)

    def test_load_rejects_multi_feature_set(self) -> None:
        """Universal: ``load`` rejects FeatureSets carrying more than one feature.

        Concrete ``load_data`` implementations all consume a single feature via
        ``next(iter(features.features))``. Passing a
        heterogeneous FeatureSet would silently use whichever feature the
        iterator yielded first; the base ``load`` now raises ``ValueError``
        instead.
        """
        cls = self.connector_reader_class()
        feat_a = self.feature_under_test()
        feat_b = Feature(f"{feat_a.name}__sibling_for_multi_feature_guard", options=feat_a.options)
        fs = FeatureSet()
        fs.add(feat_a)
        fs.add(feat_b)
        with pytest.raises(ValueError):
            cls().load(fs)

    def test_load_is_idempotent(self) -> None:
        """Running the canonical feature twice yields the same rows.

        Catches latent native-state drift (e.g. Kuzu reader's on-disk database,
        fixture mtime-cache races, NetworkX node-iteration order). Rows are
        ``dict[str, Any]`` and ``sorted`` cannot order dicts directly, so the
        sort key recursively canonicalises each row: nested dicts are sorted
        by ``repr(key)``, nested sets/frozensets are sorted as canonical
        tuples (sets carry no order), and lists/tuples preserve order. The
        leaf step uses ``repr`` rather than ``str`` so e.g. a ``datetime`` and
        its ``str()`` form do not collide on the same key (which the previous
        ``json.dumps(..., default=str)`` form silently allowed). The recursion
        is the load-bearing part: two equal rows containing nested dicts with
        different insertion order produce the same key, so ``sorted()`` pairs
        them correctly. A non-empty first-result gate keeps the assertion
        load-bearing: two empty lists would compare equal regardless of the
        reader's idempotence.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        connector_id = self.connector_reader_class().CONNECTOR_ID
        creds = self.valid_credentials()[connector_id]
        feat = self.feature_under_test()
        first = run_query(connector_id, creds, feat)
        second = run_query(connector_id, creds, feat)

        assert first, (
            f"{self.connector_reader_class().__name__}.feature_under_test() returned no rows; "
            f"idempotence test is vacuous. Supply a feature whose canonical load yields >=1 row."
        )

        def _canonical(obj: Any) -> Any:
            if isinstance(obj, dict):
                return ("__dict__", tuple(sorted((repr(k), _canonical(v)) for k, v in obj.items())))
            if isinstance(obj, (list, tuple)):
                tag = "__list__" if isinstance(obj, list) else "__tuple__"
                return (tag, tuple(_canonical(item) for item in obj))
            if isinstance(obj, (set, frozenset)):
                tag = "__set__" if isinstance(obj, set) else "__frozenset__"
                return (tag, tuple(sorted(repr(_canonical(item)) for item in obj)))
            return ("__leaf__", repr(obj))

        def _key(row: Any) -> str:
            return repr(_canonical(row))

        assert sorted(first, key=_key) == sorted(second, key=_key), (
            f"{self.connector_reader_class().__name__} produced different rows across two identical loads; "
            f"first={first!r}, second={second!r}"
        )

    def test_load_does_not_mutate_credentials(self) -> None:
        """The credential slot passed through ``run_query`` survives the load unchanged.

        Asserts a pipeline-level contract: nothing along ``run_query`` ->
        ``DataAccessCollection`` -> ``mloda.run_all`` -> reader may mutate the
        slot dict the caller supplied. Today ``DataAccessCollection`` wraps the
        dict in a ``HashableDict`` by reference (no deepcopy), so a regression
        in the reader surfaces here directly; if a future mloda release starts
        deepcopying credentials, this test still guards the documented "slot
        is read-only" contract at the pipeline boundary even though the
        reader-side check becomes vacuous.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        connector_id = self.connector_reader_class().CONNECTOR_ID
        slot = self.valid_credentials()[connector_id]
        snapshot = copy.deepcopy(slot)
        run_query(connector_id, slot, self.feature_under_test())
        assert slot == snapshot, (
            f"{self.connector_reader_class().__name__} mutated the credential slot during load; "
            f"before={snapshot!r}, after={slot!r}"
        )

    def test_load_does_not_mutate_options(self) -> None:
        """The feature's ``options.group`` and ``options.context`` survive the load unchanged.

        Pipeline-level contract: nothing along ``run_query`` ->
        ``mloda.run_all`` -> reader may mutate the caller's ``Feature.options``.
        ``Options.__eq__`` only compares ``group``; comparing the underlying
        ``group``/``context`` dicts directly is required to catch a mutation
        of ``context`` (where every KG per-call key currently lives).
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        connector_id = self.connector_reader_class().CONNECTOR_ID
        feat = self.feature_under_test()
        group_snapshot = copy.deepcopy(feat.options.group)
        context_snapshot = copy.deepcopy(feat.options.context)
        run_query(connector_id, self.valid_credentials()[connector_id], feat)
        assert feat.options.group == group_snapshot, (
            f"{self.connector_reader_class().__name__} mutated feature.options.group during load; "
            f"before={group_snapshot!r}, after={feat.options.group!r}"
        )
        assert feat.options.context == context_snapshot, (
            f"{self.connector_reader_class().__name__} mutated feature.options.context during load; "
            f"before={context_snapshot!r}, after={feat.options.context!r}"
        )

    def test_calculate_feature_runs_end_to_end(self) -> None:
        """Run the feature via the real mloda discovery + run_all path.

        Covers ``CONNECTOR_ID`` matching, ``is_valid_credentials``, the
        ``DataAccessCollection`` wiring, and ``KgPythonDictFramework`` row
        consumption (the KG-aware ``PythonDictFramework`` adapter that wraps
        native rows as ``{feature_name: row}`` during column slicing). A
        regression in any of these surfaces here.

        Pre-check that ``is_valid_credentials`` accepts
        ``valid_credentials()`` so an adapter whose canonical slot is itself
        contract-non-conformant fails here with a clear diagnostic rather
        than opaquely deep inside ``mloda.run_all``. Overlaps
        ``test_credentials_match_connector_id`` intentionally — both should
        fail loudly when the adapter is broken.

        Enforce a universal ``len(result) >= 1`` floor.
        ``expected_row_shape()`` is concrete-supplied and could in principle
        be ``lambda r: isinstance(r, list)`` — silently accepting zero rows.
        The canonical feature in every adapter is required to return at
        least one row; concrete ``expected_row_shape`` predicates then
        assert shape, not size. The ``len(...)`` call also tightens the
        contract to "result must support ``len()``" — generator-shaped
        results are intentionally not admissible, since the size floor is a
        universal invariant and ``sum(1 for _ in result)`` would consume the
        generator before the concrete's own assertions could inspect it.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        cls = self.connector_reader_class()
        creds_dict = self.valid_credentials()
        connector_id = cls.CONNECTOR_ID
        assert cls.is_valid_credentials(HashableDict(creds_dict)) is True, (
            f"{cls.__name__}.valid_credentials() returned a slot that fails is_valid_credentials; "
            f"the adapter's canonical credentials are not contract-conformant. creds={creds_dict!r}"
        )
        feat = self.feature_under_test()

        result = run_query(connector_id, creds_dict[connector_id], feat)
        assert self.expected_row_shape()(result), (
            f"{cls.__name__} returned result of shape {type(result).__name__} "
            f"that failed expected_row_shape predicate. result={result!r}"
        )
        assert len(result) >= 1, (
            f"{cls.__name__} returned zero rows for the canonical feature {feat.name!r}; "
            f"adapters must seed at least one row so expected_row_shape asserts shape, not size."
        )


# Re-exported for convenience in concrete tests.
__all__ = [
    "KgConnectorContractBase",
    "DataAccessCollection",
    "Feature",
    "HashableDict",
    "InvalidCredentialShape",
    "MissingEnvVarError",
    "MissingRequiredKeysError",
    "MissingRequiredParamsError",
]
