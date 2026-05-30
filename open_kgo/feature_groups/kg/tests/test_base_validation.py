"""Direct unit tests for ``KgConnectorReaderBase`` validation helpers.

These tests target behaviours that the per-family contract suite does not
exercise on its own: empty-string env-var values on the opt-in
``_resolve_env`` helper (issue #5 item 5) and malformed credential slots
(non-dict values, issue #5 item 6). The earlier auth_method ↔ env-name
conditional-required-keys checks (#5 item 7) were dropped when the
decorative auth surface was removed from the universal base — see issue
#32 item 2 — so this module no longer exercises them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.provider import HashableDict

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase
from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingEnvVarError,
    MissingRequiredKeysError,
)


class _FakeReader(KgConnectorReaderBase):
    """Minimal concrete reader used for direct base-class assertions."""

    CONNECTOR_ID: ClassVar[str] = "fake_connector_for_tests"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = ()


# --- Item 5: _resolve_env empty-string handling ----------------------------


def test_resolve_env_returns_none_when_credential_key_absent() -> None:
    assert _FakeReader._resolve_env({}, "auth_token_env") is None


def test_resolve_env_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KG_TEST_BASE_VALIDATION_TOKEN", "abc")
    creds: dict[str, Any] = {"auth_token_env": "KG_TEST_BASE_VALIDATION_TOKEN"}
    assert _FakeReader._resolve_env(creds, "auth_token_env") == "abc"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  abc", "abc"),
        ("abc  ", "abc"),
        ("\tabc\n", "abc"),
        (" \tabc def\n", "abc def"),
    ],
)
def test_resolve_env_strips_surrounding_whitespace(raw: str, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Surrounding whitespace is stripped before the value is returned.

    The rejection rule for whitespace-only values exists because downstream
    auth chokes on whitespace tokens. By the same rationale, a value with
    stray surrounding whitespace (a common ``.env``/copy-paste artifact)
    must not leak through unchanged either.
    """
    monkeypatch.setenv("KG_TEST_BASE_VALIDATION_STRIP", raw)
    creds: dict[str, Any] = {"auth_token_env": "KG_TEST_BASE_VALIDATION_STRIP"}
    assert _FakeReader._resolve_env(creds, "auth_token_env") == expected


def test_resolve_env_raises_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KG_TEST_BASE_VALIDATION_UNSET", raising=False)
    creds: dict[str, Any] = {"auth_token_env": "KG_TEST_BASE_VALIDATION_UNSET"}
    with pytest.raises(MissingEnvVarError):
        _FakeReader._resolve_env(creds, "auth_token_env")


@pytest.mark.parametrize("blank_value", ["", "   ", "\t", "\n", " \t\n "])
def test_resolve_env_raises_when_env_is_blank(blank_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An env var set to an empty or whitespace-only value is treated as missing.

    Issue #18 C4 extends the original empty-string case to whitespace-only
    values (``"   "``, ``"\\t"``, ``"\\n"``). Downstream auth chokes on
    whitespace tokens with opaque errors, so ``_resolve_env`` raises
    ``MissingEnvVarError`` for any value that does not contain a non-blank
    character.
    """
    monkeypatch.setenv("KG_TEST_BASE_VALIDATION_BLANK", blank_value)
    creds: dict[str, Any] = {"auth_token_env": "KG_TEST_BASE_VALIDATION_BLANK"}
    with pytest.raises(MissingEnvVarError):
        _FakeReader._resolve_env(creds, "auth_token_env")


def test_resolve_env_rejects_non_string_env_name() -> None:
    with pytest.raises(InvalidCredentialShape):
        _FakeReader._resolve_env({"auth_token_env": 42}, "auth_token_env")


# --- Item 6: _extract_slot raises on non-dict slot -------------------------


def test_extract_slot_returns_dict_for_dict_slot() -> None:
    creds = {_FakeReader.CONNECTOR_ID: {"locator": "/tmp/x"}}
    assert _FakeReader._extract_slot(creds) == {"locator": "/tmp/x"}


def test_extract_slot_returns_dict_for_hashabledict_slot() -> None:
    creds = HashableDict({_FakeReader.CONNECTOR_ID: HashableDict({"locator": "/tmp/x"})})
    assert _FakeReader._extract_slot(creds) == {"locator": "/tmp/x"}


def test_extract_slot_returns_none_when_slot_absent() -> None:
    assert _FakeReader._extract_slot({"some_other_id": {"locator": "x"}}) is None


def test_extract_slot_returns_none_when_slot_explicitly_none() -> None:
    """An explicit ``None`` value is treated as opt-out (caller declined to
    provide credentials), not as a malformed slot."""
    assert _FakeReader._extract_slot({_FakeReader.CONNECTOR_ID: None}) is None


def test_extract_slot_raises_on_string_slot() -> None:
    """``{CONNECTOR_ID: "/some/path"}`` (string instead of dict) used to be
    indistinguishable from "slot absent". It must now raise."""
    with pytest.raises(InvalidCredentialShape):
        _FakeReader._extract_slot({_FakeReader.CONNECTOR_ID: "/tmp/x.ttl"})


def test_extract_slot_raises_on_list_slot() -> None:
    with pytest.raises(InvalidCredentialShape):
        _FakeReader._extract_slot({_FakeReader.CONNECTOR_ID: ["a", "b"]})


def test_extract_slot_raises_on_int_slot() -> None:
    with pytest.raises(InvalidCredentialShape):
        _FakeReader._extract_slot({_FakeReader.CONNECTOR_ID: 7})


# --- Item 6 (matcher safety): is_valid_credentials swallows shape errors ---


class _OtherFakeReader(KgConnectorReaderBase):
    """Second reader with a distinct CONNECTOR_ID for multi-slot scenarios."""

    CONNECTOR_ID: ClassVar[str] = "fake_connector_other"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = ()


def test_is_valid_credentials_returns_false_on_malformed_slot() -> None:
    """Malformed wrapper (non-dict slot) must not propagate out of the
    matcher-facing API. mloda's ``match_read_db_data_access`` only catches
    ``NotImplementedError``; a raise here would abort iteration over
    unrelated reader subclasses. ``_extract_slot`` still raises directly
    (covered above), preserving loud diagnostics for explicit callers.
    """
    creds = HashableDict({_FakeReader.CONNECTOR_ID: "/bad/string"})
    assert _FakeReader.is_valid_credentials(creds) is False


def test_is_valid_credentials_returns_false_on_validate_shape_error() -> None:
    """Same matcher-safety contract for shape errors *inside* a dict slot:
    unknown keys, missing required keys, conditional-rule violations, and
    failed enum checks all surface as ``False`` (not raise) from the
    matcher-facing API. ``_validate_shape`` still raises directly.
    """
    creds = HashableDict({_FakeReader.CONNECTOR_ID: {"definitely_not_a_kg_key": "x"}})
    assert _FakeReader.is_valid_credentials(creds) is False


def test_matcher_safe_iteration_with_one_malformed_and_one_valid_slot() -> None:
    """Regression: a multi-slot ``credential_dicts`` with a malformed slot
    for one reader must not block a sibling reader from matching its own
    valid slot. Before the fix, the malformed-slot raise aborted iteration.
    """
    creds = HashableDict(
        {
            _FakeReader.CONNECTOR_ID: "/bad/string",
            _OtherFakeReader.CONNECTOR_ID: {"locator": "/some/path"},
        }
    )
    assert _FakeReader.is_valid_credentials(creds) is False
    assert _OtherFakeReader.is_valid_credentials(creds) is True


def test_is_valid_credentials_swallows_runtime_error_from_misbehaving_mapping() -> None:
    """Issue #18 item A4: ``is_valid_credentials`` must catch non-``InvalidCredentialShape``
    exceptions raised while probing the credentials object.

    The matcher (``ReadDB.match_read_db_data_access``) only catches
    ``NotImplementedError``, so any other propagating exception aborts
    iteration over unrelated readers. The reproducer is a plain ``dict``
    subclass whose ``.get`` raises ``RuntimeError``: the previous
    narrow-``except InvalidCredentialShape`` let the ``RuntimeError`` escape
    out of ``is_valid_credentials`` and broke the matcher contract.
    """

    class _BrokenMapping(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            raise RuntimeError("probe-time failure to prove matcher-safety")

    creds = _BrokenMapping()
    assert _FakeReader.is_valid_credentials(creds) is False


def test_is_valid_credentials_swallows_runtime_error_from_misbehaving_hashabledict() -> None:
    """A4 second probe site: ``HashableDict.data`` is exercised via ``credentials.data.get(...)``.

    ``_extract_slot`` has a separate branch for ``isinstance(credentials, HashableDict)``;
    swapping a misbehaving ``dict`` subclass into ``HashableDict.data`` exercises that
    second probe path (the plain-dict test above only covers the
    ``isinstance(credentials, dict)`` branch). ``except Exception`` covers both,
    but pinning both paths here means a future narrowing of either probe site
    surfaces immediately.
    """

    class _BrokenMapping(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            raise RuntimeError("probe-time failure from inside HashableDict.data")

    creds = HashableDict(_BrokenMapping())
    assert _FakeReader.is_valid_credentials(creds) is False


def test_each_reader_returns_false_independently_for_misbehaving_mapping() -> None:
    """Per-reader matcher-safety against a misbehaving Mapping.

    A ``.get`` that raises unconditionally cannot reproduce the
    iteration-continues-past-one-broken-reader scenario in a single
    ``DataAccessCollection``: every reader probing the broken Mapping
    raises and returns False, so there is no "this reader matches"
    candidate left in the iteration. That single-collection iteration
    property is already pinned by
    ``test_matcher_safe_iteration_with_one_malformed_and_one_valid_slot``
    (which uses a malformed-but-not-raising slot value, so the second
    reader still sees a probe-able dict).

    This test pins a narrower guarantee: each reader, when handed the
    misbehaving Mapping on its own call, returns False rather than
    propagating ``RuntimeError``. Without this, the matcher's outer
    loop would see the first probe blow up and abort before reaching
    any other reader at all.
    """

    class _BrokenMapping(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            raise RuntimeError("probe-time failure")

    broken = _BrokenMapping()
    assert _FakeReader.is_valid_credentials(broken) is False
    assert _OtherFakeReader.is_valid_credentials(broken) is False


# --- Loud-failure contract: _validate_shape raises directly -----------------


def test_validate_shape_raises_on_unknown_key() -> None:
    with pytest.raises(InvalidCredentialShape):
        _FakeReader._validate_shape({"definitely_not_a_kg_key": "x"})


# --- Issue #18 item A5: connect() validation parity with is_valid_credentials ----


class _RequiredKeyReader(KgConnectorReaderBase):
    """Synthetic reader with a non-empty ``REQUIRED_KEYS`` for A5 connect() tests."""

    CONNECTOR_ID: ClassVar[str] = "fake_required_key_reader"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",),)


def test_connect_raises_typed_when_required_key_missing() -> None:
    """``connect()`` runs ``_validate_shape`` after ``_require_slot``.

    A partial slot like ``{CONNECTOR_ID: {}}`` (slot exists, but empty) used
    to surface as whatever ``_connect_from_slot`` raised next: typically a
    downstream ``FileNotFoundError`` or ``KeyError`` against a missing
    locator. The shape gate now runs first, so the typed
    ``MissingRequiredKeysError`` from ``_validate_required_keys`` is the
    visible error.
    """
    creds = HashableDict({_RequiredKeyReader.CONNECTOR_ID: {}})
    with pytest.raises(MissingRequiredKeysError):
        _RequiredKeyReader.connect(creds)


def test_required_keys_presence_convention_accepts_falsey_value() -> None:
    """A present-but-falsey required credential (``""``, ``0``, ``False``) satisfies the rule.

    ``_validate_required_keys`` tests presence with ``is not None`` rather than
    truthiness, matching ``_validate_required_params`` and the ``kg_contract``
    presence convention. A required key that is absent or ``None`` is still
    rejected.
    """
    for falsey in ("", 0, False):
        # present-but-falsey: satisfied, no raise
        _RequiredKeyReader._validate_required_keys({"locator": falsey})

    # absent and explicit-None are both still unsatisfied
    with pytest.raises(MissingRequiredKeysError):
        _RequiredKeyReader._validate_required_keys({})
    with pytest.raises(MissingRequiredKeysError):
        _RequiredKeyReader._validate_required_keys({"locator": None})


def test_connect_raises_on_unknown_credential_key() -> None:
    """``connect()`` also rejects closed-world unknown keys (full shape parity).

    Mirrors ``_validate_shape``: a key not declared in ``PROPERTY_MAPPING``
    surfaces as ``InvalidCredentialShape`` from the direct-call path, not
    only from the matcher-safe ``is_valid_credentials`` path.
    """
    creds = HashableDict({_RequiredKeyReader.CONNECTOR_ID: {"locator": "/tmp/x", "definitely_not_a_kg_key": "bad"}})
    with pytest.raises(InvalidCredentialShape):
        _RequiredKeyReader.connect(creds)
