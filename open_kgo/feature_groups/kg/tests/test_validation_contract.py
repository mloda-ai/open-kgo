"""Validation-contract tests for the universal KG reader surface.

These exercise behaviors that are universal across all KG readers and do not
need a per-family adapter, so they live next to the cross-group smoke test
rather than under each family.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import HashableDict

from open_kgo.feature_groups.kg.base import (
    KgConnectorReaderBase,
    ParamReader,
    QueryReader,
    _collect_kg_known_keys,
)
from open_kgo.feature_groups.kg.citation_rest.file_fixture_citation import FileFixtureCitationReader
from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingRequiredKeysError,
    MissingRequiredParamsError,
)
from open_kgo.feature_groups.kg.lineage.dbt_manifest import DbtManifestReader
from open_kgo.feature_groups.kg.tests._discovery import (
    import_all_kg_readers,
    iter_strict_specs,
    walk_subclasses,
)

# Populate every kg family's concrete reader into the subclass tree so the
# walks in test_strict_specs_declare_allowed_values_explicitly /
# test_collect_kg_known_keys_unions_property_and_params cover everything.
# Mirrors the pattern in sibling cross-walk test modules.
import_all_kg_readers()


# -- Item 1: is_valid_credentials swallows shape errors -----------------------


def test_is_valid_credentials_returns_false_on_unknown_key_does_not_raise() -> None:
    """Discovery must not crash on a malformed slot."""
    bad = HashableDict({"dbt_manifest": {"locator": "/tmp/x.json", "definitely_not_real": "v"}})
    assert DbtManifestReader.is_valid_credentials(bad) is False


def test_is_valid_credentials_returns_false_on_strict_enum_violation() -> None:
    """A bad enum value rejects the candidate without aborting discovery.

    Uses ``lineage_direction``: the dbt concrete narrows it to
    ``{UPSTREAM, DOWNSTREAM, BOTH}`` via ``SUPPORTED_VALUES``, so ``"SIDEWAYS"``
    is reliably outside the effective allowed set. The earlier seed for this
    test was ``auth_method="evil"``; the auth surface was removed from the
    universal base, so the seed value moved to a strict
    enum that the concrete still honors.
    """
    bad = HashableDict({"dbt_manifest": {"locator": "/tmp/x.json", "lineage_direction": "SIDEWAYS"}})
    assert DbtManifestReader.is_valid_credentials(bad) is False


def test_is_valid_credentials_returns_false_on_missing_required_keys() -> None:
    bad = HashableDict({"dbt_manifest": {}})
    assert DbtManifestReader.is_valid_credentials(bad) is False


def test_validate_shape_still_raises_for_strict_callers() -> None:
    """Direct callers retain the strict typed-error contract."""
    with pytest.raises(InvalidCredentialShape):
        DbtManifestReader._validate_shape({"locator": "/tmp/x.json", "lineage_direction": "SIDEWAYS"})
    with pytest.raises(MissingRequiredKeysError):
        DbtManifestReader._validate_shape({})


# -- Item 2: allowed_values declared explicitly per spec ----------------------


def test_strict_specs_declare_allowed_values_explicitly() -> None:
    """Every strict-validation spec across every reader must carry ``allowed_values``.

    Guards the prior brittle pattern: the value space used to be inferred
    from plain string keys at the spec's top level, so adding a doc-only
    key like ``"see_also"`` would silently expand the allowed set.
    """
    for klass in walk_subclasses(KgConnectorReaderBase):
        for key, spec, layer_name in iter_strict_specs(klass):
            assert "allowed_values" in spec, (
                f"{klass.__name__}.{layer_name}[{key!r}] has strict_validation=True but no 'allowed_values' field."
            )


def test_spec_allowed_values_ignores_doc_only_keys() -> None:
    """Adding an unrelated string key to a spec must not affect the allowed set."""
    spec: dict[Any, Any] = {
        "explanation": "Probe enum.",
        "allowed_values": {"a": "A", "b": "B"},
        # Doc-only key that the prior key-derivation logic would have promoted:
        "see_also": "https://example.invalid/related",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
        DefaultOptionKeys.default: "a",
    }
    allowed = KgConnectorReaderBase._spec_allowed_values("mode", spec)
    assert allowed == {"a", "b"}
    assert "see_also" not in allowed


def test_spec_allowed_values_raises_when_missing() -> None:
    """A misconfigured spec (strict=True but no allowed_values) is itself a shape error."""
    spec: dict[Any, Any] = {
        "explanation": "Bad spec.",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
        DefaultOptionKeys.default: "a",
    }
    with pytest.raises(InvalidCredentialShape):
        KgConnectorReaderBase._spec_allowed_values("mode", spec)


def test_spec_allowed_values_accepts_iterable() -> None:
    """Allowed values may be supplied as a plain list/tuple, not only as a dict."""
    spec = {"allowed_values": ["x", "y"], DefaultOptionKeys.strict_validation: True}
    assert KgConnectorReaderBase._spec_allowed_values("mode", spec) == {"x", "y"}


# -- Item 3: typed param errors -----------------------------------------------


def test_validate_params_strict_enum_violation_raises_credential_shape() -> None:
    """Strict-enum failures on per-call params raise ``InvalidCredentialShape`` (single-rooted tree)."""
    with pytest.raises(InvalidCredentialShape):
        DbtManifestReader._validate_params({"asset_urn": "x", "lineage_direction": "SIDEWAYS"})


def test_validate_params_missing_required_raises_typed_subclass() -> None:
    """Missing ``REQUIRED_PARAMS`` raises ``MissingRequiredParamsError`` so callers can scope handlers."""
    with pytest.raises(MissingRequiredParamsError) as info:
        DbtManifestReader._validate_params({"lineage_direction": "BOTH"})
    assert info.value.connector_id == "dbt_manifest"
    assert info.value.unsatisfied_groups == (("asset_urn",),)


def test_missing_required_params_is_an_invalid_credential_shape() -> None:
    """Single-rooted tree: a generic ``InvalidCredentialShape`` handler catches the param subclass too."""
    with pytest.raises(InvalidCredentialShape):
        DbtManifestReader._validate_params({"lineage_direction": "BOTH"})


# -- Item 4: cross-reader leakage of reserved keys (warning, not raise) -------


def test_collect_kg_known_keys_unions_property_and_params() -> None:
    known = _collect_kg_known_keys()
    # Spot-check across families: universal keys (locator, result_limit),
    # PARAMS_MAPPING-only keys (asset_urn, lineage_direction, stable_id),
    # and PROPERTY_MAPPING-only keys (graph_file_format, retrieval_mode).
    for expected in (
        "locator",
        "result_limit",
        "asset_urn",
        "lineage_direction",
        "stable_id",
        "graph_file_format",
        "retrieval_mode",
    ):
        assert expected in known, f"{expected!r} should be a reserved KG key"


def test_build_params_warns_on_keys_reserved_by_other_family() -> None:
    """Passing a lineage-only key through a citation reader emits a warning and drops the key."""
    # FileFixtureCitationReader does not declare ``asset_urn`` (which is owned by lineage).
    feat = Feature(
        "file_fixture_citation__pathway",
        options=Options(context={"stable_id": "R-HSA-1640170", "asset_urn": "model.shop.fct_orders"}),
    )
    with pytest.warns(UserWarning, match=r"asset_urn"):
        params = FileFixtureCitationReader.build_params(FeatureSet([feat]))
    # The key is dropped from the params dict; the warning is the diagnostic channel.
    assert "asset_urn" not in params
    assert params.get("stable_id") == "R-HSA-1640170"


def test_build_params_does_not_warn_on_own_declared_keys() -> None:
    """Keys this reader declares (in PARAMS_MAPPING or PROPERTY_MAPPING) must not trigger the leak warning."""
    feat = Feature(
        "file_fixture_citation__pathway",
        options=Options(context={"stable_id": "R-HSA-1640170", "hierarchy_depth": 1}),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FileFixtureCitationReader.build_params(FeatureSet([feat]))
    leaks = [str(w.message) for w in caught if "reserved by other KG readers" in str(w.message)]
    assert leaks == []


def test_build_params_does_not_warn_on_unrelated_user_keys() -> None:
    """A key not in any KG reader's mapping passes through silently.

    The closed-world rule is bounded to reserved KG names; generic mloda
    keys and concrete-plugin-local keys (``operation``, ``start_node``)
    must be tolerated because ``feature.options`` is a shared surface.
    """
    feat = Feature(
        "file_fixture_citation__pathway",
        options=Options(context={"stable_id": "R-HSA-1640170", "totally_user_owned_key": "value"}),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FileFixtureCitationReader.build_params(FeatureSet([feat]))
    leaks = [str(w.message) for w in caught if "reserved by other KG readers" in str(w.message)]
    assert leaks == []


def test_query_reader_subclasses_unaffected_by_param_check() -> None:
    """The cross-reader check lives on ``ParamReader`` only; ``QueryReader``'s path is untouched."""
    assert not hasattr(QueryReader, "_warn_on_reserved_unknown_keys")
    assert hasattr(ParamReader, "_warn_on_reserved_unknown_keys")
