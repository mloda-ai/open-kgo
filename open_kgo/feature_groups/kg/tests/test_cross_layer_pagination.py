"""Cross-layer constraint: cursor_token implies a cursor-family pagination_style.

``pagination_style`` is a connector default declared in ``PROPERTY_MAPPING``;
``cursor_token`` is a per-call param declared in ``PARAMS_MAPPING``. The two
keys live in different validation layers, so before the cross-layer hook was
added there was no place a validator could see both. These tests pin the
hook's behaviour: ``cursor_token`` set with a non-cursor-family
``pagination_style`` must raise; the symmetric "first call" case
(``pagination_style`` cursor-family with no ``cursor_token``) must pass.

The literal "require cursor_token on continuation calls" wording from the
design notes isn't enforceable from a single call (no first-vs-continuation
signal exists). The symmetric "cursor_token implies cursor-family style"
rule catches the same misconfiguration class and is enforceable from one
call's worth of state.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase
from open_kgo.feature_groups.kg.citation_rest.base import CitationRestReader
from open_kgo.feature_groups.kg.errors import InvalidCredentialShape
from open_kgo.feature_groups.kg.rest_public.base import RestPublicReader
from open_kgo.feature_groups.kg.saas_authz.base import SaasAuthzReader


@pytest.mark.parametrize("reader_cls", [RestPublicReader, CitationRestReader, SaasAuthzReader])
@pytest.mark.parametrize("style", ["cursor", "cursorMark", "odata-nextLink"])
def test_cursor_token_with_cursor_family_style_passes(reader_cls: type[KgConnectorReaderBase], style: str) -> None:
    """cursor_token paired with any cursor-family style validates cleanly.

    Parametrized across all three PaginationMixin composers as a smoke test
    that each composer's MRO actually routes ``_validate_cross_layer`` to
    PaginationMixin. ``test_cooperative_super_chain_runs_pagination_check``
    is the dedicated guard against a sibling mixin shadowing the hook
    without ``super()``.
    """
    reader_cls._validate_cross_layer({"pagination_style": style}, {"cursor_token": "abc"})


@pytest.mark.parametrize("reader_cls", [RestPublicReader, CitationRestReader, SaasAuthzReader])
@pytest.mark.parametrize("style", ["page", "offset", "start_rows", "none"])
def test_cursor_token_with_non_cursor_style_raises(reader_cls: type[KgConnectorReaderBase], style: str) -> None:
    """cursor_token paired with a non-cursor style is a configuration error."""
    with pytest.raises(InvalidCredentialShape, match="cursor-family"):
        reader_cls._validate_cross_layer({"pagination_style": style}, {"cursor_token": "abc"})


@pytest.mark.parametrize("reader_cls", [RestPublicReader, CitationRestReader, SaasAuthzReader])
def test_missing_pagination_style_with_cursor_token_raises(reader_cls: type[KgConnectorReaderBase]) -> None:
    """Absent pagination_style defaults to ``"none"``; cursor_token still errors.

    Guards against the ``creds.get("pagination_style")`` returning None and
    silently bypassing the membership check. The default must be the same
    string PaginationMixin declares as the property's default value.
    """
    with pytest.raises(InvalidCredentialShape, match="cursor-family"):
        reader_cls._validate_cross_layer({}, {"cursor_token": "abc"})


@pytest.mark.parametrize("reader_cls", [RestPublicReader, CitationRestReader, SaasAuthzReader])
@pytest.mark.parametrize("style", ["cursor", "page", "none"])
def test_no_cursor_token_passes_for_any_style(reader_cls: type[KgConnectorReaderBase], style: str) -> None:
    """First-call scenario: cursor_token absent, any pagination_style is fine."""
    reader_cls._validate_cross_layer({"pagination_style": style}, {})


def test_default_validate_cross_layer_is_noop() -> None:
    """Readers without PaginationMixin inherit the no-op base hook.

    LineageReader composes ``TraversalMixin`` + ``EntityFilterParamMixin`` (no
    ``PaginationMixin``). It must not raise even when handed cursor_token-shaped
    params, because the hook is opt-in via mixin composition.
    """
    from open_kgo.feature_groups.kg.lineage.base import LineageReader

    LineageReader._validate_cross_layer({"pagination_style": "none"}, {"cursor_token": "abc"})


def test_cooperative_super_chain_runs_pagination_check() -> None:
    """A sibling mixin overriding ``_validate_cross_layer`` cooperatively must not hide PaginationMixin.

    Defines a fake mixin that adds its own check (sentinel raise on a
    ``"trip"`` sentinel), composes it with PaginationMixin and a
    ParamReader subclass. Verifies both checks fire, in the order dictated
    by MRO. If a future mixin overrides without ``super()._validate_cross_layer``,
    PaginationMixin's check would be silently bypassed; this test catches
    that regression by also asserting PaginationMixin's check still raises
    when its trigger is met and the fake mixin's trigger is absent.
    """
    from typing import Any, Mapping

    from open_kgo.feature_groups.kg.base import ParamReader
    from open_kgo.feature_groups.kg.mixins import PaginationMixin

    class _FakeSentinelMixin:
        @classmethod
        def _validate_cross_layer(cls, creds: Mapping[str, Any], params: dict[str, Any]) -> None:
            super()._validate_cross_layer(creds, params)  # type: ignore[misc]
            if creds.get("trip_sentinel"):
                raise InvalidCredentialShape("fake-mixin: tripped on sentinel")

    class _Composed(_FakeSentinelMixin, PaginationMixin, ParamReader):
        CONNECTOR_ID = "test_composed"

    # PaginationMixin's check still fires when fake's trigger is absent.
    with pytest.raises(InvalidCredentialShape, match="cursor-family"):
        _Composed._validate_cross_layer({"pagination_style": "page"}, {"cursor_token": "abc"})

    # Fake mixin's check fires when its trigger is set.
    with pytest.raises(InvalidCredentialShape, match="fake-mixin"):
        _Composed._validate_cross_layer({"trip_sentinel": True, "pagination_style": "cursor"}, {})

    # Both pass when neither trigger is met.
    _Composed._validate_cross_layer({"pagination_style": "cursor"}, {"cursor_token": "abc"})


def _feature_set_with(context: dict[str, object]) -> FeatureSet:
    """Build a real FeatureSet carrying a single feature with the given options context."""
    fs = FeatureSet()
    fs.add(Feature("rest_public__list", options=Options(context=context)))
    return fs


def test_build_params_invokes_cross_layer_when_creds_supplied() -> None:
    """End-to-end through ParamReader.build_params: passing creds engages the check.

    Uses the family base ``RestPublicReader`` directly: it keeps cursor_token
    in PARAMS_MAPPING (no concrete-level stripping) so the cross-layer check
    receives the param and raises against a non-cursor pagination_style.
    Concrete file-fixture readers strip cursor_token via ``_STRIPPED_PARAMS``;
    using one here would short-circuit at the strip-rejection step before
    reaching the cross-layer check.
    """
    features = _feature_set_with({"cursor_token": "abc"})
    with pytest.raises(InvalidCredentialShape, match="cursor-family"):
        RestPublicReader.build_params(features, {"pagination_style": "page"})


def test_build_params_skips_cross_layer_when_creds_omitted() -> None:
    """Backwards-compatible call site: build_params without creds = old behaviour."""
    features = _feature_set_with({"cursor_token": "abc"})
    params = RestPublicReader.build_params(features)
    assert params["cursor_token"] == "abc"


def test_in_process_tuple_store_load_data_engages_cross_layer_check() -> None:
    """Through ``InProcessTupleStoreReader.load_data``: a real concrete that retains cursor_token.

    InProcessTupleStoreReader is the only PaginationMixin composer that
    keeps ``cursor_token`` in PARAMS_MAPPING (no _STRIPPED_PARAMS narrowing)
    AND does not call ``build_params`` for its own purposes. Without the
    explicit ``build_params(features, ctx.slot)`` call wired into
    ``load_data`` for its validation side effect, the cross-layer hook
    would have zero practical exposure on this concrete. This test is the
    proof that the hook fires end-to-end through a concrete's ``load_data``.
    """
    from pathlib import Path

    from mloda.provider import HashableDict

    from open_kgo.feature_groups.kg.saas_authz.in_process_tuple_store import InProcessTupleStoreReader

    fixture = Path(__file__).resolve().parent.parent / "saas_authz" / "tests" / "fixtures" / "tuples.json"
    creds = HashableDict(
        {
            "in_process_tuple_store": {
                "locator": str(fixture),
                "tenant": "tenant_a",
                # page-style is non-cursor; cursor_token below should trip the check.
                "pagination_style": "page",
            }
        }
    )
    fs = FeatureSet()
    fs.add(Feature("in_process_tuple_store__viewers", options=Options(context={"cursor_token": "tok-xyz"})))

    with pytest.raises(InvalidCredentialShape, match="cursor-family"):
        InProcessTupleStoreReader.load_data(creds, fs)


def test_pagination_style_classifications_match_declared_values() -> None:
    """Every value declared in pagination_style spec must be classified by family.

    Three drift modes are guarded:

    1. Spec / family-tag drift: every key in ``_PAGINATION_STYLES`` must appear
       in the spec's ``allowed_values`` (and vice versa). A direct edit to the
       spec dict that bypasses the derivation would leave the cursor-family
       set stale.
    2. Family-tag typos: every family tag must come from a fixed enum
       (``"cursor"``, ``"counter"``, ``"none"``). A typo like ``"Cursor"``
       would silently exclude a style from ``_CURSOR_FAMILY_STYLES``.
    3. Default-value drift: the spec default must be a key in
       ``_PAGINATION_STYLES`` and must be the value the cross-layer hook
       falls back to via ``creds.get("pagination_style", <default>)``. The
       hook hard-codes ``"none"``; if the spec default ever changes, the
       fallback no longer matches.
    """
    from open_kgo.feature_groups.kg.mixins import _PAGINATION_STYLES, PaginationMixin

    style_spec = PaginationMixin.PROPERTY_MAPPING_DELTA["pagination_style"]
    declared = set(style_spec["allowed_values"].keys())
    classified = set(_PAGINATION_STYLES.keys())
    assert declared == classified, (
        f"pagination_style declarations and family classifications drifted: "
        f"declared but unclassified={declared - classified}; "
        f"classified but undeclared={classified - declared}"
    )

    valid_family_tags = {"cursor", "counter", "none"}
    actual_family_tags = {family for (family, _) in _PAGINATION_STYLES.values()}
    rogue = actual_family_tags - valid_family_tags
    assert not rogue, (
        f"_PAGINATION_STYLES contains unknown family tags {sorted(rogue)}; "
        f"only {sorted(valid_family_tags)} feed into _CURSOR_FAMILY_STYLES."
    )

    spec_default = style_spec[DefaultOptionKeys.default]
    assert spec_default in _PAGINATION_STYLES, f"pagination_style default {spec_default!r} is not a declared style."
    assert spec_default == "none", (
        "PaginationMixin._validate_cross_layer hard-codes 'none' as the fallback "
        "for missing pagination_style; the spec default must match. Update both "
        "in lockstep if the default changes."
    )
