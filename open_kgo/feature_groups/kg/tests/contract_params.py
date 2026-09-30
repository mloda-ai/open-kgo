"""Per-call param contract tests: PARAMS_MAPPING enums, stripped params, required params.

One of the four concern mixins aggregated by ``KgConnectorContractBase``
(see ``kg_contract.py``). Everything here exercises the per-call parameter
layer of ``ParamReader`` concretes; tests skip cleanly for ``QueryReader``
concretes, which carry no ``PARAMS_MAPPING``.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.base import ParamReader
from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingRequiredParamsError,
)
from open_kgo.feature_groups.kg.tests._discovery import iter_strict_specs
from open_kgo.feature_groups.kg.tests._helpers import bogus_value_for_strict_spec
from open_kgo.feature_groups.kg.tests.contract_adapters import KgContractAdapterBase


class ParamContract(KgContractAdapterBase):
    """Contract tests for the per-call param surface of a concrete KG plugin."""

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
        full_ctx = dict(raw)
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
        ctx = dict(raw)
        for group in cls.REQUIRED_PARAMS:
            for k in group:
                ctx.pop(k, None)
        stripped = Feature(feat.name, options=Options(context=ctx))

        fs = FeatureSet()
        fs.add(stripped)
        with pytest.raises(MissingRequiredParamsError):
            cls.build_params(fs)
