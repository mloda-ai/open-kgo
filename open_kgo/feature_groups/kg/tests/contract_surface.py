"""Advertised-surface honesty contract tests: enum dispositions and unconsumed keys.

One of the four concern mixins aggregated by ``KgConnectorContractBase``
(see ``kg_contract.py``). These two tests partition the advertised surface
(strict enums vs. the rest) so every key a concrete advertises has an
explicit disposition; see the "Honest credential surface" section in
``base.py`` for the rule they enforce.
"""

from __future__ import annotations

from mloda.provider import PropertySpec

from open_kgo.feature_groups.kg.tests._discovery import (
    effective_unconsumed_waivers,
    iter_nonstrict_specs,
    iter_strict_specs,
    reader_string_literals,
)
from open_kgo.feature_groups.kg.tests.contract_adapters import KgContractAdapterBase


class SurfaceHonestyContract(KgContractAdapterBase):
    """Contract tests for the honest-credential-surface rule of a concrete KG plugin."""

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
        lies: list[tuple[str, str, PropertySpec]] = []
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

    def test_no_unconsumed_advertised_keys(self) -> None:
        """Every advertised non-strict credential/param key is consumed or waived.

        Enforces the "Honest credential surface" rule (see base.py). Strict
        enums are owned by ``test_strict_enum_honored_or_waived``; this test
        owns the complementary non-strict keys. Each must either appear as an
        exact string literal in a reader method across the kg-package MRO (read
        at runtime) or be listed in ``_WAIVED_UNCONSUMED_KEYS`` (unioned across
        the MRO). The literal check is a heuristic proxy for consumption (per
        issue #22): every shipped reader reads its keys by literal, so it
        reliably turns a silent surface lie into a red build.
        """
        cls = self.connector_reader_class()
        consumed = reader_string_literals(cls)
        waived = effective_unconsumed_waivers(cls)
        lies: list[tuple[str, str]] = []
        for key, _spec, layer_name in iter_nonstrict_specs(cls):
            if key in consumed or key in waived:
                continue
            lies.append((layer_name, key))
        if lies:
            rendered = [f"{layer}.{key}" for layer, key in lies]
            raise AssertionError(
                f"{cls.__name__}: advertised non-strict key(s) never consumed at runtime and not "
                f"waived: {rendered}. Either read the key in a reader method, strip it from the "
                f"mapping (narrow_property_mapping / PARAMS narrowing), or add it to "
                f"_WAIVED_UNCONSUMED_KEYS with a one-line waiver comment. See the 'Honest credential "
                f"surface' section in base.py."
            )
