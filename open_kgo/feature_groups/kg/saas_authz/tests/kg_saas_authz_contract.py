"""Per-family contract base for saas_authz connectors."""

from __future__ import annotations

from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class SaasAuthzContractTestBase(KgConnectorContractBase):
    def test_invalid_consistency_mode_rejected(self) -> None:
        from mloda.provider import HashableDict

        from open_kgo.feature_groups.kg.errors import InvalidCredentialShape

        cls = self.connector_reader_class()
        slot = dict(next(iter(self.valid_credentials().values())))
        slot["consistency_mode"] = "evil"
        creds = HashableDict({cls.CONNECTOR_ID: slot})
        try:
            ok = cls.is_valid_credentials(creds)
        except InvalidCredentialShape:
            return
        assert ok is False
