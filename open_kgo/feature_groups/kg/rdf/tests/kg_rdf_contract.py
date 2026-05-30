"""Per-family contract base for RDF/SPARQL connectors.

Concrete plugin tests subclass ``RdfContractTestBase`` and supply the 5
adapter methods. The behavior assertions below run against any RDF/SPARQL
concrete plugin.
"""

from __future__ import annotations

from mloda.provider import HashableDict
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class RdfContractTestBase(KgConnectorContractBase):
    """Family-specific contract assertions for RDF/SPARQL connectors."""

    def test_invalid_result_format_rejected(self) -> None:
        cls = self.connector_reader_class()
        slot = dict(next(iter(self.valid_credentials().values())))
        slot["result_format"] = "application/evil+json"
        creds = HashableDict({cls.CONNECTOR_ID: slot})
        from open_kgo.feature_groups.kg.errors import InvalidCredentialShape

        try:
            ok = cls.is_valid_credentials(creds)
        except InvalidCredentialShape:
            return
        assert ok is False

    def test_invalid_reasoning_profile_rejected(self) -> None:
        cls = self.connector_reader_class()
        slot = dict(next(iter(self.valid_credentials().values())))
        slot["reasoning_profile"] = "evil-rl"
        creds = HashableDict({cls.CONNECTOR_ID: slot})
        from open_kgo.feature_groups.kg.errors import InvalidCredentialShape

        try:
            ok = cls.is_valid_credentials(creds)
        except InvalidCredentialShape:
            return
        assert ok is False

    def test_query_text_required_for_load(self) -> None:
        """build_query must raise ValueError when feature.options.context lacks query_text."""
        import pytest

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet

        cls = self.connector_reader_class()
        feat = Feature("rdflib_sparql__missing_query", options=Options(context={}))
        fs = FeatureSet()
        fs.add(feat)
        with pytest.raises(ValueError):
            cls.build_query(fs)
