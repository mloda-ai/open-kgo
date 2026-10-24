"""Per-family contract base for RDF/SPARQL connectors.

Concrete plugin tests subclass ``RdfContractTestBase`` and supply the 5
adapter methods. The behavior assertions below run against any RDF/SPARQL
concrete plugin.
"""

from __future__ import annotations

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class RdfContractTestBase(KgConnectorContractBase):
    """Family-specific contract assertions for RDF/SPARQL connectors."""

    def test_invalid_result_format_rejected(self) -> None:
        self.assert_strict_enum_value_rejected("result_format", "application/evil+json")

    def test_invalid_reasoning_profile_rejected(self) -> None:
        self.assert_strict_enum_value_rejected("reasoning_profile", "evil-rl")

    def test_remote_locator_rejected(self) -> None:
        """connect() must refuse remote schemes (no network IO at fetch time)."""
        self.assert_remote_locator_rejected("http://example.com/evil.ttl")

    def test_query_text_required_for_load(self) -> None:
        """build_query must raise ValueError when feature.options.context lacks query_text."""
        import pytest

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet

        from open_kgo.feature_groups.kg.base import QueryReader

        cls = self.connector_reader_class()
        assert issubclass(cls, QueryReader)
        feat = Feature("rdflib_sparql__missing_query", options=Options(context={}))
        fs = FeatureSet()
        fs.add(feat)
        with pytest.raises(ValueError):
            cls.build_query(fs)
