"""Per-family contract base for RDF/SPARQL connectors.

Concrete plugin tests subclass ``RdfContractTestBase`` and supply the 5
adapter methods. The behavior assertions below run against any RDF/SPARQL
concrete plugin.
"""

from __future__ import annotations

from mloda.user import Credential, DataAccessCollection, Feature, Options, mloda

from open_kgo.feature_groups.kg.base import PythonDictFramework
from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class RdfContractTestBase(KgConnectorContractBase):
    """Family-specific contract assertions for RDF/SPARQL connectors."""

    def test_projection_converts_literals_to_python_values(self) -> None:
        """``~<row_key>`` cells are Python values (an ``xsd:integer`` COUNT is ``int``, an IRI is ``str``)."""
        connector_id = self.connector_reader_class().CONNECTOR_ID
        query = (
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/> "
            "SELECT ?s (COUNT(?o) AS ?n) WHERE { ?s foaf:knows ?o } GROUP BY ?s"
        )
        names = [f"{connector_id}__counts", f"{connector_id}__counts~n", f"{connector_id}__counts~s"]
        partitions = mloda.run_all(
            [Feature(name, options=Options(context={"query_text": query})) for name in names],
            compute_frameworks=[PythonDictFramework],
            data_access_collection=DataAccessCollection(credentials=Credential(self.valid_credentials())),
        )
        rows, counts, subjects = ([v for p in partitions for v in p.get(name, [])] for name in names)
        assert rows and all(isinstance(row["n"], str) for row in rows)
        assert sorted(counts) == [1, 1, 1] and all(type(count) is int for count in counts)
        assert sorted(subjects) == sorted(str(row["s"]) for row in rows) and all(type(s) is str for s in subjects)

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

        from mloda.provider import FeatureSet

        from open_kgo.feature_groups.kg.base import QueryReader

        cls = self.connector_reader_class()
        assert issubclass(cls, QueryReader)
        feat = Feature("rdflib_sparql__missing_query", options=Options(context={}))
        fs = FeatureSet()
        fs.add(feat)
        with pytest.raises(ValueError):
            cls.build_query(fs)
