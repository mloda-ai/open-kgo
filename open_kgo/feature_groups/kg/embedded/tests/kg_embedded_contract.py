"""Per-family contract base for embedded graph connectors."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.base import ParamReader
from open_kgo.feature_groups.kg.errors import InvalidCredentialShape
from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class EmbeddedContractTestBase(KgConnectorContractBase):
    def test_invalid_operation_rejected(self) -> None:
        """Strict-enum validation on ``operation`` rejects unknown values via build_params.

        Inherited by every concrete embedded reader so the strict-enum is
        verified once per backend (NetworkX, igraph, ...) without each
        concrete test having to hand-roll it.
        """
        cls = self.connector_reader_class()
        assert issubclass(cls, ParamReader), f"{cls.__name__} must be a ParamReader to validate `operation`."
        feat = Feature(f"{cls.CONNECTOR_ID}__bad_op", options=Options(context={"operation": "evil"}))
        fs = FeatureSet()
        fs.add(feat)
        with pytest.raises(InvalidCredentialShape):
            cls.build_params(fs)
