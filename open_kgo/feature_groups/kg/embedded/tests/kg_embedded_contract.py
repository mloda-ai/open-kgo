"""Per-family contract base for embedded graph connectors."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.base import ParamReader
from open_kgo.feature_groups.kg.embedded.base import UnknownStartNodeError
from open_kgo.feature_groups.kg.errors import InvalidCredentialShape
from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class EmbeddedContractTestBase(KgConnectorContractBase):
    @classmethod
    def absent_start_node(cls) -> Any:
        """Adapter hook: a ``start_node`` value guaranteed absent from the fixture graph.

        The default sentinel does not appear in any committed embedded
        fixture; a concrete whose fixture could legitimately contain it
        overrides this hook (same adapter-method pattern as
        ``valid_credentials`` / ``feature_under_test``).
        """
        return "__kg_embedded_absent_start_node__"

    def test_remote_locator_rejected(self) -> None:
        """Remote schemes must be refused, like the other file-backed readers.

        The embedded loaders only open local paths, so a remote locator could
        never fetch anyway, but rejecting it up front keeps the file-only
        contract uniform and the error message consistent across families.
        """
        self.assert_remote_locator_rejected("http://example.com/evil.gml")

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

    def test_unknown_start_node_raises_typed_error(self) -> None:
        """``operation=neighbors`` with an absent ``start_node`` raises ``UnknownStartNodeError``.

        Family-level pin for the "backend variety, identical behavior"
        thesis: before this contract the two embedded concretes diverged
        (igraph silently returned ``[]``; NetworkX leaked a raw
        ``networkx.NetworkXError``). Inherited by every concrete so any new
        embedded backend must fail the same typed way.
        """
        cls = self.connector_reader_class()
        feat = Feature(
            f"{cls.CONNECTOR_ID}__absent_start_node",
            options=Options(context={"operation": "neighbors", "start_node": self.absent_start_node()}),
        )
        fs = FeatureSet()
        fs.add(feat)
        with pytest.raises(UnknownStartNodeError):
            cls.load_data(self.valid_credentials(), fs)
