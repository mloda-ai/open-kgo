"""Per-family contract base for citation REST connectors."""

from __future__ import annotations

from abc import abstractmethod

from open_kgo.feature_groups.kg.base import ParamReader
from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class CitationRestContractTestBase(KgConnectorContractBase):
    @classmethod
    @abstractmethod
    def connector_reader_class(cls) -> type[ParamReader]:
        """Citation-REST readers are ParamReader subclasses."""
