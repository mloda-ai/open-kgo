"""Per-family contract base for code/build/SBOM connectors."""

from __future__ import annotations

from abc import abstractmethod

from open_kgo.feature_groups.kg.base import ParamReader
from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class CodeBuildContractTestBase(KgConnectorContractBase):
    @classmethod
    @abstractmethod
    def connector_reader_class(cls) -> type[ParamReader]:
        """Code-build readers are ParamReader subclasses."""
