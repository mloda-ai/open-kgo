"""Per-family contract base for lineage connectors.

The universal contract (`KgConnectorContractBase`) plus the per-concrete
end-to-end test in `test_dbt_manifest.py` cover behavior. This base only
narrows the `connector_reader_class()` return type so subclasses don't need
type-ignore comments around `PARAMS_MAPPING` access.
"""

from __future__ import annotations

from abc import abstractmethod

from open_kgo.feature_groups.kg.base import ParamReader
from open_kgo.feature_groups.kg.tests.kg_contract import KgConnectorContractBase


class LineageContractTestBase(KgConnectorContractBase):
    @classmethod
    @abstractmethod
    def connector_reader_class(cls) -> type[ParamReader]:
        """Lineage readers are ParamReader subclasses."""
