"""KuzuDB embedded Cypher reader.

Kuzu is a property-graph engine with Cypher syntax but is embedded (no
network). The ``locator`` points to the Kuzu database directory. We open it,
run the Cypher in ``query_text``, and return rows as ``list[dict[str, Any]]``
keyed by Kuzu's column names — matching the cross-family return shape used
by the other 8 readers.

PROTOTYPE NOTE: ``read_consistency`` and ``transaction_mode`` are accepted
at the property layer but are no-ops on Kuzu (single-process embedded).

Resource lifecycle: ``kuzu.Database`` is the FD holder and is cached
process-wide by absolute path (see ``kg.fixtures.load_kuzu_database``)
so a 100-feature ``mloda.run_all`` opens the DB once instead of 100x
(repeated full re-parse of the same source, plus native FD leak). Each call
builds a fresh ``kuzu.Connection(db)`` on top of the cached Database;
the Connection is caller-owned and may be closed by direct callers
without poisoning the cache.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import kuzu

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.fixtures import load_kuzu_database
from open_kgo.feature_groups.kg.network_pg.base import (
    NetworkPropertyGraphFeatureGroup,
    NetworkPropertyGraphReader,
)


class KuzuCypherReader(NetworkPropertyGraphReader):
    CONNECTOR_ID: ClassVar[str] = "kuzu_cypher"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = (("locator",),)
    # Waive read_consistency / transaction_mode: the module docstring already
    # notes both are accepted for forward-compat but no-op on Kuzu. Real
    # network_pg backends (Neo4j, Memgraph, TypeDB) will dispatch on these,
    # so narrowing here would lock the family contract to Kuzu's single
    # honored value and force future concretes to widen.
    _WAIVED_ENUM_KEYS: ClassVar[frozenset[str]] = frozenset({"read_consistency", "transaction_mode"})

    @classmethod
    def _connect_from_slot(cls, slot: Mapping[str, Any]) -> kuzu.Connection:
        """Return a fresh ``kuzu.Connection`` on top of the path-cached Database.

        Native FDs live on the cached ``kuzu.Database``; the Connection
        layer is cheap and remains caller-owned (the matcher contract
        test closes it). Two callers with the same ``locator`` share
        one Database; external mutations to the DB directory will NOT
        refresh the cache (Kuzu writes to its own directory during
        normal operation, so mtime keying is unsafe — see
        ``load_kuzu_database`` docstring).
        """
        db = load_kuzu_database(cls.CONNECTOR_ID, slot["locator"])
        return kuzu.Connection(db)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> list[dict[str, Any]]:
        ctx = cls._prepare_load(data_access)
        conn = cls._connect_from_slot(ctx.slot)

        query_text = cls.build_query(features)
        result_obj = conn.execute(query_text)
        rows: list[dict[str, Any]] = []
        # kuzu returns either a single QueryResult or a list of QueryResult.
        results = result_obj if isinstance(result_obj, list) else [result_obj]
        for r in results:
            columns = r.get_column_names()
            while r.has_next():
                rows.append(dict(zip(columns, r.get_next())))
                if len(rows) >= ctx.result_limit:
                    return rows
        return rows


class KuzuCypherFeatureGroup(NetworkPropertyGraphFeatureGroup):
    READER_CLASS: ClassVar[type[KuzuCypherReader]] = KuzuCypherReader  # type: ignore[assignment]
