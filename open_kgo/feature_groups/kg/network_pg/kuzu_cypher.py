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
``_load_rows`` closes it, and a direct ``connect()`` caller owns and closes its own.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import kuzu

from mloda.core.abstract_plugins.components.feature_set import FeatureSet

from open_kgo.feature_groups.kg.base import LoadContext
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
        layer is cheap and caller-owned. Two callers with the same ``locator`` share
        one Database; external mutations to the DB directory will NOT
        refresh the cache (Kuzu writes to its own directory during
        normal operation, so mtime keying is unsafe — see
        ``load_kuzu_database`` docstring).
        """
        db = load_kuzu_database(cls.CONNECTOR_ID, slot["locator"])
        return kuzu.Connection(db)

    @classmethod
    def _load_rows(cls, ctx: LoadContext, connection: Any, features: FeatureSet) -> list[dict[str, Any]]:
        conn = connection
        rows: list[dict[str, Any]] = []
        try:
            result_obj = conn.execute(cls.build_query(features))
            # kuzu returns either a single QueryResult or a list of QueryResult.
            results = result_obj if isinstance(result_obj, list) else [result_obj]
            for r in results:
                columns = r.get_column_names()
                while r.has_next() and len(rows) < ctx.result_limit:
                    rows.append(dict(zip(columns, r.get_next())))
        finally:
            conn.close()
        return rows


class KuzuCypherFeatureGroup(NetworkPropertyGraphFeatureGroup):
    READER_CLASS: ClassVar[type[KuzuCypherReader]] = KuzuCypherReader
