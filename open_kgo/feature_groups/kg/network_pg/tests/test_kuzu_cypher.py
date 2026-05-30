"""Concrete tests for KuzuCypherReader.

Builds a small Kuzu database in a temp dir during setup_method, runs Cypher
queries against it, and asserts on the row shape.

PROTOTYPE NOTE: this exercises the network_pg property layout against a real
Cypher engine but does NOT exercise read_consistency / transaction_mode
semantics (Kuzu is embedded; these properties are no-ops here).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import kuzu

from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.network_pg.kuzu_cypher import KuzuCypherReader
from open_kgo.feature_groups.kg.network_pg.tests.kg_network_pg_contract import (
    NetworkPropertyGraphContractTestBase,
)


def _seed_kuzu_db(db_dir: Path) -> None:
    db = kuzu.Database(str(db_dir))
    conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE Person(name STRING, PRIMARY KEY (name))")
    conn.execute("CREATE (:Person {name: 'Alice'})")
    conn.execute("CREATE (:Person {name: 'Bob'})")
    conn.execute("CREATE (:Person {name: 'Carol'})")


class TestKuzuCypherReader(NetworkPropertyGraphContractTestBase):
    _tmp: Path | None = None

    def setup_method(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="kg_kuzu_"))
        self._db_path = self._tmpdir / "graph.kuzu"
        _seed_kuzu_db(self._db_path)
        type(self)._tmp = self._db_path

    def teardown_method(self) -> None:
        if self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        type(self)._tmp = None

    @classmethod
    def connector_reader_class(cls) -> type[KuzuCypherReader]:
        return KuzuCypherReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        # When this method runs (during invalid_credentials calls etc.), self._tmp
        # may be set by setup_method on the running test instance. We fall back
        # to a placeholder for tests that don't actually load (shape-only tests).
        path = str(cls._tmp) if cls._tmp is not None else "/tmp/kg_kuzu_placeholder"
        return {
            "kuzu_cypher": {
                "locator": path,
                "dataset": "default",
                "read_consistency": "read",
                "transaction_mode": "auto",
                "result_limit": 100,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        path = str(cls._tmp) if cls._tmp is not None else "/tmp/kg_kuzu_placeholder"
        return {"kuzu_cypher": {"locator": path, "read_consistency": "evil"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "kuzu_cypher__list_persons",
            options=Options(context={"query_text": "MATCH (p:Person) RETURN p.name"}),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) == 3

    def test_cypher_returns_three_persons(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("kuzu_cypher", self.valid_credentials()["kuzu_cypher"], feat)
        names = sorted(r["p.name"] for r in rows)
        assert names == ["Alice", "Bob", "Carol"]
