"""Concrete tests for KuzuCypherReader.

Builds a small Kuzu database in a temp dir during setup_method, runs Cypher
queries against it, and asserts on the row shape.

DESIGN NOTE (building the backend at test time is intentional, issue #20):
kuzu_cypher is the only connector that constructs its backend in setup rather
than reading a committed fixture, and that is a recorded decision, not an open
follow-up. Two parts:

1. Runtime build vs committed fixture: Kuzu's on-disk store is a binary,
   version-coupled format, so a checked-in ``.kuzu`` database could fail to open
   after a ``kuzu`` version bump. ``kuzu`` is subject to the default
   ``exclude-newer`` 7-day window (it is not in the longer ``exclude-newer-package``
   pin, which covers only the mloda packages), so it upgrades after that window
   rather than being held forever. Building a tiny three-node database at test
   time is version-agnostic and cheap. The sibling concrete ``grand_cypher`` reads
   a committed ``.gml`` text fixture, so the network_pg family already demonstrates
   both shapes (static text fixture and built backend).
2. Fresh temp path per test (not a shared, seed-once database): a shared
   seed-once database is technically viable. The reader caches ``kuzu.Database``
   process-wide keyed by absolute path (see ``kg.fixtures.load_kuzu_database``),
   and ``kg/conftest.py`` clears that cache around every test, so cache staleness
   across a reused path is already neutralised: under CPython the ``cache_clear``
   drops the only reference and the native handle (with its directory lock) is
   finalised before the next open. We keep the per-test rebuild simply because it
   is the most obviously isolated option and the cost is immaterial (a three-node
   seed; the network_pg suite runs in a couple of seconds), so collapsing it to a
   shared database would trade clarity for no measurable gain. A unique
   ``mkdtemp`` path is also incidentally robust against delayed finalisation on a
   non-refcounting interpreter, but that is not the reason for the choice.

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
import pytest

from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.network_pg.kuzu_cypher import KuzuCypherReader
from open_kgo.feature_groups.kg.network_pg.tests.kg_network_pg_contract import (
    NetworkPropertyGraphContractTestBase,
)
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials


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
        return make_valid_credentials(cls.connector_reader_class(), locator=path, dataset="default", result_limit=100)

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

    @pytest.mark.parametrize("query", ["MATCH (p:Person) RETURN p.name", "NOT CYPHER"])
    def test_load_closes_its_connection(self, query: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """``load_data`` closes the Connection it opened, also when the query fails."""
        closed: list[bool] = []
        connect = KuzuCypherReader._connect_from_slot

        def spy(slot: Any) -> kuzu.Connection:
            conn = connect(slot)
            close = conn.close
            conn.close = lambda: closed.append(True) or close()  # type: ignore[method-assign, func-returns-value]
            return conn

        monkeypatch.setattr(KuzuCypherReader, "_connect_from_slot", spy)
        features = FeatureSet()
        features.add(Feature("kuzu_cypher__q", options=Options(context={"query_text": query})))
        if query.startswith("MATCH"):
            assert len(KuzuCypherReader.load_data(self.valid_credentials(), features)) == 3
        else:
            with pytest.raises(RuntimeError):
                KuzuCypherReader.load_data(self.valid_credentials(), features)
        assert closed == [True]

    def test_cypher_returns_three_persons(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("kuzu_cypher", self.valid_credentials()["kuzu_cypher"], feat)
        names = sorted(r["p.name"] for r in rows)
        assert names == ["Alice", "Bob", "Carol"]
