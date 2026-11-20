"""Concrete tests for SpdxSbomReader.

Unlike the CycloneDX concrete (flat component list), this reader walks the
SPDX ``DEPENDS_ON`` graph, so the tests assert the family's TraversalMixin
keys (``lineage_direction`` / ``upstream_depth`` / ``downstream_depth``) are
honored end to end.

``fixtures/sample.spdx.json`` is a minimal parser fixture, not schema-valid
SPDX 2.3 (it omits ``documentNamespace``, ``creationInfo``, and per-package
``downloadLocation``); the reader only consumes ``packages`` and
``relationships``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.code_build.spdx_sbom import SpdxSbomReader
from open_kgo.feature_groups.kg.code_build.tests.kg_code_build_contract import (
    CodeBuildContractTestBase,
)
from open_kgo.feature_groups.kg.errors import InvalidCredentialShape
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials


_FIXTURE = Path(__file__).parent / "fixtures" / "sample.spdx.json"


class TestSpdxSbomReader(CodeBuildContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[SpdxSbomReader]:
        return SpdxSbomReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        # REQUIRED_KEYS lists manifest_path / locator as alternatives; the
        # per-alternative coherence contract requires every alternative present.
        return make_valid_credentials(
            cls.connector_reader_class(),
            manifest_path=str(_FIXTURE),
            locator=str(_FIXTURE),
            commit_sha="a1b2c3d4",
            branch="main",
            language_code="python",
            result_limit=100,
        )

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        # No strict-validation credential enum on this concrete; trigger the
        # closed-world unknown-key rejection.
        return {"spdx_sbom": {"manifest_path": str(_FIXTURE), "definitely_not_a_kg_key": "x"}}

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature(
            "spdx_sbom__upstream",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-app",
                    "lineage_direction": "UPSTREAM",
                    "upstream_depth": 2,
                    "downstream_depth": 0,
                }
            ),
        )

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: isinstance(result, list) and len(result) >= 1 and "name" in result[0]

    def test_upstream_walk_reaches_transitive_dependencies(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        assert {r["name"] for r in rows} == {"app", "flask", "werkzeug", "click"}

    def test_depth_one_excludes_transitive(self) -> None:
        """upstream_depth=1 stops at flask; werkzeug/click are two hops out."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__upstream_d1",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-app",
                    "lineage_direction": "UPSTREAM",
                    "upstream_depth": 1,
                    "downstream_depth": 0,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        assert {r["name"] for r in rows} == {"app", "flask"}

    def test_dependency_of_inverse_orientation_and_dangling_edge(self) -> None:
        """``DEPENDENCY_OF`` resolves to the same depender->dependency orientation, and a
        relationship pointing at a package id absent from ``packages`` is skipped
        without aborting the walk.

        Fixture: ``lib-b DEPENDENCY_OF lib-a`` (so lib-a depends on lib-b) plus a
        dangling ``lib-a DEPENDS_ON <missing>``. Walking UPSTREAM from lib-a must
        reach lib-b (inverse orientation) and drop the missing id from output.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__lib_upstream",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-lib-a",
                    "lineage_direction": "UPSTREAM",
                    "upstream_depth": 2,
                    "downstream_depth": 0,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        assert {r["name"] for r in rows} == {"lib-a", "lib-b"}

    def test_missing_transit_node_blocks_reachability(self) -> None:
        """A package reachable ONLY through a missing intermediate is not surfaced.

        Fixture chain: ``chain-app DEPENDS_ON chain-missing`` (absent from
        ``packages``) and ``chain-missing DEPENDS_ON chain-real``. Walking
        UPSTREAM from chain-app must NOT reach chain-real: the dangling transit
        node is neither emitted nor traversed, so the only row is the start.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__chain_upstream",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-chain-app",
                    "lineage_direction": "UPSTREAM",
                    "upstream_depth": 3,
                    "downstream_depth": 0,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        names = {r["name"] for r in rows}
        assert names == {"chain-app"}
        assert "chain-real" not in names

    def test_downstream_walk_finds_dependents(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__downstream",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-flask",
                    "lineage_direction": "DOWNSTREAM",
                    "upstream_depth": 0,
                    "downstream_depth": 1,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        assert {r["name"] for r in rows} == {"flask", "app"}

    def test_both_direction_unions_upstream_and_downstream(self) -> None:
        """lineage_direction=BOTH (the default) returns upstream ∪ downstream, no duplicate rows.

        Start at ``flask``: its dependencies (``werkzeug``, ``click``) and its
        dependents (``app``) are disjoint sets, so this case alone cannot catch
        double emission; the overlapping-direction guarantee is pinned by
        ``test_cycle_under_both_emits_no_duplicates`` below.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__both",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-flask",
                    "lineage_direction": "BOTH",
                    "upstream_depth": 2,
                    "downstream_depth": 2,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        names = [r["name"] for r in rows]
        assert len(names) == len(set(names))
        assert set(names) == {"flask", "werkzeug", "click", "app"}

    def test_cycle_under_both_emits_no_duplicates(self) -> None:
        """A 2-cycle walked with BOTH emits each package exactly once.

        Fixture: ``cycle-a DEPENDS_ON cycle-b`` and ``cycle-b DEPENDS_ON
        cycle-a``, so cycle-b is reachable both upstream and downstream of
        cycle-a. The two walks share one EMITTED set; without it the result
        would be [cycle-a, cycle-b, cycle-b] and result_limit would be
        double-billed.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__cycle_both",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-cycle-a",
                    "lineage_direction": "BOTH",
                    "upstream_depth": 3,
                    "downstream_depth": 3,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        names = [r["name"] for r in rows]
        assert sorted(names) == ["cycle-a", "cycle-b"]

    def test_both_direction_expands_overlap_node_reached_by_first_walk(self, tmp_path: Path) -> None:
        """A package beyond an overlap node is still reached by the second walk under BOTH.

        Topology: ``dep-a DEPENDS_ON dep-b``, ``dep-c DEPENDS_ON dep-a``,
        ``dep-b DEPENDS_ON dep-c``, ``dep-d DEPENDS_ON dep-b``. From dep-a,
        the upstream walk emits dep-b (1 hop) and dep-c (2 hops); the
        downstream walk runs dep-a -> dep-c -> dep-b -> dep-d, reaching the
        already-emitted dep-c and dep-b, which it must still EXPAND (without
        re-emitting them) so dep-d is returned at hop 3. A regression to one
        seen set shared for traversal would stop the downstream walk at dep-c
        and silently drop dep-d.
        """
        import json

        from open_kgo.feature_groups.kg.tests._helpers import run_query

        sbom = {
            "packages": [
                {"SPDXID": "SPDXRef-Package-dep-a", "name": "dep-a"},
                {"SPDXID": "SPDXRef-Package-dep-b", "name": "dep-b"},
                {"SPDXID": "SPDXRef-Package-dep-c", "name": "dep-c"},
                {"SPDXID": "SPDXRef-Package-dep-d", "name": "dep-d"},
            ],
            "relationships": [
                {
                    "spdxElementId": "SPDXRef-Package-dep-a",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": "SPDXRef-Package-dep-b",
                },
                {
                    "spdxElementId": "SPDXRef-Package-dep-c",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": "SPDXRef-Package-dep-a",
                },
                {
                    "spdxElementId": "SPDXRef-Package-dep-b",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": "SPDXRef-Package-dep-c",
                },
                {
                    "spdxElementId": "SPDXRef-Package-dep-d",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": "SPDXRef-Package-dep-b",
                },
            ],
        }
        fixture = tmp_path / "overlap.spdx.json"
        fixture.write_text(json.dumps(sbom), encoding="utf-8")
        feat = Feature(
            "spdx_sbom__overlap_both",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-dep-a",
                    "lineage_direction": "BOTH",
                    "upstream_depth": 3,
                    "downstream_depth": 3,
                }
            ),
        )
        rows = run_query("spdx_sbom", {"manifest_path": str(fixture), "locator": str(fixture)}, feat)
        names = [r["name"] for r in rows]
        assert sorted(names) == ["dep-a", "dep-b", "dep-c", "dep-d"], (
            f"expected every package exactly once (dep-d beyond the overlap nodes), got {names}"
        )

    def test_self_loop_emits_no_duplicate(self) -> None:
        """A self-loop (``self-loop DEPENDS_ON self-loop``) yields only the start row once."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__self_loop",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-self-loop",
                    "lineage_direction": "BOTH",
                    "upstream_depth": 2,
                    "downstream_depth": 2,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        assert [r["name"] for r in rows] == ["self-loop"]

    def test_upstream_depth_zero_yields_only_start(self) -> None:
        """upstream_depth=0 with UPSTREAM disables the walk; only the start row is emitted."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__upstream_d0",
            options=Options(
                context={
                    "start_spdx_id": "SPDXRef-Package-app",
                    "lineage_direction": "UPSTREAM",
                    "upstream_depth": 0,
                    "downstream_depth": 0,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        assert [r["name"] for r in rows] == ["app"]

    @pytest.mark.parametrize(
        "start",
        ["SPDXRef-Package-totally-unknown", "SPDXRef-Package-chain-missing"],
    )
    def test_missing_start_is_neither_emitted_nor_traversed(self, start: str) -> None:
        """A start absent from ``packages`` returns [] (the dangling-node rule covers the start too).

        ``chain-missing`` is the stronger case: it HAS an outgoing edge to the
        real package chain-real, but a dangling start is not traversed, so
        nothing is reachable from it.
        """
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature(
            "spdx_sbom__missing_start",
            options=Options(
                context={
                    "start_spdx_id": start,
                    "lineage_direction": "BOTH",
                    "upstream_depth": 3,
                    "downstream_depth": 3,
                }
            ),
        )
        rows = run_query("spdx_sbom", self.valid_credentials()["spdx_sbom"], feat)
        assert rows == []

    def test_result_limit_truncates_walk(self) -> None:
        """result_limit=2 stops the walk after the start row plus one dependency."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["spdx_sbom"])
        slot["result_limit"] = 2
        feat = self.feature_under_test()
        rows = run_query("spdx_sbom", slot, feat)
        names = [r["name"] for r in rows]
        assert len(names) == 2
        assert names[0] == "app"

    def test_empty_start_spdx_id_raises_typed_error(self) -> None:
        """start_spdx_id="" passes the REQUIRED_PARAMS is-not-None check but is rejected typed.

        A missing/None start never reaches load_data (MissingRequiredParamsError
        fires in build_params); the empty string is the only falsey value that
        does, and it must raise InvalidCredentialShape, not a bare ValueError.
        """
        fs = FeatureSet()
        fs.add(
            Feature(
                "spdx_sbom__empty_start",
                options=Options(context={"start_spdx_id": ""}),
            )
        )
        with pytest.raises(InvalidCredentialShape, match="non-empty SPDXID"):
            SpdxSbomReader.load_data(self.valid_credentials(), fs)

    @pytest.mark.parametrize("bad", ["2", "garbage", 1.5, True, False, -1])
    def test_non_int_or_negative_depth_rejected(self, bad: Any) -> None:
        """Depths must be non-bool ints >= 0; strings, floats, bools, and negatives raise typed."""
        for key in ("upstream_depth", "downstream_depth"):
            fs = FeatureSet()
            fs.add(
                Feature(
                    "spdx_sbom__bad_depth",
                    options=Options(context={"start_spdx_id": "SPDXRef-Package-app", key: bad}),
                )
            )
            with pytest.raises(InvalidCredentialShape, match=key):
                SpdxSbomReader.load_data(self.valid_credentials(), fs)
