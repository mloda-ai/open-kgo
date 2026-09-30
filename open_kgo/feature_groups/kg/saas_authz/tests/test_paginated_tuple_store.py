"""Concrete tests for PaginatedTupleStoreReader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingRequiredKeysError,
    UnknownTenantError,
)
from open_kgo.feature_groups.kg.saas_authz.paginated_tuple_store import (
    PaginatedTupleStoreReader,
)
from open_kgo.feature_groups.kg.saas_authz.tests.kg_saas_authz_contract import (
    SaasAuthzContractTestBase,
)
from open_kgo.feature_groups.kg.tests._helpers import make_valid_credentials


_FIXTURE = Path(__file__).parent / "fixtures" / "tuples_b.json"


class TestPaginatedTupleStoreReader(SaasAuthzContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[PaginatedTupleStoreReader]:
        return PaginatedTupleStoreReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return make_valid_credentials(
            cls.connector_reader_class(),
            locator=str(_FIXTURE),
            tenant="tenant_b",
            api_version="v1.0",
            relationship_type="viewer",
            pagination_style="cursor",
            page_size=2,
            result_limit=100,
        )

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        return {
            "paginated_tuple_store": {
                "locator": str(_FIXTURE),
                "tenant": "tenant_b",
                "consistency_mode": "evil",
            }
        }

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature("paginated_tuple_store__viewers", options=Options(context={}))

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: (
            isinstance(result, list) and len(result) >= 1 and all(r["relation"] == "viewer" for r in result)
        )

    def test_first_page_honors_page_size(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = self.feature_under_test()
        rows = run_query("paginated_tuple_store", self.valid_credentials()["paginated_tuple_store"], feat)
        assert [(r["object_id"], r["user"]) for r in rows] == [("doc1", "user:alice"), ("doc1", "user:bob")]

    def test_second_page_via_cursor_token(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        feat = Feature("paginated_tuple_store__page2", options=Options(context={"cursor_token": "offset:2"}))
        rows = run_query("paginated_tuple_store", self.valid_credentials()["paginated_tuple_store"], feat)
        assert [(r["object_id"], r["user"]) for r in rows] == [("doc2", "user:carol"), ("doc3", "user:dave")]

    def test_expand_paths_expands_group_userset(self) -> None:
        """With expand_paths, the group userset is replaced by its members."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        slot["expand_paths"] = ["member"]
        slot["page_size"] = 100
        feat = self.feature_under_test()
        rows = run_query("paginated_tuple_store", slot, feat)
        users = {r["user"] for r in rows}
        assert "user:erin" in users and "user:frank" in users
        assert "group:eng#member" not in users

    def test_without_expand_paths_group_ref_passes_through(self) -> None:
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        slot["page_size"] = 100
        feat = self.feature_under_test()
        rows = run_query("paginated_tuple_store", slot, feat)
        users = {r["user"] for r in rows}
        assert "group:eng#member" in users
        assert "user:erin" not in users

    def test_unknown_tenant_raises_typed_error(self) -> None:
        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        slot["tenant"] = "tenant_does_not_exist"
        with pytest.raises(UnknownTenantError):
            PaginatedTupleStoreReader._connect_from_slot(slot)

    def test_cursor_token_with_cursor_style_builds_params(self) -> None:
        """Happy path: cursor_token + the pinned cursor pagination_style passes the cross-layer guard.

        pagination_style is narrowed to ``cursor`` via SUPPORTED_VALUES AND
        required via REQUIRED_KEYS, so no slot that passes validation can pair
        cursor_token with a non-cursor style (omission is rejected at
        ``is_valid_credentials`` time; see
        ``test_omitted_pagination_style_rejected_at_validate_time``). This
        test pins the complementary positive: a validated slot's cursor_token
        survives ``build_params`` intact.
        """
        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        feat = Feature("paginated_tuple_store__cursor_ok", options=Options(context={"cursor_token": "offset:0"}))
        fs = FeatureSet()
        fs.add(feat)
        params = PaginatedTupleStoreReader.build_params(fs, slot)
        assert params["cursor_token"] == "offset:0"

    def test_omitted_pagination_style_rejected_at_validate_time(self) -> None:
        """A full valid slot minus ``pagination_style`` fails ``is_valid_credentials``.

        ``SUPPORTED_VALUES`` only validates keys present in the slot, and the
        family default is ``none`` (a style this reader does not honor), so
        the omission case must be closed by ``REQUIRED_KEYS``: without it, the
        slot would validate, serve a cursor-paginated page 1, and then reject
        its own continuation token at the cross-layer guard.
        """

        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        del slot["pagination_style"]
        creds = {"paginated_tuple_store": slot}
        assert PaginatedTupleStoreReader.is_valid_credentials(creds) is False
        with pytest.raises(MissingRequiredKeysError):
            PaginatedTupleStoreReader._validate_shape(slot)

    def test_malformed_cursor_token_rejected(self) -> None:
        """A malformed cursor surfaces ``InvalidCredentialShape`` from ``load_data``.

        Exercised directly against the reader (not through ``mloda.run_all``,
        which wraps reader exceptions in a bare ``Exception``) so the typed
        error contract is asserted at the reader boundary.
        """
        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        feat = Feature("paginated_tuple_store__bad_cursor", options=Options(context={"cursor_token": "garbage"}))
        fs = FeatureSet()
        fs.add(feat)
        with pytest.raises(InvalidCredentialShape):
            PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)

    def _slot_for(self, tmp_path: Path, tuples: list[list[str]], **overrides: Any) -> dict[str, Any]:
        """Write a one-tenant fixture and return a valid slot pointed at it."""
        fixture = tmp_path / "tuples.json"
        fixture.write_text(json.dumps({"t": tuples}), encoding="utf-8")
        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        slot["locator"] = str(fixture)
        slot["tenant"] = "t"
        slot["page_size"] = 100
        slot.update(overrides)
        return slot

    def test_duplicate_membership_tuple_expands_member_once(self, tmp_path: Path) -> None:
        """A group member defined by two identical membership tuples is granted once (dedup)."""
        slot = self._slot_for(
            tmp_path,
            tuples=[
                ["document", "doc", "viewer", "group:eng#member"],
                ["group", "eng", "member", "user:erin"],
                ["group", "eng", "member", "user:erin"],
            ],
            expand_paths=["member"],
        )
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)
        assert [r["user"] for r in rows] == ["user:erin"]

    def test_unresolvable_userset_yields_zero_grants(self, tmp_path: Path) -> None:
        """A userset referencing an absent/empty group expands to zero grants."""
        slot = self._slot_for(
            tmp_path,
            tuples=[["document", "doc", "viewer", "group:ghost#member"]],
            expand_paths=["member"],
        )
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)
        assert rows == []

    def test_nested_userset_passes_through_unexpanded(self, tmp_path: Path) -> None:
        """Single-pass semantics: expansion output containing a userset is NOT re-expanded.

        ``group:eng#member`` contains ``group:sub#member`` whose rel IS in
        ``expand_paths``; Zanzibar's recursive Expand would resolve through to
        ``user:zoe``, but this single structural pass emits the nested userset
        reference as-is (the documented divergence in ``_expand_usersets``).
        """
        slot = self._slot_for(
            tmp_path,
            tuples=[
                ["document", "doc", "viewer", "group:eng#member"],
                ["group", "eng", "member", "group:sub#member"],
                ["group", "sub", "member", "user:zoe"],
            ],
            expand_paths=["member"],
        )
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)
        assert [r["user"] for r in rows] == ["group:sub#member"]

    def test_self_referential_userset_terminates_and_emits_itself(self, tmp_path: Path) -> None:
        """A userset cycle (group lists itself as a member) terminates and emits the reference itself.

        Single-pass expansion has no recursion to cycle through; the
        self-membership tuple simply becomes the expansion output.
        """
        slot = self._slot_for(
            tmp_path,
            tuples=[
                ["document", "doc", "viewer", "group:eng#member"],
                ["group", "eng", "member", "group:eng#member"],
            ],
            expand_paths=["member"],
        )
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)
        assert [r["user"] for r in rows] == ["group:eng#member"]

    def test_empty_tenant_store_yields_empty_list(self, tmp_path: Path) -> None:
        """A tenant whose tuple list is empty yields ``[]`` (with expansion enabled)."""
        slot = self._slot_for(tmp_path, tuples=[], expand_paths=["member"])
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)
        assert rows == []

    def test_page_boundary_splitting_an_expansion_serves_consistent_pages(self, tmp_path: Path) -> None:
        """A page boundary landing mid-expanded-group serves non-overlapping, complete pages.

        One userset expands to three grants; ``page_size=2`` puts the boundary
        inside the expanded group. Page 1 and page 2 (via ``offset:2``) must
        not overlap and must union to the full sorted expansion.
        """
        tuples = [
            ["document", "doc", "viewer", "group:eng#member"],
            ["group", "eng", "member", "user:a"],
            ["group", "eng", "member", "user:b"],
            ["group", "eng", "member", "user:c"],
        ]
        slot = self._slot_for(tmp_path, tuples=tuples, expand_paths=["member"], page_size=2)

        fs_page1 = FeatureSet()
        fs_page1.add(self.feature_under_test())
        page1 = PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs_page1)

        fs_page2 = FeatureSet()
        fs_page2.add(Feature("paginated_tuple_store__page2", options=Options(context={"cursor_token": "offset:2"})))
        page2 = PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs_page2)

        assert [r["user"] for r in page1] == ["user:a", "user:b"]
        assert [r["user"] for r in page2] == ["user:c"]
        users_page1 = {r["user"] for r in page1}
        users_page2 = {r["user"] for r in page2}
        assert not users_page1 & users_page2
        assert users_page1 | users_page2 == {"user:a", "user:b", "user:c"}

    def test_result_limit_caps_below_page_size(self, tmp_path: Path) -> None:
        """result_limit smaller than page_size truncates the page to result_limit rows.

        Pins the ``[: ctx.result_limit]`` slice after the page slice (the same
        contract the citation sibling pins): the page is silently shortened,
        not rejected.
        """
        tuples = [
            ["document", "doc1", "viewer", "user:a"],
            ["document", "doc2", "viewer", "user:b"],
            ["document", "doc3", "viewer", "user:c"],
        ]
        slot = self._slot_for(tmp_path, tuples=tuples, page_size=100, result_limit=2)
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        rows = PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)
        assert [r["user"] for r in rows] == ["user:a", "user:b"]

    def test_bare_string_expand_paths_rejected(self) -> None:
        """A bare-string ``expand_paths`` raises through the load path.

        A plain ``str`` is iterable, so ``tuple("member")`` would silently
        become ``("m","e","m","b","e","r")`` and disable expansion with no
        error; the typed guard rejects it instead.
        """
        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        slot["expand_paths"] = "member"
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        with pytest.raises(InvalidCredentialShape):
            PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)

    @pytest.mark.parametrize("bad", [False, 0, b"", {"member": 1}, {"member"}, 7])
    def test_non_list_expand_paths_rejected(self, bad: object) -> None:
        """Only ``None`` and an empty list/tuple opt out of expansion; everything else raises.

        ``False`` / ``0`` / ``b""`` would silently opt out under a truthiness
        check, and a dict would pass an element check via key iteration; the
        guard accepts list/tuple of non-empty strings only.
        """
        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        slot["expand_paths"] = bad
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        with pytest.raises(InvalidCredentialShape):
            PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)

    @pytest.mark.parametrize("empty", [[], ()])
    def test_empty_list_or_tuple_expand_paths_opts_out(self, empty: object) -> None:
        """An explicit empty list/tuple is the documented expansion opt-out (group refs pass through)."""
        from open_kgo.feature_groups.kg.tests._helpers import run_query

        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        slot["expand_paths"] = empty
        slot["page_size"] = 100
        rows = run_query("paginated_tuple_store", slot, self.feature_under_test())
        assert "group:eng#member" in {r["user"] for r in rows}

    @pytest.mark.parametrize("bad", [0, -1, "abc", True])
    def test_invalid_page_size_rejected(self, bad: object) -> None:
        """A non-positive-int / non-numeric page_size raises InvalidCredentialShape at load time."""
        slot = dict(self.valid_credentials()["paginated_tuple_store"])
        slot["page_size"] = bad
        fs = FeatureSet()
        fs.add(self.feature_under_test())
        with pytest.raises(InvalidCredentialShape):
            PaginatedTupleStoreReader.load_data({"paginated_tuple_store": slot}, fs)
