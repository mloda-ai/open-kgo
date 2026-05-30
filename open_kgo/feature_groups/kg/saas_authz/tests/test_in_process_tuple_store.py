"""Concrete tests for InProcessTupleStoreReader."""

from __future__ import annotations

import json
from typing import Any, Callable

from mloda.user import Feature, Options

import pytest

from mloda.provider import HashableDict

from open_kgo.feature_groups.kg.errors import UnknownTenantError
from open_kgo.feature_groups.kg.saas_authz.in_process_tuple_store import (
    InProcessTupleStoreReader,
)
from open_kgo.feature_groups.kg.saas_authz.tests.kg_saas_authz_contract import (
    SaasAuthzContractTestBase,
)


class TestInProcessTupleStoreReader(SaasAuthzContractTestBase):
    @classmethod
    def connector_reader_class(cls) -> type[InProcessTupleStoreReader]:
        return InProcessTupleStoreReader

    @classmethod
    def valid_credentials(cls) -> dict[str, Any]:
        return {
            "in_process_tuple_store": {
                "tenant": "tenant_a",
                "api_version": "v1.0",
                "entity_type": "document",
                "relationship_type": "viewer",
                "consistency_mode": "minimize_latency",
                "pagination_style": "none",
                "result_limit": 100,
            }
        }

    @classmethod
    def invalid_credentials(cls) -> dict[str, Any]:
        return {
            "in_process_tuple_store": {
                "tenant": "tenant_a",
                "consistency_mode": "evil",
            }
        }

    @classmethod
    def feature_under_test(cls) -> Feature:
        return Feature("in_process_tuple_store__viewers", options=Options(context={}))

    @classmethod
    def expected_row_shape(cls) -> Callable[[Any], bool]:
        return lambda result: (
            isinstance(result, list) and len(result) >= 1 and all(r["relation"] == "viewer" for r in result)
        )

    def test_unknown_tenant_rejected_on_both_layers(self) -> None:
        """Issue #5 item 16: tenant validation runs at two layers.

        Validate-time (matcher-safe): the closed-world enum from the
        ``PROPERTY_MAPPING`` override rejects unknown tenants before any
        fixture I/O, so ``is_valid_credentials`` returns False.

        Connect-time (loud): the public ``connect()`` runs ``_validate_shape``
        first (issue #18 item A5), so the strict-enum gate also fires there
        for an unknown tenant value. ``_connect_from_slot`` itself raises
        ``UnknownTenantError`` for callers that bypass the shape gate; this
        is the typed-error defense-in-depth that mirrors agent_memory's
        ``UnknownMemoryScopeError`` and ensures a bare-``ValueError``
        regression in the fixture-lookup path stays visible.
        """
        slot = dict(self.valid_credentials()["in_process_tuple_store"])
        slot["tenant"] = "tenant_does_not_exist"
        creds = HashableDict({"in_process_tuple_store": slot})
        assert InProcessTupleStoreReader.is_valid_credentials(creds) is False
        with pytest.raises(UnknownTenantError):
            InProcessTupleStoreReader._connect_from_slot(slot)

    def test_allowed_tenants_mirror_fixture(self) -> None:
        """The strict-validation enum and the pinned fixture agree on the supported tenant set.

        Issue #5 item 16 introduces a second source of truth for "which
        tenants this concrete serves" (the ``allowed_values`` override on
        the ``tenant`` spec, alongside the pinned fixture's top-level keys).
        The two must stay aligned: a tenant in the fixture but not in the
        enum is unreachable through the matcher; a tenant in the enum but
        not in the fixture passes validation only to fail at connect with
        ``UnknownTenantError``. This test fails the build on either drift.
        """
        fixture_data = json.loads(InProcessTupleStoreReader._FIXTURE_PATH.read_text(encoding="utf-8"))
        allowed = set(InProcessTupleStoreReader.PROPERTY_MAPPING["tenant"]["allowed_values"])
        assert allowed == set(fixture_data)
