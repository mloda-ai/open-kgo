"""Load-behavior contract tests: multi-feature guard, idempotence, mutation safety, e2e.

One of the four concern mixins aggregated by ``KgConnectorContractBase``
(see ``kg_contract.py``). Everything here runs the reader's real load path,
mostly through ``run_query`` (``mloda.run_all`` + ``PythonDictFramework``),
so regressions in matching, validation, or the load-side feature-name wrap
surface here rather than silently passing.
"""

from __future__ import annotations

import copy

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import HashableDict
from mloda.user import Feature

from open_kgo.feature_groups.kg.tests._helpers import canonical_row_key, run_query
from open_kgo.feature_groups.kg.tests.contract_adapters import KgContractAdapterBase


class LoadBehaviorContract(KgContractAdapterBase):
    """Contract tests for the load path of a concrete KG plugin."""

    def test_load_rejects_multi_feature_set(self) -> None:
        """Universal: ``load`` rejects FeatureSets carrying more than one feature.

        Concrete ``load_data`` implementations all consume a single feature via
        ``next(iter(features.features))``. Passing a
        heterogeneous FeatureSet would silently use whichever feature the
        iterator yielded first; the base ``load`` now raises ``ValueError``
        instead.
        """
        cls = self.connector_reader_class()
        feat_a = self.feature_under_test()
        feat_b = Feature(f"{feat_a.name}__sibling_for_multi_feature_guard", options=feat_a.options)
        fs = FeatureSet()
        fs.add(feat_a)
        fs.add(feat_b)
        with pytest.raises(ValueError):
            cls().load(fs)

    def test_load_is_idempotent(self) -> None:
        """Running the canonical feature twice yields the same rows.

        Catches latent native-state drift (e.g. Kuzu reader's on-disk database,
        fixture mtime-cache races, NetworkX node-iteration order). Rows are
        sorted via ``canonical_row_key`` (see ``_helpers``), which
        canonicalises nested dicts/sets so two equal rows with different
        insertion order pair correctly. A non-empty first-result gate keeps
        the assertion load-bearing: two empty lists would compare equal
        regardless of the reader's idempotence.
        """
        connector_id = self.connector_reader_class().CONNECTOR_ID
        creds = self.valid_credentials()[connector_id]
        feat = self.feature_under_test()
        first = run_query(connector_id, creds, feat)
        second = run_query(connector_id, creds, feat)

        assert first, (
            f"{self.connector_reader_class().__name__}.feature_under_test() returned no rows; "
            f"idempotence test is vacuous. Supply a feature whose canonical load yields >=1 row."
        )

        assert sorted(first, key=canonical_row_key) == sorted(second, key=canonical_row_key), (
            f"{self.connector_reader_class().__name__} produced different rows across two identical loads; "
            f"first={first!r}, second={second!r}"
        )

    def test_load_does_not_mutate_credentials(self) -> None:
        """The credential slot passed through ``run_query`` survives the load unchanged.

        Asserts a pipeline-level contract: nothing along ``run_query`` ->
        ``DataAccessCollection`` -> ``mloda.run_all`` -> reader may mutate the
        slot dict the caller supplied. Today ``DataAccessCollection`` wraps the
        dict in a ``HashableDict`` by reference (no deepcopy), so a regression
        in the reader surfaces here directly; if a future mloda release starts
        deepcopying credentials, this test still guards the documented "slot
        is read-only" contract at the pipeline boundary even though the
        reader-side check becomes vacuous.
        """
        connector_id = self.connector_reader_class().CONNECTOR_ID
        slot = self.valid_credentials()[connector_id]
        snapshot = copy.deepcopy(slot)
        run_query(connector_id, slot, self.feature_under_test())
        assert slot == snapshot, (
            f"{self.connector_reader_class().__name__} mutated the credential slot during load; "
            f"before={snapshot!r}, after={slot!r}"
        )

    def test_load_does_not_mutate_options(self) -> None:
        """The feature's ``options.group`` and ``options.context`` survive the load unchanged.

        Pipeline-level contract: nothing along ``run_query`` ->
        ``mloda.run_all`` -> reader may mutate the caller's ``Feature.options``.
        ``Options.__eq__`` only compares ``group``; comparing the underlying
        ``group``/``context`` dicts directly is required to catch a mutation
        of ``context`` (where every KG per-call key currently lives).
        """
        connector_id = self.connector_reader_class().CONNECTOR_ID
        feat = self.feature_under_test()
        group_snapshot = copy.deepcopy(feat.options.group)
        context_snapshot = copy.deepcopy(feat.options.context)
        run_query(connector_id, self.valid_credentials()[connector_id], feat)
        assert feat.options.group == group_snapshot, (
            f"{self.connector_reader_class().__name__} mutated feature.options.group during load; "
            f"before={group_snapshot!r}, after={feat.options.group!r}"
        )
        assert feat.options.context == context_snapshot, (
            f"{self.connector_reader_class().__name__} mutated feature.options.context during load; "
            f"before={context_snapshot!r}, after={feat.options.context!r}"
        )

    def test_calculate_feature_runs_end_to_end(self) -> None:
        """Run the feature via the real mloda discovery + run_all path.

        Covers ``CONNECTOR_ID`` matching, ``is_valid_credentials``, the
        ``DataAccessCollection`` wiring, and the load-side wrap of native
        rows into a ``{feature_name: [row, ...]}`` column consumed by the
        stock ``PythonDictFramework``. A regression in any of these
        surfaces here.

        Pre-check that ``is_valid_credentials`` accepts
        ``valid_credentials()`` so an adapter whose canonical slot is itself
        contract-non-conformant fails here with a clear diagnostic rather
        than opaquely deep inside ``mloda.run_all``. Overlaps
        ``test_credentials_match_connector_id`` intentionally — both should
        fail loudly when the adapter is broken.

        Enforce a universal ``len(result) >= 1`` floor.
        ``expected_row_shape()`` is concrete-supplied and could in principle
        be ``lambda r: isinstance(r, list)`` — silently accepting zero rows.
        The canonical feature in every adapter is required to return at
        least one row; concrete ``expected_row_shape`` predicates then
        assert shape, not size. The ``len(...)`` call also tightens the
        contract to "result must support ``len()``" — generator-shaped
        results are intentionally not admissible, since the size floor is a
        universal invariant and ``sum(1 for _ in result)`` would consume the
        generator before the concrete's own assertions could inspect it.
        """
        cls = self.connector_reader_class()
        creds_dict = self.valid_credentials()
        connector_id = cls.CONNECTOR_ID
        assert cls.is_valid_credentials(HashableDict(creds_dict)) is True, (
            f"{cls.__name__}.valid_credentials() returned a slot that fails is_valid_credentials; "
            f"the adapter's canonical credentials are not contract-conformant. creds={creds_dict!r}"
        )
        feat = self.feature_under_test()

        result = run_query(connector_id, creds_dict[connector_id], feat)
        assert self.expected_row_shape()(result), (
            f"{cls.__name__} returned result of shape {type(result).__name__} "
            f"that failed expected_row_shape predicate. result={result!r}"
        )
        assert len(result) >= 1, (
            f"{cls.__name__} returned zero rows for the canonical feature {feat.name!r}; "
            f"adapters must seed at least one row so expected_row_shape asserts shape, not size."
        )
