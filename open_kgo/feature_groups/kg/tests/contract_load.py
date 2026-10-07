"""Load-behavior contract tests: multi-feature guard, idempotence, mutation safety, e2e, identity, row counts.

One of the four concern mixins aggregated by ``KgConnectorContractBase``
(see ``kg_contract.py``). Everything here runs the reader's real load path,
mostly through ``run_query`` (``mloda.run_all`` + ``PythonDictFramework``),
so regressions in matching, validation, or the load-side feature-name wrap
surface here rather than silently passing.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import BaseInputData
from mloda.user import Credential, DataAccessCollection, Feature, Options, mloda

from open_kgo.feature_groups.kg.base import ParamReader, PythonDictFramework
from open_kgo.feature_groups.kg.tests._helpers import canonical_row_key, run_query, run_scoped_query, scoped_feature
from open_kgo.feature_groups.kg.tests.contract_adapters import KgContractAdapterBase

_SECRET = "s3cr3t-must-not-leak"


class LoadBehaviorContract(KgContractAdapterBase):
    """Contract tests for the load path of a concrete KG plugin."""

    def test_load_rejects_multi_feature_set(self) -> None:
        """Universal: ``load`` rejects FeatureSets carrying more than one base feature.

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

    def test_row_key_projection_matches_whole_rows_in_one_run(self) -> None:
        """``<feature>`` and ``<feature>~<row_key>`` in one run: whole rows, and that key's values from the same rows."""
        connector_id = self.connector_reader_class().CONNECTOR_ID
        creds = self.valid_credentials()[connector_id]
        feat = self.feature_under_test()
        rows = run_query(connector_id, creds, feat)
        assert rows, f"{self.connector_reader_class().__name__}: projection test needs >= 1 row."
        key = sorted(rows[0])[0]
        # Fresh options per feature: mloda writes the matched reader into them.
        pair = [
            Feature(name, options=Options(group=dict(feat.options.group), context=dict(feat.options.context)))
            for name in (feat.name, f"{feat.name}~{key}")
        ]
        dac = DataAccessCollection(credentials=Credential({connector_id: creds}))
        requested: list[Feature | str] = list(pair)
        partitions = mloda.run_all(requested, compute_frameworks=[PythonDictFramework], data_access_collection=dac)
        for feature, expected in zip(pair, (rows, [row.get(key) for row in rows])):
            got = [value for partition in partitions for value in partition.get(feature.name, [])]
            assert sorted(got, key=canonical_row_key) == sorted(expected, key=canonical_row_key), feature.name

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
        slot dict the caller supplied. Today ``Credential`` copies only the outer
        dict, so the reader sees the caller's slot by reference and a regression
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
        assert cls.is_valid_credentials(creds_dict) is True, (
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

    def test_feature_scoped_data_access_matches_the_global_path(self) -> None:
        """Credentials on the feature's options under the reader's name select only this reader, same rows.

        The pinned slot reaches the plan redacted, like a ``DataAccessCollection`` credential.
        """
        cls = self.connector_reader_class()
        slot = self.valid_credentials()[cls.CONNECTOR_ID]
        feat = self.feature_under_test()
        scoped = run_scoped_query(cls, slot, feat)
        assert scoped, f"{cls.__name__}: feature-scoped access returned no rows."
        assert sorted(scoped, key=canonical_row_key) == sorted(
            run_query(cls.CONNECTOR_ID, slot, feat), key=canonical_row_key
        )

        def planned_access(features: list[Feature], **kwargs: Any) -> list[str]:
            plan = mloda.explain(features, compute_frameworks=[PythonDictFramework], **kwargs)
            return [repr(step.reader_data_access[1]) for step in plan if step.reader_data_access is not None]

        pinned = planned_access([scoped_feature(cls, slot, feat)])
        assert pinned, f"{cls.__name__}: the plan carries no reader data access."
        dac = DataAccessCollection(credentials=Credential({cls.CONNECTOR_ID: slot}))
        assert pinned == planned_access([feat], data_access_collection=dac)

    def test_data_access_identity_names_the_source_and_no_secret(self) -> None:
        """The identity extenders see on INPUT_DATA_LOAD is the source (or mloda's default), never a secret."""
        cls = self.connector_reader_class()
        creds = copy.deepcopy(self.valid_credentials())
        slot = creds[cls.CONNECTOR_ID]
        slot["password"] = _SECRET
        identity = cls.data_access_identity(creds)
        assert _SECRET not in identity, identity

        source = slot.get(cls.SOURCE_SLOT) if cls.SOURCE_SLOT else None
        if source is not None and os.path.exists(str(source)):
            assert identity == str(source)
            for alias in cls._source_keys()[1:]:
                assert cls.data_access_identity({cls.CONNECTOR_ID: {alias: source}}) == str(source)
        else:
            assert identity == BaseInputData.data_access_identity(creds)

        if cls.SOURCE_SLOT:
            slot[cls.SOURCE_SLOT] = f"file://user:{_SECRET}@host/graph.ttl?token={_SECRET}#{_SECRET}"
            assert _SECRET not in cls.data_access_identity(creds)
        assert isinstance(cls.data_access_identity({cls.CONNECTOR_ID: "not-a-slot"}), str)

    def test_count_rows_matches_the_load(self) -> None:
        """``count_rows`` equals the load's row count, capped by ``result_limit``; None unless ``ROWS_FROM_SLOT``."""
        cls = self.connector_reader_class()
        creds = self.valid_credentials()
        count = cls.count_rows(creds, PythonDictFramework)
        if not cls.ROWS_FROM_SLOT:
            assert count is None
            return

        assert issubclass(cls, ParamReader) and not cls.PARAMS_MAPPING, (
            f"{cls.__name__}: ROWS_FROM_SLOT needs a ParamReader without per-call params."
        )
        rows = run_query(cls.CONNECTOR_ID, creds[cls.CONNECTOR_ID], self.feature_under_test())
        assert count == len(rows)
        assert len(rows) > 1, f"{cls.__name__}: the cap check below needs a fixture with more than one row."
        capped = copy.deepcopy(creds)
        capped[cls.CONNECTOR_ID]["result_limit"] = 1
        assert cls.count_rows(capped, PythonDictFramework) == 1
