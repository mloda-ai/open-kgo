"""Load-behavior contract tests: multi-feature guard, idempotence, mutation safety, e2e, identity, row counts.

One of the four concern mixins aggregated by ``KgConnectorContractBase``
(see ``kg_contract.py``). Everything here runs the reader's real load path,
mostly through ``run_query`` (``mloda.run_all`` + ``PythonDictFramework``),
so regressions in matching, validation, or the load-side feature-name wrap
surface here rather than silently passing.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from mloda.provider import FeatureSet
from mloda.steward import Extender, ExtenderHook, HookContext
from mloda.user import Credential, DataAccessCollection, Feature, FeatureResolutionError, Options, mloda
from mloda.user.pyarrow import PyArrowTable

from open_kgo.feature_groups.kg.base import ParamReader, PythonDictFramework, QueryReader
from open_kgo.feature_groups.kg.tests._helpers import (
    canonical_row_key,
    feature_group_of,
    run_query,
    run_scoped_query,
    scoped_feature,
)
from open_kgo.feature_groups.kg.tests.contract_adapters import KgContractAdapterBase

_SECRET = "s3cr3t-must-not-leak"


class _MatchHook(Extender):
    """INPUT_DATA_MATCH extender refusing ``group``'s source probe."""

    def __init__(self, group: type) -> None:
        self.group = group

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is not None and context.reader_class is self.group:
            raise PermissionError("refused by the test gate")
        return func(*args, **kwargs)


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
            cls.load(self.valid_credentials(), fs)

    def test_row_key_projection_matches_whole_rows_in_one_run(self) -> None:
        """``<feature>`` and ``<feature>~<row_key>`` in one run: whole rows, and that key's projected values."""
        cls = self.connector_reader_class()
        connector_id = cls.CONNECTOR_ID
        creds = self.valid_credentials()[connector_id]
        feat = self.feature_under_test()
        rows = run_query(connector_id, creds, feat)
        assert rows, f"{self.connector_reader_class().__name__}: projection test needs >= 1 row."
        key = sorted(rows[0])[0]
        pair = [Feature(name, options=feat.options) for name in (feat.name, f"{feat.name}~{key}")]
        dac = DataAccessCollection(credentials=Credential({connector_id: creds}))
        requested: list[Feature | str] = list(pair)
        partitions = mloda.run_all(requested, compute_frameworks=[PythonDictFramework], data_access_collection=dac)
        for feature, expected in zip(pair, (rows, [cls._project_value(row.get(key)) for row in rows])):
            got = [value for partition in partitions for value in partition.get(feature.name, [])]
            assert sorted(got, key=canonical_row_key) == sorted(expected, key=canonical_row_key), feature.name

    def test_row_key_projection_loads_off_python_dict(self) -> None:
        """Whole rows refuse PyArrow; a scalar ``~<row_key>`` projection loads there, or on PythonDict when listed first."""
        cls = self.connector_reader_class()
        creds = self.valid_credentials()[cls.CONNECTOR_ID]
        feat = self.feature_under_test()
        rows = run_query(cls.CONNECTOR_ID, creds, feat)
        dac = DataAccessCollection(credentials=Credential({cls.CONNECTOR_ID: creds}))
        whole = Feature(feat.name, options=feat.options)
        with pytest.raises(FeatureResolutionError, match="supports_compute_framework rejected"):
            mloda.run_all([whole], compute_frameworks=[PyArrowTable], data_access_collection=dac)
        scalar = (str, int, float)
        keys = sorted(k for k in rows[0] if all(isinstance(cls._project_value(row.get(k)), scalar) for row in rows))
        if not keys:
            pytest.skip(f"{cls.__name__}: no row key holds scalars in every row.")
        projected = Feature(f"{feat.name}~{keys[0]}", options=feat.options)
        tables = mloda.run_all([projected], compute_frameworks=[PyArrowTable], data_access_collection=dac)
        got = [value for table in tables for value in table[projected.name].to_pylist()]
        assert sorted(map(str, got)) == sorted(str(cls._project_value(row[keys[0]])) for row in rows)
        both = [PythonDictFramework, PyArrowTable]
        kept = mloda.run_all([projected], compute_frameworks=both, data_access_collection=dac)
        assert kept and all(isinstance(partition, dict) for partition in kept), [type(p) for p in kept]

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

    def _planned_identities(self, features: list[Feature | str], **kwargs: Any) -> list[str | None]:
        """The plan's ``data_access_identity`` for each step served by this connector's FeatureGroup."""
        group = feature_group_of(self.connector_reader_class())
        plan = mloda.explain(features, compute_frameworks=[PythonDictFramework], **kwargs)
        return [step.data_access_identity for step in plan if step.feature_group is group]

    def test_feature_scoped_data_access_matches_the_global_path(self) -> None:
        """A pointer under the FeatureGroup's name selects only this group: same rows, same plan identity.

        A pointer with the wrong slot key, or on a name the group does not claim, fails naming the group.
        """
        cls = self.connector_reader_class()
        group = feature_group_of(cls).get_class_name()
        slot = self.valid_credentials()[cls.CONNECTOR_ID]
        feat = self.feature_under_test()
        scoped = run_scoped_query(cls, slot, feat)
        assert scoped, f"{cls.__name__}: feature-scoped access returned no rows."
        assert sorted(scoped, key=canonical_row_key) == sorted(
            run_query(cls.CONNECTOR_ID, slot, feat), key=canonical_row_key
        )

        dac = DataAccessCollection(credentials=Credential({cls.CONNECTOR_ID: slot}))
        pinned = self._planned_identities([scoped_feature(cls, slot, feat)])
        assert pinned == self._planned_identities([feat], data_access_collection=dac)
        assert pinned == [cls.data_access_identity({cls.CONNECTOR_ID: slot})]

        wrong_slot = {**feat.options.context, group: {"wrong_key": slot}}
        for mispinned in (
            Feature(feat.name, options=Options(group=dict(feat.options.group), context=wrong_slot)),
            Feature("unclaimed__feature", options=Options(context={group: {cls.CONNECTOR_ID: slot}})),
        ):
            with pytest.raises(FeatureResolutionError, match=group):
                mloda.run_all([mispinned], compute_frameworks=[PythonDictFramework])

    def test_match_refusal_declines_both_paths(self) -> None:
        """An INPUT_DATA_MATCH refusal of the group's source probe declines the pointed and the collection path."""
        cls = self.connector_reader_class()
        slot = self.valid_credentials()[cls.CONNECTOR_ID]
        feat = self.feature_under_test()
        dac = DataAccessCollection(credentials=Credential({cls.CONNECTOR_ID: slot}))
        for request, dac_or_none in ((scoped_feature(cls, slot, feat), None), (feat, dac)):
            with pytest.raises(FeatureResolutionError, match="refused at INPUT_DATA_MATCH"):
                mloda.run_all(
                    [request],
                    compute_frameworks=[PythonDictFramework],
                    data_access_collection=dac_or_none,
                    function_extender={_MatchHook(feature_group_of(cls))},
                )

    def test_two_credentials_for_one_connector_need_a_handle(self) -> None:
        """Two different valid slots are ambiguous and ``data_access_handle`` picks one; an identical copy is one source."""
        cls = self.connector_reader_class()
        slot = self.valid_credentials()[cls.CONNECTOR_ID]
        other = {**slot, "result_limit": slot.get("result_limit", 1000) + 1}
        feat = self.feature_under_test()
        expected = sorted(run_query(cls.CONNECTOR_ID, slot, feat), key=canonical_row_key)

        def rows(handle: Any, credentials: dict[str, Any]) -> list[Any]:
            context = {**feat.options.context, **({"data_access_handle": handle} if handle else {})}
            request = Feature(feat.name, options=Options(group=dict(feat.options.group), context=context))
            dac = DataAccessCollection(credentials=credentials)
            partitions = mloda.run_all([request], compute_frameworks=[PythonDictFramework], data_access_collection=dac)
            return sorted(
                (row for partition in partitions for row in partition.get(feat.name, [])), key=canonical_row_key
            )

        two = {"a": {cls.CONNECTOR_ID: slot}, "b": {cls.CONNECTOR_ID: other}}
        with pytest.raises(ValueError, match="several sources"):
            rows(None, two)
        assert rows("a", two) == expected
        assert rows(None, {"a": {cls.CONNECTOR_ID: slot}, "b": {cls.CONNECTOR_ID: dict(slot)}}) == expected
        with pytest.raises(FeatureResolutionError, match="data_access_handle 'zzz' is unknown"):
            rows("zzz", two)
        with pytest.raises(FeatureResolutionError, match="failed validation for .data_access_handle."):
            rows(1, {"a": {cls.CONNECTOR_ID: slot}})

    def test_query_family_without_query_text_names_the_missing_option(self) -> None:
        """A query family declines a request with a missing or blank ``query_text`` and says why."""
        cls = self.connector_reader_class()
        if not issubclass(cls, QueryReader):
            pytest.skip("only query families claim through query_text")
        feat = self.feature_under_test()
        dac = DataAccessCollection(credentials=Credential(self.valid_credentials()))
        context = {k: v for k, v in feat.options.context.items() if k != "query_text"}
        for variant, reason in (
            (context, "needs the option 'query_text'"),
            ({**context, "query_text": "  "}, "query_text.*a non-empty query string"),
        ):
            bare = Feature(feat.name, options=Options(group=dict(feat.options.group), context=variant))
            with pytest.raises(FeatureResolutionError, match=reason):
                mloda.run_all([bare], compute_frameworks=[PythonDictFramework], data_access_collection=dac)

    def test_data_access_identity_names_the_source_and_no_secret(self) -> None:
        """The identity extenders and the plan see is the source (or ``{CONNECTOR_ID}``), never a secret."""
        cls = self.connector_reader_class()
        creds = copy.deepcopy(self.valid_credentials())
        slot = creds[cls.CONNECTOR_ID]
        slot["password"] = _SECRET
        identity = cls.data_access_identity(creds)
        assert _SECRET not in identity, identity

        source = next((slot[key] for key in cls._source_keys() if slot.get(key) is not None), None)
        if source is not None:
            assert identity == str(source)
            for alias in cls._source_keys()[1:]:
                assert cls.data_access_identity({cls.CONNECTOR_ID: {alias: source}}) == str(source)
        else:
            assert identity == f"{{{cls.CONNECTOR_ID}}}"

        if cls.SOURCE_SLOT:
            slot[cls.SOURCE_SLOT] = f"file://user:{_SECRET}@host/graph.ttl?token={_SECRET}#{_SECRET}"
            assert _SECRET not in cls.data_access_identity(creds)
        assert isinstance(cls.data_access_identity({cls.CONNECTOR_ID: "not-a-slot"}), str)

    def test_count_rows_matches_the_load(self) -> None:
        """``session.count_rows(step)`` equals the load's row count, capped by ``result_limit``; None unless ``ROWS_FROM_SLOT``."""
        cls = self.connector_reader_class()
        creds = self.valid_credentials()
        group = feature_group_of(cls)
        feat = self.feature_under_test()

        def count(credentials: dict[str, Any]) -> int | None:
            dac = DataAccessCollection(credentials=Credential(credentials))
            session = mloda.prepare([feat], compute_frameworks=[PythonDictFramework], data_access_collection=dac)
            (step,) = [step for step in session.resolved_plan() if step.feature_group is group]
            return session.count_rows(step)

        if not cls.ROWS_FROM_SLOT:
            assert count(creds) is None
            return

        assert issubclass(cls, ParamReader) and not cls.PARAMS_MAPPING, (
            f"{cls.__name__}: ROWS_FROM_SLOT needs a ParamReader without per-call params."
        )
        rows = run_query(cls.CONNECTOR_ID, creds[cls.CONNECTOR_ID], feat)
        assert count(creds) == len(rows)
        assert len(rows) > 1, f"{cls.__name__}: the cap check below needs a fixture with more than one row."
        capped = copy.deepcopy(creds)
        capped[cls.CONNECTOR_ID]["result_limit"] = 1
        assert count(capped) == 1
