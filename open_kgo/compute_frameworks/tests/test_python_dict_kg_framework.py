"""Direct unit tests for ``KgPythonDictFramework``.

The KG-aware ``PythonDictFramework`` adapter wraps native KG rows under the
requested feature name during column slicing, so the wrap-for-column-matching
strategy lives in a framework-specific layer instead of leaking into the
universal ``KgConnectorReaderBase.load``. These tests pin down the adapter's
public behaviour independently of any concrete connector.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.user import FeatureName

from open_kgo.compute_frameworks.python_dict_kg_framework import KgPythonDictFramework


def _make_framework() -> KgPythonDictFramework:
    """Construct an instance with the bare minimum mloda runtime args.

    Tracks ``ComputeFramework.__init__`` (private mloda surface): currently
    requires ``mode`` and ``children_if_root``. These tests exercise pure
    data-shaping behaviour, not parallelization or DAG wiring, so any sync
    mode and an empty children set suffice. If mloda renames either kwarg or
    adds a required one, every test here fails together — fix the helper.
    """
    return KgPythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


def test_select_data_wraps_native_rows_under_feature_name() -> None:
    """Each native row is returned as ``{feature_name: row}``."""
    fw = _make_framework()
    native_rows = [{"s": "x", "p": "y", "o": "z"}, {"s": "a", "p": "b", "o": "c"}]
    result = fw.select_data_by_column_names(native_rows, {FeatureName("my_feature")})
    assert result == [
        {"my_feature": {"s": "x", "p": "y", "o": "z"}},
        {"my_feature": {"s": "a", "p": "b", "o": "c"}},
    ]


def test_select_data_wrap_is_unconditional_even_when_feature_name_collides() -> None:
    """Wrap mirrors the prior ``KgConnectorReaderBase.load`` semantics: always wrap.

    Conditional wrapping (skip when the feature name already appears as a
    row key) would silently change behaviour for the rare collision case.
    Wrapping unconditionally preserves the historical contract.
    """
    fw = _make_framework()
    rows = [{"my_feature": "shadowed", "other": 1}]
    result = fw.select_data_by_column_names(rows, {FeatureName("my_feature")})
    assert result == [{"my_feature": {"my_feature": "shadowed", "other": 1}}]


def test_select_data_rejects_multi_feature_call() -> None:
    """Adapter mirrors the reader's single-feature contract.

    KG readers dispatch one feature at a time; passing two feature names to
    the framework's column slicer would silently pick one. Reject loudly.
    """
    fw = _make_framework()
    with pytest.raises(ValueError):
        fw.select_data_by_column_names([{"a": 1}], {FeatureName("feat_a"), FeatureName("feat_b")})


def test_select_data_returns_empty_list_for_empty_input() -> None:
    """Empty input returns ``[]``.

    The parent ``PythonDictFramework`` raises on ``[]`` because for tabular
    data an empty selection is usually a schema-discovery failure. KG
    semantics differ: a query with no matches is a legitimate outcome, so
    the adapter relaxes the parent's "empty is fatal" guard. The
    cardinality and ``column_ordering`` guards are orthogonal and still
    fire (see the dedicated tests below); this test only pins the
    relaxation on the data dimension.
    """
    fw = _make_framework()
    result = fw.select_data_by_column_names([], {FeatureName("my_feature")})
    assert result == []


def test_select_data_empty_input_still_rejects_multi_feature_call() -> None:
    """Cardinality guard fires even on ``[]`` (caller-shape bug is orthogonal to row count)."""
    fw = _make_framework()
    with pytest.raises(ValueError):
        fw.select_data_by_column_names([], {FeatureName("feat_a"), FeatureName("feat_b")})


def test_select_data_empty_input_still_rejects_column_ordering() -> None:
    """``column_ordering`` guard fires even on ``[]`` (caller-shape bug is orthogonal to row count)."""
    fw = _make_framework()
    with pytest.raises(ValueError):
        fw.select_data_by_column_names([], {FeatureName("my_feature")}, column_ordering="alphabetical")


def test_select_data_rejects_column_ordering() -> None:
    """``column_ordering`` is meaningless for a single-feature wrap; reject loudly.

    The parent ``PythonDictFramework`` honors ``column_ordering`` via
    ``identify_naming_convention``. The KG adapter bypasses that hook because
    each wrapped row has exactly one key (the feature name). Silently
    accepting the parameter would be a lie about supported semantics.
    """
    fw = _make_framework()
    with pytest.raises(ValueError):
        fw.select_data_by_column_names(
            [{"a": 1}],
            {FeatureName("my_feature")},
            column_ordering="alphabetical",
        )
