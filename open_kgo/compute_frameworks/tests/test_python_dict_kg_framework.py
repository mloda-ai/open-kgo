"""Direct unit tests for ``KgPythonDictFramework``.

The KG-aware ``PythonDictFramework`` adapter wraps native KG rows into a single
column named after the requested feature during ``transform`` (native ->
columnar), so the wrap-for-column-matching strategy lives in a
framework-specific layer instead of leaking into the universal
``KgConnectorReaderBase.load``. These tests pin down the adapter's public
behaviour independently of any concrete connector.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode

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


def test_transform_wraps_native_rows_under_feature_name() -> None:
    """Native rows become one column keyed by the feature, each cell a full row."""
    fw = _make_framework()
    native_rows = [{"s": "x", "p": "y", "o": "z"}, {"s": "a", "p": "b", "o": "c"}]
    result = fw.transform(native_rows, {"my_feature"})
    assert result == {"my_feature": [{"s": "x", "p": "y", "o": "z"}, {"s": "a", "p": "b", "o": "c"}]}


def test_transform_wrap_is_unconditional_even_when_feature_name_collides() -> None:
    """Wrap always applies, even when the feature name already appears as a row key.

    Conditional wrapping (skip when the feature name is already a row key)
    would silently change behaviour for the rare collision case.
    """
    fw = _make_framework()
    rows = [{"my_feature": "shadowed", "other": 1}]
    result = fw.transform(rows, {"my_feature"})
    assert result == {"my_feature": [{"my_feature": "shadowed", "other": 1}]}


def test_transform_rejects_multi_feature_call() -> None:
    """KG readers dispatch one feature at a time; two feature names is a caller-shape bug."""
    fw = _make_framework()
    with pytest.raises(ValueError):
        fw.transform([{"a": 1}], {"feat_a", "feat_b"})


def test_transform_empty_input_yields_schema_bearing_zero_row_column() -> None:
    """Empty results return ``{feature_name: []}``, not the schema-less ``{}`` mloda rejects.

    A KG query with no matches is a legitimate zero-row outcome, but mloda
    requires a schema (at least one column), so the empty result keeps the
    feature's column with no rows.
    """
    fw = _make_framework()
    assert fw.transform([], {"my_feature"}) == {"my_feature": []}


def test_transform_empty_input_still_rejects_multi_feature_call() -> None:
    """Cardinality guard fires even on ``[]`` (caller-shape bug is orthogonal to row count)."""
    fw = _make_framework()
    with pytest.raises(ValueError):
        fw.transform([], {"feat_a", "feat_b"})
