"""Direct unit tests for the feature-name wrap in ``KgConnectorReaderBase.load``.

``load`` wraps native KG rows into the columnar single-column frame
``{feature_name: [row, ...]}`` that the stock ``PythonDictFramework``
(mloda >= 0.9.0) accepts as-is. These tests pin down that wrap independently
of any concrete connector; they were migrated from the retired
``KgPythonDictFramework`` adapter, which used to perform the wrap in
``transform``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.user import Feature, Options

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase


class _StubRowSource:
    """Stands in for the reader returned by ``init_reader``; serves preset rows."""

    def __init__(self, rows: Any) -> None:
        self._rows = rows

    def load_data(self, data_access: Any, features: FeatureSet) -> Any:
        return self._rows


class _WrapFakeReader(KgConnectorReaderBase):
    """Minimal concrete reader whose load path serves preset native rows.

    The class-level ``load_data`` is intentionally NOT overridden so the
    inherited ``NotImplementedError`` probe behaviour stays intact for
    discovery-walk tests that sweep concrete subclasses.
    """

    CONNECTOR_ID: ClassVar[str] = "fake_connector_for_load_wrap_tests"
    REQUIRED_KEYS: ClassVar[tuple[tuple[str, ...], ...]] = ()

    def __init__(self, rows: Any = None) -> None:
        super().__init__()
        self._rows = rows

    def init_reader(self, options: Any) -> tuple[Any, Any]:
        return _StubRowSource(self._rows), None


def _feature_set(*features: Feature) -> FeatureSet:
    fs = FeatureSet()
    for feature in features:
        fs.add(feature)
    return fs


def _single_feature_set(name: str) -> FeatureSet:
    return _feature_set(Feature(name))


def test_load_wraps_native_rows_under_feature_name() -> None:
    """Native rows become one column keyed by the feature, each cell a full row."""
    native_rows = [{"s": "x", "p": "y", "o": "z"}, {"s": "a", "p": "b", "o": "c"}]
    result = _WrapFakeReader(native_rows).load(_single_feature_set("my_feature"))
    assert result == {"my_feature": [{"s": "x", "p": "y", "o": "z"}, {"s": "a", "p": "b", "o": "c"}]}


def test_load_wrap_is_unconditional_even_when_feature_name_collides() -> None:
    """Wrap always applies, even when the feature name already appears as a row key."""
    rows = [{"my_feature": "shadowed", "other": 1}]
    result = _WrapFakeReader(rows).load(_single_feature_set("my_feature"))
    assert result == {"my_feature": [{"my_feature": "shadowed", "other": 1}]}


def test_load_empty_result_yields_schema_bearing_zero_row_column() -> None:
    """Empty results return ``{feature_name: []}``, not the schema-less ``{}`` mloda rejects."""
    result = _WrapFakeReader([]).load(_single_feature_set("my_feature"))
    assert result == {"my_feature": []}


def test_load_rejects_non_list_load_data_result() -> None:
    """A concrete drifting to a single dict return is a typed error at the base."""
    with pytest.raises(TypeError):
        _WrapFakeReader({"s": "x"}).load(_single_feature_set("my_feature"))


def test_load_rejects_non_dict_rows() -> None:
    """A concrete drifting to non-dict rows is a typed error at the base."""
    with pytest.raises(TypeError):
        _WrapFakeReader([("s", "x")]).load(_single_feature_set("my_feature"))


def test_load_projects_row_keys_for_sibling_features_in_one_load() -> None:
    """``name~key`` yields that key's values (None where a row lacks it) next to the whole-row base column."""
    rows = [{"s": "a", "n": 1}, {"s": "b"}]
    fs = _feature_set(Feature("my_feature"), Feature("my_feature~s"), Feature("my_feature~n"))
    assert _WrapFakeReader(rows).load(fs) == {"my_feature": rows, "my_feature~s": ["a", "b"], "my_feature~n": [1, None]}
    assert _WrapFakeReader([]).load(_single_feature_set("my_feature~s")) == {"my_feature~s": []}


@pytest.mark.parametrize(("name", "rows"), [("my_feature~typo", [{"s": "a"}]), ("my_feature~", [])])
def test_load_rejects_a_row_key_no_row_carries(name: str, rows: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="no row carries key"):
        _WrapFakeReader(rows).load(_single_feature_set(name))


@pytest.mark.parametrize(
    "other",
    [
        Feature("other_feature~s"),
        Feature("my_feature~n", options=Options(context={"query_text": "different"})),
        Feature("my_feature~n", options=Options(group={"variant": "b"}, context={"query_text": "q"})),
    ],
    ids=["other_base", "other_context", "other_group"],
)
def test_load_rejects_siblings_that_would_share_one_query_wrongly(other: Feature) -> None:
    """Only projections of one base feature with equal options share a load."""
    first = Feature("my_feature~s", options=Options(context={"query_text": "q"}))
    with pytest.raises(ValueError, match="exactly one feature per call"):
        _WrapFakeReader([{"s": "a", "n": 1}]).load(_feature_set(first, other))
