"""KG-aware adapter over ``PythonDictFramework``.

Native KG rows (SPARQL bindings, Cypher rows, REST records, BFS hops, ...)
have their own keys (``s``, ``p``, ``o``, ``name``, ``ancestor``, ...) — none
of which match the user-defined feature name. ``PythonDictFramework`` matches
columns by feature-name presence in each row (see
``select_data_by_column_names`` in
``mloda_plugins.compute_framework.base_implementations.python_dict``), so an
unmatched feature would yield ``[{}, {}, ...]`` and silently lose every row.

The wrap that satisfies that contract used to live on the universal
``KgConnectorReaderBase.load``, which made the framework-matching strategy
leak into every reader regardless of compute framework. This adapter moves
the wrap into a ``PythonDictFramework`` subclass that owns it as an internal
concern: ``load_data`` returns native KG rows; this framework wraps them as
``{feature_name: row}`` immediately before column slicing. Other compute
frameworks never see the wrap.

Pinned by ``KgConnectorFeatureGroupBase.compute_framework_rule`` so every KG
FeatureGroup runs through this adapter by default.
"""

from __future__ import annotations

from typing import Any, Optional

from mloda.user import FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class KgPythonDictFramework(PythonDictFramework):
    """``PythonDictFramework`` that wraps native KG rows under the requested feature name.

    Mirrors the prior universal-base behavior: every row is wrapped as
    ``{feature_name: row}`` before column slicing, regardless of whether
    ``feature_name`` already appears as a key in the row. Wrapping
    unconditionally preserves the historical contract, a row whose own keys
    happen to collide with the feature name still loses access to the
    colliding key, but that was already the case under the prior wrap and
    changing it here would be a silent semantic shift.

    The parent's ``identify_naming_convention`` hook (which supports the
    ``feature_name~suffix`` multi-column pattern) is intentionally bypassed:
    after the wrap, every row has exactly one key, the feature name, so
    convention-based suffix matching has nothing to expand against. Native KG
    rows have their own keys (``s``, ``p``, ``o``, ...) that never match the
    user-defined feature name with or without suffixes, so this restriction
    matches existing reality. KG features that need the suffix pattern would
    need a different adapter; surfacing that explicitly is preferable to
    silent loss.

    Empty results are returned as ``[]`` instead of raising. The
    parent ``PythonDictFramework`` treats ``[]`` as fatal because for tabular
    data an empty selection is usually a schema-discovery failure (no rows
    means no column types to infer, no naming convention to expand). KG
    semantics are different: a SPARQL ``SELECT`` with no matches, a citation
    lookup against an unknown ``stable_id``, an ``agent_memory`` query that
    finds nothing, or a ``saas_authz`` filter that excludes every tuple all
    legitimately return zero rows. Forcing those onto the parent's
    "empty is fatal" path made ``mloda.run_all`` hostile to data users with
    real empty-result queries (the prior workaround had to bypass the run
    entirely). The cardinality and ``column_ordering`` guards still fire
    even on empty data, because those are caller-shape bugs that are
    orthogonal to row count.
    """

    def select_data_by_column_names(
        self,
        data: list[dict[str, Any]],
        selected_feature_names: set[FeatureName],
        column_ordering: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if len(selected_feature_names) != 1:
            raise ValueError(
                f"{type(self).__name__}.select_data_by_column_names expects exactly one feature per call "
                f"(KG concrete load_data implementations all dispatch one feature at a time); "
                f"the reader's ``load`` should reject this earlier, this branch is defense-in-depth. "
                f"Got {sorted(str(n) for n in selected_feature_names)}."
            )

        if column_ordering is not None:
            raise ValueError(
                f"{type(self).__name__} does not honor column_ordering "
                f"(single-feature wrap has no ordering to apply); got {column_ordering!r}."
            )

        if not data:
            return []

        feature_name = str(next(iter(selected_feature_names)))
        return [{feature_name: row} for row in data]
