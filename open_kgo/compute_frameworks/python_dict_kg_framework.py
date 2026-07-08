"""KG-aware adapter over ``PythonDictFramework``.

Native KG rows (SPARQL bindings, Cypher rows, REST records, BFS hops, ...)
have their own keys (``s``, ``p``, ``o``, ``name``, ``ancestor``, ...) — none
of which match the user-defined feature name. ``PythonDictFramework`` (mloda
>=0.9.0) stores tabular data COLUMNAR (``dict[str, list]``) and its
``transform`` pivots a reader's native list-of-row-dicts into that shape, so
the KG columns would never match the requested feature name and every row
would be dropped during column selection.

This adapter overrides ``transform`` to instead wrap the native rows into a
single column named after the requested feature, one cell per row holding the
whole native row dict. The parent's columnar ``select_data_by_column_names``
then selects that column unchanged, and each cell is unwrapped by feature name
downstream. Other compute frameworks never see the wrap.

Pinned by ``KgConnectorFeatureGroupBase.compute_framework_rule`` so every KG
FeatureGroup runs through this adapter by default.
"""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class KgPythonDictFramework(PythonDictFramework):
    """``PythonDictFramework`` that wraps native KG rows under the requested feature name.

    KG concrete ``load_data`` implementations dispatch one feature at a time,
    so ``transform`` wraps unconditionally under that single feature name,
    regardless of whether it already appears as a key in the native row (a
    colliding key still loses direct access, matching the prior wrap). An empty
    result yields the zero-row single-column frame ``{feature_name: []}``
    (schema-bearing, not the schema-less ``{}`` mloda rejects), so a SPARQL
    ``SELECT`` with no matches, a citation lookup against an unknown
    ``stable_id``, an ``agent_memory`` query that finds nothing, or a
    ``saas_authz`` filter that excludes every tuple all return zero rows
    without raising.
    """

    def transform(self, data: Any, feature_names: set[str]) -> dict[str, list[Any]]:
        if len(feature_names) != 1:
            raise ValueError(
                f"{type(self).__name__}.transform expects exactly one feature per call "
                f"(KG concrete load_data implementations all dispatch one feature at a time); "
                f"got {sorted(str(n) for n in feature_names)}."
            )
        if isinstance(data, list):
            feature_name = str(next(iter(feature_names)))
            return {feature_name: list(data)}
        return super().transform(data, feature_names)
