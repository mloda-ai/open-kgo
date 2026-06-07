"""Compute frameworks for open-kgo.

Currently holds ``KgPythonDictFramework`` (``python_dict_kg_framework.py``), the
KG-aware ``PythonDictFramework`` adapter that wraps native KG rows as
``{feature_name: row}`` before column slicing. It is pinned by
``open_kgo.feature_groups.kg.base.KgConnectorFeatureGroupBase.compute_framework_rule``
so every KG FeatureGroup runs through it by default.
"""
