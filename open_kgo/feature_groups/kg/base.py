"""Facade for the KG connector base layer.

The universal base layer used to live in this module as one large file; it is
now split by concern and re-exported here, so existing import sites
(``from open_kgo.feature_groups.kg.base import ...``) keep working unchanged:

- ``reader_base.py`` owns ``LoadContext``, the property-mapping composition
  helpers (``compose_property_mapping`` / ``narrow_property_mapping``), and
  ``KgConnectorReaderBase``. The conceptual rules referenced across the
  package as "see base.py" (the "Honest credential surface" rule and the
  "Source-slot convention") are documented on that module's docstring.
- ``readers.py`` owns the two per-call input flavors a family base picks
  between: ``QueryReader`` (query string) and ``ParamReader`` (typed param
  dict on ``PARAMS_MAPPING``).
- ``feature_group.py`` owns ``KgConnectorFeatureGroupBase``, the thin
  FeatureGroup that delegates to a reader.
- ``spec.py`` wraps mloda core's ``property_spec`` builder that spec dicts
  are authored through.
"""

from open_kgo.feature_groups.kg.feature_group import KgConnectorFeatureGroupBase
from open_kgo.feature_groups.kg.reader_base import (
    KgConnectorReaderBase,
    LoadContext,
    _collect_kg_known_keys,
    compose_property_mapping,
    narrow_property_mapping,
)
from open_kgo.feature_groups.kg.readers import ParamReader, QueryReader

__all__ = [
    "KgConnectorFeatureGroupBase",
    "KgConnectorReaderBase",
    "LoadContext",
    "ParamReader",
    "QueryReader",
    "_collect_kg_known_keys",
    "compose_property_mapping",
    "narrow_property_mapping",
]
