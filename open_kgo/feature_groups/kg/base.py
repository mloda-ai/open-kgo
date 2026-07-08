"""Facade for the KG connector base layer.

The universal base layer used to live in this module as one large file; it is
now split by concern and re-exported here, so existing import sites
(``from open_kgo.feature_groups.kg.base import ...``) keep working unchanged:

- ``reader_base.py`` owns ``LoadContext`` and ``KgConnectorReaderBase``. The
  conceptual rules referenced across the package as "see base.py" (the
  "Honest credential surface" rule and the "Source-slot convention") are
  documented on that module's docstring.
- ``composition.py`` owns the property-mapping composition helpers
  (``compose_property_mapping`` / ``narrow_property_mapping``) that
  ``reader_base.py``'s family-base composition delegates to.
- ``readers.py`` owns the two per-call input flavors a family base picks
  between: ``QueryReader`` (query string) and ``ParamReader`` (typed param
  dict on ``PARAMS_MAPPING``).
- ``feature_group.py`` owns ``KgConnectorFeatureGroupBase``, the thin
  FeatureGroup that delegates to a reader.
- ``spec.py`` wraps mloda core's ``property_spec`` builder that spec dicts
  are authored through.

``PythonDictFramework`` (the stock mloda compute framework every KG
FeatureGroup pins) is re-exported here as well, so consumers avoid the deep
``mloda_plugins.compute_framework.base_implementations...`` path; mloda
offers no shorter public import for it.
"""

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from open_kgo.feature_groups.kg.composition import compose_property_mapping, narrow_property_mapping
from open_kgo.feature_groups.kg.feature_group import KgConnectorFeatureGroupBase
from open_kgo.feature_groups.kg.reader_base import (
    KgConnectorReaderBase,
    LoadContext,
    _collect_kg_known_keys,
)
from open_kgo.feature_groups.kg.readers import ParamReader, QueryReader

__all__ = [
    "KgConnectorFeatureGroupBase",
    "KgConnectorReaderBase",
    "LoadContext",
    "ParamReader",
    "PythonDictFramework",
    "QueryReader",
    "_collect_kg_known_keys",
    "compose_property_mapping",
    "narrow_property_mapping",
]
