"""Family base for code / build / SBOM KG connectors."""

from __future__ import annotations

from typing import Any, ClassVar

from open_kgo.feature_groups.kg.base import KgConnectorFeatureGroupBase, ParamReader
from open_kgo.feature_groups.kg.mixins import EntityFilterParamMixin, TraversalMixin
from open_kgo.feature_groups.kg.spec import property_spec


_FAMILY_PROPERTIES: dict[str, Any] = {
    "manifest_path": property_spec(
        (
            "Path to the manifest/database/SBOM artifact. This family deliberately keys on "
            "manifest_path instead of the shared 'locator' slot (a richer address paired with "
            "commit_sha/branch/language_code); 'locator' is still accepted as a fallback. See the "
            "DESIGN NOTE in this package's __init__.py."
        ),
    ),
    "commit_sha": property_spec(
        "Source commit SHA the artifact was produced from.",
    ),
    "branch": property_spec(
        "Source branch the artifact was produced on.",
    ),
    "language_code": property_spec(
        "Language code (e.g. 'java', 'python') for language-scoped artifacts.",
    ),
}


class CodeBuildReader(
    TraversalMixin,
    EntityFilterParamMixin,
    ParamReader,
    family_properties=_FAMILY_PROPERTIES,
):
    # Declared source-slot rename (issue #18, enforced per issue #21): this
    # family keys its address on manifest_path; 'locator' stays advertised as
    # a fallback. See the DESIGN NOTE in this package's __init__.py and the
    # "Source-slot convention" in kg/base.py.
    SOURCE_SLOT: ClassVar[str | None] = "manifest_path"

    # Honest surface (option 3, see base.py): VCS/build provenance the SBOM
    # concretes never consult (they parse manifest_path/locator), reserved for a
    # provenance-aware concrete.
    _WAIVED_UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"commit_sha", "branch", "language_code"})


class CodeBuildFeatureGroup(KgConnectorFeatureGroupBase):
    """Abstract family base; each concrete pins ``READER_CLASS``."""
