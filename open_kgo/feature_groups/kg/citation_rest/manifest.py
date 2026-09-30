"""mloda entry-point manifest for the citation_rest family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.citation_rest.file_fixture_citation import FileFixtureCitationFeatureGroup
from open_kgo.feature_groups.kg.citation_rest.paginated_citation import PaginatedCitationFeatureGroup

FEATURE_GROUPS = (FileFixtureCitationFeatureGroup, PaginatedCitationFeatureGroup)
