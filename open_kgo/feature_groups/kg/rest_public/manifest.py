"""mloda entry-point manifest for the rest_public family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.rest_public.file_fixture_paged_rest import FileFixturePagedRestFeatureGroup
from open_kgo.feature_groups.kg.rest_public.file_fixture_rest import FileFixtureRestFeatureGroup

FEATURE_GROUPS = (FileFixturePagedRestFeatureGroup, FileFixtureRestFeatureGroup)
