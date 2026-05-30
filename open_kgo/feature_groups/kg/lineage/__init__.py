"""Metadata / lineage KG connectors (DataHub, OpenMetadata, Atlas, dbt manifest, ...).

Hidden-KG family. Locator points at a metadata service or file artifact;
``asset_urn`` is the addressing key, ``lineage_direction`` and depth properties
come from ``TraversalMixin``.

PROTOTYPE NOTE: ``DbtManifestReader`` is the deepest exerciser of
``TraversalMixin`` (honors ``lineage_direction`` + ``upstream_depth`` +
``downstream_depth``), but ``entity_type``, ``relationship_type``, and
``expand_paths`` are accepted at the property layer and never read at
runtime — the manifest walk is type-agnostic and follows every edge in
``parent_map`` / ``child_map`` regardless of these filters.
"""
