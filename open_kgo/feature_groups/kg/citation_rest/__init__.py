"""Citation/scientific REST connectors (Reactome, OpenAlex citation graph, ...).

A specialised flavor of REST non-SPARQL with stable_id-based addressing,
hierarchy_depth traversal, and species/release version pinning. Inherits
``PaginationMixin``.

PROTOTYPE NOTE: ``FileFixtureCitationReader`` reads ``locator``,
``stable_id``, and ``hierarchy_depth`` at runtime. ``pagination_style``
(strict-validated), ``page_size``, ``cursor_token``, ``entity_type``,
``species_prefix``, and ``dataset_version`` are accepted at the property
layer but never read — the catalog dict is loaded eagerly and ancestor
walks are bounded by ``hierarchy_depth`` alone.
"""
