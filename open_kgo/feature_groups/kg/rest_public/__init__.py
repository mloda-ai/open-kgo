"""REST non-SPARQL public KG connectors (OpenAlex, Reactome, STRING, ConceptNet, ...).

The locator is a base URL, ``dataset`` is null (one URL is the corpus). Family
adds: ``entity_type``, ``dataset_version``, ``user_agent``, ``rate_limit_pace``.
Inherits ``PaginationMixin`` (cursor / page / cursorMark / etc.).

PROTOTYPE NOTE: the only concrete plugin is ``FileFixtureRestReader``, which
walks ``page_*.json`` files on disk. ``pagination_style`` is strict-validated
against the 7-value enum but the page walker hardcodes cursor-style
termination on ``meta.next_cursor`` and does not switch on the enum value.
``rate_limit_pace``, ``user_agent``, ``entity_type``, and ``dataset_version``
are accepted at the property layer and never read at runtime — there is no
real HTTP client to apply them to.
"""
