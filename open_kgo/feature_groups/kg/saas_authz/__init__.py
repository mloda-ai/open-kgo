"""SaaS / authz / wiki KG connectors (Microsoft Graph, OpenFGA, SpiceDB, Notion, ...).

Hidden-KG family. ``tenant`` instead of ``dataset`` (six observed shapes:
subdomain, instance_url, store_id, token-implicit, wiki_url, vault_path).
``consistency_token`` and ``consistency_mode`` for Zanzibar-style systems.
``expand_paths`` for OData / permission-tree expansion.

PROTOTYPE NOTE: ``InProcessTupleStoreReader`` is pinned to a canonical
``fixtures/tuples.json`` (ships ``tenant_a``); the ``locator`` credential
slot is dropped on this concrete because the fixture is baked into the
class. A ``tenant`` outside the closed ``allowed_values`` enum is rejected
at ``is_valid_credentials`` time (matcher-safe); direct callers that bypass
validation hit ``UnknownTenantError`` at ``connect()``. An unreadable /
malformed fixture raises ``FixtureLoadError``.
"""
