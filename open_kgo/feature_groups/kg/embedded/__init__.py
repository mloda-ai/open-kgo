"""Embedded / in-memory graph KG connectors (NetworkX, igraph, KuzuDB embedded, ...).

The credential key is ``locator`` and it carries a filesystem path (no URL;
the embedded family has no network endpoint). The family base declares
``REQUIRED_KEYS = ()`` because pure object-reference cases (NetworkX object
already in memory) are valid; concrete plugins set ``REQUIRED_KEYS``
explicitly when their backend needs a path.

PROTOTYPE NOTE: ``read_only`` and ``max_threads`` are advisory-only. The
concrete ``NetworkxEmbeddedReader`` validates them at the property layer
but never enforces them at runtime; NetworkX has no read-only mode and
the reader is single-threaded.
"""
