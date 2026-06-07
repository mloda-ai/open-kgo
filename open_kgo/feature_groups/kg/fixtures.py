"""Shared resource-cache helpers for KG concrete readers.

File-backed concretes (rdflib, dbt manifest, CycloneDX SBOM, citation, REST
page-directory, NetworkX memory, in-process tuple store) would otherwise pay
a full disk read + parse on every ``load_data`` call. A data provider running
N features through a single ``mloda.run_all`` therefore pays N parses against
the same source artifact. This module centralises the
parse-once / share-many pipeline so every file-backed concrete routes
through one cache and the connector-lifecycle contract stays consistent.

Three loaders are provided:

- ``load_json_fixture(connector_id, locator)`` — open + ``json.load`` + dict
  shape check, memoised by ``(absolute path, mtime_ns)``. Used by the JSON
  artefacts: dbt manifest, CycloneDX SBOM, citation catalog, REST page
  files, NetworkX memory store, in-process tuple store. The returned dict
  is shared across calls; callers MUST treat it as read-only (see
  ``KgConnectorReaderBase`` connection-lifecycle docstring).
- ``load_rdf_graph(connector_id, locator)`` — parse Turtle/N-Triples/RDF-XML
  into ``rdflib.Graph``, memoised by ``(absolute path, mtime_ns)``. The
  default Memory store's ``close()`` is a no-op, so the shared graph
  survives the contract test that closes ``connect()``'s return value
  (verified empirically against rdflib 7.x).
- ``load_kuzu_database(connector_id, locator)`` — open a ``kuzu.Database``
  directory, memoised by ``absolute path`` only. Mtime keying does NOT
  apply: Kuzu mutates the database directory as it runs its own queries,
  so an mtime-keyed cache would invalidate after every internal write
  and defeat the whole point. Callers therefore MUST treat the database
  as a long-lived process-level handle; external mutations to the dir
  will not refresh the cache. Per-call ``kuzu.Connection`` instances are
  cheap and remain caller-owned.

The cache lives for the process lifetime (``lru_cache(maxsize=64)``).
Test suites that build fresh fixtures in temp dirs and rely on cache-key
freshness should call the ``cache_clear()`` method on the underlying inner
loader between tests; mtime-keyed JSON / RDF loaders naturally invalidate
when the underlying file changes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# SAXException is used purely as a parent class for narrowing the
# ``except`` list in ``_read_rdf_graph_cached``; no SAX parser is
# instantiated here. Bandit's B406 fires on any ``xml.sax`` import, so
# the nosec comment documents the intent.
from xml.sax import SAXException  # nosec B406

import rdflib
import rdflib.exceptions

from open_kgo.feature_groups.kg.errors import FixtureLoadError


class _FixtureLoadProblem(Exception):
    """Internal signal carrying the connector-agnostic ``reason`` string.

    Raised inside cached IO functions (which cannot embed ``connector_id``
    in their cache key without bloating the cache); the outer wrapper
    catches it and re-raises as ``FixtureLoadError`` with the connector_id
    attached to the message. Never escapes this module.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def copy_cached_row(value: Any) -> Any:
    """Shallow-copy a row taken from a shared cached fixture before handing it out.

    The JSON / RDF loaders memoise and SHARE their parsed objects across
    calls (see module docstring), so a connector that appended a row dict
    by reference would let a downstream consumer mutate the cache and
    poison every subsequent load. Connectors route each emitted row through
    here so the cache stays read-only at the row level. Non-dict values (a
    cached scalar or list a malformed fixture produced) are passed through
    unchanged — there is nothing to alias-protect and the connector return
    contract already tolerates them.
    """
    return {**value} if isinstance(value, dict) else value


def _rejected_scheme(locator: Any) -> str | None:
    """Return the rejected scheme (lowercase) if ``locator`` is remote, else ``None``.

    Single source of truth for the file-backed scheme rule: only local
    file paths or ``file://`` URLs are admissible (mirrors PR #7's
    rdflib guard). Single-letter "schemes" are Windows drive letters
    (e.g. ``C:\\path``); IANA-registered URL schemes are 2+ chars, so we
    only reject schemes longer than one character.

    Returning the rejected scheme rather than raising lets two callers
    surface two different typed errors from the same rule: the file
    loaders below raise ``FixtureLoadError``; ``RdfLibSparqlReader``
    preserves its pre-existing ``ValueError`` contract for the
    no-locator carve-out without duplicating the parsing logic.
    """
    scheme = urlparse(str(locator)).scheme.lower()
    if scheme not in ("", "file") and len(scheme) > 1:
        return scheme
    return None


def _validate_local_locator(connector_id: str, locator: Any) -> Path:
    """Reject remote schemes and return the locator as a ``Path``."""
    locator_str = str(locator)
    bad = _rejected_scheme(locator)
    if bad is not None:
        raise FixtureLoadError(
            connector_id,
            locator_str,
            f"locator scheme {bad!r} is not permitted; only local file paths or file:// URLs are allowed.",
        )
    return Path(locator_str)


def load_json_fixture(connector_id: str, locator: Any) -> dict[str, Any]:
    """Resolve ``locator`` to a local file, read it as UTF-8 JSON, validate top-level shape.

    All failure modes (remote URI, missing file, malformed JSON, non-dict
    top level) raise ``FixtureLoadError`` (a subclass of
    ``InvalidCredentialShape``) carrying the connector_id and locator so
    the typed-error contract holds at ``connect()`` time. Mtime-keyed
    caching means callers can invoke this on every ``load_data`` without
    re-reading disk; the returned dict is shared across calls so callers
    MUST treat it as read-only (shallow-copy any row appended into a
    result list — see ``FileFixtureCitationReader.load_data``).
    """
    path = _validate_local_locator(connector_id, locator)
    locator_str = str(locator)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise FixtureLoadError(connector_id, locator_str, f"cannot stat locator path: {exc}") from exc
    abs_path = str(path.resolve())
    try:
        return _read_json_cached(abs_path, mtime_ns)
    except _FixtureLoadProblem as exc:
        raise FixtureLoadError(connector_id, abs_path, exc.reason) from exc.__cause__


@lru_cache(maxsize=64)
def _read_json_cached(abs_path: str, mtime_ns: int) -> dict[str, Any]:
    """Open, parse, and shape-check a JSON file; results memoised by (path, mtime).

    Raises ``_FixtureLoadProblem`` (not ``FixtureLoadError``) so the
    cache key can stay connector-agnostic; the outer wrapper attaches
    the connector_id to the user-visible error. lru_cache memoises only
    successful returns, so a malformed fixture is re-attempted on every
    call (loud failure rather than cached failure).
    """
    try:
        with open(abs_path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise _FixtureLoadProblem(f"cannot open locator file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _FixtureLoadProblem(f"locator is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _FixtureLoadProblem(f"locator JSON must be an object at the top level, got {type(data).__name__}.")
    return data


def load_rdf_graph(connector_id: str, locator: Any) -> rdflib.Graph:
    """Parse an RDF file into an ``rdflib.Graph``; memoised by ``(path, mtime_ns)``.

    A real Turtle file is 1-100MB and rdflib's parse pass is the expensive
    step (syntax + namespace + serialization); the graph object itself is
    cheap to reuse. Callers MUST treat the returned graph as read-only:
    ``rdflib.Graph.close()`` is a no-op on the default Memory store
    (verified against rdflib 7.x) so the contract-test path that closes
    ``connect()``'s return survives the cache; ``add`` / ``remove`` calls
    would corrupt subsequent loads in the same process.
    """
    path = _validate_local_locator(connector_id, locator)
    locator_str = str(locator)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise FixtureLoadError(connector_id, locator_str, f"cannot stat locator path: {exc}") from exc
    abs_path = str(path.resolve())
    try:
        return _read_rdf_graph_cached(abs_path, mtime_ns)
    except _FixtureLoadProblem as exc:
        raise FixtureLoadError(connector_id, abs_path, exc.reason) from exc.__cause__


@lru_cache(maxsize=64)
def _read_rdf_graph_cached(abs_path: str, mtime_ns: int) -> rdflib.Graph:
    """Parse ``abs_path`` with rdflib; memoised by ``(path, mtime_ns)``.

    Wraps rdflib's parse errors as ``_FixtureLoadProblem`` so the outer
    helper can attach the connector_id. rdflib auto-detects the format
    from the file extension; an unparseable file surfaces as a typed
    ``FixtureLoadError`` rather than a raw rdflib exception. The except
    list enumerates the exception hierarchies rdflib actually raises
    from parse failures (Turtle/N3 ``BadSyntax`` inherits from
    ``SyntaxError``; RDF/XML uses ``xml.sax.SAXException``; rdflib's
    own ``ParserError`` lives under ``rdflib.exceptions.Error``;
    plus ``ValueError`` / ``UnicodeDecodeError`` for bad bytes). Bare
    ``except Exception`` was previously used; narrowing it stops the
    block from masking programmer errors like ``AttributeError``.
    """
    graph = rdflib.Graph()
    try:
        graph.parse(abs_path)
    except OSError as exc:
        raise _FixtureLoadProblem(f"cannot open RDF locator file: {exc}") from exc
    except (rdflib.exceptions.Error, SyntaxError, SAXException, ValueError, UnicodeDecodeError) as exc:
        raise _FixtureLoadProblem(f"locator is not parseable as RDF: {exc}") from exc
    return graph


def load_kuzu_database(connector_id: str, locator: Any) -> Any:
    """Open a ``kuzu.Database`` for ``locator`` and cache the handle by path.

    Returns ``Any`` to keep the kuzu dependency optional at import time
    (the import is local; tests that don't exercise kuzu don't need it).
    Cache key is the absolute path WITHOUT mtime: Kuzu mutates its own
    database directory as it runs queries, so mtime keying would
    invalidate after every internal write and re-open the FDs we're
    trying to keep alive (the native FD leak this cache exists to
    prevent). External mutations to the dir therefore will NOT refresh
    the cache; callers that need to swap the underlying database between
    test cases must call ``_open_kuzu_database_cached.cache_clear()``.

    Caller-side contract: the returned ``kuzu.Database`` is shared and
    process-lived; callers MUST NOT call ``close()`` on it (and SHOULD
    build a fresh ``kuzu.Connection(db)`` per call, since Connections
    are cheap and remain caller-owned — closing a Connection does not
    poison the cached Database).
    """
    path = _validate_local_locator(connector_id, locator)
    locator_str = str(locator)
    # Explicit exists() check: kuzu.Database(path) silently creates the
    # database directory if it does not exist, so without this guard a
    # typoed locator would surface as an empty fresh DB instead of a
    # typed FixtureLoadError.
    if not path.exists():
        raise FixtureLoadError(connector_id, locator_str, f"kuzu locator path does not exist: {locator_str!r}")
    abs_path = str(path.resolve())
    try:
        return _open_kuzu_database_cached(abs_path)
    except _FixtureLoadProblem as exc:
        raise FixtureLoadError(connector_id, abs_path, exc.reason) from exc.__cause__


@lru_cache(maxsize=64)
def _open_kuzu_database_cached(abs_path: str) -> Any:
    """Open ``kuzu.Database(abs_path)``; memoised by path only.

    The kuzu import is deferred to call time so test environments without
    kuzu installed can still import this module (the JSON / RDF loaders
    above have no kuzu dependency).
    """
    import kuzu

    try:
        return kuzu.Database(abs_path)
    except Exception as exc:
        raise _FixtureLoadProblem(f"cannot open kuzu database: {exc}") from exc
