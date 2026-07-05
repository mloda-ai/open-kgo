"""Shared resource-cache helpers for KG concrete readers.

File-backed concretes (rdflib, dbt manifest, CycloneDX SBOM, citation, REST
page-directory, NetworkX memory, in-process tuple store) would otherwise pay
a full disk read + parse on every ``load_data`` call. A data provider running
N features through a single ``mloda.run_all`` therefore pays N parses against
the same source artifact. This module centralises the
parse-once / share-many pipeline so every file-backed concrete routes
through one cache and the connector-lifecycle contract stays consistent.

Four loaders are provided:

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
- ``load_oxigraph_store(connector_id, locator)`` — parse the same RDF
  serialisations into an in-memory ``pyoxigraph.Store``, memoised by
  ``(absolute path, mtime_ns)``. The oxigraph sibling of ``load_rdf_graph``;
  the pyoxigraph import is deferred to call time so the module imports
  without the ``kg-rdf`` extra. The returned store is shared across calls;
  callers run read-only SPARQL against it and MUST NOT mutate it.
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
from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeVar
from urllib.parse import urlparse

# SAXException is used purely as a parent class for narrowing the
# ``except`` list in ``_read_rdf_graph_cached``; no SAX parser is
# instantiated here. Bandit's B406 fires on any ``xml.sax`` import, so
# the nosec comment documents the intent.
from xml.sax import SAXException  # nosec B406

from open_kgo.feature_groups.kg.errors import FixtureLoadError

# rdflib is imported lazily inside ``_read_rdf_graph_cached`` (like the
# deferred pyoxigraph / kuzu loaders) so non-RDF families can import this
# shared module without the ``kg-rdf`` extra. Only the type checker needs the
# symbol here; ``from __future__ import annotations`` keeps the ``rdflib.Graph``
# return annotations as strings at runtime.
if TYPE_CHECKING:
    import rdflib


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


def copy_cached_rows(rows: Iterable[Any], limit: int) -> list[dict[str, Any]]:
    """Copy up to ``limit`` rows from a shared cached fixture into a fresh list.

    The "route each emitted row through ``copy_cached_row``, stop at the
    cap" loop used to be spelled inline by every flat-list fixture reader
    (REST page walks, CycloneDX components). Centralising it keeps the
    cache-immutability enforcement point impossible to forget when a new
    fixture-backed concrete lands: the copy and the bound travel together.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        # Bound check before the append so a non-positive limit yields [] (the
        # callers' result_limit arithmetic keeps it >= 1 today, but a shared
        # primitive should not over-emit on the edge).
        if len(out) >= limit:
            break
        out.append(copy_cached_row(row))
    return out


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


_T = TypeVar("_T")


def _load_cached(connector_id: str, locator: Any, cached_loader: Callable[..., _T], *, mtime_keyed: bool = True) -> _T:
    """Shared validate -> stat -> resolve -> cached-call -> error-translation pipeline.

    Every public loader below is this wrapper plus a format-specific cached
    inner function. ``mtime_keyed=True`` (JSON / RDF / oxigraph) stats the
    path and calls ``cached_loader(abs_path, mtime_ns)`` so the cache
    invalidates when the file changes; ``mtime_keyed=False`` (kuzu) calls
    ``cached_loader(abs_path)`` so the handle survives the backend's own
    directory writes (see ``load_kuzu_database`` for why). Failure modes:
    a remote scheme or unstattable path raises ``FixtureLoadError`` against
    the caller-supplied locator string; a ``_FixtureLoadProblem`` from the
    inner loader is re-raised as ``FixtureLoadError`` against the resolved
    absolute path (matching the cache key the problem occurred under).
    """
    path = _validate_local_locator(connector_id, locator)
    locator_str = str(locator)
    key_args: tuple[int, ...] = ()
    if mtime_keyed:
        try:
            key_args = (path.stat().st_mtime_ns,)
        except OSError as exc:
            raise FixtureLoadError(connector_id, locator_str, f"cannot stat locator path: {exc}") from exc
    abs_path = str(path.resolve())
    try:
        return cached_loader(abs_path, *key_args)
    except _FixtureLoadProblem as exc:
        raise FixtureLoadError(connector_id, abs_path, exc.reason) from exc.__cause__


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
    return _load_cached(connector_id, locator, _read_json_cached)


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
    would corrupt subsequent loads in the same process. The rdflib import is
    deferred to call time (like the oxigraph / kuzu loaders) so this module
    imports without the ``kg-rdf`` extra.
    """
    return _load_cached(connector_id, locator, _read_rdf_graph_cached)


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
    block from masking programmer errors like ``AttributeError``. The rdflib
    import is deferred here so non-RDF families can import this module without
    the ``kg-rdf`` extra.
    """
    import rdflib
    import rdflib.exceptions

    graph = rdflib.Graph()
    try:
        graph.parse(abs_path)
    except OSError as exc:
        raise _FixtureLoadProblem(f"cannot open RDF locator file: {exc}") from exc
    except (rdflib.exceptions.Error, SyntaxError, SAXException, ValueError, UnicodeDecodeError) as exc:
        raise _FixtureLoadProblem(f"locator is not parseable as RDF: {exc}") from exc
    return graph


_OXIGRAPH_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".ttl": "TURTLE",
    ".nt": "N_TRIPLES",
    ".n3": "N3",
    ".trig": "TRIG",
    ".nq": "N_QUADS",
    ".rdf": "RDF_XML",
    ".xml": "RDF_XML",
    ".jsonld": "JSON_LD",
}


def load_oxigraph_store(connector_id: str, locator: Any) -> Any:
    """Parse an RDF file into an in-memory ``pyoxigraph.Store``; memoised by ``(path, mtime_ns)``.

    The oxigraph sibling of ``load_rdf_graph``: a real Turtle file is
    1-100MB and oxigraph's parse pass is the expensive step, so a
    100-feature ``mloda.run_all`` pays one parse instead of one hundred.
    Returns ``Any`` to keep the pyoxigraph dependency optional at import
    time (the import is deferred to call time, mirroring
    ``load_kuzu_database``), so test environments without the ``kg-rdf``
    extra can still import this module. The returned store is shared
    across calls; callers run read-only SPARQL queries against it and MUST
    NOT mutate it.
    """
    return _load_cached(connector_id, locator, _read_oxigraph_store_cached)


@lru_cache(maxsize=64)
def _read_oxigraph_store_cached(abs_path: str, mtime_ns: int) -> Any:
    """Load ``abs_path`` into a ``pyoxigraph.Store``; memoised by ``(path, mtime_ns)``.

    The RDF serialisation format is selected from the file extension
    (defaulting to Turtle) since oxigraph's loader needs it explicitly,
    unlike rdflib's content auto-detection. A suffix outside the known map
    keeps the documented Turtle default, but the parse-failure message then
    names the assumed format and the unknown suffix so a misleading "not
    parseable as RDF" on e.g. a ``.owl`` file is self-explaining. The file
    object is passed to ``Store.load`` directly (pyoxigraph 0.5.x accepts
    binary I/O objects) so the 1-100MB artifacts the module docstring sizes
    are streamed rather than buffered via ``f.read()``. pyoxigraph raises
    the builtin ``SyntaxError`` on malformed RDF (verified against
    pyoxigraph 0.5.x); ``ValueError`` / ``UnicodeDecodeError`` cover bad
    bytes. The pyoxigraph import is deferred so the module imports without
    the ``kg-rdf`` extra.
    """
    import pyoxigraph

    suffix = Path(abs_path).suffix.lower()
    fmt_name = _OXIGRAPH_FORMAT_BY_SUFFIX.get(suffix)
    if fmt_name is None:
        fmt_name = "TURTLE"
        format_note = (
            f"parsed as TURTLE based on suffix {suffix!r} which is not in the known suffix map "
            f"{sorted(_OXIGRAPH_FORMAT_BY_SUFFIX)}"
        )
    else:
        format_note = f"parsed as {fmt_name} based on suffix {suffix!r}"
    rdf_format = getattr(pyoxigraph.RdfFormat, fmt_name)
    store = pyoxigraph.Store()
    try:
        with open(abs_path, "rb") as f:
            store.load(f, format=rdf_format)
    except OSError as exc:
        raise _FixtureLoadProblem(f"cannot open RDF locator file: {exc}") from exc
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise _FixtureLoadProblem(f"locator is not parseable as RDF ({format_note}): {exc}") from exc
    return store


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
    return _load_cached(connector_id, locator, _open_kuzu_database_cached, mtime_keyed=False)


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
