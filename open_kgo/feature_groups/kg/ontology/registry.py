"""Ontology registry for KG connector typed traversal.

Loads a YAML ontology definition file and provides lookups for:
- valid outgoing relationship types per entity type
- domain / range constraints per relationship type
- edge validity checks

The registry is keyed by namespace so rules are reusable across any connector
or dataset that operates in the same domain (e.g. every movie KG shares the
``movie`` namespace ontology regardless of whether it lives in Neo4j, NetworkX,
or RDF). The parsed ontology shape (``NamespaceOntology`` / ``RelationshipRule``)
and the YAML parser live in the sibling ``models`` module; this module owns the
process-wide cache and namespace/path lookup.

Usage::

    namespace = OntologyRegistry.load_file("path/to/movie_ontology.yaml")
    OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by")  # True
    OntologyRegistry.is_valid_edge("movie", "Genre", "directed_by")  # False
    OntologyRegistry.get_range_type("movie", "directed_by")           # "Person"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from open_kgo.feature_groups.kg.ontology.models import NamespaceOntology, parse_ontology


class OntologyRegistry:
    """Process-wide registry of namespace ontologies loaded from YAML files.

    Two indexes are maintained for O(1) lookup in both directions:
    - ``_by_path``: resolved file path → NamespaceOntology (cache; avoids re-parse)
    - ``_by_namespace``: namespace name → NamespaceOntology (primary lookup)

    Two files declaring the same namespace raise ``ValueError`` at load time
    rather than silently overwriting, so duplicate registrations surface
    immediately.

    Call ``_clear()`` in tests to reset global state between runs. The KG
    ``conftest.py`` autouse fixture does this automatically for every KG test.
    """

    _by_path: ClassVar[dict[str, NamespaceOntology]] = {}
    _by_namespace: ClassVar[dict[str, NamespaceOntology]] = {}

    @classmethod
    def load_file(cls, path: str) -> str:
        """Load and register a YAML ontology file; return the declared namespace.

        Idempotent for the same file (cached by resolved path). Raises
        ``ValueError`` if a different file tries to register an already-known
        namespace. Raises ``ImportError`` if pyyaml is not installed.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for ontology support. Install with: uv sync --extra kg-ontology"
            ) from exc

        resolved = str(Path(path).resolve())
        if resolved in cls._by_path:
            return cls._by_path[resolved].namespace

        with open(resolved, encoding="utf-8") as f:
            raw: Any = yaml.safe_load(f)

        ontology = cls._parse(raw)

        existing = cls._by_namespace.get(ontology.namespace)
        if existing is not None:
            raise ValueError(
                f"Namespace {ontology.namespace!r} is already registered. "
                f"Each namespace must come from exactly one ontology file."
            )

        cls._by_path[resolved] = ontology
        cls._by_namespace[ontology.namespace] = ontology
        return ontology.namespace

    @classmethod
    def _parse(cls, data: Any) -> NamespaceOntology:
        """Parse a loaded YAML mapping into a ``NamespaceOntology``; see ``models.parse_ontology``."""
        return parse_ontology(data)

    @classmethod
    def get(cls, namespace: str) -> NamespaceOntology | None:
        """Return the ontology for ``namespace``, or None if not registered."""
        return cls._by_namespace.get(namespace)

    @classmethod
    def is_valid_edge(cls, namespace: str, entity_type: str, relationship: str) -> bool:
        """Return True if this edge is valid in ``namespace``.

        Returns True (pass-through) when no ontology is registered for the
        namespace so connectors without an ontology are unaffected.
        """
        ontology = cls.get(namespace)
        if ontology is None:
            return True
        return ontology.is_valid_edge(entity_type, relationship)

    @classmethod
    def get_range_type(cls, namespace: str, relationship: str) -> str | None:
        """Return the range entity type for ``relationship`` in ``namespace``."""
        ontology = cls.get(namespace)
        if ontology is None:
            return None
        return ontology.get_range_type(relationship)

    @classmethod
    def valid_next_hops(cls, namespace: str, entity_type: str) -> frozenset[str]:
        """Return valid outgoing relationship types for ``entity_type`` in ``namespace``."""
        ontology = cls.get(namespace)
        if ontology is None:
            return frozenset()
        return ontology.valid_next_hops(entity_type)

    @classmethod
    def _clear(cls) -> None:
        """Reset all cached ontologies. For test isolation only."""
        cls._by_path.clear()
        cls._by_namespace.clear()
