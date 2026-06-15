"""Ontology registry for KG connector typed traversal.

Loads a YAML ontology definition file and provides lookups for:
- valid outgoing relationship types per entity type
- domain / range constraints per relationship type
- edge validity checks

The registry is keyed by namespace so rules are reusable across any connector
or dataset that operates in the same domain (e.g. every movie KG shares the
``movie`` namespace ontology regardless of whether it lives in Neo4j, NetworkX,
or RDF).

Usage::

    namespace = OntologyRegistry.load_file("path/to/movie_ontology.yaml")
    OntologyRegistry.is_valid_edge("movie", "Movie", "directed_by")  # True
    OntologyRegistry.is_valid_edge("movie", "Genre", "directed_by")  # False
    OntologyRegistry.get_range_type("movie", "directed_by")           # "Person"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class RelationshipRule:
    """Domain and range type constraints for one relationship type."""

    domain: str
    range_type: str
    weight: float = 1.0


@dataclass
class NamespaceOntology:
    """All ontology rules for one namespace / domain."""

    namespace: str
    entity_valid_outgoing: dict[str, frozenset[str]]
    relationships: dict[str, RelationshipRule]

    def is_valid_edge(self, entity_type: str, relationship: str) -> bool:
        """Return True if ``relationship`` is a valid outgoing edge from ``entity_type``.

        Unknown entity types pass through (True) so connectors that carry
        entities not declared in the ontology are not silently broken.
        """
        valid = self.entity_valid_outgoing.get(entity_type)
        if valid is None:
            return True
        return relationship in valid

    def get_range_type(self, relationship: str) -> str | None:
        """Return the entity type at the far end of ``relationship``, or None."""
        rule = self.relationships.get(relationship)
        return rule.range_type if rule else None

    def valid_next_hops(self, entity_type: str) -> frozenset[str]:
        """Return the set of valid outgoing relationship types for ``entity_type``."""
        return self.entity_valid_outgoing.get(entity_type, frozenset())


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
        """Parse a loaded YAML mapping into a ``NamespaceOntology``.

        Raises ``ValueError`` (the same error type ``load_file`` already
        documents for duplicate namespaces) on a malformed file — a non-mapping
        top level, a missing ``namespace``, or a relationship that omits its
        ``domain`` / ``range`` — so a bad ontology surfaces a clear, typed
        error rather than a raw ``KeyError`` / ``TypeError`` from indexing.
        """
        if not isinstance(data, dict):
            raise ValueError(f"ontology file must contain a YAML mapping at the top level, got {type(data).__name__}.")
        if "namespace" not in data:
            raise ValueError("ontology file is missing the required 'namespace' key.")
        namespace = str(data["namespace"])

        entity_valid_outgoing: dict[str, frozenset[str]] = {}
        for name, spec in (data.get("entities") or {}).items():
            valid_outgoing = spec.get("valid_outgoing") if isinstance(spec, dict) else None
            entity_valid_outgoing[str(name)] = frozenset(valid_outgoing or [])

        relationships: dict[str, RelationshipRule] = {}
        for name, spec in (data.get("relationships") or {}).items():
            if not isinstance(spec, dict) or "domain" not in spec or "range" not in spec:
                raise ValueError(
                    f"ontology relationship {name!r} must be a mapping declaring both 'domain' and 'range'."
                )
            relationships[str(name)] = RelationshipRule(
                domain=str(spec["domain"]),
                range_type=str(spec["range"]),
                weight=float(spec.get("weight", 1.0)),
            )

        return NamespaceOntology(
            namespace=namespace,
            entity_valid_outgoing=entity_valid_outgoing,
            relationships=relationships,
        )

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
