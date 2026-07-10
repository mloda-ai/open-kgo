"""Ontology data model and YAML parsing for ``OntologyRegistry``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        """True if ``relationship`` is a valid outgoing edge from ``entity_type`` (unknown types pass through)."""
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


def parse_ontology(data: Any) -> NamespaceOntology:
    """Parse a loaded YAML mapping into a ``NamespaceOntology``; raises ``ValueError`` on a malformed file."""
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
            raise ValueError(f"ontology relationship {name!r} must be a mapping declaring both 'domain' and 'range'.")
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
