"""RDF / SPARQL family of KG connectors.

Family-base properties (beyond the universal layer) follow Version B of
``docs/kg-connector-base-classes.md``: ``default_graph_uris``,
``named_graph_uris``, ``update_endpoint``, ``result_format``, plus
``reasoning_profile`` from ``InferenceMixin``.

PROTOTYPE NOTE: ``RdfLibSparqlReader`` validates property *shape* only for
several keys. ``result_format`` is strict-validated against the SPARQL
results-media-type enum but the reader always returns Python dicts
regardless of the value. ``default_graph_uris``, ``named_graph_uris``, and
``update_endpoint`` are accepted but never read at runtime (rdflib's
in-memory graph has no notion of named graphs in this prototype).
``reasoning_profile`` is strict-validated against
``{none, rdfs, owl-rl, owl-dl, custom}`` but only ``"none"`` works; any
other value would raise ``NotImplementedError`` if the inference path were
wired in (it isn't — see ``mixins.py``).
"""
