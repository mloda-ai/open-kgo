"""RDF / SPARQL family of KG connectors.

Family-base properties (beyond the universal layer) follow Version B of
``open_kgo/feature_groups/kg/README.md``: ``default_graph_uris``,
``named_graph_uris``, ``update_endpoint``, ``result_format``, plus
``reasoning_profile`` from ``InferenceMixin``.

PROTOTYPE NOTE: two SPARQL engines back this family — ``RdfLibSparqlReader``
(``rdflib``) and ``OxigraphSparqlReader`` (``pyoxigraph``, a Rust-backed
embedded store). Both validate property *shape* only for several keys and
return Python dicts, so ``result_format`` stays narrowed to JSON on both.
``default_graph_uris``, ``named_graph_uris``, and ``update_endpoint`` are
accepted but never read at runtime. ``reasoning_profile`` is strict-validated
against ``{none, rdfs, owl-rl, owl-dl, custom}`` but neither engine has an
inference path wired in, so both narrow it to ``"none"``. A second SPARQL
engine on the same family base is backend variety, not new surface.
"""
