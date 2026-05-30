"""Network property-graph KG connectors (Neo4j, Memgraph, Neptune-Gremlin, KuzuDB-with-Cypher, ...).

Family base for graph DBs with property-graph data model and a vendor query
language (Cypher / Gremlin / GSQL / nGQL / TypeQL). Adds ``dataset`` (database
name), ``read_consistency``, ``transaction_mode``.

PROTOTYPE NOTE: the only concrete plugin shipped here is KuzuDB embedded.
Kuzu is property-graph + Cypher but has no network endpoint, so
``read_consistency`` / ``transaction_mode`` are no-ops on it. The base
validates property *shape* only.
"""
