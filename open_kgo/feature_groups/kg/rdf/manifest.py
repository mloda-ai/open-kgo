"""mloda entry-point manifest for the rdf family (``PluginLoader.all()`` discovery)."""

from __future__ import annotations

from open_kgo.feature_groups.kg.rdf.oxigraph_sparql import OxigraphSparqlFeatureGroup
from open_kgo.feature_groups.kg.rdf.rdflib_sparql import RdfLibSparqlFeatureGroup

FEATURE_GROUPS = (OxigraphSparqlFeatureGroup, RdfLibSparqlFeatureGroup)
