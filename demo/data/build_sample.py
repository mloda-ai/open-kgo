"""Build metaqa_sample.gml: 1-hop induced subgraph around every QA topic entity.

The output is the minimum MetaQA subgraph needed to answer the 1-hop QA
splits. Nodes are typed (Movie / Person / Genre / Year / Language / Tag /
Rating / Votes) by inferring from the relations they participate in. Output
is byte-deterministic for a given input.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import networkx as nx

from . import KB_FILE, QA_FILE, SAMPLE_FILE

DEFAULT_QA_FILES: tuple[Path, ...] = (QA_FILE,)

TOPIC_PATTERN = re.compile(r"\[([^\]]+)\]")

OBJECT_TYPE_BY_RELATION: dict[str, str] = {
    "directed_by": "Person",
    "written_by": "Person",
    "starred_actors": "Person",
    "has_genre": "Genre",
    "release_year": "Year",
    "in_language": "Language",
    "has_tags": "Tag",
    "has_imdb_rating": "Rating",
    "has_imdb_votes": "Votes",
}


def load_kb(path: Path) -> nx.MultiDiGraph:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            s, p, o = parts
            g.add_edge(s, o, relation=p)
    return g


def read_topic_entities(qa_files: Iterable[Path]) -> set[str]:
    entities: set[str] = set()
    for qa_file in qa_files:
        with qa_file.open(encoding="utf-8") as f:
            for line in f:
                for match in TOPIC_PATTERN.finditer(line):
                    entities.add(match.group(1))
    return entities


def infer_types(g: nx.MultiDiGraph) -> dict[str, str]:
    types: dict[str, str] = {}
    for u, v, data in g.edges(data=True):
        rel = data.get("relation", "")
        if rel in OBJECT_TYPE_BY_RELATION:
            types.setdefault(u, "Movie")
            types.setdefault(v, OBJECT_TYPE_BY_RELATION[rel])
    return types


def build_sample(
    kb_path: Path = KB_FILE,
    qa_files: Iterable[Path] = DEFAULT_QA_FILES,
    out_path: Path = SAMPLE_FILE,
) -> Path:
    """Build sample.gml from kb_path and qa_files. Deterministic per input."""
    g = load_kb(kb_path)
    topics = read_topic_entities(qa_files)
    seeds = topics & set(g.nodes())
    nodes: set[str] = set(seeds)
    for s in seeds:
        nodes.update(g.successors(s))
        nodes.update(g.predecessors(s))

    sub: nx.MultiDiGraph = nx.MultiDiGraph()
    sub.add_nodes_from(sorted(nodes))
    for u, v, data in g.edges(data=True):
        if u in nodes and v in nodes:
            sub.add_edge(u, v, **data)

    types = infer_types(sub)
    for n in sub.nodes():
        sub.nodes[n]["type"] = types.get(n, "Unknown")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(sub, out_path)
    return out_path


if __name__ == "__main__":
    import sys

    out = build_sample()
    print(f"Wrote {out}", file=sys.stderr)
