"""Offline regression tests for the committed demo sample graph.

These tests exercise the exact data path the demo notebooks use (build the
sample graph from the committed KB/QA, load the ontology, traverse). They are
designed to fail if the sample, the builder, or the ontology drift out of sync
so that a broken demo is caught in CI instead of at notebook startup.
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import pytest

from demo.data import KB_FILE, QA_FILE, build_sample
from open_kgo.feature_groups.kg.ontology.registry import OntologyRegistry

NAMESPACE = "movie"
ONTOLOGY_YAML = (
    Path(__file__).parent.parent
    / "open_kgo"
    / "feature_groups"
    / "kg"
    / "ontology"
    / "tests"
    / "fixtures"
    / "metaqa_ontology.yaml"
)
TOPIC_PATTERN = re.compile(r"\[([^\]]+)\]")


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> nx.MultiDiGraph:
    out = tmp_path_factory.mktemp("demo_data") / "sample.gml"
    build_sample.build_sample(KB_FILE, [QA_FILE], out)
    return nx.read_gml(str(out))


@pytest.fixture(scope="module")
def _ontology() -> None:
    OntologyRegistry._clear()
    OntologyRegistry.load_file(str(ONTOLOGY_YAML))


def test_committed_sample_files_present_and_nonempty() -> None:
    assert KB_FILE.exists(), "sample_kb.txt missing"
    assert QA_FILE.exists(), "sample_qa.txt missing"
    assert KB_FILE.read_text(encoding="utf-8").strip(), "sample_kb.txt is empty"
    assert QA_FILE.read_text(encoding="utf-8").strip(), "sample_qa.txt is empty"


def test_graph_builds_with_typed_movie_nodes(graph: nx.MultiDiGraph) -> None:
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    movies = [n for n, d in graph.nodes(data=True) if d.get("type") == "Movie"]
    assert movies, "sample graph has no Movie nodes"
    assert all(d.get("type", "Unknown") != "Unknown" for _, d in graph.nodes(data=True))


def test_every_edge_is_ontology_valid(graph: nx.MultiDiGraph, _ontology: None) -> None:
    """Arch-2 traversal raises on any edge whose (domain, range) violates the ontology.

    If this fails, demo_kg_ontology and eval_qa_accuracy will throw at runtime.
    """
    violations: list[str] = []
    for src, dst, data in graph.edges(data=True):
        rel = data.get("relation", "")
        src_type = graph.nodes[src].get("type", "Unknown")
        dst_type = graph.nodes[dst].get("type", "Unknown")
        if not OntologyRegistry.is_valid_edge(NAMESPACE, src_type, rel):
            violations.append(f"domain: ({src_type}) -{rel}-> {dst}")
            continue
        expected_range = OntologyRegistry.get_range_type(NAMESPACE, rel)
        if expected_range is not None and dst_type != expected_range:
            violations.append(f"range: {rel} expected {expected_range} but {dst} is {dst_type}")
    assert not violations, "ontology violations in sample graph:\n" + "\n".join(violations)


def test_all_qa_topics_present_in_graph(graph: nx.MultiDiGraph) -> None:
    """Every bracketed QA topic must resolve to a node, else the eval silently skips it."""
    missing: list[str] = []
    for line in QA_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for topic in TOPIC_PATTERN.findall(line):
            if topic not in graph:
                missing.append(topic)
    assert not missing, f"QA topics absent from graph: {sorted(set(missing))}"


def test_forward_questions_return_gold_answers(graph: nx.MultiDiGraph) -> None:
    """At least one forward question per relation must produce a hit.

    Guards against a graph that loads but is vacuous: confirms the demo
    produces signal, not just absence of errors.
    """
    hit_relations: set[str] = set()
    for line in QA_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question, _, raw_answers = line.partition("\t")
        gold = {a.strip() for a in raw_answers.split("|") if a.strip()}
        match = TOPIC_PATTERN.search(question)
        if not match:
            continue
        topic = match.group(1)
        if topic not in graph or graph.nodes[topic].get("type") != "Movie":
            continue
        for _, dst, data in graph.out_edges(topic, data=True):
            if dst in gold:
                hit_relations.add(data.get("relation", ""))
    # The seven movie relations the ontology defines, exercised forward.
    expected = {
        "directed_by",
        "written_by",
        "starred_actors",
        "has_genre",
        "release_year",
        "in_language",
        "has_tags",
    }
    assert expected <= hit_relations, f"relations with no forward hit: {sorted(expected - hit_relations)}"
