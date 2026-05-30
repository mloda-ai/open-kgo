"""Tests for demo.data.build_sample (QA-anchored 1-hop subgraph extraction)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from demo.data import build_sample

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "data"
TINY_KB = FIXTURE_DIR / "tiny_kb.txt"
TINY_QA = FIXTURE_DIR / "tiny_qa.txt"


def test_load_kb_parses_pipe_separated_triples() -> None:
    g = build_sample.load_kb(TINY_KB)
    assert ("Movie1", "Director1") in g.edges()
    relations = {data["relation"] for _, _, data in g.edges(data=True)}
    assert {"directed_by", "starred_actors", "has_genre"} <= relations


def test_read_topic_entities_extracts_brackets() -> None:
    topics = build_sample.read_topic_entities([TINY_QA])
    assert topics == {"Movie1", "Movie2"}


def test_infer_types_assigns_movie_to_subjects() -> None:
    g = build_sample.load_kb(TINY_KB)
    types = build_sample.infer_types(g)
    assert types["Movie1"] == "Movie"
    assert types["Director1"] == "Person"
    assert types["Comedy"] == "Genre"


def test_build_sample_includes_only_qa_anchored_subgraph(tmp_path: Path) -> None:
    out = tmp_path / "sample.gml"
    build_sample.build_sample(TINY_KB, [TINY_QA], out)
    g = nx.read_gml(str(out))
    labels = set(g.nodes())
    # Movie1, Movie2 seeds plus their 1-hop neighbors.
    assert labels == {
        "Movie1",
        "Movie2",
        "Director1",
        "Director2",
        "Actor1",
        "Comedy",
        "Drama",
    }
    # Movie3 and its neighbors must be excluded (no QA query references it).
    assert "Movie3" not in labels
    assert "Actor2" not in labels


def test_build_sample_assigns_types(tmp_path: Path) -> None:
    out = tmp_path / "sample.gml"
    build_sample.build_sample(TINY_KB, [TINY_QA], out)
    g = nx.read_gml(str(out))
    types_by_label = {n: data["type"] for n, data in g.nodes(data=True)}
    assert types_by_label["Movie1"] == "Movie"
    assert types_by_label["Director1"] == "Person"
    assert types_by_label["Comedy"] == "Genre"


def test_build_sample_is_deterministic(tmp_path: Path) -> None:
    out1 = tmp_path / "sample_a.gml"
    out2 = tmp_path / "sample_b.gml"
    build_sample.build_sample(TINY_KB, [TINY_QA], out1)
    build_sample.build_sample(TINY_KB, [TINY_QA], out2)
    assert out1.read_bytes() == out2.read_bytes()
