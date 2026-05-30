# Sample movie graph

This directory holds the small movie knowledge graph used by the demo
notebooks. It runs entirely offline: no download, no network, no external
services.

## Files

- `sample_kb.txt` -- knowledge base as `subject|relation|object` triples
  (e.g. `The Matrix|directed_by|Lana Wachowski`).
- `sample_qa.txt` -- question/answer pairs, tab-separated, with the topic
  entity in brackets and pipe-separated gold answers
  (e.g. `who directed [The Matrix]\tLana Wachowski|Lilly Wachowski`).
- `build_sample.py` -- builds `metaqa_sample.gml`, the 1-hop induced subgraph
  around every QA topic entity, with node types inferred from the relations
  each node participates in. Byte-deterministic for a given input.
- `metaqa_sample.gml` -- the built graph (gitignored; regenerated on demand).

The notebooks call `ensure_data()` at startup, which builds `metaqa_sample.gml`
if it is missing, so you do not normally run anything here by hand.

## Provenance and license

`sample_kb.txt` / `sample_qa.txt` are hand-authored and contain only public,
factual movie data (directors, cast, genres, release years, languages). They
follow the triple format and relation names of the MetaQA dataset
(Zhang, Yuyu et al., "Variational Reasoning for Question Answering with
Knowledge Graph", AAAI 2018, https://github.com/yuyuz/MetaQA) so that the same
ontology and traversal code applies.

This sample is **not** derived from the MetaQA dataset files: factual data is
not copyrightable and the format is a convention, so the sample carries no
license obligation. MetaQA itself is licensed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/legalcode) and is not
redistributed here. Its license text is kept in the repo at
[`metaqa/METAQA-CC-BY-3.0.txt`](metaqa/METAQA-CC-BY-3.0.txt) for attribution.

## Running against the full MetaQA dataset

The demos ship with the small sample so they work out of the box. To reproduce
the experiments at full scale, swap in the real MetaQA benchmark:

1. **Download** the dataset from the upstream project. MetaQA is distributed via
   Google Drive only (no stable direct-download URL), so this step is manual:
   open https://github.com/yuyuz/MetaQA and follow the Google Drive link in its
   README. You need the knowledge base `kb.txt` and, for the QA accuracy demo,
   the 1-hop split `1-hop/vanilla/qa_test.txt`.

2. **Place** the files in the dedicated `metaqa/` folder next to this README:
   `demo/data/metaqa/kb.txt`, `demo/data/metaqa/qa_test.txt`, and the dataset's
   license as `demo/data/metaqa/METAQA-CC-BY-3.0.txt`. That folder is gitignored
   (`metaqa/*`), so the CC BY 3.0 data cannot be committed by accident. See
   [`metaqa/README.md`](metaqa/README.md).

3. **Build** the graph from the full KB:

   ```python
   from demo.data import build_sample

   build_sample.build_sample(
       kb_path="demo/data/metaqa/kb.txt",
       qa_files=["demo/data/metaqa/qa_test.txt"],
       out_path="demo/data/metaqa_sample.gml",
   )
   ```

   This overwrites `metaqa_sample.gml` with the full 1-hop subgraph.

4. **Point** the QA demo at the full split: in `demo/eval_qa_accuracy.py`, set
   `QA_TEST = DATA_DIR / "metaqa" / "qa_test.txt"` instead of the sample file.

5. **Re-run** any notebook. The ontology, traversal, and eval code is unchanged;
   only the input graph is larger. Name-collision range violations
   (Experiment 3b) appear only at this scale.

When you use the full dataset, keep the MetaQA attribution (CC BY 3.0) in any
output you publish.
