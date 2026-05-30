# Third-Party Notices

`open-kgo` is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). Its runtime
and connector dependencies (`mloda`, `rdflib`, `networkx`, `kuzu`, `pyyaml`) are
all permissive (Apache-2.0 / BSD-3-Clause / MIT) and impose no additional
distribution obligations on this project.

## Demo extra (`pip install open-kgo[demo]`)

The optional `demo` extra pulls in [`marimo`](https://github.com/marimo-team/marimo)
(Apache-2.0) for the example notebooks. `marimo` brings in two transitive
dependencies worth noting for SBOM/compliance completeness — neither is a
dependency of any `kg-*` connector extra:

- **`pathspec`** — MPL-2.0 (Mozilla Public License 2.0). Weak, file-level
  copyleft: obligations attach only if you modify `pathspec`'s own source
  files. Using it as an unmodified dependency carries no obligation for
  open-kgo.
- **`loro`** (loro-py, a CRDT library) — MIT upstream; the published wheel ships
  without embedded license metadata. Recorded here so automated SBOM scans that
  flag the missing metadata have a documented reference.

These notes cover the demo tooling only; installing open-kgo without the `demo`
extra does not include them.

## MetaQA evaluation data

The MetaQA dataset (CC BY 3.0, Zhang et al., 2018) is **not** redistributed by
this repository — only a small hand-authored sample is committed. See
[`demo/data/README.md`](demo/data/README.md) and
[`demo/data/metaqa/README.md`](demo/data/metaqa/README.md) for the license
boundary and attribution.
