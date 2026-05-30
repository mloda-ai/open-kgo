"""Code / build / SBOM KG connectors (CodeQL, Bazel, CycloneDX, SPDX, ...).

Hidden-KG family. Locator is replaced by ``manifest_path`` semantics; address
includes ``commit_sha``, ``branch``, ``language_code``. Inherits ``TraversalMixin``.

PROTOTYPE NOTE: ``CycloneDxSbomReader`` reads ``manifest_path`` (with
``locator`` fallback) and returns the SBOM's ``components`` list — nothing
more. The entire ``TraversalMixin`` surface (``lineage_direction``,
``upstream_depth``, ``downstream_depth``, ``entity_type``,
``relationship_type``, ``expand_paths``) is accepted at the property layer
and ignored at runtime. The CycloneDX ``dependencies`` array, which carries
the actual build-graph edges, is **not walked** — so the family inherits a
traversal mixin it does not exercise. ``commit_sha``, ``branch``, and
``language_code`` are also accepted but unused.
"""
