"""Registry-vs-discovery integrity gates over the shared ``_family_cases`` registry.

One of the three holistic modules (see ``_family_cases`` for the split). These
tests cross-check the hand-written ``CASES`` registry against independent package
discovery: no gaps, no ghosts, every family floored at two concretes, and every
tag drawn from the documented vocabularies.
"""

from __future__ import annotations

from collections import defaultdict
from importlib.metadata import EntryPoint, entry_points

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase
from open_kgo.feature_groups.kg.tests._discovery import family_of, family_subpackages, walk_subclasses
from open_kgo.feature_groups.kg.tests._family_cases import (
    CASES,
    FIXTURE_SCOPES,
    INPUT_SHAPES,
    LOCATOR_KEYS,
    SETUPS,
    discovered_connector_ids,
    is_production_connector,
)


def test_registry_covers_all_discovered_connectors() -> None:
    """The CASES registry must cover exactly the concretes discovery finds: no gaps, no ghosts.

    Add a 19th connector module without a CASES entry and ``missing`` is non-empty -> red.
    This is the prospective gap detector; the smoke module only proves today's 18 work.
    """
    discovered = discovered_connector_ids()
    registered = {c.connector_id for c in CASES}
    missing = discovered - registered
    ghosts = registered - discovered
    assert not missing, f"connectors with no end-to-end usage recipe in CASES: {sorted(missing)}"
    assert not ghosts, f"CASES entries for connector_ids that no longer exist: {sorted(ghosts)}"


def test_every_family_has_at_least_two_concretes() -> None:
    """Floor each family at >= 2 concretes. The whole premise of this branch is "two backends per base".

    ``test_cross_group_contract`` only floors at >= 1 per family; this is the >= 2 guard that
    catches a family quietly regressing to a single concrete.
    """
    by_family: dict[str, set[str]] = defaultdict(set)
    for sub in walk_subclasses(KgConnectorReaderBase):
        if not is_production_connector(sub):
            continue
        fam = family_of(sub)
        assert fam is not None  # guaranteed by is_production_connector; narrows for mypy
        by_family[fam].add(sub.CONNECTOR_ID)

    thin = {fam: sorted(ids) for fam, ids in by_family.items() if len(ids) < 2}
    assert not thin, f"families with fewer than two concrete plugins: {thin}"

    # Cross-check against the package layout: every connector subpackage must appear above,
    # so a family that registers zero concretes (not just < 2) is also caught.
    missing_families = family_subpackages() - set(by_family)
    assert not missing_families, f"connector subpackages registering no concrete reader: {sorted(missing_families)}"


def test_registry_tags_use_known_vocabularies() -> None:
    """Every CASES entry classifies into the known structural vocabularies.

    A connector whose invocation shape, fixture sourcing, or setup style is not one of the
    documented kinds trips this, forcing whoever adds it to extend the vocabulary consciously
    instead of introducing an unlabelled new shape that the asymmetry catalog would miss.
    """
    for case in CASES:
        assert case.input_shape in INPUT_SHAPES, f"{case.connector_id}: unknown input_shape {case.input_shape!r}"
        assert case.locator_key in LOCATOR_KEYS, f"{case.connector_id}: unknown locator_key {case.locator_key!r}"
        assert case.fixture_scope in FIXTURE_SCOPES, (
            f"{case.connector_id}: unknown fixture_scope {case.fixture_scope!r}"
        )
        assert case.setup in SETUPS, f"{case.connector_id}: unknown setup {case.setup!r}"


def _open_kgo_entry_points(group: str) -> list[EntryPoint]:
    eps = [ep for ep in entry_points(group=group) if ep.dist is not None and ep.dist.name == "open-kgo"]
    assert eps, f"no open-kgo '{group}' entry points installed; run `uv sync` to refresh the package metadata."
    return eps


def _family(module: str) -> str:
    return module.removeprefix("open_kgo.feature_groups.kg.").split(".")[0]


def test_entry_points_register_every_connector_once_per_family() -> None:
    """One ``mloda.feature_groups`` entry point per family, together covering exactly the discovered connectors."""
    by_family: dict[str, set[str]] = {}
    for ep in _open_kgo_entry_points("mloda.feature_groups"):
        family = _family(ep.module)
        assert family not in by_family, f"two entry points for family {family!r}"
        classes = ep.load()
        assert {_family(c.__module__) for c in classes} == {family}, f"{ep.name} mixes families: {classes}"
        by_family[family] = {c.READER_CLASS.CONNECTOR_ID for c in classes}

    assert set(by_family) == family_subpackages()
    assert set().union(*by_family.values()) == discovered_connector_ids()
