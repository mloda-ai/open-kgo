"""Cross-group registry smoke test.

Asserts that every kg subpackage registers at least one ``KgConnectorReaderBase``
subclass with a non-empty ``CONNECTOR_ID``. Discovery of plugin modules and the
expected family set both come from ``_discovery``, so adding a new family is a
single new package — no parallel edits to import lists or hardcoded id tuples.

Catches deletions / accidental skips that the per-family inherited tests would
not flag: ``test_minimum_family_count`` floors the family count so removing an
entire subpackage trips this test even though parametrize would otherwise just
run over fewer cases.
"""

from __future__ import annotations

import pytest

from open_kgo.feature_groups.kg.base import KgConnectorReaderBase
from open_kgo.feature_groups.kg.tests._discovery import (
    family_of,
    family_subpackages,
    import_all_kg_readers,
    walk_subclasses,
)

import_all_kg_readers()


def test_minimum_family_count() -> None:
    """Floor guard so removing a whole subpackage cannot silently shrink coverage.

    Discovery cannot detect deletion the way a hand-maintained id list could.
    If a family is intentionally removed, bump this floor in the same change.
    """
    families = sorted(family_subpackages())
    assert len(families) >= 9, f"a kg subpackage was deleted; bump the floor or restore it (found {families})"


@pytest.mark.parametrize("family", sorted(family_subpackages()))
def test_family_has_registered_connector(family: str) -> None:
    """Every kg.<family>/ subpackage must register at least one concrete reader."""
    found = {
        sub.CONNECTOR_ID
        for sub in walk_subclasses(KgConnectorReaderBase)
        if sub.CONNECTOR_ID and family_of(sub) == family
    }
    assert found, (
        f"Subpackage {family!r} registered no concrete KgConnectorReaderBase "
        f"with a non-empty CONNECTOR_ID. Either add a concrete reader or remove "
        f"the empty subpackage."
    )


def test_no_duplicate_connector_ids() -> None:
    """Two concrete plugins with the same CONNECTOR_ID would silently shadow each other.

    An id must also be a clean feature-name prefix: no chain (``__``) or column (``~``) separator, no trailing ``_``.
    """
    seen: dict[str, type[KgConnectorReaderBase]] = {}
    for sub in walk_subclasses(KgConnectorReaderBase):
        if not sub.CONNECTOR_ID:
            continue
        assert "__" not in sub.CONNECTOR_ID and "~" not in sub.CONNECTOR_ID and not sub.CONNECTOR_ID.endswith("_"), (
            f"CONNECTOR_ID={sub.CONNECTOR_ID!r} on {sub.__module__}.{sub.__name__} is not a clean feature-name prefix"
        )
        if sub.CONNECTOR_ID in seen:
            raise AssertionError(
                f"Duplicate CONNECTOR_ID={sub.CONNECTOR_ID!r} on "
                f"{sub.__module__}.{sub.__name__} and "
                f"{seen[sub.CONNECTOR_ID].__module__}.{seen[sub.CONNECTOR_ID].__name__}"
            )
        seen[sub.CONNECTOR_ID] = sub
