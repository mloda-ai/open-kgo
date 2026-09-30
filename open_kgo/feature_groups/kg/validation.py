"""Shared runtime validation helpers for KG connector readers.

Home for small value-level guards that more than one reader (or more than one
layer) needs. The first resident is ``parse_bounded_int``, which used to exist
in four near-identical spellings: ``parse_page_size`` in ``mixins.py``
(``int >= 1``), ``_parse_depth`` in ``code_build/spdx_sbom.py`` (``int >= 0``),
an inline ``hierarchy_depth`` check in ``citation_rest/paginated_citation.py``,
and ``_validate_result_limit`` in ``credentials.py``. All four rejected bool
explicitly (it is an ``int`` subclass, but a count or depth expressed as a
truth-value is almost always a caller mistake); centralising them means a
future validator cannot forget that guard.

This module sits below ``reader_base`` in the import graph (it imports only
``errors``), so both the universal base and the mixins/concretes can use it
without cycles.
"""

from __future__ import annotations

from typing import Any

from open_kgo.feature_groups.kg.errors import InvalidCredentialShape


def _bound_phrase(min_value: int) -> str:
    """Human-readable bound description matching the historical error wording."""
    if min_value == 1:
        return "a positive int (>= 1, bool not accepted)"
    if min_value == 0:
        return "a non-negative int (bool not accepted)"
    return f"an int >= {min_value} (bool not accepted)"


def parse_bounded_int(
    connector_id: str,
    key: str,
    value: Any,
    *,
    min_value: int,
    default: int | None = None,
) -> int:
    """Validate and return a bounded int, optionally falling back to ``default`` when unset.

    The single source of truth for the "optional int with a minimum bound"
    guard. ``None`` maps to ``default`` when one is given (absent / opt-out
    semantics, the params-layer convention); when ``default`` is ``None`` the
    value is mandatory and ``None`` fails the type check like any other
    non-int. Bool is rejected explicitly before the ``int`` check because
    ``True``/``False`` are ``int`` subclasses in Python. Strings, floats, and
    out-of-bound integers raise ``InvalidCredentialShape`` so a typo surfaces
    a typed error rather than a raw ``ValueError`` mid-load or a silently
    wrong slice/walk.
    """
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < min_value:
        raise InvalidCredentialShape(
            f"{connector_id}: {key} must be {_bound_phrase(min_value)}, got {type(value).__name__} {value!r}."
        )
    return value
