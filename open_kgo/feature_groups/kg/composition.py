"""Family-surface composition helpers for KG connector readers.

Runtime companion to the two class-definition-time/runtime concern modules
``kg.class_guards`` and ``kg.credentials``: this module owns the
``PROPERTY_MAPPING`` / ``PARAMS_MAPPING`` composition machinery that
``KgConnectorReaderBase.__init_subclass__`` (in ``reader_base.py``) delegates
to. Extracted from ``reader_base.py`` so that module stays focused on the
reader lifecycle; see its module docstring for the concern-module split this
belongs to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from open_kgo.feature_groups.kg.errors import NonDictSpecError, PropertyMappingCollision

if TYPE_CHECKING:
    from open_kgo.feature_groups.kg.reader_base import KgConnectorReaderBase


def compose_property_mapping(*sources: dict[str, Any], context: str = "") -> dict[str, Any]:
    """Merge property-mapping dicts and raise on duplicate keys.

    Replaces the spread-merge ``{**A, **B, **C}`` idiom used by family bases.
    The spread-merge silently overwrites duplicates; this helper raises
    ``PropertyMappingCollision`` so structural collisions surface at import
    time. ``context`` is the family/mixin name carried into the error message.

    Also rejects non-dict spec values at composition time via
    ``NonDictSpecError``. A ``None`` spec would otherwise pass downstream
    ``mapping.get(key) is None`` lookups and surface as a self-contradicting
    "unknown credential key" error: the key is listed in ``allowed`` but
    ``_validate_mapping`` cannot distinguish a missing key from a ``None``
    spec via ``.get``. Catching it here keeps the closed-world check honest.
    ``NonDictSpecError`` is a sibling of ``PropertyMappingCollision`` under
    ``InvalidCredentialShape`` so callers can scope a single typed handler
    across the whole structural-error family.
    """
    merged: dict[str, Any] = {}
    for source in sources:
        for key, spec in source.items():
            if key in merged:
                raise PropertyMappingCollision(key, context=context)
            if not isinstance(spec, dict):
                raise NonDictSpecError(key, spec, context=context)
            merged[key] = spec
    return merged


def narrow_property_mapping(source: dict[str, Any], *exclude: str) -> dict[str, Any]:
    """Return ``source`` minus the ``exclude`` keys — the narrowing companion to ``compose_property_mapping``.

    Concrete plugins drop family-level keys they do not honor (advertising
    a key the reader ignores would be a surface lie that the closed-world
    credential check then rejects). This is option 1 of the "Honest credential
    surface" rule in ``reader_base``'s module docstring. Several concretes
    spelled this as an inline ``{k: v for k, v in Parent.PROPERTY_MAPPING.items()
    if k not in {...}}`` comprehension; centralising it names the intent and
    keeps the narrowing rule in one place. Keys in ``exclude`` that are absent
    from ``source`` are silently ignored (narrowing is idempotent).
    """
    excluded = set(exclude)
    return {k: v for k, v in source.items() if k not in excluded}


def compose_family_surface(
    cls: type["KgConnectorReaderBase"],
    family_properties: dict[str, Any],
    family_params: dict[str, Any],
) -> None:
    """Auto-compose ``PROPERTY_MAPPING`` / ``PARAMS_MAPPING`` for a family base.

    Family bases used to spell the same ritual: ``PROPERTY_MAPPING =
    compose_property_mapping(Parent.PROPERTY_MAPPING, Mixin.PROPERTY_
    MAPPING_DELTA, {family keys}, context=...)`` and a parallel call for
    ``PARAMS_MAPPING``, which made it possible to inherit a mixin yet
    forget to merge one of its delta layers. Declaring the surface as
    class keyword arguments instead::

        class CitationRestReader(PaginationMixin, ParamReader,
                                 family_properties={...},
                                 family_params={...}):

    routes through here (via ``KgConnectorReaderBase.__init_subclass__``):
    the inherited mapping (the parent reader's), every
    ``PROPERTY_MAPPING_DELTA`` / ``PARAMS_MAPPING_DELTA`` found in the MRO
    (the declared mixins), and the family keys are composed with
    ``compose_property_mapping`` (duplicate keys raise, as before) before
    the class-definition guards run.

    Rules:
    - Opt-in: only classes passing ``family_properties`` and/or
      ``family_params`` are touched; concrete plugins keep assembling
      their narrowed mappings explicitly in the class body.
    - Mixing styles is rejected: a class that opts in must not also
      assign either mapping in its class body (the body value would
      silently win over the parent/mixin layers).
    - ``PARAMS_MAPPING`` is composed only for readers that have the
      attribute (``ParamReader`` descendants). A ``QueryReader`` family
      may inherit a mixin with a params delta (agent_memory inherits
      ``PaginationMixin``); the delta is deliberately ignored because the
      flavor has no ``build_params`` to honor it, and passing
      ``family_params`` on such a family raises.
    - Deltas are collected from the WHOLE MRO. A future family base that
      subclasses another family base which already absorbed a mixin would
      re-collect that mixin's delta and fail loudly with a
      ``PropertyMappingCollision`` at import; such a chain needs explicit
      composition instead of this opt-in.
    """
    for layer in ("PROPERTY_MAPPING", "PARAMS_MAPPING"):
        if layer in cls.__dict__:
            raise ValueError(
                f"{cls.__name__} passes family_properties/family_params but also assigns {layer} "
                f"in its class body; declare the surface one way only."
            )
    property_deltas = [
        klass.__dict__["PROPERTY_MAPPING_DELTA"] for klass in cls.__mro__ if "PROPERTY_MAPPING_DELTA" in klass.__dict__
    ]
    cls.PROPERTY_MAPPING = compose_property_mapping(
        cls.PROPERTY_MAPPING, *property_deltas, family_properties, context=cls.__name__
    )
    params_mapping = getattr(cls, "PARAMS_MAPPING", None)
    if params_mapping is None:
        if family_params:
            raise ValueError(
                f"{cls.__name__} passes family_params but is not a ParamReader descendant; "
                f"query-flavored families have no PARAMS_MAPPING layer."
            )
        return
    params_deltas = [
        klass.__dict__["PARAMS_MAPPING_DELTA"] for klass in cls.__mro__ if "PARAMS_MAPPING_DELTA" in klass.__dict__
    ]
    if params_deltas or family_params:
        # setattr because PARAMS_MAPPING is declared on ParamReader, not on
        # this universal base; the getattr gate above already proved cls is
        # a params-flavored reader.
        setattr(  # noqa: B010
            cls,
            "PARAMS_MAPPING",
            compose_property_mapping(
                params_mapping, *params_deltas, family_params, context=f"{cls.__name__}.PARAMS_MAPPING"
            ),
        )
