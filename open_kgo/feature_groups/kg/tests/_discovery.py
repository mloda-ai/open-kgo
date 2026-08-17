"""Auto-discovery helpers for cross-group KG smoke tests.

Walks ``open_kgo.feature_groups.kg`` with ``pkgutil.walk_packages`` and
imports every non-test submodule, so ``KgConnectorReaderBase.__subclasses__()``
is fully populated without per-test ``# noqa: F401`` import lists.

Trade-off vs. a hand-maintained CONNECTOR_ID list: discovery makes adding a
10th family a one-package edit, but it cannot detect *deletion* of a whole
family — if a subpackage is removed, ``family_subpackages()`` shrinks with it
and the parametrized smoke test silently runs over fewer cases. A counted
floor in ``test_cross_group_contract.py`` keeps the deletion alarm without
re-introducing the per-id list.

Discovery is exposed as an explicit function (``import_all_kg_readers``) rather
than a module-level side-effect so test modules opt in. ``kg_contract.py`` and
the per-family contract bases must not pay the kg-wide import cost just to use
``run_query``.

The leading-underscore filename signals "private to ``kg/tests/`` siblings" —
sibling test modules import these helpers, but consumers outside the kg test
suite should not.
"""

from __future__ import annotations

import ast
import gc
import importlib
import inspect
import pkgutil
import sys
import textwrap
from contextlib import contextmanager
from typing import Any, Iterator, Literal

from mloda.provider import PropertySpec

import open_kgo.feature_groups.kg as _kg_pkg
from open_kgo.feature_groups.kg.base import KgConnectorReaderBase, ParamReader

_KG_PACKAGE_NAME: str = _kg_pkg.__name__

_LayerName = Literal["PROPERTY_MAPPING", "PARAMS_MAPPING"]


def import_all_kg_readers() -> None:
    """Import every non-test submodule of the kg package.

    Side-effect: each concrete ``KgConnectorReaderBase`` subclass gets registered
    with Python's class hierarchy, making it discoverable via ``__subclasses__``.
    Import errors propagate — a broken module must not silently shrink the
    registered set.
    """
    for module_info in pkgutil.walk_packages(_kg_pkg.__path__, prefix=f"{_KG_PACKAGE_NAME}."):
        if "tests" in module_info.name.split("."):
            continue
        # ``walk_packages`` auto-imports packages so it can read ``__path__``,
        # but non-package modules are returned without being imported. The
        # explicit ``import_module`` is therefore mandatory — without it,
        # concrete readers in plain ``.py`` files would never register.
        importlib.import_module(module_info.name)


_NON_CONNECTOR_PACKAGES: frozenset[str] = frozenset({"tests", "ontology"})


def family_subpackages() -> set[str]:
    """Top-level subpackage names under kg, excluding shared infrastructure packages.

    ``tests`` is the shared test package. ``ontology`` is a cross-cutting registry
    module, not a connector family — it ships no ``KgConnectorReaderBase`` subclass.
    """
    return {
        info.name
        for info in pkgutil.iter_modules(_kg_pkg.__path__)
        if info.ispkg and info.name not in _NON_CONNECTOR_PACKAGES
    }


def walk_subclasses(cls: type[KgConnectorReaderBase]) -> set[type[KgConnectorReaderBase]]:
    """Return every (transitive) subclass of ``cls``, excluding ``cls`` itself."""
    out: set[type[KgConnectorReaderBase]] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out.update(walk_subclasses(sub))
    return out


def family_of(sub: type[KgConnectorReaderBase]) -> str | None:
    """Return the kg subpackage name that owns ``sub``, or None if it lives elsewhere."""
    prefix = f"{_KG_PACKAGE_NAME}."
    if not sub.__module__.startswith(prefix):
        return None
    return sub.__module__[len(prefix) :].split(".", 1)[0]


@contextmanager
def clean_kg_subclass_registry() -> Iterator[None]:
    """Detect synthetic-class leaks into ``KgConnectorReaderBase.__subclasses__``.

    ``__subclasses__`` is process-global. A synthetic test class held alive past
    its test by a closure, a module-level binding, or a long-lived collection
    pollutes every subsequent ``walk_subclasses`` call. The pollution surfaces
    not at the offending test but as a flaky cross-group-contract failure days
    later, which is the failure mode this helper is designed to short-circuit.

    Snapshot the set on enter; on exit, force ``gc.collect`` so weakly-
    referenced test classes have a chance to be reclaimed, then assert no
    persistent additions. If the assertion fires, the leaked class is named so
    the offending test surfaces immediately instead of weeks of cross-test
    pollution.

    Synthetic classes that raise during ``__init_subclass__`` (the invariant
    tests in ``test_property_mapping_integrity.py``) do not leak in practice:
    when class creation fails, Python never binds the local name, so the
    weakref in ``__subclasses__`` is cleared as soon as the test frame
    discards. Wrapping those tests still pins the contract — a future refactor
    that moves the synthetic class to module scope (a plausible deduplication)
    would fire this immediately.

    If the body itself raises, the leak check is skipped so the body's
    exception propagates unmasked. A test that already failed almost always
    leaves partial state behind; double-reporting would replace the test's
    own diagnostic with a noisy "leak detected" header and the original
    failure would only survive in ``__context__``.
    """
    before = walk_subclasses(KgConnectorReaderBase)
    try:
        yield
    finally:
        # ``return`` from a ``finally`` would discard an in-flight exception
        # from the body. Use a guarded branch instead: only run the leak
        # check when no exception is propagating, so the body's exception
        # passes through unmasked.
        if sys.exc_info()[0] is None:
            gc.collect()
            leaked = walk_subclasses(KgConnectorReaderBase) - before
            if leaked:
                names = sorted(f"{c.__module__}.{c.__name__}" for c in leaked)
                raise AssertionError(
                    f"clean_kg_subclass_registry: {len(leaked)} subclass(es) leaked past the block: "
                    f"{names}. Define the synthetic class inside a factory function so its local "
                    f"binding is reclaimed when the factory returns, or release any external "
                    f"reference before exiting the block."
                )


def iter_strict_specs(
    reader_class: type[KgConnectorReaderBase],
) -> Iterator[tuple[str, PropertySpec, _LayerName]]:
    """Yield ``(key, spec, layer_name)`` for every strict-validation spec on ``reader_class``.

    Walks ``PROPERTY_MAPPING`` for every reader and ``PARAMS_MAPPING`` for
    ``ParamReader`` subclasses only (``QueryReader`` subclasses don't carry
    one). The ``isinstance(spec, PropertySpec)`` guard is defensive: class-time
    ``_validate_mapping_spec_shapes`` already rejects malformed specs, so
    in practice the guard is unreachable through a valid class. The
    integrity check for those lives in ``test_property_mapping_integrity.py``.

    Single-place iteration so adding a new spec layer in the future is a
    one-edit change for every contract test that walks strict enums.
    """
    layers: list[tuple[_LayerName, dict[str, Any]]] = [
        ("PROPERTY_MAPPING", reader_class.PROPERTY_MAPPING),
    ]
    if issubclass(reader_class, ParamReader):
        layers.append(("PARAMS_MAPPING", reader_class.PARAMS_MAPPING))
    for layer_name, mapping in layers:
        for key, spec in mapping.items():
            if not isinstance(spec, PropertySpec):
                continue
            if spec.strict_validation is True:
                yield key, spec, layer_name


def iter_nonstrict_specs(
    reader_class: type[KgConnectorReaderBase],
) -> Iterator[tuple[str, PropertySpec, _LayerName]]:
    """Yield ``(key, spec, layer_name)`` for every NON-strict-validation spec on ``reader_class``.

    The complement of ``iter_strict_specs`` over the same layers, so the
    surface-honesty contract (non-strict keys) and the strict-enum contract
    partition the advertised surface without overlap.
    """
    layers: list[tuple[_LayerName, dict[str, Any]]] = [
        ("PROPERTY_MAPPING", reader_class.PROPERTY_MAPPING),
    ]
    if issubclass(reader_class, ParamReader):
        layers.append(("PARAMS_MAPPING", reader_class.PARAMS_MAPPING))
    for layer_name, mapping in layers:
        for key, spec in mapping.items():
            if not isinstance(spec, PropertySpec):
                continue
            if spec.strict_validation is not True:
                yield key, spec, layer_name


def reader_string_literals(reader_class: type[KgConnectorReaderBase]) -> set[str]:
    """Return the exact string-literal values in any reader method across the kg-package MRO.

    The consumption signal for the surface-honesty contract. AST-collects every
    string ``Constant`` from the methods of each kg-package class in the MRO
    (concrete, family bases, mixins, universal base); non-package classes are
    skipped. A key a reader reads appears as an exact literal (``slot["locator"]``,
    ``params.get("stable_id")``); a key only *declared* in a mapping is a
    class-level attribute, absent from method source. Exact set membership means
    docstrings and error-message fragments cannot masquerade as consumption.

    ``textwrap.dedent`` before ``ast.parse`` strips the method's class
    indentation; getsource/parse failures are skipped defensively. Caveat: a
    flush-left multi-line string in a method body defeats dedent and drops that
    method's literals, but the failure mode is a loud false red build (a key
    read only there would be flagged), never a silent miss. No shipped method
    hits it.
    """
    prefix = f"{_KG_PACKAGE_NAME}."
    literals: set[str] = set()
    for klass in reader_class.__mro__:
        module = getattr(klass, "__module__", "")
        if module != _KG_PACKAGE_NAME and not module.startswith(prefix):
            continue
        for attr in vars(klass).values():
            # Unwrap classmethod / staticmethod to the underlying function.
            func = getattr(attr, "__func__", attr)
            if not inspect.isfunction(func):
                continue
            try:
                source = textwrap.dedent(inspect.getsource(func))
            except (OSError, TypeError):
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    literals.add(node.value)
    return literals


def effective_unconsumed_waivers(reader_class: type[KgConnectorReaderBase]) -> set[str]:
    """Union the locally-declared ``_WAIVED_UNCONSUMED_KEYS`` across ``reader_class.__mro__``.

    Merges each class's own set (not the shadowing attribute read) so a family
    base can waive family-wide keys while a concrete adds its own.
    """
    waived: set[str] = set()
    for klass in reader_class.__mro__:
        local = klass.__dict__.get("_WAIVED_UNCONSUMED_KEYS")
        if local:
            waived |= set(local)
    return waived
