"""Shared abstract test base for KG connector contract suites.

Mirrors ``DataOpsTestBase`` from
``mloda-registry/mloda/testing/feature_groups/data_operations/base.py``: a
small set of abstract adapter methods that concrete plugin tests implement
once, plus a body of contract tests that those plugins inherit for free.

The contract body is split by concern into four sibling modules, each a
mixin over the shared adapter ABC (``contract_adapters.py``):

- ``contract_credentials.py`` — credential-slot matching, shape validation,
  ``REQUIRED_KEYS``, env-var resolution.
- ``contract_params.py`` — per-call ``PARAMS_MAPPING`` validation, stripped
  params, ``REQUIRED_PARAMS`` (ParamReader concretes only).
- ``contract_surface.py`` — the "Honest credential surface" rule: every
  advertised key has an explicit disposition (narrowed, waived, consumed).
- ``contract_load.py`` — load-path behavior: multi-feature guard,
  idempotence, mutation safety, end-to-end ``mloda.run_all``.

``KgConnectorContractBase`` aggregates all four, so concrete plugin tests
and per-family contract bases (e.g. ``RdfContractTestBase`` in
``rdf/tests/kg_rdf_contract.py``) keep subclassing this one name and inherit
the full suite; the split is internal navigation structure only.

The cross-group contract suite (``test_cross_group_contract.py``) walks every
``KgConnectorContractBase`` subclass and runs the universal contract
assertions against each.
"""

from __future__ import annotations

from mloda.user import DataAccessCollection, Feature

from open_kgo.feature_groups.kg.errors import (
    InvalidCredentialShape,
    MissingEnvVarError,
    MissingRequiredKeysError,
    MissingRequiredParamsError,
)
from open_kgo.feature_groups.kg.tests.contract_credentials import CredentialContract
from open_kgo.feature_groups.kg.tests.contract_load import LoadBehaviorContract
from open_kgo.feature_groups.kg.tests.contract_params import ParamContract
from open_kgo.feature_groups.kg.tests.contract_surface import SurfaceHonestyContract


class KgConnectorContractBase(CredentialContract, ParamContract, SurfaceHonestyContract, LoadBehaviorContract):
    """Abstract test base every concrete KG plugin's tests inherit from.

    Subclasses implement 5 adapter methods (declared on
    ``KgContractAdapterBase``, or inherited from a per-family intermediate
    base). Concrete contract tests are inherited for free from the four
    concern mixins.
    """


# Re-exported for convenience in concrete tests.
__all__ = [
    "KgConnectorContractBase",
    "DataAccessCollection",
    "Feature",
    "InvalidCredentialShape",
    "MissingEnvVarError",
    "MissingRequiredKeysError",
    "MissingRequiredParamsError",
]
