"""Universal source FeatureGroup that delegates source discovery and loading to a KG connector reader.

Re-exported through ``kg.base``. Family bases subclass ``KgConnectorFeatureGroupBase`` and stay abstract;
each concrete pins ``READER_CLASS``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.input_data.source_feature_group import HANDLE_SPEC  # not in mloda.provider
from mloda.provider import (
    CHAIN_SEPARATOR,
    COLUMN_SEPARATOR,
    HANDLE_OPTION,
    INPUT_DATA_STAGE,
    QUERY_OPTION,
    ClaimRoute,
    ComputeFramework,
    FeatureSet,
    NamePolicy,
    PropertySpec,
    SourceFeatureGroup,
    SourceMatch,
    record_match_rejection,
)
from mloda.user import Credential, DataAccessCollection, FeatureName, Options

from mloda.user.python_dict import PythonDictFramework

from open_kgo.feature_groups.kg.reader_base import KgConnectorReaderBase
from open_kgo.feature_groups.kg.readers import QueryReader


_QUERY_SPEC = PropertySpec(
    "Query text the reader runs.",
    default=None,
    match_guard=lambda value: isinstance(value, str) and bool(value.strip()),
    expected="a non-empty query string",
)


class _UnpinnedReader:
    """``READER_CLASS`` placeholder: keeps a group abstract until a concrete pins a reader."""

    __isabstractmethod__ = True


class KgConnectorFeatureGroupBase(SourceFeatureGroup):
    """Claims ``<CONNECTOR_ID>__<name>`` from the credential slot of its ``READER_CLASS``.

    Query families claim only with ``query_text`` set. Whole rows load on ``PythonDictFramework``, which takes
    the reader's ``{feature_name: [row, ...]}`` frame as-is; a ``~<row_key>`` projection loads on any framework.
    """

    READER_CLASS: ClassVar[type[KgConnectorReaderBase]] = _UnpinnedReader  # type: ignore[assignment]
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {HANDLE_OPTION: HANDLE_SPEC}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        pinned = issubclass(cls.READER_CLASS, KgConnectorReaderBase)
        query = pinned and issubclass(cls.READER_CLASS, QueryReader)
        if query and QUERY_OPTION not in cls.PROPERTY_MAPPING:
            cls.PROPERTY_MAPPING = {**cls.PROPERTY_MAPPING, QUERY_OPTION: _QUERY_SPEC}
        super().__init_subclass__(**kwargs)
        if pinned:
            required = (QUERY_OPTION,) if query else ()
            cls.CLAIM_ROUTES = (ClaimRoute("credentials", NamePolicy.DECLARED, True, required),)

    @classmethod
    def declared_name_prefixes(cls) -> tuple[str, ...]:
        return (f"{cls.READER_CLASS.CONNECTOR_ID}{CHAIN_SEPARATOR}",)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        """Gate on the KG name shape first; record why a pointed or ``query_text``-less request declines."""
        name, group, reader = str(feature_name), cls.get_class_name(), cls.READER_CLASS
        if issubclass(reader, KgConnectorReaderBase):
            if not reader.claims_feature_name(name):
                if cls.is_pointed(name, options, data_access_collection):
                    reason = f"{group} claims only '{reader.CONNECTOR_ID}__<name>' names, not '{name}'"
                    record_match_rejection(group, reason, stage=INPUT_DATA_STAGE)
                return False
            # An absent required option skips the claim route without a reason; a blank one fails _QUERY_SPEC.
            if cls.serves_query_text() and options.get(QUERY_OPTION) is None:
                record_match_rejection(
                    group, f"{group} needs the option '{QUERY_OPTION}' for '{name}'", stage=INPUT_DATA_STAGE
                )
                return False
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        """Valid slots from the pointer, else the collection narrowed by ``data_access_handle``."""
        reader = cls.READER_CLASS
        pointer = next((key for key in cls.pointer_keys() if key in options), None)
        if pointer is not None:
            value = options.get(pointer)
            candidates = [(pointer, value.data if isinstance(value, Credential) else value)]
        elif data_access_collection is None:
            return []
        else:
            registry = data_access_collection.credentials
            handle = options.get(HANDLE_OPTION)
            if isinstance(handle, str) and handle not in registry:
                reason = f"data_access_handle '{handle}' is unknown; credential handles: {sorted(registry)}"
                record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
            candidates = [
                (name, creds) for name, creds in registry.items() if not isinstance(handle, str) or name == handle
            ]
        valid: list[tuple[str, Any]] = []
        for handle, creds in candidates:
            if reader.is_valid_credentials(creds) and all(creds != seen for _, seen in valid):
                valid.append((handle, creds))
        identities = [reader.data_access_identity(creds) for _, creds in valid]
        # Sources must be unique per access: same-source slots differ by handle, so they abort as ambiguous.
        return [
            SourceMatch(identity if identities.count(identity) == 1 else f"{identity}@{handle}", creds)
            for identity, (handle, creds) in zip(identities, valid)
        ]

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: FeatureSet) -> Any:
        return cls.READER_CLASS.load(match.access, features)

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        return cls.READER_CLASS.count_rows(match.access, compute_framework)

    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        group = cls.get_class_name()
        return (
            f"select one credential with data_access_handle, or point {group} at one with "
            f"options={{{group!r}: Credential({cls.READER_CLASS.CONNECTOR_ID}={{...}})}}."
        )

    @classmethod
    def supports_compute_framework(
        cls, feature_name: FeatureName | str, options: Options, compute_framework: type[ComputeFramework]
    ) -> bool:
        """Whole rows hold native objects only PythonDict keeps; a ``~<row_key>`` projection is a plain column."""
        if not issubclass(compute_framework, PythonDictFramework) and COLUMN_SEPARATOR not in str(feature_name):
            return False
        return super().supports_compute_framework(feature_name, options, compute_framework)
