"""Versioning and compatibility helpers for scope-profiler HDF5 files."""

from __future__ import annotations

from numbers import Integral

SCHEMA_ATTRIBUTE = "scope_profiler_schema"
CURRENT_SCHEMA_VERSION = 2


class HDF5SchemaError(ValueError):
    """Raised when a profiling file has an invalid or unsupported schema."""


def read_schema_version(h5file) -> int:
    """Return and validate the schema version stored on an HDF5 file.

    Files written before schema versioning was introduced have no attribute;
    they are the original layout and are therefore treated as version 1.
    """
    raw_version = h5file.attrs.get(SCHEMA_ATTRIBUTE, 1)
    if isinstance(raw_version, bool) or not isinstance(raw_version, Integral):
        raise HDF5SchemaError(
            f"HDF5 schema version attribute {SCHEMA_ATTRIBUTE!r} must be an integer, "
            f"got {raw_version!r}",
        )

    version = int(raw_version)
    if version < 1:
        raise HDF5SchemaError(
            f"Unsupported scope-profiler HDF5 schema version {version}; "
            "versions must be positive",
        )
    if version > CURRENT_SCHEMA_VERSION:
        raise HDF5SchemaError(
            f"Unsupported scope-profiler HDF5 schema version {version}; "
            f"this package supports versions through {CURRENT_SCHEMA_VERSION}. "
            "Upgrade scope-profiler to read this file.",
        )

    return version


def migrate_schema(h5file, version: int) -> int:
    """Normalize a supported file schema for the reader.

    This is intentionally a separate dispatch point even while version 1 is
    the only schema. Future readers can add transformations for older layouts
    here without spreading version checks through the HDF5 parser. Migration
    is read-only: the input file is never modified.
    """
    if version in {1, 2}:
        return version
    raise HDF5SchemaError(f"No migration registered for schema version {version}")
