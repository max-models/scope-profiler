"""Keep user-facing format documentation aligned with reader constants."""

import re
from pathlib import Path

import pytest

from scope_profiler.h5schema import CURRENT_SCHEMA_VERSION
from scope_profiler.json_export import FORMAT_VERSION as JSON_FORMAT_VERSION
from scope_profiler.native_trace import FORMAT_VERSION as NATIVE_FORMAT_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FORMAT_GUIDE = REPOSITORY_ROOT / "docs/source/guide/hdf5_and_python_api.qmd"


@pytest.mark.skipif(
    not FORMAT_GUIDE.exists(), reason="repository documentation not installed"
)
def test_hdf5_guide_names_the_current_schema():
    text = FORMAT_GUIDE.read_text(encoding="utf-8")
    match = re.search(r"New files currently use schema version `([0-9]+)`", text)

    assert match, "the HDF5 guide must state the current schema version"
    assert int(match.group(1)) == CURRENT_SCHEMA_VERSION


def test_format_versions_are_deliberately_independent():
    """Changing one format must not silently renumber the other formats."""
    assert CURRENT_SCHEMA_VERSION == 3
    assert JSON_FORMAT_VERSION == 1
    assert NATIVE_FORMAT_VERSION == 2
