"""Keep user-facing format documentation aligned with reader constants."""

import re
from pathlib import Path

import pytest

from scope_profiler.h5schema import CURRENT_SCHEMA_VERSION
from scope_profiler.json_export import FORMAT_VERSION as JSON_FORMAT_VERSION
from scope_profiler.native_trace import FORMAT_VERSION as NATIVE_FORMAT_VERSION
from scope_profiler.plotting_scripts._utils import (
    PLOT_DATA_FORMAT,
    PLOT_DATA_FORMAT_VERSION,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FORMAT_GUIDE = REPOSITORY_ROOT / "docs/source/guide/hdf5_and_python_api.qmd"
PLOTLY_PACKAGE = REPOSITORY_ROOT / "packages/plotly/src/index.js"


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


@pytest.mark.skipif(
    not PLOTLY_PACKAGE.exists(), reason="repository packages not installed"
)
def test_the_plotly_package_reads_the_plot_data_version_we_write():
    """@scope-profiler/plotly refuses a document newer than it supports.

    That check is only as good as the constant behind it: bump the exporter's
    plot-data version without bumping the package and every dashboard rejects
    the files the CLI has just written, with a message telling users to
    upgrade a package that is already current.
    """
    text = PLOTLY_PACKAGE.read_text(encoding="utf-8")
    fmt = re.search(r'PLOT_DATA_FORMAT = "([^"]+)"', text)
    version = re.search(r"SUPPORTED_FORMAT_VERSION = ([0-9]+)", text)

    assert fmt and version, "the plotly package must state the format it reads"
    assert fmt.group(1) == PLOT_DATA_FORMAT
    assert int(version.group(1)) == PLOT_DATA_FORMAT_VERSION
