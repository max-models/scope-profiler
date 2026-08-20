"""Tests for the Textual HDF5 browser's data model."""

import pytest

from scope_profiler.__main__ import main as cli_main
from scope_profiler.tests.test_inspection import _write_sample_h5
from scope_profiler.tui import build_browser_model, node_detail_text

NS = 1_000_000_000


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "profiling_data.h5"
    _write_sample_h5(
        path,
        {
            0: {"setup": ([0], [NS]), "solve": ([2 * NS], [4 * NS])},
            1: {"solve": ([2 * NS], [5 * NS])},
        },
        metadata={"timestamp": "2026-08-20T12:00:00", "modules": ["gcc", "python"]},
        sources={
            "solve": ("kernels.py", 7, "with profile_region('solve'):\n    pass\n")
        },
    )
    return path


def _find(node, label):
    if node.label == label:
        return node
    for child in node.children:
        found = _find(child, label)
        if found is not None:
            return found
    return None


def test_browser_model_exposes_major_sections(sample_file):
    model = build_browser_model(sample_file)

    assert [child.label for child in model.root.children] == [
        "Overview",
        "Metadata",
        "Regions",
        "Raw HDF5",
    ]

    regions = _find(model.root, "Regions")
    assert [child.label for child in regions.children] == ["setup", "solve"]


def test_region_details_include_ranks_calls_source_and_raw_hdf5(sample_file):
    model = build_browser_model(sample_file)

    solve = _find(model.root, "solve")
    assert solve is not None
    assert {"Summary", "Rank 0", "Rank 1", "Source"} <= {
        child.label for child in solve.children
    }
    assert "Calls: 2" in node_detail_text(solve)
    assert "kernels.py:7" in node_detail_text(_find(solve, "Source"))

    calls = _find(_find(solve, "Rank 0"), "Calls")
    assert "#  Start" in node_detail_text(calls)
    assert "2" in node_detail_text(calls)

    start_times = _find(model.root, "start_times")
    assert "HDF5 dataset" in node_detail_text(start_times)
    assert "Dtype: int64" in node_detail_text(start_times)


def test_tui_help_does_not_require_textual(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["tui", "--help"])

    assert exc_info.value.code == 0
    assert "scope-profiler tui" in capsys.readouterr().out


def test_tui_is_listed_in_top_level_help(capsys):
    with pytest.raises(SystemExit):
        cli_main(["--help"])

    assert "tui" in capsys.readouterr().out
