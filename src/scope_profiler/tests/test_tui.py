"""Tests for the Textual HDF5 browser's data model."""

import numpy as np
import pytest

from scope_profiler.__main__ import main as cli_main
from scope_profiler.h5writer import ProfilingWriter
from scope_profiler.profile_manager import RankPayload
from scope_profiler.tests.test_inspection import _write_sample_h5
from scope_profiler.tui import (
    _matplotlib_child_script,
    build_browser_model,
    node_detail_text,
    render_plot,
)

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


@pytest.fixture
def line_profile_file(tmp_path):
    path = tmp_path / "line_profile.h5"
    record = {
        "region": "solve",
        "filename": str(tmp_path / "app.py"),
        "function": "solve",
        "first_lineno": 10,
        "line_numbers": np.asarray([11, 12]),
        "hits": np.asarray([1, 5]),
        "times": np.asarray([10.0, 25.0]),
        "unit": 1e-9,
    }
    (tmp_path / "app.py").write_text(
        "\n" * 10 + "total = 0\nfor i in range(5):\n", encoding="utf-8"
    )
    payload = RankPayload(
        regions={"solve": (np.asarray([0]), np.asarray([NS]))},
        likwid={},
        likwid_environment={},
        line_profile=[record],
    )
    with ProfilingWriter(path) as writer:
        writer.write_rank(0, payload)
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
        "Plots",
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
    assert {"Summary", "Calls", "Source"} <= {child.label for child in solve.children}
    assert "Rank 0" not in {child.label for child in solve.children}
    assert "Calls: 2" in node_detail_text(solve)
    assert "kernels.py:7" in node_detail_text(_find(solve, "Source"))

    calls = _find(solve, "Calls")
    assert "Rank" in node_detail_text(calls)
    rank_calls = _find(calls, "Rank 0")
    rank_call_details = node_detail_text(rank_calls)
    assert all(column in rank_call_details for column in ("#", "Start", "Duration"))
    assert "2" in rank_call_details

    start_times = _find(model.root, "start_times")
    assert "HDF5 dataset" in node_detail_text(start_times)
    assert "Dtype: int64" in node_detail_text(start_times)


def test_line_profile_records_are_clickable(line_profile_file):
    model = build_browser_model(line_profile_file)

    assert [child.label for child in model.root.children] == [
        "Overview",
        "Plots",
        "Metadata",
        "Regions",
        "Raw HDF5",
    ]

    solve = _find(_find(model.root, "Regions"), "solve")
    line_profile = _find(solve, "Line Profile")
    assert line_profile is not None
    assert "Function" in node_detail_text(line_profile)

    rank = _find(line_profile, "Rank 0")
    details = node_detail_text(rank)
    assert "Rank 0 | solve | solve" in details
    assert "Line" in details and "Hits" in details and "Time [s]" in details
    assert "----" not in details
    assert "11" in details and "1e-08" in details
    assert "total = 0" in details


def test_plot_section_exposes_existing_plot_kinds(sample_file):
    model = build_browser_model(sample_file)
    plots = _find(model.root, "Plots")

    assert [child.label for child in plots.children] == [
        "Gantt",
        "Flame",
        "Durations",
        "Timeseries",
        "Histogram",
        "Imbalance",
    ]
    assert "Matplotlib" in node_detail_text(plots.children[0])


def test_render_plot_dispatches_to_existing_renderer(sample_file, monkeypatch, tmp_path):
    model = build_browser_model(sample_file)
    plot = _find(model.root, "Durations")
    calls = []

    def fake_plot(results, **kwargs):
        calls.append((results, kwargs))

    monkeypatch.setattr("scope_profiler.tui.plot_durations", fake_plot)
    output = tmp_path / "durations.png"

    assert render_plot(plot, filepath=output) == str(output)
    assert calls[0][0] is model.results
    assert calls[0][1]["filepath"] == str(output)
    assert calls[0][1]["show"] is False


def test_render_plot_passes_plot_settings(sample_file, monkeypatch):
    model = build_browser_model(sample_file)
    plot = _find(model.root, "Durations")
    captured = {}

    def fake_plot(results, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("scope_profiler.tui.plot_durations", fake_plot)
    render_plot(
        plot,
        settings={
            "include": "solve,setup",
            "exclude": "debug",
            "ranks": "0,2-3",
            "cmap": "viridis",
            "metrics": "avg,max",
            "sort_by": "avg",
            "top_n": "5",
            "log_scale": True,
        },
    )

    assert captured["include"] == ["solve", "setup"]
    assert captured["exclude"] == ["debug"]
    assert captured["ranks"] == [0, 2, 3]
    assert captured["cmap"] == "viridis"
    assert captured["metrics"] == ["avg", "max"]
    assert captured["sort_by"] == "avg"
    assert captured["top_n"] == 5
    assert captured["log_scale"] is True


def test_matplotlib_child_script_is_valid_python():
    compile(_matplotlib_child_script(), "<matplotlib-child>", "exec")


def test_tui_help_does_not_require_textual(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["tui", "--help"])

    assert exc_info.value.code == 0
    assert "scope-profiler tui" in capsys.readouterr().out


def test_tui_is_listed_in_top_level_help(capsys):
    with pytest.raises(SystemExit):
        cli_main(["--help"])

    assert "tui" in capsys.readouterr().out
