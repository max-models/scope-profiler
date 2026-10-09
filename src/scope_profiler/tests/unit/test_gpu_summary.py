"""GPU summaries keep device time separate from CPU enqueue time."""

from io import StringIO

import h5py
import numpy as np
import pytest

from scope_profiler import MPIRegion, ProfilingResults, Region
from scope_profiler.diff import diff_files, diff_rows
from scope_profiler.html_report import create_html_report
from scope_profiler.inspection import inspect_file
from scope_profiler.summary import format_region_table, gpu_timing_warnings, region_rows


def _region(starts, ends, gpu=None):
    return Region(np.array(starts), np.array(ends), gpu_durations=gpu)


@pytest.fixture
def results():
    return ProfilingResults(
        {
            "a": MPIRegion("a", {0: _region([0], [100])}),
            "b": MPIRegion("b", {0: _region([100], [200])}),
            "work": MPIRegion(
                "work",
                {
                    0: _region([10, 110], [20, 120], [100, 300]),
                    1: _region([0], [10]),
                },
            ),
        }
    )


def test_gpu_summary_pools_by_path_and_filters_ranks(results):
    rows = region_rows(results, ranks=[0])
    work = [row for row in rows if row["name"] == "work"]
    assert [row["gpu_total"] for row in work] == pytest.approx([1e-7, 3e-7])
    assert [row["gpu_avg"] for row in work] == pytest.approx([1e-7, 3e-7])
    columns, _ = format_region_table(rows)
    assert [key for key, _ in columns][-2:] == ["gpu_total", "gpu_avg"]
    assert gpu_timing_warnings(rows)
    cpu_rows = region_rows(results, ranks=[1])
    assert all(row["gpu_total"] is None for row in cpu_rows)
    assert not gpu_timing_warnings(cpu_rows)
    assert "gpu_total" not in dict(format_region_table(cpu_rows)[0])
    assert "gpu_total" not in dict(format_region_table(rows, ["region", "total"])[0])


def test_gpu_diff_pools_all_paths_and_only_gpu_calls(results):
    for metric, expected in (("gpu_total", 4e-7), ("gpu_avg", 2e-7)):
        rows = diff_rows(results, results, metric=metric)
        assert len(rows) == 1
        assert rows[0]["a"] == pytest.approx(expected)
        assert rows[0]["delta"] == 0


def test_gpu_aggregate_summary():
    region = Region(
        np.array([], dtype=int),
        np.array([], dtype=int),
        aggregate={
            "count": 3,
            "total": 30,
            "minimum": 10,
            "maximum": 10,
            "gpu_count": 2,
            "gpu_total": 400,
        },
        event_data_available=False,
    )
    results = ProfilingResults({"work": MPIRegion("work", {0: region})})
    row = region_rows(results)[0]
    assert row["gpu_total"] == pytest.approx(4e-7)
    assert row["gpu_avg"] == pytest.approx(2e-7)


def test_gpu_file_inspect_diff_and_report(tmp_path):
    path = tmp_path / "gpu.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("rank0/regions/work")
        group["start_times"] = [0, 10_000_000]
        group["end_times"] = [1_000_000, 11_000_000]
        group["gpu_durations"] = [5_000_000, 7_000_000]
    stream = StringIO()
    inspect_file(path, stream=stream)
    assert "gpu total [s]" in stream.getvalue()
    assert "0.012000" in stream.getvalue()
    assert "asynchronous GPU work" in stream.getvalue()
    stream = StringIO()
    diff_files(path, path, metric="gpu_total", stream=stream)
    assert "0.012" in stream.getvalue()
    report = create_html_report(path, tmp_path / "report.html", include_charts=False)
    document = report.read_text()
    assert "gpu total [s]" in document
    assert "asynchronous GPU work" in document


def test_gpu_finalize_and_summary_only_roundtrip(tmp_path, capsys):
    from scope_profiler import GPUOptions, ProfileManager, ProfilingOptions
    from scope_profiler.profile_io import read_profile_summary

    class Backend:
        def record_event(self):
            return object()

        def elapsed_time_ns(self, start, end):
            return 1_000_000_000

        def synchronize(self):
            pass

    path = tmp_path / "profile.h5"
    ProfileManager.setup(
        file_path=str(path),
        options=ProfilingOptions(
            gpu=GPUOptions(
                timing=True,
                backend=Backend(),
                sync_on_exit=True,
            )
        ),
    )
    with ProfileManager.profile_region("work"):
        pass
    ProfileManager.finalize()
    output = capsys.readouterr().out
    assert "gpu total [s]" in output
    assert "asynchronous GPU work" in output
    results = read_profile_summary(path)
    region = results.get_region("work").regions[0]
    assert not region.has_event_data
    row = next(row for row in region_rows(results) if row["name"] == "work")
    assert row["gpu_total"] == 1.0
    assert row["gpu_avg"] == 1.0
    stream = StringIO()
    diff_files(path, path, metric="gpu_avg", stream=stream)
    assert "gpu avg [s]" in stream.getvalue()


def test_gpu_sync_toml_option(tmp_path):
    from scope_profiler.profile_config import load_profiling_config

    path = tmp_path / "profile.toml"
    path.write_text(
        '[profiling.gpu]\ntiming = true\nbackend = "cupy"\nsync_on_exit = true\n'
    )
    options = load_profiling_config(path)
    assert options["gpu_sync_on_exit"] is True
    assert options["use_gpu_timing"] is True
