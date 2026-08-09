import json

import h5py
import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.call_stack import build_call_stack
from scope_profiler.plotting_scripts import (
    _duration_timeseries,
    plot_duration_timeseries,
    plot_durations,
    plot_flame,
    plot_gantt,
    plot_speedup,
)
from scope_profiler.post_processing import main


def _write_sample_h5(path, rank_regions, metadata=None):
    with h5py.File(path, "w") as h5file:
        if metadata:
            meta_grp = h5file.create_group("metadata")
            for key, value in metadata.items():
                meta_grp.attrs[key] = value
        for rank, regions in rank_regions.items():
            rank_group = h5file.create_group(f"rank{rank}")
            regions_group = rank_group.create_group("regions")
            for region_name, (start_times, end_times) in regions.items():
                region_group = regions_group.create_group(region_name)
                region_group.create_dataset(
                    "start_times",
                    data=np.asarray(start_times, dtype=np.int64),
                )
                region_group.create_dataset(
                    "end_times",
                    data=np.asarray(end_times, dtype=np.int64),
                )


def _seconds(nanoseconds):
    """Timestamps are written in nanoseconds; the reader reports seconds."""
    return [value / 1e9 for value in nanoseconds]


def _sample_file_data(rank_count, setup_duration, solve_duration):
    return {
        rank: {
            "setup": ([0], [setup_duration]),
            "solve": ([20], [20 + solve_duration]),
        }
        for rank in range(rank_count)
    }


def test_plot_durations_comparison(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    out_file = tmp_path / "durations_plot.png"

    _write_sample_h5(file_one, _sample_file_data(2, 10, 20))
    _write_sample_h5(file_two, _sample_file_data(2, 20, 40))

    readers = [read_h5(file_one), read_h5(file_two)]

    saved_paths = plot_durations(
        readers,
        filepath=out_file,
        show=False,
        verbose=False,
        metrics=["avg", "min", "max", "total"],
    )

    assert len(saved_paths) == 4
    for metric in ("avg", "min", "max", "total"):
        metric_file = tmp_path / f"durations_plot_{metric}.png"
        assert metric_file.exists()
        assert metric_file.stat().st_size > 0


def test_duration_timeseries_bands_span_ranks(tmp_path):
    file_path = tmp_path / "run.h5"
    # Two calls of "solve" per rank, with rank 1 slower on the second call, so
    # the band has to widen between the two points.
    _write_sample_h5(
        file_path,
        {
            0: {"solve": ([0, 100], [10, 110])},
            1: {"solve": ([0, 100], [20, 140])},
        },
    )
    reader = read_h5(file_path)

    series = _duration_timeseries(reader.get_region("solve"), None, 0.0)

    assert list(series["num_ranks"]) == [2, 2]
    assert series["min"] == pytest.approx(_seconds([10, 10]))
    assert series["max"] == pytest.approx(_seconds([20, 40]))
    assert series["mean"] == pytest.approx(_seconds([15, 25]))
    assert series["time"] == pytest.approx(_seconds([0, 100]))


def test_duration_timeseries_handles_ragged_call_counts(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(
        file_path,
        {
            0: {"solve": ([0, 100], [10, 130])},
            1: {"solve": ([0], [20])},
        },
    )
    reader = read_h5(file_path)

    series = _duration_timeseries(reader.get_region("solve"), None, 0.0)

    # The second call exists on rank 0 only, so its band collapses to a point.
    assert list(series["num_ranks"]) == [2, 1]
    assert series["min"] == pytest.approx(_seconds([10, 30]))
    assert series["max"] == pytest.approx(_seconds([20, 30]))


def test_duration_timeseries_respects_rank_selection(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(
        file_path,
        {
            0: {"solve": ([0], [10])},
            1: {"solve": ([0], [50])},
        },
    )
    reader = read_h5(file_path)

    series = _duration_timeseries(reader.get_region("solve"), [0], 0.0)

    assert list(series["num_ranks"]) == [1]
    assert series["min"] == pytest.approx(series["max"])
    assert series["max"] == pytest.approx(_seconds([10]))


def test_plot_duration_timeseries_export_data_json(tmp_path):
    file_path = tmp_path / "run.h5"
    data_file = tmp_path / "duration_timeseries_data.json"

    _write_sample_h5(file_path, _sample_file_data(2, 10, 20))
    reader = read_h5(file_path)

    plot_duration_timeseries(
        reader,
        filepath=tmp_path / "duration_timeseries_plot.png",
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
    )

    assert (tmp_path / "duration_timeseries_plot.png").stat().st_size > 0
    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert {point["region"] for point in payload["points"]} == {"setup", "solve"}
    for point in payload["points"]:
        assert point["min_duration_seconds"] <= point["mean_duration_seconds"]
        assert point["mean_duration_seconds"] <= point["max_duration_seconds"]
        assert point["num_ranks"] == 2
    assert set(payload["colors"]) == {"setup", "solve"}


def test_plot_gantt_combined(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    out_file = tmp_path / "gantt_plot.png"

    _write_sample_h5(file_one, _sample_file_data(2, 10, 20))
    _write_sample_h5(file_two, _sample_file_data(2, 20, 40))

    readers = [read_h5(file_one), read_h5(file_two)]

    plot_gantt(readers, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_gantt_puts_every_call_of_a_region_on_one_lane(tmp_path, monkeypatch):
    """Repeated calls of a region share one row per rank, not a row each."""
    h5_path = tmp_path / "run.h5"
    _write_sample_h5(
        h5_path,
        {
            rank: {
                "setup": ([0], [10]),
                "solve": ([20, 40, 60], [30, 50, 70]),
            }
            for rank in range(2)
        },
    )

    bars_by_lane = {}

    class _RecordingCanvas:
        """Collect what plot_gantt draws instead of rendering it."""

        def __init__(self, *args, **kwargs):
            self.lanes = None

        def gantt(self, tasks, start_times, durations, **kwargs):
            self.lanes = tasks
            for lane, start in enumerate(start_times):
                if not np.isnan(start):
                    bars_by_lane.setdefault(lane, []).append(float(start))

        def __getattr__(self, name):
            # set_yticks, set_xlim, set_title, ... are all no-ops here.
            return lambda *args, **kwargs: None

    import scope_profiler.plotting_scripts as plotting_scripts

    canvas = _RecordingCanvas()
    monkeypatch.setattr(
        plotting_scripts, "_get_canvas", lambda: lambda *a, **kw: canvas
    )
    monkeypatch.setattr(plotting_scripts, "_render", lambda *a, **kw: None)

    plot_gantt(read_h5(h5_path), show=False, verbose=False)

    # One lane per (region, rank), and solve's three calls share their lane.
    assert canvas.lanes == [
        "setup (rank 0)",
        "setup (rank 1)",
        "solve (rank 0)",
        "solve (rank 1)",
    ]
    assert sorted(bars_by_lane) == [0, 1, 2, 3]
    assert [len(bars_by_lane[lane]) for lane in sorted(bars_by_lane)] == [1, 1, 3, 3]


def test_build_call_stack_reconstructs_nesting(tmp_path):
    # "outer" [0, 100) encloses two sequential "inner" calls, [10, 40) and
    # [50, 90), which in turn each enclose a "leaf" call.
    rank_regions = {
        0: {
            "outer": ([0], [100]),
            "inner": ([10, 50], [40, 90]),
            "leaf": ([15, 55], [20, 60]),
        }
    }
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, rank_regions)
    reader = read_h5(file_path)
    calls = build_call_stack(reader.get_regions(), rank=0)

    # Region.start_times converts stored nanoseconds to seconds.
    depths = {(call["name"], call["start"]): call["depth"] for call in calls}
    assert depths[("outer", 0.0)] == 0
    assert depths[("inner", 10e-9)] == 1
    assert depths[("inner", 50e-9)] == 1
    assert depths[("leaf", 15e-9)] == 2
    assert depths[("leaf", 55e-9)] == 2


def test_plot_flame_reconstructs_recursive_calls(tmp_path):
    out_file = tmp_path / "flame_plot.png"
    file_path = tmp_path / "run.h5"

    # Three nested "fib" calls emulating a self-recursive region: the
    # buffer-slot fix means each recursive call gets its own (start, end)
    # pair rather than overwriting the outer call's.
    rank_regions = {
        0: {
            "fib": ([0, 10, 60], [100, 90, 80]),
        }
    }
    _write_sample_h5(file_path, rank_regions)
    reader = read_h5(file_path)

    plot_flame(reader, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0

    calls = build_call_stack(reader.get_regions(), rank=0)
    assert len(calls) == 3
    depths = sorted(call["depth"] for call in calls)
    assert depths == [0, 1, 2]


def test_plot_speedup(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    file_four = tmp_path / "run_4.h5"
    out_file = tmp_path / "speedup_plot.png"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))
    _write_sample_h5(file_four, _sample_file_data(4, 25, 50))

    readers = [
        read_h5(file_one),
        read_h5(file_two),
        read_h5(file_four),
    ]

    plot_speedup(readers, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_speedup_x_field_omp_num_threads(tmp_path):
    file_4 = tmp_path / "threads_4.h5"
    file_1 = tmp_path / "threads_1.h5"
    file_2 = tmp_path / "threads_2.h5"

    # Written out of numeric order to confirm the x-axis is sorted
    # numerically rather than following file/CLI order.
    _write_sample_h5(
        file_4, _sample_file_data(1, 25, 50), metadata={"omp_num_threads": 4}
    )
    _write_sample_h5(
        file_1, _sample_file_data(1, 100, 200), metadata={"omp_num_threads": 1}
    )
    _write_sample_h5(
        file_2, _sample_file_data(1, 50, 100), metadata={"omp_num_threads": 2}
    )
    readers = [
        read_h5(file_4),
        read_h5(file_1),
        read_h5(file_2),
    ]

    data_file = tmp_path / "speedup_data.csv"
    plot_speedup(
        readers,
        x_field="omp_num_threads",
        show=False,
        verbose=False,
        data_filepath=data_file,
    )

    rows = [row.split(",") for row in data_file.read_text().strip().splitlines()[1:]]
    thread_values = sorted({int(row[1]) for row in rows})
    assert thread_values == [1, 2, 4]

    # Baseline (1 thread) should have speedup 1.0 for both regions.
    baseline_speedups = {float(row[2]) for row in rows if row[1] == "1"}
    assert baseline_speedups == {1.0}


def test_plot_speedup_x_field_total_cores(tmp_path):
    file_small = tmp_path / "small.h5"
    file_big = tmp_path / "big.h5"

    _write_sample_h5(
        file_small,
        _sample_file_data(1, 100, 200),
        metadata={"omp_num_threads": 1},
    )
    _write_sample_h5(
        file_big,
        _sample_file_data(2, 25, 50),
        metadata={"omp_num_threads": 2},
    )
    readers = [read_h5(file_small), read_h5(file_big)]

    data_file = tmp_path / "speedup_data.csv"
    plot_speedup(
        readers,
        x_field="total_cores",
        show=False,
        verbose=False,
        data_filepath=data_file,
    )

    rows = [row.split(",") for row in data_file.read_text().strip().splitlines()[1:]]
    core_values = sorted({int(row[1]) for row in rows})
    # file_small: 1 rank * 1 thread = 1; file_big: 2 ranks * 2 threads = 4.
    assert core_values == [1, 4]


def test_plot_speedup_categorical_field_preserves_cli_order_and_skips_ideal_line(
    tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    file_b = tmp_path / "b.h5"
    file_a = tmp_path / "a.h5"

    # Intentionally not alphabetically ordered on disk, so a value-based sort
    # would reorder them; the CLI order below (b, then a) must be preserved.
    _write_sample_h5(
        file_b, _sample_file_data(1, 50, 100), metadata={"build_variant": "b_variant"}
    )
    _write_sample_h5(
        file_a,
        _sample_file_data(1, 100, 200),
        metadata={"build_variant": "a_variant"},
    )
    readers = [read_h5(file_b), read_h5(file_a)]

    captured = {}
    original_close = plt.close

    def fake_close(fig=None):
        ax = fig.get_axes()[0]
        captured["labels"] = [line.get_label() for line in ax.get_lines()]
        captured["xticklabels"] = [t.get_text() for t in ax.get_xticklabels()]
        original_close(fig)

    monkeypatch.setattr(plt, "close", fake_close)

    # A figure is only built when the plot is actually saved or shown, so
    # write one out to inspect the axes maxplotlib produced.
    plot_speedup(
        readers,
        x_field="build_variant",
        filepath=str(tmp_path / "speedup.png"),
        show=False,
        verbose=False,
    )

    assert captured["xticklabels"] == ["b_variant", "a_variant"]
    assert "Ideal scaling" not in captured["labels"]


def test_plot_speedup_unknown_metadata_field_raises(tmp_path):
    file_a = tmp_path / "a.h5"
    file_b = tmp_path / "b.h5"
    _write_sample_h5(file_a, _sample_file_data(1, 10, 20))
    _write_sample_h5(file_b, _sample_file_data(2, 5, 10))
    readers = [read_h5(file_a), read_h5(file_b)]

    with pytest.raises(ValueError, match="not found"):
        plot_speedup(readers, x_field="nonexistent_field", show=False, verbose=False)


def test_post_processing_cli_supports_multiple_files(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    file_four = tmp_path / "run_4.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))
    _write_sample_h5(file_four, _sample_file_data(4, 25, 50))

    main([str(file_one), str(file_two), str(file_four), "-o", str(output_dir)])

    gantt_plot = output_dir / "gantt_plot.png"
    flame_plot = output_dir / "flame_plot.png"
    timeseries_plot = output_dir / "duration_timeseries_plot.png"
    speedup_plot = output_dir / "speedup_plot.png"
    stats_json = output_dir / "region_statistics.json"

    assert gantt_plot.exists()
    assert gantt_plot.stat().st_size > 0
    assert flame_plot.exists()
    assert flame_plot.stat().st_size > 0
    durations_plot = output_dir / "durations_plot.png"
    assert durations_plot.exists()
    assert durations_plot.stat().st_size > 0
    assert timeseries_plot.exists()
    assert timeseries_plot.stat().st_size > 0
    assert speedup_plot.exists()
    assert speedup_plot.stat().st_size > 0
    assert stats_json.exists()
    assert stats_json.stat().st_size > 0
    payload = json.loads(stats_json.read_text(encoding="utf-8"))
    assert payload["units"]["durations"] == "seconds"
    assert payload["common_regions"] == ["setup", "solve"]
    assert len(payload["files"]) == 3
    assert payload["files"][0]["region_statistics"]["setup"]["count"] == 1
    assert payload["files"][1]["region_statistics"]["setup"]["count"] == 2
    assert payload["files"][2]["region_statistics"]["setup"]["count"] == 4


def test_post_processing_cli_supports_wildcard_file_patterns(tmp_path):
    file_one = tmp_path / "file_1.h5"
    file_two = tmp_path / "file_2.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))

    wildcard_pattern = str(tmp_path / "file_*.h5")
    main([wildcard_pattern, "-o", str(output_dir)])

    gantt_plot = output_dir / "gantt_plot.png"
    flame_plot = output_dir / "flame_plot.png"
    speedup_plot = output_dir / "speedup_plot.png"
    stats_json = output_dir / "region_statistics.json"

    assert gantt_plot.exists()
    assert gantt_plot.stat().st_size > 0
    assert flame_plot.exists()
    assert flame_plot.stat().st_size > 0
    durations_plot = output_dir / "durations_plot.png"
    assert durations_plot.exists()
    assert durations_plot.stat().st_size > 0
    assert speedup_plot.exists()
    assert speedup_plot.stat().st_size > 0
    assert stats_json.exists()
    payload = json.loads(stats_json.read_text(encoding="utf-8"))
    assert len(payload["files"]) == 2
    assert payload["common_regions"] == ["setup", "solve"]


def test_plot_gantt_export_data_json(tmp_path):
    file_path = tmp_path / "run.h5"
    data_file = tmp_path / "gantt_data.json"

    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))
    reader = read_h5(file_path)

    plot_gantt(
        reader,
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
    )

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert {"setup", "solve"} <= set(payload["colors"])
    assert all(color.startswith("#") for color in payload["colors"].values())
    regions = {interval["region"] for interval in payload["intervals"]}
    assert regions == {"setup", "solve"}


def test_plot_flame_export_data_json(tmp_path):
    file_path = tmp_path / "run.h5"
    data_file = tmp_path / "flame_data.json"

    rank_regions = {0: {"fib": ([0, 10, 60], [100, 90, 80])}}
    _write_sample_h5(file_path, rank_regions)
    reader = read_h5(file_path)

    plot_flame(
        reader,
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
    )

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert payload["colors"]["fib"].startswith("#")
    depths = sorted(call["depth"] for call in payload["calls"])
    assert depths == [0, 1, 2]


def test_plot_durations_export_data_json(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    data_file = tmp_path / "durations_data.json"

    _write_sample_h5(file_one, _sample_file_data(2, 10, 20))
    _write_sample_h5(file_two, _sample_file_data(2, 20, 40))
    readers = [read_h5(file_one), read_h5(file_two)]

    plot_durations(
        readers,
        filepath=tmp_path / "durations_plot.png",
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
        metrics=["avg", "min", "max", "total"],
    )

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert set(payload["metrics"]) == {"avg", "min", "max", "total"}
    assert set(payload["colors"]) == {"run_one", "run_two"}
    assert all(color.startswith("#") for color in payload["colors"].values())
    assert {bar["metric"] for bar in payload["bars"]} == {"avg", "min", "max", "total"}


def test_plot_speedup_export_data_json(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    data_file = tmp_path / "speedup_data.json"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))
    readers = [read_h5(file_one), read_h5(file_two)]

    plot_speedup(
        readers,
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
    )

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert {"setup", "solve"} <= set(payload["colors"])
    assert all(color.startswith("#") for color in payload["colors"].values())
    assert {point["region"] for point in payload["points"]} == {"setup", "solve"}


def test_post_processing_cli_export_data_format_json(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))

    main(
        [
            str(file_one),
            str(file_two),
            "-o",
            str(output_dir),
            "--export-data",
            "--export-data-format",
            "json",
        ]
    )

    for name in (
        "gantt_data.json",
        "flame_data.json",
        "durations_data.json",
        "speedup_data.json",
    ):
        data_file = output_dir / name
        assert data_file.exists()
        json.loads(data_file.read_text(encoding="utf-8"))
        assert not (output_dir / name.replace(".json", ".csv")).exists()


def test_post_processing_cli_skip_plot_images(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))

    main(
        [
            str(file_one),
            str(file_two),
            "-o",
            str(output_dir),
            "--export-data",
            "--export-data-format",
            "json",
            "--skip-plot-images",
        ]
    )

    for name in (
        "gantt_data.json",
        "flame_data.json",
        "durations_data.json",
        "speedup_data.json",
        "region_statistics.json",
    ):
        assert (output_dir / name).exists()

    assert list(output_dir.glob("*.png")) == []


def test_post_processing_cli_skip_plot_images_requires_export_data(tmp_path):
    file_one = tmp_path / "run_1.h5"
    output_dir = tmp_path / "figures"
    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))

    with pytest.raises(SystemExit):
        main([str(file_one), "-o", str(output_dir), "--skip-plot-images"])
