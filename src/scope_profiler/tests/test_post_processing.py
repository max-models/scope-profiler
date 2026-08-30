import csv
import json
import sys
import types

import h5py
import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.call_stack import build_call_stack
from scope_profiler.likwid_data import LikwidRegionResult
from scope_profiler.plotting_scripts import (
    _display_matplotlib_figure_in_notebook,
    _duration_timeseries,
    _group_regions,
    _hover_summary,
    _render,
    _set_xticks,
    _stacked_segments,
    available_likwid_metrics,
    collect_region_statistics,
    plot_duration_histogram,
    plot_duration_timeseries,
    plot_durations,
    plot_flame,
    plot_gantt,
    plot_imbalance,
    plot_likwid,
    plot_rank_heatmap,
    plot_scaling_efficiency,
    plot_speedup,
    plot_weak_scaling,
)
from scope_profiler.post_processing import export_main, main
from scope_profiler.results import ProfilingResults


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
    """Timestamps are written in nanoseconds; the API reports seconds."""
    return [value / 1e9 for value in nanoseconds]


def _sample_file_data(rank_count, setup_duration, solve_duration):
    return {
        rank: {
            "setup": ([0], [setup_duration]),
            "solve": ([20], [20 + solve_duration]),
        }
        for rank in range(rank_count)
    }


def test_matplotlib_show_uses_ipython_display_in_jupyter(monkeypatch):
    displayed = []
    fig = object()

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: types.SimpleNamespace(config={"IPKernelApp": {}})
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = displayed.append

    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    assert _display_matplotlib_figure_in_notebook(fig) is True
    assert displayed == [fig]


def test_matplotlib_show_falls_back_outside_jupyter(monkeypatch):
    shown = []

    class _Figure:
        def savefig(self, *args, **kwargs):
            raise AssertionError("show-only render should not save")

    class _Canvas:
        def render(self, **kwargs):
            return _Figure(), []

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: types.SimpleNamespace(config={})
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda fig: shown.append(("display", fig))

    import matplotlib.pyplot as plt

    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)
    monkeypatch.setattr(plt, "show", lambda: shown.append(("show", None)))

    _render(_Canvas(), filepath=None, show=True, backend="matplotlib")

    assert shown == [("show", None)]


def test_set_xticks_reports_whether_label_options_are_supported():
    class _NewCanvas:
        def __init__(self):
            self.calls = []

        def set_xticks(self, ticks, labels=None, **kwargs):
            self.calls.append((list(ticks), labels, kwargs))

    class _OldCanvas:
        def __init__(self):
            self.calls = []

        def set_xticks(self, ticks, labels=None):
            self.calls.append((list(ticks), labels))

    new_canvas = _NewCanvas()
    old_canvas = _OldCanvas()

    assert _set_xticks(new_canvas, [0], labels=["a"], rotation=45, ha="right") is True
    assert new_canvas.calls == [([0], ["a"], {"rotation": 45, "ha": "right"})]

    assert _set_xticks(old_canvas, [0], labels=["a"], rotation=45, ha="right") is False
    assert old_canvas.calls == [([0], ["a"])]


def test_plot_durations_comparison(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    out_file = tmp_path / "durations_plot.png"

    _write_sample_h5(file_one, _sample_file_data(2, 10, 20))
    _write_sample_h5(file_two, _sample_file_data(2, 20, 40))

    runs = [read_h5(file_one), read_h5(file_two)]

    saved_paths = plot_durations(
        runs,
        metric="avg",
        filepath=out_file,
        show=False,
        verbose=False,
    )

    assert len(saved_paths) == 1
    metric_file = out_file
    assert metric_file.exists()
    assert metric_file.stat().st_size > 0


def _nested_file_data(rank_count=1):
    """One outer region per rank, with two children and some self time."""
    return {
        rank: {
            "step": ([0, 1000], [500, 1500]),
            "assemble": ([10], [110]),
            "solve": ([200], [400]),
            "inner": ([250], [300]),
        }
        for rank in range(rank_count)
    }


def test_stacked_segments_split_self_time_from_children(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())
    run = read_h5(file_path)

    _, members = _group_regions([region.name for region in run.get_regions()], None)
    segments = _stacked_segments(run, members)

    # "step" spans 0-500 and 1000-1500: 100 ns in assemble, 200 in solve,
    # and everything else its own.
    assert segments["step"] == {"self": 700.0, "assemble": 100.0, "solve": 200.0}
    # "inner" is a child of "solve", not of "step" -- only direct children
    # become segments, and solve's self time excludes it.
    assert segments["solve"] == {"self": 150.0, "inner": 50.0}
    assert segments["inner"] == {"self": 50.0}


def test_stacked_segments_fold_same_bar_nesting_into_self(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())
    run = read_h5(file_path)

    names = [region.name for region in run.get_regions()]
    _, members = _group_regions(names, {"solver": ["solve", "inner"]})
    segments = _stacked_segments(run, members)

    # "inner" nests inside "solve", and both are the same bar, so its time
    # stays in that bar's self time instead of becoming a segment of itself.
    assert segments["solver"] == {"self": 200.0}
    assert segments["step"]["solver"] == 200.0


def test_plot_durations_stacked_bars_sum_to_the_plain_bar(tmp_path):
    file_path = tmp_path / "nested.h5"
    data_path = tmp_path / "durations.csv"
    _write_sample_h5(file_path, _nested_file_data())

    saved_paths = plot_durations(
        read_h5(file_path),
        metric="total",
        stack_children=True,
        filepath=tmp_path / "durations.png",
        data_filepath=data_path,
        show=False,
        verbose=False,
    )

    assert len(saved_paths) == 1
    rows = list(csv.DictReader(data_path.open()))
    assert {row["segment"] for row in rows} == {"self", "assemble", "solve", "inner"}
    stacked = sum(
        float(row["value_seconds"]) for row in rows if row["region"] == "step"
    )
    # 500 + 500 ns of "step", split across its own time and its children.
    assert stacked == pytest.approx(1000 / 1e9)


def test_plot_durations_rejects_stacking_a_min_or_max(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())

    with pytest.raises(ValueError, match="stack_children"):
        plot_durations(
            read_h5(file_path),
            metric="max",
            stack_children=True,
            show=False,
            verbose=False,
        )


def _plotly_figure(plot_func, *args, **kwargs):
    """Render one plot to a Plotly figure, hover text included."""
    pytest.importorskip("plotly")
    return plot_func(
        *args, backend="plotly", return_fig=True, show=False, verbose=False, **kwargs
    )


def _hover_texts(figure):
    """Every hover string in a figure, flattened out of its traces."""
    texts = []
    for trace in figure.data:
        hovertext = getattr(trace, "hovertext", None)
        if hovertext is None:
            continue
        if isinstance(hovertext, str):
            texts.append(hovertext)
            continue
        for entry in hovertext:
            texts.extend(entry if isinstance(entry, (list, tuple)) else [entry])
    return [text for text in texts if text]


def test_hover_summary_reports_every_statistic_the_region_does(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())
    region = read_h5(file_path).get_region("step")

    text = _hover_summary(region)
    lines = text.split("<br>")

    assert lines[0] == "<b>step</b>"
    # The hover box is the region's own get_summary(), not a second list of
    # statistics that could drift from it.
    # The heading replaces "name"; "inclusive_duration" is an alias of
    # "total_duration" and is not repeated.
    assert len(lines) == 1 + len(region.get_summary()) - 2
    assert "calls: 2" in lines
    assert any(line.startswith("total: 1e-06 s") for line in lines)


def test_hover_summary_survives_a_broken_call_graph(tmp_path):
    file_path = tmp_path / "overlapping.h5"
    # "solve" starts inside "setup" and ends after it: no nesting, so no
    # exclusive time -- but the other statistics are still recorded.
    _write_sample_h5(
        file_path,
        {0: {"setup": ([0], [100]), "solve": ([20], [220])}},
    )
    region = read_h5(file_path).get_region("setup")

    text = _hover_summary(region)

    assert "calls: 1" in text
    assert "self:" not in text


def test_plotly_duration_bars_hover_with_the_region_summary(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())

    figure = _plotly_figure(plot_durations, read_h5(file_path))

    texts = _hover_texts(figure)
    assert len(texts) == 4  # one per region bar
    assert any(text.startswith("<b>step</b>") for text in texts)
    assert all("calls: " in text and "total: " in text for text in texts)


def test_plotly_combined_duration_bar_names_its_members(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())

    figure = _plotly_figure(
        plot_durations,
        read_h5(file_path),
        combine_regions={"work": ["assemble", "solve"]},
    )

    # A combined bar has no region object behind it, so it says what it
    # pools instead of borrowing one member's summary.
    combined = [text for text in _hover_texts(figure) if text.startswith("<b>work")]
    assert combined and "combines: assemble, solve" in combined[0]


def test_plotly_gantt_hover_names_the_rank_and_the_call(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data(rank_count=2))

    figure = _plotly_figure(plot_gantt, read_h5(file_path))

    texts = _hover_texts(figure)
    assert any(text.startswith("<b>step (rank 1)</b>") for text in texts)
    assert all("this call: " in text for text in texts)


def test_plotly_flame_hover_names_each_call(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())

    figure = _plotly_figure(plot_flame, read_h5(file_path))

    # maxplotlib draws flame frames as a single Bar trace, one frame per
    # point, and carries hover text on that trace directly.
    assert len(figure.data) == 1
    texts = _hover_texts(figure)
    assert len(texts) == 5  # one per call
    assert any(
        "call: step &gt; solve" in text or "call: step > solve" in text
        for text in texts
    )


def test_plotly_heatmap_hovers_each_rank_region_cell(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data(rank_count=2))

    figure = _plotly_figure(plot_rank_heatmap, read_h5(file_path))

    texts = _hover_texts(figure)
    assert len(texts) == 2 * 4  # ranks x regions
    assert any(text.startswith("<b>solve (rank 1)</b>") for text in texts)


def test_plotly_scaling_hover_describes_the_run_behind_each_point(tmp_path):
    slow, fast = tmp_path / "slow.h5", tmp_path / "fast.h5"
    _write_sample_h5(slow, _sample_file_data(1, 10, 40), {"num_ranks": 1})
    _write_sample_h5(fast, _sample_file_data(2, 10, 20), {"num_ranks": 2})

    figure = _plotly_figure(plot_speedup, [read_h5(slow), read_h5(fast)])

    texts = _hover_texts(figure)
    assert any("@ num_ranks = 2" in text for text in texts)
    assert all("speedup: " in text for text in texts)


def test_plotly_timeseries_hover_sits_on_the_line_not_the_band(tmp_path):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())

    figure = _plotly_figure(plot_duration_timeseries, read_h5(file_path))

    with_hover = [trace for trace in figure.data if trace.hovertext is not None]
    # One band plus one line per region; only the lines carry hover.
    assert len(figure.data) == 8
    assert len(with_hover) == 4


def test_plotly_hover_is_written_into_the_exported_html(tmp_path):
    pytest.importorskip("plotly")
    file_path = tmp_path / "nested.h5"
    out_file = tmp_path / "imbalance.html"
    _write_sample_h5(file_path, _nested_file_data(rank_count=2))

    # plot_imbalance renders without handing back a figure, so the exported
    # document is where its hover text can be checked.
    plot_imbalance(
        read_h5(file_path),
        filepath=str(out_file),
        backend="plotly",
        show=False,
        verbose=False,
    )

    assert "step (rank 1)" in out_file.read_text(encoding="utf-8")


def test_matplotlib_plots_skip_building_hover_text(tmp_path, monkeypatch):
    file_path = tmp_path / "nested.h5"
    _write_sample_h5(file_path, _nested_file_data())

    def fail(*args, **kwargs):
        raise AssertionError("hover text must not be built for matplotlib")

    monkeypatch.setattr("scope_profiler.plotting_scripts._hover_summary", fail)
    plot_durations(
        read_h5(file_path),
        filepath=tmp_path / "durations.png",
        show=False,
        verbose=False,
    )


def test_simple_plotext_backend_writes_terminal_plot(tmp_path):
    file_path = tmp_path / "run.h5"
    out_file = tmp_path / "durations.txt"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))

    saved_paths = plot_durations(
        read_h5(file_path),
        metric="avg",
        filepath=out_file,
        backend="plotext",
        show=False,
        verbose=False,
    )

    assert saved_paths == [out_file]
    assert "Region duration comparison" in out_file.read_text()


def test_pyvis_backend_rejects_flame_before_rendering(tmp_path, capsys):
    """PyVis only renders the explicit callgraph, not time-based charts."""
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))

    with pytest.raises(SystemExit):
        main(["flame", str(file_path), "--backend", "pyvis"])

    assert "pyvis supports the interactive callgraph only" in capsys.readouterr().err


def test_plot_helpers_can_return_rendered_figures(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))
    results = read_h5(file_path)

    fig, axes = plot_gantt(results, return_fig=True, show=False, verbose=False)
    assert fig is not None
    assert axes is not None

    fig, axes = plot_durations(
        results,
        metric="avg",
        return_fig=True,
        show=False,
        verbose=False,
    )
    assert fig is not None
    assert axes is not None


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
    results = read_h5(file_path)

    series = _duration_timeseries(results.get_region("solve"), None, 0.0)

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
    results = read_h5(file_path)

    series = _duration_timeseries(results.get_region("solve"), None, 0.0)

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
    results = read_h5(file_path)

    series = _duration_timeseries(results.get_region("solve"), [0], 0.0)

    assert list(series["num_ranks"]) == [1]
    assert series["min"] == pytest.approx(series["max"])
    assert series["max"] == pytest.approx(_seconds([10]))


def test_plot_duration_timeseries_export_data_json(tmp_path):
    file_path = tmp_path / "run.h5"
    data_file = tmp_path / "duration_timeseries_data.json"

    _write_sample_h5(file_path, _sample_file_data(2, 10, 20))
    results = read_h5(file_path)

    plot_duration_timeseries(
        results,
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

    runs = [read_h5(file_one), read_h5(file_two)]

    plot_gantt(runs, filepath=out_file, show=False, verbose=False)

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

    from scope_profiler import plotting_scripts

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
    results = read_h5(file_path)
    calls = build_call_stack(results.get_regions(), rank=0)

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
    results = read_h5(file_path)

    fig, _ = plot_flame(
        results, filepath=out_file, show=False, verbose=False, return_fig=True
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert fig.legends
    assert fig.legends[0].get_title().get_text() == "Regions"

    calls = build_call_stack(results.get_regions(), rank=0)
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

    runs = [
        read_h5(file_one),
        read_h5(file_two),
        read_h5(file_four),
    ]

    plot_speedup(runs, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_weak_scaling(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    file_four = tmp_path / "run_4.h5"
    out_file = tmp_path / "weak_scaling_plot.png"
    data_file = tmp_path / "weak_scaling_data.json"

    # Constant per-region runtime is ideal weak scaling.
    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 100, 200))
    _write_sample_h5(file_four, _sample_file_data(4, 100, 200))
    runs = [read_h5(path) for path in (file_one, file_two, file_four)]

    plot_weak_scaling(
        runs,
        filepath=out_file,
        data_filepath=data_file,
        data_format="json",
        show=False,
        verbose=False,
    )

    assert out_file.exists()
    points = json.loads(data_file.read_text())["points"]
    assert {point["normalized_runtime"] for point in points} == {1.0}


def test_plot_rank_heatmap(tmp_path):
    file_path = tmp_path / "run.h5"
    out_file = tmp_path / "rank_heatmap.png"
    _write_sample_h5(
        file_path,
        {
            0: {"setup": ([0], [10]), "solve": ([20], [120])},
            1: {"setup": ([0], [20]), "solve": ([20], [220])},
        },
    )
    results = read_h5(file_path)

    fig, axes = plot_rank_heatmap(
        results, filepath=out_file, return_fig=True, show=False, verbose=False
    )

    assert out_file.exists()
    assert fig is not None
    assert axes is not None


def test_plot_scaling_efficiency(tmp_path):
    paths = [tmp_path / f"run_{n}.h5" for n in (1, 2, 4)]
    # Runtime grows by 1.0, 1.25, 1.5 while ideal speedup grows by 1, 2, 4.
    for path, ranks, duration in zip(paths, (1, 2, 4), (100, 125, 150)):
        _write_sample_h5(path, _sample_file_data(ranks, 10, duration))
    runs = [read_h5(path) for path in paths]
    data_file = tmp_path / "efficiency.json"

    plot_scaling_efficiency(
        runs, data_filepath=data_file, data_format="json", show=False, verbose=False
    )

    efficiencies = {
        point["efficiency"]
        for point in json.loads(data_file.read_text())["points"]
        if point["region"] == "solve"
    }
    assert efficiencies == {1.0, 0.4, 1 / 6}


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
    runs = [
        read_h5(file_4),
        read_h5(file_1),
        read_h5(file_2),
    ]

    data_file = tmp_path / "speedup_data.csv"
    plot_speedup(
        runs,
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
    runs = [read_h5(file_small), read_h5(file_big)]

    data_file = tmp_path / "speedup_data.csv"
    plot_speedup(
        runs,
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
    runs = [read_h5(file_b), read_h5(file_a)]

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
        runs,
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
    runs = [read_h5(file_a), read_h5(file_b)]

    with pytest.raises(ValueError, match="not found"):
        plot_speedup(runs, x_field="nonexistent_field", show=False, verbose=False)


def test_post_processing_cli_supports_multiple_files(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    file_four = tmp_path / "run_4.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))
    _write_sample_h5(file_four, _sample_file_data(4, 25, 50))

    main(["quick", str(file_one), str(file_two), str(file_four), "-o", str(output_dir)])

    speedup_plot = output_dir / "speedup_plot.png"
    stats_json = output_dir / "region_statistics.json"

    durations_plot = output_dir / "durations_plot.png"
    assert durations_plot.exists()
    assert durations_plot.stat().st_size > 0
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


def test_post_processing_cli_defaults_when_plot_kind_is_omitted(tmp_path):
    file_one = tmp_path / "run_1.h5"
    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    output_dir = tmp_path / "figures"

    main([str(file_one), "-o", str(output_dir)])

    assert (output_dir / "gantt_plot.png").exists()
    assert (output_dir / "durations_plot.png").exists()
    assert not (output_dir / "flame_plot.png").exists()
    assert not (output_dir / "duration_timeseries_plot.png").exists()
    assert not (output_dir / "speedup_plot.png").exists()


def test_post_processing_cli_supports_wildcard_file_patterns(tmp_path):
    file_one = tmp_path / "file_1.h5"
    file_two = tmp_path / "file_2.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))

    wildcard_pattern = str(tmp_path / "file_*.h5")
    main(["quick", wildcard_pattern, "-o", str(output_dir)])

    speedup_plot = output_dir / "speedup_plot.png"
    stats_json = output_dir / "region_statistics.json"

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
    results = read_h5(file_path)

    plot_gantt(
        results,
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


def test_plot_flame_export_data_json(tmp_path, monkeypatch):
    file_path = tmp_path / "run.h5"
    data_file = tmp_path / "flame_data.json"

    rank_regions = {
        0: {
            "phase_a": ([0], [100]),
            "phase_b": ([200], [300]),
            "work": ([10, 210], [90, 290]),
        }
    }
    _write_sample_h5(file_path, rank_regions)
    with h5py.File(file_path, "a") as h5file:
        source_attrs = h5file["rank0/regions/work"].attrs
        source_attrs["source_file"] = "solver.py"
        source_attrs["source_lineno"] = 42
        source_attrs["source_text"] = "with ProfileManager.profile_region('work'):"
    results = read_h5(file_path)

    frame_labels = []

    class _RecordingCanvas:
        def __init__(self, *args, **kwargs):
            pass

        def flame_chart(self, labels, *args, **kwargs):
            frame_labels.extend(labels)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    from scope_profiler import plotting_scripts

    monkeypatch.setattr(plotting_scripts, "_get_canvas", lambda: _RecordingCanvas)
    monkeypatch.setattr(plotting_scripts, "_render", lambda *args, **kwargs: None)

    plot_flame(
        results,
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
    )

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert payload["colors"]["work"].startswith("#")
    calls = {call["call_path"]: call for call in payload["calls"]}
    assert set(calls) == {"phase_a", "phase_a > work", "phase_b", "phase_b > work"}
    assert calls["phase_a > work"]["parent_call_id"] == calls["phase_a"]["call_id"]
    assert calls["phase_b > work"]["parent_call_id"] == calls["phase_b"]["call_id"]
    assert calls["phase_a > work"]["exclusive_duration_seconds"] == pytest.approx(80e-9)
    assert calls["phase_a > work"]["source_file"] == "solver.py"
    assert calls["phase_a > work"]["source_lineno"] == 42
    assert frame_labels == ["phase_a", "phase_a > work", "phase_b", "phase_b > work"]


def test_plot_durations_export_data_json(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    data_file = tmp_path / "durations_data.json"

    _write_sample_h5(file_one, _sample_file_data(2, 10, 20))
    _write_sample_h5(file_two, _sample_file_data(2, 20, 40))
    runs = [read_h5(file_one), read_h5(file_two)]

    plot_durations(
        runs,
        filepath=tmp_path / "durations_plot.png",
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
        metric="avg",
    )

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert set(payload["metrics"]) == {"avg"}
    assert set(payload["colors"]) == {"run_one", "run_two"}
    assert all(color.startswith("#") for color in payload["colors"].values())
    assert {bar["metric"] for bar in payload["bars"]} == {"avg"}


def test_collect_region_statistics_includes_total_time(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(
        file_path,
        {0: {"solve": ([100 * 1_000_000_000], [130 * 1_000_000_000])}},
        metadata={
            "start_time_ns": 80 * 1_000_000_000,
            "finalize_time_ns": 140 * 1_000_000_000,
        },
    )

    payload = collect_region_statistics(read_h5(file_path))

    assert payload["files"][0]["total_time_seconds"] == pytest.approx(60.0)


def test_collect_region_statistics_total_time_none_without_metadata(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))

    payload = collect_region_statistics(read_h5(file_path))

    assert payload["files"][0]["total_time_seconds"] is None


def test_plot_durations_sort_by_and_top_n(tmp_path):
    file_path = tmp_path / "run.h5"
    # "solve" totals more than "setup" for every rank, so descending sort by
    # total puts it first regardless of region-declaration order.
    _write_sample_h5(
        file_path,
        {
            0: {"setup": ([0], [10]), "solve": ([20], [220])},
            1: {"setup": ([0], [10]), "solve": ([20], [220])},
        },
    )
    results = read_h5(file_path)
    data_file = tmp_path / "durations_data.json"

    plot_durations(
        results,
        filepath=tmp_path / "durations_plot.png",
        show=False,
        verbose=False,
        metric="total",
        sort_by="total",
        top_n=1,
        data_filepath=data_file,
        data_format="json",
    )

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert {bar["region"] for bar in payload["bars"]} == {"solve"}


def test_plot_durations_log_scale_renders(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(2, 10, 20))
    results = read_h5(file_path)
    out_file = tmp_path / "durations_plot.png"

    plot_durations(
        results,
        filepath=out_file,
        show=False,
        verbose=False,
        metric="total",
        log_scale=True,
    )

    assert out_file.stat().st_size > 0


def test_plot_durations_combine_regions_pools_stats(tmp_path):
    file_path = tmp_path / "run.h5"
    ns = 1_000_000_000
    _write_sample_h5(
        file_path,
        {
            0: {
                "setup: read_input": ([0], [10 * ns]),
                "setup: init_grid": ([10 * ns], [15 * ns]),
                "solve": ([15 * ns], [35 * ns]),
            }
        },
    )
    results = read_h5(file_path)
    data_file = tmp_path / "durations_data.json"

    plot_durations(
        results,
        filepath=tmp_path / "durations_plot.png",
        show=False,
        verbose=False,
        metric="total",
        combine_regions={"setup": ["^setup:.*"]},
        data_filepath=data_file,
        data_format="json",
    )

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    bars = {
        (bar["region"], bar["metric"]): bar["value_seconds"] for bar in payload["bars"]
    }
    assert {bar["region"] for bar in payload["bars"]} == {"setup", "solve"}
    assert bars[("setup", "total")] == pytest.approx(15.0)
    assert ("setup", "avg") not in bars
    assert bars[("solve", "total")] == pytest.approx(20.0)


def test_plot_durations_combine_regions_no_match_raises(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))
    results = read_h5(file_path)

    with pytest.raises(ValueError, match="matched no regions"):
        plot_durations(
            results,
            filepath=tmp_path / "durations_plot.png",
            show=False,
            verbose=False,
            combine_regions={"nope": ["^does_not_exist.*"]},
        )


def test_plot_durations_combine_regions_name_collision_raises(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))
    results = read_h5(file_path)

    with pytest.raises(ValueError, match="collide"):
        plot_durations(
            results,
            filepath=tmp_path / "durations_plot.png",
            show=False,
            verbose=False,
            combine_regions={"solve": ["^setup$"]},
        )


def test_post_processing_cli_combine_regions(tmp_path):
    file_path = tmp_path / "run.h5"
    output_dir = tmp_path / "figures"
    _write_sample_h5(
        file_path,
        {
            0: {
                "setup: read_input": ([0], [10]),
                "setup: init_grid": ([10], [15]),
                "solve": ([15], [35]),
            }
        },
    )

    main(
        [
            "durations",
            str(file_path),
            "-o",
            str(output_dir),
            "--combine-regions",
            "setup=^setup:.*",
        ]
    )

    plot_file = output_dir / "durations_plot.png"
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0


def test_post_processing_cli_combine_regions_bad_spec_errors(tmp_path, capsys):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))

    with pytest.raises(SystemExit):
        main(["durations", str(file_path), "--combine-regions", "not-a-valid-spec"])

    assert "NAME=PATTERN" in capsys.readouterr().err


def test_plot_duration_histogram_export_data_json(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(
        file_path,
        {
            0: {"solve": ([0, 100, 200], [10, 130, 260])},
            1: {"solve": ([0, 100], [15, 135])},
        },
    )
    results = read_h5(file_path)
    data_file = tmp_path / "histogram_data.json"

    plot_duration_histogram(
        results,
        filepath=tmp_path / "histogram_plot.png",
        show=False,
        verbose=False,
        bins=5,
        data_filepath=data_file,
        data_format="json",
    )

    assert (tmp_path / "histogram_plot.png").stat().st_size > 0
    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert {b["region"] for b in payload["bins"]} == {"solve"}
    # Every recorded call lands in exactly one bin.
    assert sum(b["count"] for b in payload["bins"]) == 5


def test_plot_imbalance_export_data_json(tmp_path):
    file_path = tmp_path / "run.h5"
    # Rank 1 is twice as slow as rank 0 in "solve" -- an obvious imbalance.
    _write_sample_h5(
        file_path,
        {
            0: {"solve": ([0], [10])},
            1: {"solve": ([0], [20])},
        },
    )
    results = read_h5(file_path)
    data_file = tmp_path / "imbalance_data.json"

    plot_imbalance(
        results,
        metric="total",
        filepath=tmp_path / "imbalance_plot.png",
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
    )

    assert (tmp_path / "imbalance_plot.png").stat().st_size > 0
    payload = json.loads(data_file.read_text(encoding="utf-8"))
    points = {point["rank"]: point["value_seconds"] for point in payload["points"]}
    assert points[0] == pytest.approx(_seconds([10])[0])
    assert points[1] == pytest.approx(_seconds([20])[0])
    mean_values = {point["mean_over_ranks_seconds"] for point in payload["points"]}
    assert len(mean_values) == 1
    assert mean_values.pop() == pytest.approx(_seconds([15])[0])


def _likwid_results(rank_values: dict[int, float]) -> ProfilingResults:
    """Build a ProfilingResults with one LIKWID region ("solve") per rank."""
    likwid = {
        rank: {
            "solve": LikwidRegionResult(
                tag="solve",
                group_name="FLOPS_DP",
                cpus=[rank],
                times=np.array([1.0]),
                call_counts=np.array([1]),
                metric_names=["MFlops/s"],
                metrics=np.array([[value]]),
            )
        }
        for rank, value in rank_values.items()
    }
    return ProfilingResults(
        regions={},
        num_ranks=len(rank_values),
        likwid=likwid,
        file_path="synthetic.h5",
    )


def test_available_likwid_metrics_lists_metrics_and_events():
    results = _likwid_results({0: 500.0, 1: 550.0})

    assert available_likwid_metrics(results) == ["MFlops/s"]


def test_plot_likwid_export_data_json(tmp_path):
    results = _likwid_results({0: 500.0, 1: 550.0})
    data_file = tmp_path / "likwid_data.json"

    plot_likwid(
        results,
        metric="MFlops/s",
        filepath=tmp_path / "likwid_plot.png",
        show=False,
        verbose=False,
        data_filepath=data_file,
        data_format="json",
    )

    assert (tmp_path / "likwid_plot.png").stat().st_size > 0
    payload = json.loads(data_file.read_text(encoding="utf-8"))
    assert payload["metric"] == "MFlops/s"
    values = {bar["series"]: bar["value"] for bar in payload["bars"]}
    assert values["rank 0"] == pytest.approx(500.0)
    assert values["rank 1"] == pytest.approx(550.0)


def test_plot_likwid_without_likwid_data_raises(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))
    results = read_h5(file_path)

    with pytest.raises(ValueError, match="LIKWID"):
        plot_likwid(results, metric="MFlops/s", show=False, verbose=False)


def test_post_processing_cli_plots_likwid_requires_metric(tmp_path):
    file_path = tmp_path / "run.h5"
    _write_sample_h5(file_path, _sample_file_data(1, 10, 20))

    with pytest.raises(SystemExit):
        main(["likwid", str(file_path), "-o", str(tmp_path / "figures")])


def test_post_processing_cli_new_plots_and_options(tmp_path):
    file_path = tmp_path / "run.h5"
    output_dir = tmp_path / "figures"
    _write_sample_h5(file_path, _sample_file_data(2, 10, 20))

    main(["durations", str(file_path), "-o", str(output_dir), "--sort-by", "total"])
    main(
        [
            "histogram",
            str(file_path),
            "-o",
            str(output_dir),
            "--log-scale",
            "--bins",
            "5",
        ]
    )
    main(["imbalance", str(file_path), "-o", str(output_dir), "--metric", "avg"])

    for name in ("durations_plot.png", "histogram_plot.png", "imbalance_plot.png"):
        plot_file = output_dir / name
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0


def test_plot_speedup_export_data_json(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    data_file = tmp_path / "speedup_data.json"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))
    runs = [read_h5(file_one), read_h5(file_two)]

    plot_speedup(
        runs,
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

    export_main(
        [
            "plot-data",
            str(file_one),
            str(file_two),
            "-o",
            str(output_dir),
            "--format",
            "json",
            "--plots",
            "durations",
            "speedup",
        ]
    )

    for name in (
        "durations_data.json",
        "speedup_data.json",
    ):
        data_file = output_dir / name
        assert data_file.exists()
        json.loads(data_file.read_text(encoding="utf-8"))
        assert not (output_dir / name.replace(".json", ".csv")).exists()


def test_post_processing_cli_export_plot_data_without_images(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))

    export_main(
        [
            "plot-data",
            str(file_one),
            str(file_two),
            "-o",
            str(output_dir),
            "--format",
            "json",
            "--plots",
            "durations",
            "speedup",
        ]
    )

    for name in (
        "durations_data.json",
        "speedup_data.json",
        "region_statistics.json",
    ):
        assert (output_dir / name).exists()

    assert list(output_dir.glob("*.png")) == []


def test_post_processing_cli_single_plot_can_write_file(tmp_path):
    file_one = tmp_path / "run_1.h5"
    output_file = tmp_path / "duration.png"
    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))

    main(["durations", str(file_one), "-o", str(output_file)])

    assert output_file.exists()
    assert output_file.stat().st_size > 0
