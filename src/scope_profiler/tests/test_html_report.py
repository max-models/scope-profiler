"""Tests for standalone HTML profiling reports."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from scope_profiler.__main__ import main as cli_main
from scope_profiler.h5reader import read_h5
from scope_profiler.html_report import create_html_report
from scope_profiler.likwid_data import LikwidRegionResult
from scope_profiler.perf_events import PerfEventTotals
from scope_profiler.results import ProfilingResults

from .test_post_processing import _sample_file_data, _write_sample_h5


def test_report_command_writes_escaped_region_statistics_and_metadata(tmp_path, capsys):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "nested" / "report.html"
    _write_sample_h5(
        profile,
        {0: {"<solve & verify>": ([0], [10])}},
        metadata={"host": "a < b & c"},
    )

    assert cli_main(["report", str(profile), "-o", str(report)]) == 0

    document = report.read_text(encoding="utf-8")
    assert "scope-profiler report" in document
    assert "&lt;solve &amp; verify&gt;" in document
    assert "a &lt; b &amp; c" in document
    assert "total [s]" in document
    assert str(report) in capsys.readouterr().out


def test_report_command_filters_regions(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    cli_main(["report", str(profile), "-o", str(report), "--include", "solve"])

    document = report.read_text(encoding="utf-8")
    assert ">solve<" in document
    assert ">setup<" not in document


def test_report_can_omit_embedded_charts(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    assert "<h2>Charts</h2>" not in report.read_text(encoding="utf-8")


def test_report_show_opens_generated_file(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    cli_main(["report", str(profile), "-o", str(report), "--no-charts", "--show"])

    assert opened == [report.resolve().as_uri()]


def test_report_overview_flags_hot_spot_and_imbalance(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(
        profile,
        {
            0: {"solve": ([0], [10])},
            1: {"solve": ([0], [20])},
        },
    )

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert '<ul class="findings">' in document
    assert "<code>solve</code></a> is the largest bottleneck" in document
    assert 'href="#run-0-region-0"' in document
    assert "unevenly distributed across" in document
    # The callout points at the load-balance section, which only MPI runs get.
    assert 'href="#run-0-balance">load balance</a>' in document
    assert '<div id="run-0-balance"><h3>Load balance</h3>' in document
    assert "Rank imbalance" in document
    assert "Points far from the mean identify stragglers." in document


def test_report_navigation_links_to_runs_regions_and_chart_controls(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert '<nav class="toc" aria-label="Report contents">' in document
    assert '<a href="#run-0-summary">Summary</a>' in document
    assert '<a href="#run-0-table">Region statistics</a>' in document
    assert '<h1 id="top">profile</h1>' in document
    assert '<a href="#charts">Charts</a>' in document
    assert 'id="run-0-region-0"' in document
    assert 'href="#run-0-region-' in document
    assert "Expand all charts" in document
    assert "Collapse all charts" in document
    assert 'href="#top">Back to top</a>' in document


def test_report_cross_highlights_regions_between_tables_and_charts(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert 'data-region="solve" data-run="profile"' in document
    assert "window.scopeProfilerSelectRegion = select" in document
    assert "window.scopeProfilerOnRegionSelect" in document
    assert 'target.on("plotly_click"' in document
    assert (
        "highlightFigure(chart, build(chart.payload, options), selectedRegion)"
        in document
    )
    assert "region-selected" in document
    assert 'id="region-selection"' in document
    assert 'id="clear-region-selection"' in document
    assert 'scrollIntoView({ behavior: "smooth", block: "center" })' in document


def test_report_overview_flags_frequent_short_calls(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    starts = np.arange(1001, dtype=np.int64) * 2
    _write_sample_h5(profile, {0: {"tiny": (starts, starts + 1)}})

    create_html_report(profile, report, include_charts=False)

    document = report.read_text(encoding="utf-8")
    assert "1001 times" in document
    assert "timer overhead itself measurable" in document


def test_report_explains_regions_missing_from_the_selected_ranks(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(
        profile,
        {0: {"solve": ([0], [10])}, 1: {"rank-one-only": ([0], [10])}},
    )

    create_html_report(profile, report, ranks=[0], include_charts=False)

    document = report.read_text(encoding="utf-8")
    assert "1 region(s) recorded no calls on the selected ranks" in document


def test_report_handles_an_out_of_range_rank_selection(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, {0: {"solve": ([0], [10])}})

    create_html_report(profile, report, ranks=[9], include_charts=False)

    document = report.read_text(encoding="utf-8")
    assert "No timed regions to summarize" in document
    assert '<div class="bottlenecks">' not in document


def test_report_rejects_an_empty_run_list(tmp_path):
    with pytest.raises(ValueError, match="At least one profiling result"):
        create_html_report([], tmp_path / "report.html")


def test_report_includes_recorded_hardware_counter_tables(tmp_path):
    report = tmp_path / "report.html"
    likwid = LikwidRegionResult(
        tag="solve",
        group_id=0,
        group_name="CLOCK",
        cpus=[0],
        times=np.asarray([0.5]),
        call_counts=np.asarray([3]),
        event_names=["INSTR_RETIRED_ANY"],
        counter_names=["FIXC0"],
        events=np.asarray([[1234.0]]),
        metric_names=["CPI"],
        metrics=np.asarray([[0.5]]),
        source="full_api",
    )
    results = ProfilingResults(
        {},
        num_ranks=1,
        likwid={0: {"solve": likwid}},
        perf_events={
            0: {
                "solve": PerfEventTotals(
                    calls=3,
                    values={"cycles": 100, "instructions": 250},
                ),
            },
        },
        file_path="hardware.h5",
    )

    create_html_report(results, report, include_charts=False)

    document = report.read_text(encoding="utf-8")
    assert '<section id="hardware-counters">' in document
    assert "LIKWID: hardware, rank 0, group CLOCK" in document
    assert "INSTR_RETIRED_ANY" in document
    assert ">1234<" in document
    assert "Linux perf events: hardware, rank 0" in document
    assert ">cycles<" in document
    assert ">instructions<" in document
    assert '<a href="#hardware-counters">Hardware counters</a>' in document


def test_report_plots_the_first_available_likwid_metric(tmp_path, monkeypatch):
    report = tmp_path / "report.html"
    likwid = LikwidRegionResult(
        tag="solve",
        group_id=0,
        group_name="CLOCK",
        cpus=[0],
        times=np.asarray([0.5]),
        call_counts=np.asarray([1]),
        event_names=[],
        counter_names=[],
        events=np.empty((0, 1)),
        metric_names=["CPI"],
        metrics=np.asarray([[0.5]]),
        source="full_api",
    )
    results = ProfilingResults(
        {},
        num_ranks=1,
        likwid={0: {"solve": likwid}},
        file_path="hardware.h5",
    )

    from scope_profiler import plotting_scripts

    def unavailable(*args, **kwargs):
        raise ValueError("no timing data")

    def fake_likwid(*args, data_filepath, metric, **kwargs):
        assert metric == "CPI"
        Path(data_filepath).write_text(
            json.dumps({"plot": "likwid", "metric": metric, "bars": []}),
            encoding="utf-8",
        )

    for name in (
        "plot_gantt",
        "plot_durations",
        "plot_duration_timeseries",
        "plot_rank_heatmap",
        "plot_callgraph",
        "plot_flame",
        "plot_flame_graph",
    ):
        monkeypatch.setattr(plotting_scripts, name, unavailable)
    monkeypatch.setattr(plotting_scripts, "plot_likwid", fake_likwid)

    create_html_report(results, report)

    document = report.read_text(encoding="utf-8")
    assert "LIKWID: CPI" in document
    assert "Bars compare the selected LIKWID metric" in document
    assert '"plot": "likwid"' in document


def test_report_omits_hardware_section_when_no_counters_were_recorded(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    assert "Hardware counters" not in report.read_text(encoding="utf-8")


def test_report_region_rows_are_clickable_with_call_site_detail(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert 'class="region-row"' in document
    assert 'class="region-detail" hidden' in document
    assert "class='rank-table'" in document
    assert "region-row" in document and 'addEventListener("click"' in document


def test_report_embeds_plotly_chart_fragments(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    def fake_plot(*args, data_filepath, data_format, backend, **kwargs):
        assert data_format == "json"
        assert backend == "data-only"
        is_durations = "durations" in Path(data_filepath).name
        Path(data_filepath).write_text(
            json.dumps(
                {
                    "format": "scope-profiler-plot-data",
                    "format_version": 1,
                    "plot": "durations" if is_durations else "gantt",
                    "bars" if is_durations else "intervals": [],
                    **({"options": {"stack_children": True}} if is_durations else {}),
                },
            ),
            encoding="utf-8",
        )

    from scope_profiler import plotting_scripts

    monkeypatch.setattr(plotting_scripts, "plot_gantt", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_durations", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_duration_timeseries", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_callgraph", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_rank_heatmap", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame_graph", fake_plot)

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert "Timeline: profile" in document
    # The flame views are left out of reports for now.
    assert "Flame chart" not in document
    assert "Flame graph" not in document
    # One run on one rank, each region entered once: the rank views have one
    # rank to show and there are no repeated calls to follow over time. The
    # call graph repeats the tree.
    assert "Region durations" in document
    assert "Rank heatmap" not in document
    assert "Rank imbalance" not in document
    assert "Duration over time" not in document
    assert "Call graph" not in document
    assert 'id="scope-profiler-chart-0"' in document
    assert "const scopeProfilerCharts = " in document
    # The per-chart options reach the builder, now by way of the object the
    # region filter extends with its predicate.
    assert "build(chart.payload, options)" in document
    assert "...chart.options" in document
    # The timeline labels every row with its region; its legend is redundant.
    assert '"options": {"layout": {"showlegend": false}}' in document
    assert "Each bar is one recorded region call on rank 0." in document
    # The timeline starts open; the durations bars wait, collapsed.
    panels = re.findall(
        r'<details class="chart-panel"( open)?>.*?aria-level="3">([^<]*)', document
    )
    assert [title for is_open, title in panels if is_open] == [
        "Timeline: profile",
    ]
    assert [title for is_open, title in panels if not is_open] == [
        "Region durations",
    ]
    assert 'data-chart-action="expand"' in document
    assert 'data-chart-action="collapse"' in document
    # Every chart can be opened on a page of its own.
    assert document.count('class="chart-open" data-chart="scope-profiler-chart-') == 2
    assert "const openInNewTab = (chart) =>" in document
    assert '<script id="scope-profiler-plotly-runtime">' in document
    assert "plotly.js" in document
    assert "<script src=" not in document
    assert 'import("https://' not in document


def test_report_limits_gantt_and_uses_exclusive_rank_heatmap(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(2, 10, 20))

    from scope_profiler import plotting_scripts

    captured = {}

    def write_payload(data_filepath):
        Path(data_filepath).write_text(
            json.dumps({"plot": "gantt", "intervals": []}),
            encoding="utf-8",
        )

    def fake_gantt(*args, data_filepath, **kwargs):
        captured["gantt"] = kwargs
        write_payload(data_filepath)

    def fake_heatmap(*args, data_filepath, **kwargs):
        captured["heatmap"] = kwargs
        write_payload(data_filepath)

    def fake_plot(*args, data_filepath, **kwargs):
        write_payload(data_filepath)

    monkeypatch.setattr(plotting_scripts, "plot_gantt", fake_gantt)
    monkeypatch.setattr(plotting_scripts, "plot_durations", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_duration_timeseries", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_callgraph", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_rank_heatmap", fake_heatmap)
    monkeypatch.setattr(plotting_scripts, "plot_imbalance", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame_graph", fake_plot)

    cli_main(["report", str(profile), "-o", str(report)])

    assert captured["gantt"]["ranks"] == [0]
    assert captured["heatmap"]["exclusive"] is True
    assert "exclusive timings" in report.read_text(encoding="utf-8")


def test_single_run_report_stacks_region_durations(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(2, 10, 20))

    from scope_profiler import plotting_scripts

    captured = {}

    def fake_plot(*args, data_filepath, **kwargs):
        is_durations = "durations" in Path(data_filepath).name
        if is_durations:
            captured.update(kwargs)
        Path(data_filepath).write_text(
            json.dumps(
                {
                    "plot": "durations" if is_durations else "gantt",
                    "bars" if is_durations else "intervals": [],
                    **(
                        {"options": {"stack_children": kwargs.get("stack_children")}}
                        if is_durations
                        else {}
                    ),
                },
            ),
            encoding="utf-8",
        )

    for name in (
        "plot_gantt",
        "plot_durations",
        "plot_imbalance",
        "plot_rank_heatmap",
    ):
        monkeypatch.setattr(plotting_scripts, name, fake_plot)

    create_html_report(profile, report)

    document = report.read_text(encoding="utf-8")
    # One run's bars show where each region's time went, summed over ranks.
    assert captured["stack_children"] is True
    assert captured["metric"] == "total"
    assert captured["sort_by"] == "total"
    assert "The stacked segments divide that time" in document
    panels = re.findall(
        r'<details class="chart-panel"( open)?>.*?aria-level="3">([^<]*)', document
    )
    assert "Region durations" in [title for is_open, title in panels if not is_open]
    assert [title for is_open, title in panels if is_open] == [
        "Timeline: profile (rank 0)",
    ]


def test_aggregated_report_draws_unstacked_region_durations(tmp_path):
    pytest.importorskip("plotly")
    from scope_profiler import ProfileManager

    with ProfileManager.session(
        file_path=str(tmp_path / "aggregate.h5"),
        aggregation_mode=True,
        return_results=True,
        verbose=False,
    ) as run:
        with ProfileManager.profile_region("outer"):
            with ProfileManager.profile_region("inner"):
                pass

    report = create_html_report(run.results, tmp_path / "report.html")

    document = report.read_text(encoding="utf-8")
    # No call timestamps: nothing to nest the bars by, and no timeline.
    assert '"stack_children": false' in document
    assert "summed over the selected ranks, longest first." in document
    assert "The stacked segments" not in document
    # The durations bars are the only chart, so they are not left shut.
    panels = re.findall(
        r'<details class="chart-panel"( open)?>.*?aria-level="3">([^<]*)', document
    )
    assert [title for is_open, title in panels if is_open] == ["Region durations"]


def test_reports_leave_out_duration_over_time(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    starts = np.array([30, 40, 50])
    _write_sample_h5(profile, {0: {"setup": ([0], [10]), "step": (starts, starts + 5)}})

    from scope_profiler import plotting_scripts

    def unexpected(*args, **kwargs):
        raise AssertionError("duration over time is not part of reports")

    monkeypatch.setattr(plotting_scripts, "plot_duration_timeseries", unexpected)

    single = create_html_report(profile, tmp_path / "single.html")
    both = create_html_report(
        [profile, profile], tmp_path / "both.html", individual_reports=False
    )
    for report in (single, both):
        assert "Duration over time" not in report.read_text(encoding="utf-8")


def _scaling_profiles(tmp_path, rank_counts, threads=None):
    """One profile per rank count: solve's work split over the ranks."""
    paths = []
    for index, count in enumerate(rank_counts):
        path = tmp_path / f"r{index}.h5"
        metadata = {"omp_num_threads": threads[index]} if threads else None
        _write_sample_h5(
            path,
            {
                rank: {"setup": ([0], [10]), "solve": ([20], [20 + 80 // count])}
                for rank in range(count)
            },
            metadata=metadata,
        )
        paths.append(path)
    return paths


def _chart_titles(document):
    return re.findall(
        r'<details class="chart-panel"(?: open)?>.*?aria-level="3">([^<]*)', document
    )


def test_comparison_of_run_sizes_shows_a_speedup_chart(tmp_path):
    pytest.importorskip("plotly")
    paths = _scaling_profiles(tmp_path, [1, 2, 4])

    report = create_html_report(
        paths, tmp_path / "scaling.html", individual_reports=False
    )
    document = report.read_text(encoding="utf-8")

    titles = _chart_titles(document)
    assert titles[0] == "Speedup"
    assert (
        '<details class="chart-panel" open><summary><span class="chart-heading"'
        ' role="heading" aria-level="3">Speedup' in document
    )
    assert '"plot": "speedup"' in document and '"x_field": "num_ranks"' in document
    assert "the run with the fewest MPI ranks" in document
    # Across run sizes, the durations bars show a call, not a sum over ranks.
    assert "each region&#x27;s mean call duration" in document or (
        "each region's mean call duration" in document
    )
    assert '"metrics": ["avg"]' in document


def test_comparison_of_run_sizes_compares_time_per_rank(tmp_path):
    paths = _scaling_profiles(tmp_path, [1, 4])

    report = create_html_report(
        paths, tmp_path / "scaling.html", include_charts=False, individual_reports=False
    )
    document = report.read_text(encoding="utf-8")
    table = document[document.index('<table class="compare-table metric-total"') :]
    table = table[: table.index("</table>")]
    # solve: 80 on one rank, 20 on each of four -- per rank, 4x faster; summed
    # over the ranks it would read as unchanged.
    solve = table[table.index('data-region="solve"') :]
    solve = solve[: solve.index("</tr>")]
    assert '<span class="delta-text">−75.0%</span>' in solve
    assert "total and own times are per rank" in document


def test_comparison_of_thread_counts_scales_over_threads(tmp_path):
    pytest.importorskip("plotly")
    threads_only = _scaling_profiles(tmp_path, [1, 1], threads=[1, 4])
    document = create_html_report(
        threads_only, tmp_path / "threads.html", individual_reports=False
    ).read_text(encoding="utf-8")
    assert '"x_field": "omp_num_threads"' in document
    assert "<th>threads</th>" in document

    both = _scaling_profiles(tmp_path, [1, 2], threads=[1, 4])
    document = create_html_report(
        both, tmp_path / "both.html", individual_reports=False
    ).read_text(encoding="utf-8")
    assert '"x_field": "total_cores"' in document


def test_comparison_of_equal_sizes_has_no_speedup_chart(tmp_path):
    from scope_profiler.html_report import _scaling_field, _scaling_value, _threads

    pytest.importorskip("plotly")
    paths = _scaling_profiles(tmp_path, [2, 2])
    document = create_html_report(
        paths, tmp_path / "same.html", individual_reports=False
    ).read_text(encoding="utf-8")
    assert "Speedup" not in _chart_titles(document)
    assert "<th>threads</th>" not in document
    assert '"metrics": ["total"]' in document

    class _Run:
        def __init__(self, num_ranks, metadata):
            self.num_ranks, self.metadata = num_ranks, metadata

    # Unrecorded or unreadable thread counts do not count as different.
    assert _threads(_Run(1, {"omp_num_threads": "x"})) is None
    assert _scaling_field([_Run(1, {}), _Run(1, {"omp_num_threads": 4})]) is None
    assert _scaling_value(_Run(2, {"omp_num_threads": 3}), "total_cores") == 6
    assert _scaling_value(_Run(2, {}), "omp_num_threads") == 1


def test_region_durations_compare_multiple_runs_without_stacking(tmp_path, monkeypatch):
    profiles = [tmp_path / "one.h5", tmp_path / "two.h5"]
    report = tmp_path / "report.html"
    for profile in profiles:
        _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    from scope_profiler import plotting_scripts

    captured = {}

    def fake_plot(*args, data_filepath, **kwargs):
        if kwargs.get("sort_by") == "total":
            captured.update(kwargs)
        is_durations = "durations" in Path(data_filepath).name
        Path(data_filepath).write_text(
            json.dumps(
                {
                    "plot": "durations" if is_durations else "gantt",
                    "bars" if is_durations else "intervals": [],
                    **(
                        {"options": {"stack_children": kwargs.get("stack_children")}}
                        if is_durations
                        else {}
                    ),
                },
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(plotting_scripts, "plot_gantt", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_durations", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_duration_timeseries", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_callgraph", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_rank_heatmap", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame_graph", fake_plot)

    # The individual reports each stack their own run's bars.
    create_html_report(profiles, report, individual_reports=False)

    assert captured["stack_children"] is False
    assert "Grouped bars compare each region's total recorded duration" in (
        report.read_text(encoding="utf-8")
    )


def test_report_adds_a_change_chart_for_exactly_two_runs(tmp_path):
    baseline = tmp_path / "baseline.h5"
    candidate = tmp_path / "candidate.h5"
    _write_sample_h5(baseline, _sample_file_data(1, 10, 20))
    _write_sample_h5(candidate, _sample_file_data(1, 10, 40))
    report = tmp_path / "report.html"

    create_html_report([baseline, candidate], report, include_charts=False)
    assert '"plot": "region_statistics"' not in report.read_text(encoding="utf-8")

    create_html_report([baseline, candidate], report)
    document = report.read_text(encoding="utf-8")
    assert "Change: candidate vs baseline" in document
    assert '"plot": "region_statistics"' in document
    assert '"comparison": "percent"' in document
    assert "Percent change in each region's total duration" in document


def test_report_omits_the_change_chart_for_one_or_three_runs(tmp_path):
    one = tmp_path / "one.h5"
    two = tmp_path / "two.h5"
    three = tmp_path / "three.h5"
    for path in (one, two, three):
        _write_sample_h5(path, _sample_file_data(1, 10, 20))

    single_report = tmp_path / "single.html"
    create_html_report([one], single_report)
    document = single_report.read_text(encoding="utf-8")
    assert "chart-heading" in document
    assert '"plot": "region_statistics"' not in document

    triple_report = tmp_path / "triple.html"
    create_html_report([one, two, three], triple_report)
    assert '"plot": "region_statistics"' not in triple_report.read_text(
        encoding="utf-8",
    )


def test_report_escapes_profile_text_inside_embedded_chart_json(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    run = read_h5(profile)
    run.label = "</script><script>bad()</script>"
    create_html_report(run, report)

    document = report.read_text(encoding="utf-8")
    assert "\\u003c/script>\\u003cscript>bad()\\u003c/script>" in document
    assert "</script><script>bad()" not in document


def test_report_keeps_tables_when_all_chart_payloads_fail(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    from scope_profiler import plotting_scripts

    def fail(*args, **kwargs):
        raise ValueError("no plottable calls")

    monkeypatch.setattr(plotting_scripts, "plot_gantt", fail)
    monkeypatch.setattr(plotting_scripts, "plot_durations", fail)
    monkeypatch.setattr(plotting_scripts, "plot_duration_timeseries", fail)
    monkeypatch.setattr(plotting_scripts, "plot_callgraph", fail)
    monkeypatch.setattr(plotting_scripts, "plot_rank_heatmap", fail)
    monkeypatch.setattr(plotting_scripts, "plot_flame", fail)
    monkeypatch.setattr(plotting_scripts, "plot_flame_graph", fail)

    create_html_report(profile, report)

    document = report.read_text(encoding="utf-8")
    assert "Region statistics" in document
    assert "No charts could be rendered." in document
    assert "Unavailable chart(s):" in document
    assert "no plottable calls" in document
    assert "const scopeProfilerCharts" not in document


def test_report_keeps_tables_when_plotly_is_not_installed(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    import builtins

    real_import = builtins.__import__

    def without_plotly(name, *args, **kwargs):
        if name == "plotly.offline":
            raise ImportError("plotly unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plotly)

    create_html_report(profile, report)

    document = report.read_text(encoding="utf-8")
    assert "Region statistics" in document
    assert "Charts require" in document
    assert "scope-profiler[pproc]" in document


def test_bundled_plotly_builders_match_the_npm_package_source():
    repository = Path(__file__).parents[3]
    npm_source = repository / "packages" / "plotly" / "src" / "index.js"
    bundled = (
        repository
        / "src"
        / "scope_profiler"
        / "_assets"
        / "scope-profiler-plotly-0.2.0.js"
    )

    assert bundled.read_bytes() == npm_source.read_bytes()


def test_report_region_table_headers_are_sortable_and_show_a_trend_column(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(
        profile,
        {0: {"solve": ([0, 5, 10], [1, 7, 13])}},
    )

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    document = report.read_text(encoding="utf-8")
    assert 'class="region-stats"' in document
    assert '<th data-key="total">' in document
    assert 'data-total="' in document
    assert '<svg class="spark"' in document


def test_report_summary_names_the_bottleneck_not_its_enclosing_region(tmp_path):
    """Rank bottlenecks by exclusive time.

    An enclosing region's total is mostly its children's, so ranking by the
    inclusive total just names whatever sits nearest the top of the call tree
    -- a wrapper that does no work of its own.
    """
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(
        profile,
        {
            0: {
                # wrapper spans the whole run but does 20 ms of its own work;
                # kernel, nested inside it, does 80 ms.
                "wrapper": ([0], [100_000_000]),
                "kernel": ([10_000_000], [90_000_000]),
            },
        },
    )

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    document = report.read_text(encoding="utf-8")
    assert "<code>kernel</code></a> (in wrapper) is the largest bottleneck" in document
    assert "<code>wrapper</code></a> is the largest bottleneck" not in document
    # The wrapper's own 20 ms is a leaf of the tree too, and says so.
    bottlenecks = document[document.index('<div class="bottlenecks">') :]
    assert re.findall(r'<li data-region="([^"]*)"', bottlenecks) == [
        "kernel",
        "wrapper",
    ]
    assert '<strong>wrapper</strong><span class="bn-tag"' in bottlenecks


def test_report_summary_of_a_single_region_profile(tmp_path):
    """One region is the bottleneck, with nothing to rank it against."""
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, {0: {"solve": ([0], [10_000_000])}})

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    document = report.read_text(encoding="utf-8")
    assert "<code>solve</code></a> is the largest bottleneck" in document
    assert "three largest bottlenecks" not in document
    assert '<div class="bottlenecks">' not in document


def test_report_charts_cdn_links_the_runtime_instead_of_embedding_it(tmp_path):
    """`--charts-cdn` trades ~4.7 MB of inlined Plotly for a network fetch."""
    pytest.importorskip("plotly")
    from plotly.offline._plotlyjs_version import __plotlyjs_version__

    profile = tmp_path / "profile.h5"
    embedded = tmp_path / "embedded.html"
    linked = tmp_path / "linked.html"
    _write_sample_h5(profile, {0: {"solve": ([0, 5], [1, 7])}})

    cli_main(["report", str(profile), "-o", str(embedded)])
    cli_main(["report", str(profile), "-o", str(linked), "--charts-cdn"])

    embedded_text = embedded.read_text(encoding="utf-8")
    linked_text = linked.read_text(encoding="utf-8")

    # Pinned to the version this plotly would have inlined, so both modes draw
    # with the same runtime.
    assert (
        f'<script src="https://cdn.plot.ly/plotly-{__plotlyjs_version__}.min.js"'
        in linked_text
    )
    assert '<script src="https://cdn.plot.ly' not in embedded_text
    # The builders still travel with the report; only the runtime is remote.
    assert "buildFigure" in linked_text
    assert "const scopeProfilerCharts = " in linked_text
    assert len(linked_text) < len(embedded_text) / 10

    # A blocked CDN has to say so once, not five TypeErrors later.
    assert "could not reach" in linked_text
    assert "if (!globalThis.Plotly)" in linked_text


def test_report_region_filter_targets_every_region_row(tmp_path):
    """The filter hangs off `data-region`, not the chosen sort columns.

    `--columns` decides which `data-<key>` attributes a row carries, so the
    filter would stop working for anyone who drops the name column.
    """
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(
        profile,
        {0: {"solve": ([0, 5], [1, 7]), "setup": ([10], [12])}},
    )

    cli_main(
        [
            "report",
            str(profile),
            "-o",
            str(report),
            "--no-charts",
            "--columns",
            "total",
        ],
    )

    document = report.read_text(encoding="utf-8")
    assert 'id="region-filter"' in document
    assert 'data-region="solve"' in document
    assert 'data-region="setup"' in document
    assert 'data-name="' not in document
    # A filter that matches nothing needs something to say so.
    assert 'class="region-empty"' in document
    assert "No regions match the filter." in document
    assert "scopeProfilerOnRegionFilter" in document


def test_report_region_filter_drives_the_charts_too(tmp_path):
    """The chart module redraws through the same box the tables listen to."""
    pytest.importorskip("plotly")
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, {0: {"solve": ([0, 5], [1, 7])}})

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    # Registered against the hook, redrawing with the package's own option
    # rather than a second filtering implementation.
    assert "globalThis.scopeProfilerOnRegionFilter" in document
    assert "filterRegion:" in document
    # react(), not newPlot(): typing must not tear every chart down.
    assert "Plotly.react(" in document


def test_report_escapes_a_region_name_in_the_filter_hook(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, {0: {'sol"><script>ve': ([0], [1])}})

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    document = report.read_text(encoding="utf-8")
    assert "<script>ve" not in document
    assert 'data-region="sol&quot;&gt;&lt;script&gt;ve"' in document


def test_report_region_table_is_a_collapsible_call_tree(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(
        profile,
        # "inner" (2..5) nests entirely inside "outer" (0..10).
        {0: {"outer": ([0], [10]), "inner": ([2], [5])}},
    )

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    document = report.read_text(encoding="utf-8")
    assert "Call tree" not in document
    assert 'data-path="outer" data-depth="0"' in document
    assert 'data-path="outer &gt; inner" data-depth="1"' in document
    # Only a region with nested regions gets a toggle.
    outer = document[document.index('data-path="outer"') :]
    assert outer.index('<button class="tree-toggle"') < outer.index(
        'data-path="outer &gt; inner"'
    )
    assert document.count('<button class="tree-toggle"') == 1
    assert 'data-tree-action="collapse" data-table="run-0-regions"' in document
    assert '<table class="region-stats" id="run-0-regions">' in document
    assert 'tabindex="0" aria-expanded="false"' in document
    assert "window.scopeProfilerRevealRegion" in document


def test_report_line_profile_section_only_appears_when_recorded(tmp_path):
    import numpy as np

    from scope_profiler.h5writer import ProfilingWriter
    from scope_profiler.profile_manager import RankPayload

    with_lp = tmp_path / "with_lp.h5"
    without_lp = tmp_path / "without_lp.h5"
    record = {
        "region": "solve",
        "filename": "app.py",
        "function": "solve",
        "first_lineno": 10,
        "line_numbers": np.asarray([11, 12]),
        "hits": np.asarray([1, 5]),
        "times": np.asarray([10.0, 25.0]),
        "unit": 1e-9,
    }
    with ProfilingWriter(with_lp) as writer:
        writer.write_rank(
            0,
            RankPayload(
                regions={"solve": (np.asarray([0]), np.asarray([1]))},
                likwid={},
                likwid_environment={},
                line_profile=[record],
            ),
        )
    with ProfilingWriter(without_lp) as writer:
        writer.write_rank(
            0,
            RankPayload(
                regions={"solve": (np.asarray([0]), np.asarray([1]))},
                likwid={},
                likwid_environment={},
                line_profile=[],
            ),
        )

    report_with = tmp_path / "with.html"
    report_without = tmp_path / "without.html"
    cli_main(["report", str(with_lp), "-o", str(report_with), "--no-charts"])
    cli_main(["report", str(without_lp), "-o", str(report_without), "--no-charts"])

    with_doc = report_with.read_text(encoding="utf-8")
    without_doc = report_without.read_text(encoding="utf-8")
    assert "Line profile" in with_doc
    assert '<span class="lp-func">solve</span>' in with_doc
    assert 'title="app.py">app.py:10</span>' in with_doc
    assert "28.57%" in with_doc
    # Without the source file, only the timed lines are listed; the slower
    # of the two is marked.
    assert '<tr class="lp-hot"><td class="lp-lineno">12</td>' in with_doc
    assert '<tr class="lp-idle">' not in with_doc
    assert "Line profile" not in without_doc


def _nested_results(**kwargs):
    """A session with a loop calling solve, and solve under two parents."""
    from scope_profiler import MPIRegion, Region

    def region(name, starts, ends):
        return MPIRegion(
            name,
            {0: Region(np.array(starts) * 100_000_000, np.array(ends) * 100_000_000)},
        )

    return ProfilingResults(
        {
            "scope_profiler.session": region("scope_profiler.session", [0], [100]),
            "loop": region("loop", [10, 40], [39, 90]),
            "solve": region("solve", [12, 20, 45, 92], [18, 35, 58, 98]),
            "final": region("final", [91], [99]),
        },
        **kwargs,
    )


def _region_table_html(document):
    start = document.index('<table class="region-stats"')
    # Detail rows nest per-rank tables, so end at the note after the table.
    return document[
        start : document.index('</table><p class="muted table-note">', start)
    ]


def test_report_region_table_matches_the_terminal_summary_layout(tmp_path):
    report = create_html_report(
        _nested_results(), tmp_path / "report.html", include_charts=False
    )
    document = report.read_text(encoding="utf-8")
    table = _region_table_html(document)

    thead = table[: table.index("</thead>")]
    assert [th.rsplit(">", 1)[-1] for th in thead.split("</th>")[:-1]] == [
        "region",
        "% session",
        "total [s]",
        "trend",
    ]
    # Rows follow the call tree (sort="start"), with the same tree glyphs,
    # inline call counts and (own) rows as the terminal table.
    names = re.findall(
        r'<tr class="(?:region-row|own-row)"[^>]*><td>'
        r'(?:<button class="tree-toggle"[^>]*>[^<]*</button>|<span class="tree-toggle">'
        r"</span>)<span>(?:<span class=\"indent\">[^<]*</span>)?([^<]*)</span>",
        table,
    )
    assert names == [
        "scope_profiler.session",
        "(own)",
        "loop (2x)",
        "(own)",
        "solve (3x)",
        "final",
        "(own)",
        "solve",
    ]
    assert '<span class="indent">│ └─ </span>solve (3x)' in table
    assert "10.000000" in table and "100.00%" in table
    assert "(1x)" not in table and "TOTAL" not in table
    # loop runs 7.9 s, 3.4 s of it in solve; its (own) row sits in its tbody.
    loop_body = table[table.index('data-region="loop"') :]
    loop_body = loop_body[: loop_body.index('class="region-detail"')]
    assert '<tr class="own-row">' in loop_body
    assert "4.500000" in loop_body and "45.00%" in loop_body
    assert "(own) rows show a region&#x27;s time excluding its children." in document


def test_report_links_only_the_first_row_of_a_region_on_several_call_paths(tmp_path):
    report = create_html_report(
        _nested_results(), tmp_path / "report.html", include_charts=False
    )
    document = report.read_text(encoding="utf-8")
    solve_rows = document.count('<tbody data-region="solve"')
    assert solve_rows == 2
    ids = [part.split('"', 1)[0] for part in document.split(' id="run-0-region-')[1:]]
    assert len(ids) == len(set(ids))


def test_report_region_table_scripts_handle_own_rows_and_flat_sorting(tmp_path):
    report = create_html_report(
        _nested_results(), tmp_path / "report.html", include_charts=False
    )
    document = report.read_text(encoding="utf-8")
    # The detail row follows any (own) row, so it is found within the tbody.
    assert 'row.parentNode.querySelector(".region-detail")' in document
    assert "nextElementSibling" not in document
    # Sorting by a column breaks the hierarchy; the indentation is hidden.
    assert 'table.classList.add("flat")' in document
    assert ".region-stats.flat .indent { display: none; }" in document


def test_report_region_table_omits_percent_without_the_session_root(tmp_path):
    report = create_html_report(
        _nested_results(),
        tmp_path / "report.html",
        include=["loop", "solve"],
        include_charts=False,
    )
    table = _region_table_html(report.read_text(encoding="utf-8"))
    assert "% session" not in table
    assert '<th data-key="total">total [s]</th>' in table


def test_report_region_table_explicit_columns_match_the_terminal(tmp_path):
    report = create_html_report(
        _nested_results(),
        tmp_path / "report.html",
        columns=["region", "ranks", "calls", "parent_percent"],
        include_charts=False,
    )
    table = _region_table_html(report.read_text(encoding="utf-8"))
    # A calls column replaces the inline "(2x)" counts, as in the terminal.
    assert "(2x)" not in table and '<th data-key="calls">n</th>' in table
    loop_body = table[table.index('<tbody data-region="loop"') :]
    loop_body = loop_body[: loop_body.index(">")]
    assert 'data-ranks="1"' in loop_body
    assert 'data-parent_percent="79.0"' in loop_body


def test_report_bottlenecks_are_the_leaves_of_the_call_tree(tmp_path):
    report = create_html_report(
        _nested_results(), tmp_path / "report.html", include_charts=False
    )
    document = report.read_text(encoding="utf-8")
    bottlenecks = document[document.index('<div class="bottlenecks">') :]
    bottlenecks = bottlenecks[: bottlenecks.index("</ol>")]
    names = re.findall(r'<li data-region="([^"]*)"', bottlenecks)
    # One entry per call path: solve is 3.4 s under loop and 0.6 s under
    # final. loop's 4.5 s and final's 0.2 s outside solve are their own time.
    assert names == ["loop", "solve", "scope_profiler.session", "solve", "final"]
    assert '<span class="bn-share">45.0%</span>' in bottlenecks
    assert '<span class="bn-share">34.0%</span>' in bottlenecks
    assert '<span class="bn-path">in loop</span>' in bottlenecks
    assert '<span class="bn-path">in final</span>' in bottlenecks
    assert "3.4 s · 3 calls · 1.133 s/call" in bottlenecks
    # The session root's own time is the time no other region covers.
    assert "<em>outside any region</em>" in bottlenecks
    assert 'class="bottleneck" type="button"' in bottlenecks
    # The filter hides bottlenecks too, and the whole list once none match.
    assert 'document.querySelectorAll(".bottlenecks")' in document
    assert 'block.querySelector("li[data-filter-region]:not([hidden])")' in document


def test_report_bottlenecks_need_two_entries(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, {0: {"solve": ([0], [10])}})

    create_html_report(profile, report, include_charts=False)

    assert '<div class="bottlenecks">' not in report.read_text(encoding="utf-8")


def test_report_run_header_names_file_time_host_and_scale(tmp_path):
    results = _nested_results(
        metadata={
            "timestamp": "2026-09-14T13:48:01+02:00",
            "hostname": "node042",
            "chip_information": "Apple M1",
        },
        file_path=str(tmp_path / "run.h5"),
    )
    report = create_html_report(results, tmp_path / "report.html", include_charts=False)
    document = report.read_text(encoding="utf-8")

    meta = document[document.index('<p class="run-meta">') :]
    meta = meta[: meta.index("</p>")]
    assert "<code>run.h5</code>" in meta
    assert f'title="{tmp_path / "run.h5"}"' in meta
    assert "2026-09-14 11:48 UTC" in meta
    assert "node042" in meta and "Apple M1" in meta
    assert "1 rank<" in meta and "4 regions<" in meta
    assert "10 s profiled" in meta
    # The overview no longer repeats these facts.
    assert "Profiled <strong>" not in document


def test_format_timestamp_handles_naive_and_unparseable_values():
    from scope_profiler.html_report import _format_timestamp

    assert _format_timestamp("2026-09-14T11:48:59") == "2026-09-14 11:48"
    assert _format_timestamp("yesterday") == "yesterday"


def test_report_flags_time_outside_every_region(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(
        profile,
        {0: {"scope_profiler.session": ([0], [100]), "solve": ([10], [40])}},
    )

    create_html_report(profile, report, include_charts=False)

    document = report.read_text(encoding="utf-8")
    assert "70% of the session" in document
    assert "is outside every region" in document
    assert '<span class="kpi-value">30%</span>' in document
    # The session root is not a bottleneck to name.
    assert "<code>solve</code></a> is the largest bottleneck" in document


def test_report_does_not_guess_outside_time_without_a_call_tree(tmp_path):
    from scope_profiler import MPIRegion, Region

    def region(name, starts, ends):
        return MPIRegion(name, {0: Region(np.array(starts), np.array(ends))})

    # Partially overlapping calls cannot form a call tree, and in-memory
    # results carry no stored exclusive totals to fall back on.
    results = ProfilingResults(
        {
            "scope_profiler.session": region("scope_profiler.session", [0], [100]),
            "a": region("a", [10], [50]),
            "b": region("b", [40], [60]),
        }
    )

    report = create_html_report(results, tmp_path / "report.html", include_charts=False)

    document = report.read_text(encoding="utf-8")
    assert "outside every region" not in document
    assert "outside any region" not in document


def test_report_change_chart_uses_the_comparison_builder(tmp_path):
    baseline = tmp_path / "baseline.h5"
    candidate = tmp_path / "candidate.h5"
    _write_sample_h5(baseline, _sample_file_data(1, 10, 20))
    _write_sample_h5(candidate, _sample_file_data(1, 10, 40))
    report = tmp_path / "report.html"

    create_html_report([baseline, candidate], report)

    document = report.read_text(encoding="utf-8")
    # buildFigure would draw the payload as a ranked summary, not a change.
    assert (
        "const build = options.comparison ? buildComparisonFigure : buildFigure;"
        in document
    )
    assert "export function buildComparisonFigure" in document
    panels = re.findall(
        r'<details class="chart-panel"( open)?>.*?aria-level="3">([^<]*)', document
    )
    # A comparison keeps only the charts that compare runs.
    assert [title for is_open, title in panels if is_open] == [
        "Change: candidate vs baseline",
        "Region durations",
    ]
    assert "Timeline:" not in document and "Flame graph:" not in document


def test_report_total_bars_are_drawn(tmp_path):
    report = create_html_report(
        _nested_results(), tmp_path / "report.html", include_charts=False
    )
    document = report.read_text(encoding="utf-8")
    # `.bar-cell span` once matched the bar itself and hid every bar.
    assert ".bar-cell > span:not(.bar)" in document
    assert ".bar-cell span {" not in document
    assert '<span class="bar" style="width:100%"></span>' in document


def test_report_omits_empty_findings(tmp_path):
    """A lone session root has no bottleneck or flag to report."""
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, {0: {"scope_profiler.session": ([0], [10])}})

    create_html_report(profile, report, include_charts=False)

    document = report.read_text(encoding="utf-8")
    assert '<ul class="findings">' not in document
    assert '<div class="bottlenecks">' not in document
    assert "In regions" not in document
    assert "scope_profiler.session" in document


def _ranked_results(rank_work, file_path="run.h5", extra_regions=0, num_ranks=None):
    """A session per rank: ``work`` for a rank-dependent time, then ``wait``.

    Every rank finishes at 100, so whatever ``work`` does not take, ``wait``
    does -- the shape of ranks meeting at a barrier.
    """
    from scope_profiler import MPIRegion, Region

    ranks = range(len(rank_work))
    scale = 10_000_000

    def region(name, spans):
        return MPIRegion(
            name,
            {
                rank: Region(
                    np.array([start for start, _ in spans[rank]]) * scale,
                    np.array([end for _, end in spans[rank]]) * scale,
                )
                for rank in ranks
            },
        )

    regions = {
        "scope_profiler.session": region(
            "scope_profiler.session", {rank: [(0, 100)] for rank in ranks}
        ),
        "work": region(
            "work", {rank: [(0, 10 + w)] for rank, w in enumerate(rank_work)}
        ),
        "wait": region(
            "wait", {rank: [(10 + w, 100)] for rank, w in enumerate(rank_work)}
        ),
    }
    for index in range(extra_regions):
        # Nested in work's first 10 units, one unit each, the same on every rank.
        regions[f"extra{index}"] = region(
            f"extra{index}",
            {rank: [(index * 0.5, index * 0.5 + 0.4)] for rank in ranks},
        )
    return ProfilingResults(regions, file_path=file_path, num_ranks=num_ranks)


def test_duration_text_picks_a_short_unit():
    from scope_profiler.html_report import _duration_text, _signed_pct

    assert _duration_text(None) == "-"
    assert _duration_text(0) == "0 s"
    assert _duration_text(5e-10) == "0.5 ns"
    assert _duration_text(2.5e-5) == "25 µs"
    assert _duration_text(0.0512) == "51.2 ms"
    assert _duration_text(12.3456) == "12.35 s"
    assert _duration_text(12345.6) == "12,346 s"
    assert _signed_pct(-12.34) == "−12.3%"
    assert _signed_pct(5) == "+5.0%"


def test_line_profile_shows_the_function_source_dedented(tmp_path):
    from scope_profiler import MPIRegion, Region

    source = tmp_path / "app.py"
    source.write_text(
        "class Solver:\n"
        "    def solve(self, n):\n"
        "        # sum the squares\n"
        "        total = 0\n"
        "        for i in range(n):\n"
        "            total += i * i\n"
        "        return total\n",
        encoding="utf-8",
    )

    def record(times, rank_scale=1):
        return {
            "region": "solve",
            "filename": str(source),
            "function": "solve",
            "first_lineno": 2,
            "line_numbers": np.asarray([4, 5, 6, 7]),
            "hits": np.asarray([1, 11, 10, 1]) * rank_scale,
            "times": np.asarray(times, dtype=float),
            "unit": 1e-6,
        }

    helper = {
        "region": "setup",
        "filename": str(source),
        "function": "helper",
        "first_lineno": 1,
        "line_numbers": np.asarray([1]),
        "hits": np.asarray([1]),
        "times": np.asarray([1.0]),
        "unit": 1e-6,
    }
    # The session's own frame is scope-profiler code, not the user's.
    own_frame = {
        **helper,
        "filename": str(Path(__file__).parents[1] / "profile_manager.py"),
    }
    results = ProfilingResults(
        {
            "solve": MPIRegion(
                "solve",
                {
                    0: Region(np.array([0]), np.array([1])),
                    1: Region(np.array([0]), np.array([1])),
                },
            )
        },
        line_profile={
            0: [record([1, 10, 80, 9]), own_frame],
            1: [record([1, 10, 80, 9]), helper],
        },
    )

    report = create_html_report(results, tmp_path / "report.html", include_charts=False)
    document = report.read_text(encoding="utf-8")

    assert "profile_manager.py" not in document
    # solve's line profile is in its table row's detail.
    body = document[document.index('<tbody data-region="solve"') :]
    # The row's own tbody ends where the next region's begins.
    body = body[: body.index('<tbody class="region-empty"')]
    assert (
        '<div class="lp-block"><p class="lp-head"><span class="lp-func">solve' in body
    )
    assert 'id="run-0-regions-0-lp-0"' in body
    # Ranks are summed.
    assert "2 ranks" in body and "200 µs" in body
    # The function reads as written: unrecorded lines included, the common
    # indentation stripped, nested indentation kept.
    assert (
        '<td class="lp-src"><span class="tk-kw">def</span> '
        '<span class="tk-fn">solve</span>(<span class="tk-self">self</span>, n):</td>'
        in body
    )
    assert '<tr class="lp-idle"><td class="lp-lineno">3</td>' in body
    assert (
        '<td class="lp-src">    <span class="tk-com"># sum the squares</span></td>'
        in body
    )
    assert '<td class="lp-src">        total += i * i</td>' in body
    assert '<tr class="lp-hot"><td class="lp-lineno">6</td>' in body
    assert "<td>22</td>" in body and 'style="--pct:80%">80.00%</td>' in body
    assert "Click a row for its call site, line profile and per-rank" in document

    # helper's region is not in the table, so it is listed on its own,
    # collapsed to one row with its share of the time.
    section = document[document.index('id="run-0-lines"') :]
    assert '<a href="#run-0-lines">Line profile</a>' in document
    assert 'lp-func">solve' not in section
    assert (
        '<details class="lp-function"><summary><span class="lp-func">helper' in section
    )
    assert " open>" not in section
    assert '<span class="lp-share">100%</span>' in section


def test_report_balance_sections_compare_ranks(tmp_path):
    results = _ranked_results([10, 10, 10, 30])
    report = create_html_report(results, tmp_path / "report.html", include_charts=False)
    document = report.read_text(encoding="utf-8")

    balance = document[document.index('<div id="run-0-balance">') :]
    table = balance[: balance.index("</table>")]
    # work: 20, 20, 20, 40 -> mean 25, excess 15; wait mirrors it, excess 5.
    assert re.findall(r'<tr class="select-row" data-region="([^"]*)"', table) == [
        "work",
        "wait",
    ]
    assert "<td>250 ms</td><td>200 ms</td><td>400 ms</td><td>3</td>" in table
    assert '<td class="flag">+60%</td><td>150 ms</td>' in table
    assert "excluding nested regions" in balance
    # Each rank against the region mean; rank 3 does more work but the ranks
    # wait for each other, so none is slowest overall.
    matrix = balance[balance.index('<table class="rank-matrix">') :]
    assert '<td class="heat" style="background:rgba(220,38,38,0.55)"' in matrix
    assert "rank 3: 400 ms (+60.0% vs mean)" in matrix
    assert '<th class="slowest"' not in matrix
    assert "the differences above are ranks waiting for each other" in balance
    assert "Rank 3 is the slowest rank" not in document
    assert 'data-show-all="' not in document
    # The summary reports the worst region's imbalance.
    assert '<span class="kpi-label">Load imbalance</span>' in document
    assert "work, slowest on rank 3" in document
    assert "rank 3 spends 60% more time in it" in document


def test_report_balance_names_a_slowest_rank_and_trims_large_runs(tmp_path):
    from scope_profiler.html_report import (
        _MATRIX_RANKS,
        _rank_balance,
        _rank_matrix_html,
    )

    # 30 regions and 70 ranks: more than the tables show.
    results = _ranked_results([0] * 69 + [20], extra_regions=28)
    report = create_html_report(results, tmp_path / "report.html", include_charts=False)
    document = report.read_text(encoding="utf-8")
    assert 'data-show-all="run-0-balance-table"' in document
    assert '<tr class="select-row extra"' in document
    assert "The 20 regions with the most time per rank are shown." in document
    assert f"The first {_MATRIX_RANKS} of 70 ranks are shown" in document
    assert '<table class="rank-matrix compact">' in document

    # When ranks really differ in their time inside regions, say which.
    balance = _rank_balance(
        results, [{"name": "work", "total": 1.0, "call_path": "work"}], None
    )
    balance["totals"] = np.arange(70, dtype=float)
    matrix = _rank_matrix_html(results, balance, "run-0")
    assert '<th class="slowest" title="slowest rank overall">63</th>' in matrix
    assert "Rank 63, in red in the header" in matrix


def test_report_summary_names_a_rank_slowest_in_most_regions(tmp_path):
    from scope_profiler import MPIRegion, Region

    def region(name, start, totals):
        return MPIRegion(
            name,
            {
                rank: Region(
                    np.array([start * 10**10]),
                    np.array([int((start * 10 + total) * 1e9)]),
                )
                for rank, total in enumerate(totals)
            },
        )

    # One after the other on every rank, so none nests in another.
    results = ProfilingResults(
        {
            "a": region("a", 0, [1, 2]),
            "b": region("b", 1, [1, 3]),
            "c": region("c", 2, [2, 1]),
        }
    )
    report = create_html_report(results, tmp_path / "report.html", include_charts=False)
    document = report.read_text(encoding="utf-8")
    assert "Rank 1 is the slowest rank in 2 of 3 regions." in document


def test_rank_balance_needs_two_ranks_and_time(tmp_path):
    from scope_profiler.html_report import _deviation_style, _rank_balance

    single = _ranked_results([10])
    assert _rank_balance(single, [], None) is None
    results = _ranked_results([10, 10])
    assert _rank_balance(results, [], None) is None
    assert _rank_balance(results, [], [0]) is None
    assert _deviation_style(1.0, 0.0) == ""


def test_report_balance_uses_totals_without_a_call_tree(tmp_path):
    from scope_profiler import MPIRegion, Region

    def region(name, spans):
        return MPIRegion(
            name,
            {
                rank: Region(np.array([start]), np.array([end]))
                for rank, (start, end) in enumerate(spans)
            },
        )

    # Partially overlapping calls cannot form a call tree.
    results = ProfilingResults(
        {"a": region("a", [(10, 50), (10, 90)]), "b": region("b", [(40, 60), (40, 60)])}
    )
    report = create_html_report(results, tmp_path / "report.html", include_charts=False)
    document = report.read_text(encoding="utf-8")
    assert "Each region&#x27;s total time per rank." in document or (
        "Each region's total time per rank." in document
    )


def _comparison_runs(tmp_path):
    from scope_profiler import MPIRegion, Region

    def region(name, starts, ends):
        return MPIRegion(
            name,
            {0: Region(np.array(starts) * 10_000_000, np.array(ends) * 10_000_000)},
        )

    baseline = ProfilingResults(
        {
            "scope_profiler.session": region("scope_profiler.session", [0], [100]),
            "solve": region("solve", [0], [60]),
            "kernel": region("kernel", [10], [50]),
            "io": region("io", [60], [70]),
            "old": region("old", [70], [99]),
        },
        file_path=tmp_path / "base.h5",
    )
    candidate = ProfilingResults(
        {
            "scope_profiler.session": region("scope_profiler.session", [0], [80]),
            "solve": region("solve", [0], [40]),
            "kernel": region("kernel", [10], [30]),
            "fresh": region("fresh", [32], [38]),
            "io": region("io", [40], [70]),
        },
        file_path=tmp_path / "cand.h5",
    )
    return baseline, candidate


def test_comparison_report_links_individual_reports(tmp_path):
    baseline, candidate = _comparison_runs(tmp_path)
    output = tmp_path / "out" / "compare.html"

    create_html_report([baseline, candidate], output, include_charts=False)

    document = output.read_text(encoding="utf-8")
    assert (tmp_path / "out" / "compare-0-base.html").exists()
    assert (tmp_path / "out" / "compare-1-cand.html").exists()
    assert '<a href="compare-0-base.html">full report</a>' in document
    assert '<h1 id="top">base vs cand</h1>' in document
    # No per-run detail: that is what the individual reports are for.
    assert 'class="region-stats"' not in document
    assert '<div class="bottlenecks">' not in document
    individual = (tmp_path / "out" / "compare-1-cand.html").read_text(encoding="utf-8")
    assert 'class="region-stats"' in individual
    assert "scope-profiler comparison" not in individual

    runs = document[document.index('<table class="runs-table">') :]
    assert '<span class="badge base">baseline</span>' in runs
    assert "<td>1 s</td>" in runs and "<td>800 ms</td>" in runs
    assert '<span class="delta-text">−20.0%</span>' in runs


def test_comparison_report_without_individual_reports(tmp_path):
    baseline, candidate = _comparison_runs(tmp_path)
    output = tmp_path / "compare.html"

    create_html_report(
        [baseline, candidate], output, include_charts=False, individual_reports=False
    )

    assert sorted(path.name for path in tmp_path.glob("*.html")) == ["compare.html"]
    document = output.read_text(encoding="utf-8")
    assert "full report</a>" not in document
    assert "scope-profiler report RUN.h5 -o RUN.html" in document


def test_comparison_table_aligns_call_paths(tmp_path):
    baseline, candidate = _comparison_runs(tmp_path)
    output = tmp_path / "compare.html"
    create_html_report(
        [baseline, candidate], output, include_charts=False, individual_reports=False
    )
    document = output.read_text(encoding="utf-8")

    table = document[document.index('<table class="compare-table metric-total"') :]
    table = table[: table.index("</table>")]
    # The candidate's new region slots in under its parent, after the
    # baseline's children of that parent.
    assert re.findall(r'data-region="([^"]*)" data-filter-region', table) == [
        "scope_profiler.session",
        "solve",
        "kernel",
        "fresh",
        "io",
        "old",
    ]
    assert 'style="padding-left:2.2em">kernel</span>' in table
    assert '<span class="badge new">new</span>' in table
    assert '<span class="badge gone">gone</span>' in table
    # kernel: 400 ms -> 200 ms in total, faster; io: 100 ms -> 300 ms, slower.
    assert '<span class="delta faster" title="−200 ms · 2.00× faster">' in table
    assert '<span class="delta slower" title="+200 ms · 3.00× slower">' in table
    # Calls are unchanged, which is neither good nor bad.
    assert '<span class="delta same" title="+0 calls">' in table
    # Sorting by the change of each metric.
    assert 'data-change-total="0.2"' in table and 'data-change-calls="0"' in table
    for metric in ("total", "own", "avg", "calls"):
        assert f'data-compare-metric="{metric}"' in document
    assert 'data-compare-sort="change"' in document


def test_comparison_summary_lists_what_changed(tmp_path):
    baseline, candidate = _comparison_runs(tmp_path)
    output = tmp_path / "compare.html"
    create_html_report(
        [baseline, candidate], output, include_charts=False, individual_reports=False
    )
    document = output.read_text(encoding="utf-8")

    changed = document[document.index('<section id="compare-changes">') :]
    changed = changed[: changed.index("</section>")]
    assert "<h3>cand vs base</h3>" in changed
    assert "1 s → 800 ms" in changed and "1.25× faster" in changed
    faster = changed[
        changed.index("<h4>Faster</h4>") : changed.index("<h4>Slower</h4>")
    ]
    slower = changed[changed.index("<h4>Slower</h4>") :]
    # Own time: old went away (290 ms), kernel halved (-200 ms) and solve's own
    # time shrank from 200 to 140 ms.
    assert re.findall(
        r'<li class="change-item" data-filter-region="([^"]*)"', faster
    ) == [
        "old",
        "kernel",
        "solve",
    ]
    assert re.findall(
        r'<li class="change-item" data-filter-region="([^"]*)"', slower
    ) == [
        "io",
        "scope_profiler.session",
        "fresh",
    ]
    assert "(new)" in slower
    assert '<span class="kpi-label">Largest regression</span>' in changed


def test_comparison_of_identical_runs_reports_no_changes(tmp_path):
    from scope_profiler.html_report import _change_html, _wall_kpi

    baseline, _ = _comparison_runs(tmp_path)
    output = tmp_path / "compare.html"
    create_html_report(
        [baseline, baseline, baseline],
        output,
        include_charts=False,
        individual_reports=False,
    )
    document = output.read_text(encoding="utf-8")
    assert '<h1 id="top">base vs base vs base</h1>' in document
    assert document.count('<div class="compare-candidate">') == 2
    assert '<span class="kpi-value">none</span>' in document
    assert '<p class="muted">None.</p>' in document

    empty = ProfilingResults({})
    assert _wall_kpi(empty, baseline) == (
        '<div class="kpi"><span class="kpi-label">Wall time</span>'
        '<span class="kpi-value">1 s</span></div>'
    )
    assert _change_html(None, None, "total") == ("-", None)
    assert _change_html(0.0, 1.0, "total")[0] == '<span class="badge new">new</span>'
    assert "delta slower" in _change_html(1.0, 0.0, "total")[0] or "faster" in (
        _change_html(1.0, 0.0, "total")[0]
    )
    assert (
        'class="delta neutral down" title="-2 calls"' in _change_html(4, 2, "calls")[0]
    )


def test_comparison_entries_place_a_new_root_last():
    from scope_profiler.html_report import _comparison_entries

    entries = _comparison_entries(
        [
            [{"name": "a", "call_path": "a", "depth": 0, "total": 1.0}],
            [
                {"name": "b", "call_path": "b", "depth": 0, "total": 1.0},
                {"name": "c", "call_path": "x > c", "depth": 1, "total": 1.0},
                {"name": "flat", "total": 1.0},
            ],
        ]
    )
    assert [entry["key"] for entry in entries] == [
        "path:a",
        "path:b",
        "path:x > c",
        "name:flat",
    ]
    assert entries[3]["context"] == []


def test_report_cli_can_skip_individual_reports(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_sample_h5(first, _sample_file_data(1, 10, 20))
    _write_sample_h5(second, _sample_file_data(1, 10, 40))
    output = tmp_path / "compare.html"

    cli_main(
        [
            "report",
            str(first),
            str(second),
            "-o",
            str(output),
            "--no-charts",
            "--no-individual-reports",
        ]
    )
    assert sorted(path.name for path in tmp_path.glob("*.html")) == ["compare.html"]

    cli_main(["report", str(first), str(second), "-o", str(output), "--no-charts"])
    assert sorted(path.name for path in tmp_path.glob("*.html")) == [
        "compare-0-first.html",
        "compare-1-second.html",
        "compare.html",
    ]


def test_report_file_names_are_safe_slugs(tmp_path):
    from scope_profiler.html_report import _report_file_name

    assert _report_file_name(tmp_path / "r.html", 2, "a b/c").name == "r-2-a-b-c.html"
    assert _report_file_name(tmp_path / "r.html", 0, "///").name == "r-0-run.html"


def test_report_chart_clicks_highlight_without_scrolling(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert (
        '<div class="region-toast" id="region-toast" role="status" hidden>' in document
    )
    assert 'region, runFromPoint(chart, point, region), "toast");' in document
    assert 'else if (mode === "toast") showToast(selectedRegion, target);' in document
    # Bottlenecks still jump to the table row: finding it is their purpose.
    assert 'select(button.dataset.region, button.dataset.run, "scroll");' in document


def test_rank_balance_skips_regions_without_time_on_the_selected_ranks():
    from scope_profiler import MPIRegion, Region
    from scope_profiler.html_report import _is_own_frame, _rank_balance

    results = ProfilingResults(
        {
            "busy": MPIRegion(
                "busy",
                {rank: Region(np.array([0]), np.array([10])) for rank in (0, 1)},
            ),
            # Entered, but for no measurable time.
            "instant": MPIRegion(
                "instant",
                {rank: Region(np.array([20]), np.array([20])) for rank in (0, 1)},
            ),
        }
    )
    rows = [
        {"name": "busy", "total": 1e-8, "call_path": "busy"},
        {"name": "instant", "total": 0.0, "call_path": "instant"},
    ]
    balance = _rank_balance(results, rows, None)
    assert [entry["name"] for entry in balance["regions"]] == ["busy"]
    # A file name no path can be made of is not scope-profiler's own frame.
    assert _is_own_frame("bad\0name") is False


def _line_profile_results(tmp_path, functions):
    """In-memory results with a line profile for each (name, source, times)."""
    from scope_profiler import MPIRegion, Region

    records = []
    for name, source, times in functions:
        path = tmp_path / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        records.append(
            {
                "region": name,
                "filename": str(path),
                "function": name,
                "first_lineno": 1,
                "line_numbers": np.asarray([number for number, _ in times]),
                "hits": np.ones(len(times), dtype=int),
                "times": np.asarray([time for _, time in times], dtype=float),
                "unit": 1e-6,
            }
        )
    return ProfilingResults(
        {"work": MPIRegion("work", {0: Region(np.array([0]), np.array([1]))})},
        line_profile={0: records},
    )


def test_line_profile_shows_every_line_of_a_long_function(tmp_path):
    # Line 20 of 40 costs everything; the box scrolls rather than folds.
    source = "def long():\n" + "".join(f"    x{n} = {n}\n" for n in range(2, 41))
    times = [(n, 1000.0 if n == 20 else 0.1) for n in range(2, 41)]
    results = _line_profile_results(tmp_path, [("long", source, times)])

    report = create_html_report(results, tmp_path / "report.html", include_charts=False)
    document = report.read_text(encoding="utf-8")
    table = document[document.index('<table class="lp-table" id="run-0-lp-0">') :]
    table = table[: table.index("</table>")]

    numbers = re.findall(r'<td class="lp-lineno">(\d+)', table)
    assert numbers == [str(n) for n in range(1, 41)]
    assert '<tr class="lp-hot"><td class="lp-lineno">20</td>' in table
    assert "lines hidden" not in document
    assert 'data-show-all="run-0-lp-0"' not in document


def test_region_detail_puts_ranks_above_the_line_profile(tmp_path):
    source = "def work():\n    a = 1\n"
    results = _line_profile_results(tmp_path, [("work", source, [(2, 5.0)])])

    document = create_html_report(
        results, tmp_path / "report.html", include_charts=False
    ).read_text(encoding="utf-8")
    detail = document[document.index('<tr class="region-detail"') :]
    assert detail.index("Per rank") < detail.index("Line profile")


def test_line_profile_lists_the_largest_functions_first(tmp_path):
    from scope_profiler.html_report import _LP_FUNCTIONS

    functions = [
        (f"f{index:02d}", f"def f{index:02d}():\n    pass\n", [(2, float(index + 1))])
        for index in range(_LP_FUNCTIONS + 2)
    ]
    results = _line_profile_results(tmp_path, functions)

    document = create_html_report(
        results, tmp_path / "report.html", include_charts=False
    ).read_text(encoding="utf-8")
    names = re.findall(r'<span class="lp-func">([^<]*)</span>', document)
    assert names[:2] == ["f11", "f10"]
    assert document.count('<details class="lp-function lp-extra">') == 2
    assert 'data-show-all="run-0-lp-functions"' in document
    assert "Show all 12 functions" in document


def test_report_output_defaults_to_report_html(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))
    monkeypatch.chdir(tmp_path)

    cli_main(["report", str(profile), "--no-charts"])

    assert (tmp_path / "report.html").exists()


def test_line_profile_of_a_region_on_two_call_paths_fills_both_rows(tmp_path):
    source = tmp_path / "solve.py"
    source.write_text("def solve():\n    return 1\n", encoding="utf-8")
    record = {
        "region": "solve",
        "filename": str(source),
        "function": "solve",
        "first_lineno": 1,
        "line_numbers": np.asarray([2]),
        "hits": np.asarray([4]),
        "times": np.asarray([5.0]),
        "unit": 1e-6,
    }
    results = _nested_results(line_profile={0: [record]})

    document = create_html_report(
        results, tmp_path / "report.html", include_charts=False
    ).read_text(encoding="utf-8")
    # solve sits under loop and under final: a row, and a line table, for each.
    ids = re.findall(r'<table class="lp-table" id="([^"]*)"', document)
    assert len(ids) == 2 and len(set(ids)) == 2
    assert 'id="run-0-lines"' not in document


def test_highlight_python_marks_tokens_and_escapes_text():
    from scope_profiler.html_report import _highlight_python

    lines = [
        "@app.route('/x')",
        "def run(self, items):",
        '    """Doc with <b>."""',
        "    total = len(items) + 0x1F  # count < limit",
        "    flag = None if items else True",
        "    label = f'{total} items'",
        "    return self.max(a @ b)",
    ]
    rendered = _highlight_python(lines)
    assert rendered[0] == (
        '<span class="tk-dec">@</span><span class="tk-dec">app</span>'
        '<span class="tk-dec">.</span><span class="tk-dec">route</span>'
        '(<span class="tk-str">&#x27;/x&#x27;</span>)'
    )
    assert '<span class="tk-fn">run</span>' in rendered[1]
    assert rendered[2] == (
        '    <span class="tk-str">&quot;&quot;&quot;Doc with &lt;b&gt;.'
        "&quot;&quot;&quot;</span>"
    )
    assert '<span class="tk-bi">len</span>' in rendered[3]
    assert '<span class="tk-num">0x1F</span>' in rendered[3]
    assert '<span class="tk-com"># count &lt; limit</span>' in rendered[3]
    assert rendered[4].count('class="tk-const"') == 2
    assert 'class="tk-kw">if</span>' in rendered[4]
    assert 'class="tk-str"' in rendered[5]
    # A method called max is not the builtin, and matrix @ is no decorator.
    assert 'tk-bi">max' not in rendered[6]
    assert "tk-dec" not in rendered[6]


def test_highlight_python_survives_a_slice_ending_mid_statement():
    from scope_profiler.html_report import _highlight_python

    # A function's lines can stop inside a call or an unterminated string.
    rendered = _highlight_python(["x = call(", "    1,", "y = '''open", "still <open"])
    assert rendered[0] == "x = call("
    assert rendered[1].endswith('<span class="tk-num">1</span>,')
    assert rendered[3] == "still &lt;open"
    # A multi-line string is marked on each of its lines.
    rendered = _highlight_python(['s = """a', 'b"""'])
    assert rendered == [
        's = <span class="tk-str">&quot;&quot;&quot;a</span>',
        '<span class="tk-str">b&quot;&quot;&quot;</span>',
    ]


def test_line_profile_source_travels_with_the_profile(tmp_path, monkeypatch):
    """The report shows the code without the source file at hand."""
    import importlib.util

    pytest.importorskip("line_profiler")
    from scope_profiler import ProfileManager

    code = tmp_path / "code" / "kernels.py"
    code.parent.mkdir()
    code.write_text(
        "def kernel(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += i * i\n"
        "    return total\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("kernels", code)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)

    # Recorded from another directory, with the default (minimal) metadata.
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    profile = workdir / "run.h5"
    with ProfileManager.session(
        file_path=str(profile), use_line_profiler=True, verbose=False
    ):
        ProfileManager.profile("kernel")(kernels.kernel)(2000)

    records = read_h5(profile).line_profile[0]
    # Only the user's function: scope-profiler's own frames are not kept.
    assert [record["function"] for record in records] == ["kernel"]
    assert records[0]["filename"] == "kernels.py"
    assert records[0]["source"].startswith("def kernel(n):")
    assert records[0]["source_first_lineno"] == 1

    # The source file is gone, yet the report shows the code.
    code.unlink()
    import linecache

    linecache.clearcache()
    document = create_html_report(
        profile, tmp_path / "report.html", include_charts=False
    ).read_text(encoding="utf-8")
    assert '<span class="tk-kw">for</span> i <span class="tk-kw">in</span>' in document
    assert '<td class="lp-src"></td>' not in document


def test_line_profile_says_when_it_has_no_source(tmp_path):
    from scope_profiler import MPIRegion, Region

    # As recorded before profiles stored their source, by file name only.
    record = {
        "region": "work",
        "filename": "gone.py",
        "function": "work",
        "first_lineno": 10,
        "line_numbers": np.asarray([11]),
        "hits": np.asarray([1]),
        "times": np.asarray([5.0]),
        "unit": 1e-6,
    }
    results = ProfilingResults(
        {"work": MPIRegion("work", {0: Region(np.array([0]), np.array([1]))})},
        line_profile={0: [record]},
    )
    document = create_html_report(
        results, tmp_path / "report.html", include_charts=False
    ).read_text(encoding="utf-8")
    assert "No source: this profile does not store it, and <code>gone.py</code>" in (
        document
    )


def test_line_profile_reads_a_module_path_from_sys_path(tmp_path, monkeypatch):
    from scope_profiler import MPIRegion, Region
    from scope_profiler.html_report import _source_path

    # A file outside the working directory is recorded by its module path.
    module = tmp_path / "lib" / "pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text("def work():\n    total = 41 + 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "lib"))
    _source_path.cache_clear()
    record = {
        "region": "work",
        "filename": "pkg/mod.py",
        "function": "work",
        "first_lineno": 1,
        "line_numbers": np.asarray([2]),
        "hits": np.asarray([1]),
        "times": np.asarray([5.0]),
        "unit": 1e-6,
    }
    results = ProfilingResults(
        {"work": MPIRegion("work", {0: Region(np.array([0]), np.array([1]))})},
        line_profile={0: [record]},
    )
    document = create_html_report(
        results, tmp_path / "report.html", include_charts=False
    ).read_text(encoding="utf-8")
    _source_path.cache_clear()
    assert "total = " in document
    assert "No source" not in document


def test_region_source_snippet_gives_way_to_line_profiles(tmp_path):
    from scope_profiler import MPIRegion, Region

    def results(line_profile):
        region = Region(
            np.array([0]),
            np.array([1]),
            source_file="app.py",
            source_lineno=3,
            source_text="with region('solve'):\n    solve()",
        )
        return ProfilingResults(
            {"solve": MPIRegion("solve", {0: region})}, line_profile=line_profile
        )

    record = {
        "region": "other",
        "filename": "app.py",
        "function": "other",
        "first_lineno": 1,
        "line_numbers": np.asarray([2]),
        "hits": np.asarray([1]),
        "times": np.asarray([1.0]),
        "unit": 1e-6,
    }
    without = create_html_report(
        results({}), tmp_path / "without.html", include_charts=False
    ).read_text(encoding="utf-8")
    with_lines = create_html_report(
        results({0: [record]}), tmp_path / "with.html", include_charts=False
    ).read_text(encoding="utf-8")

    # Without line profiling, the captured snippet is the only view of the code.
    assert "<pre><code>with region(&#x27;solve&#x27;):" in without
    # With it, the code is shown with its timings; the snippet would repeat it.
    assert "<pre><code>" not in with_lines
    # The call site itself stays.
    assert "<p class='muted'>app.py:3</p>" in with_lines
