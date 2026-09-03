"""Tests for standalone HTML profiling reports."""

import json
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
    assert '<div class="overview">' in document
    assert "<code>solve</code></a> dominates the recorded time" in document
    assert 'href="#run-0-region-0"' in document
    assert "unevenly distributed across" in document
    assert "Rank imbalance" in document
    assert "Points far from the mean identify stragglers." in document


def test_report_navigation_links_to_runs_regions_and_chart_controls(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert '<nav class="toc" aria-label="Report contents">' in document
    assert '<a href="#run-0">profile</a>' in document
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
    assert "highlightFigure(chart, buildFigure(chart.payload, options), selectedRegion)" in document
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
    assert "Call tree unavailable: no data for the selected ranks" in document


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
                    calls=3, values={"cycles": 100, "instructions": 250}
                )
            }
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
        {}, num_ranks=1, likwid={0: {"solve": likwid}}, file_path="hardware.h5"
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
                }
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
    assert "Region durations" in document
    assert "Duration over time" in document
    assert "Rank heatmap" in document
    assert "Flame chart: profile" in document
    assert "Flame graph: profile" in document
    assert "Call graph: profile (rank 0)" in document
    assert 'id="scope-profiler-chart-0"' in document
    assert "const scopeProfilerCharts = " in document
    # The per-chart options reach the builder, now by way of the object the
    # region filter extends with its predicate.
    assert "buildFigure(chart.payload, options)" in document
    assert "...chart.options" in document
    assert 'class="chart chart-duration"' in document
    assert '"options": {"layout": {"height": 680}}' in document
    assert "Each bar is one recorded region call on rank 0." in document
    assert "Each bar shows a region's total recorded duration." in document
    assert "This heatmap uses exclusive timings." in document
    assert "Each frame is one recorded call on the selected ranks." in document
    assert "Repeated calls with the same call path are combined." in document
    assert "Each line follows a region's mean call duration" in document
    assert "Nodes are regions and links show caller-to-callee" in document
    assert 'class="chart-panel" open' in document
    assert 'data-chart-action="expand"' in document
    assert 'data-chart-action="collapse"' in document
    assert "plotly.js" in document
    assert "<script src=" not in document
    assert 'import("https://' not in document


def test_report_limits_gantt_and_uses_exclusive_rank_heatmap(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    from scope_profiler import plotting_scripts

    captured = {}

    def write_payload(data_filepath):
        Path(data_filepath).write_text(
            json.dumps({"plot": "gantt", "intervals": []}), encoding="utf-8"
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
    monkeypatch.setattr(plotting_scripts, "plot_flame", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame_graph", fake_plot)

    cli_main(["report", str(profile), "-o", str(report)])

    assert captured["gantt"]["ranks"] == [0]
    assert captured["heatmap"]["exclusive"] is True
    assert "exclusive timings" in report.read_text(encoding="utf-8")


def test_region_durations_chart_is_stacked_and_sorted_by_total(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    from scope_profiler import plotting_scripts

    captured = {}

    def fake_plot_durations(*args, data_filepath, **kwargs):
        captured.update(kwargs)
        Path(data_filepath).write_text(
            json.dumps({"plot": "gantt", "intervals": []}), encoding="utf-8"
        )

    def fake_plot(*args, data_filepath, **kwargs):
        Path(data_filepath).write_text(
            json.dumps({"plot": "gantt", "intervals": []}), encoding="utf-8"
        )

    monkeypatch.setattr(plotting_scripts, "plot_gantt", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_durations", fake_plot_durations)
    monkeypatch.setattr(plotting_scripts, "plot_duration_timeseries", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_callgraph", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_rank_heatmap", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame_graph", fake_plot)

    cli_main(["report", str(profile), "-o", str(report)])

    assert captured["sort_by"] == "total"
    assert captured["stack_children"] is True


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
                }
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

    create_html_report(profiles, report)

    assert captured["stack_children"] is False
    assert "Grouped bars compare each region's total recorded duration" in (
        report.read_text(encoding="utf-8")
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


def test_report_overview_names_the_hot_spot_not_its_enclosing_region(tmp_path):
    """Rank the hot spot by exclusive time.

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
            }
        },
    )

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    document = report.read_text(encoding="utf-8")
    assert "<code>kernel</code></a> dominates the recorded time" in document
    assert "in the region itself, excluding nested regions" in document
    assert "<code>wrapper</code></a> dominates" not in document
    # ...and say why the region at the top of the table is not the one named.
    assert "<code>wrapper</code></a> has the largest total" in document
    assert "is spent in the regions nested inside it" in document


def test_report_overview_omits_the_nesting_note_for_a_flat_profile(tmp_path):
    """One region cannot be both the hot spot and a misleading wrapper."""
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, {0: {"solve": ([0], [10_000_000])}})

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    document = report.read_text(encoding="utf-8")
    assert "<code>solve</code></a> dominates the recorded time" in document
    assert "has the largest total" not in document


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
        ]
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


def test_report_call_tree_shows_region_nesting(tmp_path):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(
        profile,
        # "inner" (0..5) nests entirely inside "outer" (0..10).
        {0: {"outer": ([0], [10]), "inner": ([2], [5])}},
    )

    cli_main(["report", str(profile), "-o", str(report), "--no-charts"])

    document = report.read_text(encoding="utf-8")
    assert "Call tree" in document
    outer_pos = document.find("<code>outer</code>")
    inner_pos = document.find("<code>inner</code>")
    assert outer_pos != -1 and inner_pos != -1
    assert outer_pos < inner_pos


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
    assert "solve (app.py:10)" in with_doc
    assert "28.57" in with_doc
    assert "Line profile" not in without_doc
