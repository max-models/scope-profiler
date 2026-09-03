"""Tests for standalone HTML profiling reports."""

import json
from pathlib import Path

from scope_profiler.__main__ import main as cli_main
from scope_profiler.h5reader import read_h5
from scope_profiler.html_report import create_html_report

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
    assert "<code>solve</code> dominates the recorded time" in document
    assert "unevenly distributed across" in document


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
                }
            ),
            encoding="utf-8",
        )

    from scope_profiler import plotting_scripts

    monkeypatch.setattr(plotting_scripts, "plot_gantt", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_durations", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_rank_heatmap", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame_graph", fake_plot)

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert "Timeline: profile" in document
    assert "Region durations" in document
    assert "Rank heatmap" in document
    assert "Flame chart: profile" in document
    assert "Flame graph: profile" in document
    assert 'id="scope-profiler-chart-0"' in document
    assert "const scopeProfilerCharts = " in document
    assert "buildFigure(chart.payload, chart.options)" in document
    assert 'class="chart chart-duration"' in document
    assert '"options": {"layout": {"height": 680}}' in document
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
        Path(data_filepath).write_text(
            json.dumps({"plot": "gantt", "intervals": []}), encoding="utf-8"
        )

    monkeypatch.setattr(plotting_scripts, "plot_gantt", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_durations", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_rank_heatmap", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame", fake_plot)
    monkeypatch.setattr(plotting_scripts, "plot_flame_graph", fake_plot)

    create_html_report(profiles, report)

    assert captured["stack_children"] is False


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
