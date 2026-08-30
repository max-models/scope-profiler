"""Tests for standalone HTML profiling reports."""

from scope_profiler.__main__ import main as cli_main

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

    class Figure:
        def to_html(self, *, full_html, include_plotlyjs):
            assert full_html is False
            return f"<div data-plotlyjs={include_plotlyjs!r}>chart</div>"

    from scope_profiler import plotting_scripts

    monkeypatch.setattr(
        plotting_scripts, "plot_gantt", lambda *args, **kwargs: Figure()
    )
    monkeypatch.setattr(
        plotting_scripts, "plot_durations", lambda *args, **kwargs: Figure()
    )
    monkeypatch.setattr(
        plotting_scripts, "plot_rank_heatmap", lambda *args, **kwargs: Figure()
    )
    monkeypatch.setattr(
        plotting_scripts, "plot_flame", lambda *args, **kwargs: Figure()
    )

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert "Timeline: profile" in document
    assert "Region durations" in document
    assert "Rank heatmap" in document
    assert "Flame: profile" in document
    assert "data-plotlyjs=True" in document
    assert "data-plotlyjs=False" in document


def test_region_durations_chart_is_stacked_and_sorted_by_total(tmp_path, monkeypatch):
    profile = tmp_path / "profile.h5"
    report = tmp_path / "report.html"
    _write_sample_h5(profile, _sample_file_data(1, 10, 20))

    from scope_profiler import plotting_scripts

    captured = {}

    def fake_plot_durations(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(plotting_scripts, "plot_gantt", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotting_scripts, "plot_durations", fake_plot_durations)
    monkeypatch.setattr(
        plotting_scripts, "plot_rank_heatmap", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(plotting_scripts, "plot_flame", lambda *args, **kwargs: None)

    cli_main(["report", str(profile), "-o", str(report)])

    assert captured["sort_by"] == "total"
    assert captured["stack_children"] is True


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
