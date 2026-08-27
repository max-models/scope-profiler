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

    import scope_profiler.plotting_scripts as plotting_scripts

    monkeypatch.setattr(
        plotting_scripts, "plot_gantt", lambda *args, **kwargs: Figure()
    )
    monkeypatch.setattr(
        plotting_scripts, "plot_durations", lambda *args, **kwargs: Figure()
    )

    cli_main(["report", str(profile), "-o", str(report)])

    document = report.read_text(encoding="utf-8")
    assert "Timeline: profile" in document
    assert "Region durations" in document
    assert "data-plotlyjs=True" in document
    assert "data-plotlyjs=False" in document
