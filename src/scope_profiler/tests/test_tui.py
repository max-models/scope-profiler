"""Tests for the Textual HDF5 browser's data model."""

import re
import threading
import time

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
    render_plotext_text,
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
        "\n" * 10 + "    if enabled:\n        total = 0\n", encoding="utf-8"
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


def _find_tree_node(node, label):
    if str(node.label) == label:
        return node
    for child in node.children:
        found = _find_tree_node(child, label)
        if found is not None:
            return found
    return None


def _find_tree_data(node, data):
    if node.data is data:
        return node
    for child in node.children:
        found = _find_tree_data(child, data)
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
    solve_details = node_detail_text(solve)
    assert "Calls" in solve_details and "2" in solve_details
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


def test_overview_shows_dashboard_metrics_and_top_regions(sample_file):
    model = build_browser_model(sample_file)

    overview = node_detail_text(_find(model.root, "Overview"))

    assert "Top regions by total time" in overview
    assert "Calls" in overview
    assert "solve" in overview
    assert "█" in overview


def test_metadata_landing_page_is_compact(sample_file):
    model = build_browser_model(sample_file)

    metadata = node_detail_text(_find(model.root, "Metadata"))

    assert "metadata entries recorded" in metadata
    assert "Select a section below" in metadata


def test_metadata_values_wrap_inside_the_table(sample_file):
    model = build_browser_model(sample_file)
    section = _find(model.root, "Run")
    section.payload["entries"] = [("uname", "Darwin " * 30)]

    details = node_detail_text(section)

    assert len(max(details.splitlines(), key=len)) < 110
    assert details.count("Darwin") == 30


def test_metadata_with_unprintable_value_does_not_crash(sample_file):
    class BrokenValue:
        def __str__(self):
            raise TypeError("broken metadata")

    model = build_browser_model(sample_file)
    section = _find(model.root, "Run")
    section.payload["entries"] = [("broken", BrokenValue())]

    assert "<unprintable BrokenValue>" in node_detail_text(section)


def test_raw_hdf5_attributes_wrap_inside_the_table(sample_file):
    model = build_browser_model(sample_file)
    dataset = _find(model.root, "start_times")
    dataset.payload["attrs"] = {"description": "attribute " * 30}

    details = node_detail_text(dataset)

    assert len(max(details.splitlines(), key=len)) < 110
    assert details.count("attribute") == 30


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
    assert "if enabled:" in details
    assert "    total = 0" in details
    assert "    if enabled:" not in details


def test_line_profile_handles_missing_source_file(line_profile_file):
    model = build_browser_model(line_profile_file)
    rank = _find(_find(model.root, "Line Profile"), "Rank 0")
    line_profile_file.with_name("app.py").unlink()

    assert "<source unavailable>" in node_detail_text(rank)


def test_plot_section_exposes_existing_plot_kinds(sample_file):
    model = build_browser_model(sample_file)
    plots = _find(model.root, "Plots")

    assert [child.label for child in plots.children] == [
        "Gantt",
        "Durations",
        "Flame chart",
        "Flame graph",
        "Callgraph",
        "Timeseries",
        "Histogram",
        "Imbalance",
    ]
    assert "Matplotlib" in node_detail_text(plots.children[0])


def test_render_plot_dispatches_to_existing_renderer(
    sample_file, monkeypatch, tmp_path
):
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
            "metric": "avg",
            "sort_by": "avg",
            "top_n": "5",
            "log_scale": True,
            "backend": "plotly",
        },
    )

    assert captured["include"] == ["solve", "setup"]
    assert captured["exclude"] == ["debug"]
    assert captured["ranks"] == [0, 2, 3]
    assert captured["cmap"] == "viridis"
    assert captured["metric"] == "avg"
    assert captured["sort_by"] == "avg"
    assert captured["top_n"] == 5
    assert captured["log_scale"] is True
    assert captured["backend"] == "plotly"


def test_matplotlib_child_script_is_valid_python():
    compile(_matplotlib_child_script(), "<matplotlib-child>", "exec")


def test_render_plotext_text_captures_terminal_output(sample_file, monkeypatch):
    model = build_browser_model(sample_file)
    plot = _find(model.root, "Durations")

    def fake_plot(results, **kwargs):
        print("plotext chart")

    monkeypatch.setattr("scope_profiler.tui.render_plot", fake_plot)

    assert render_plotext_text(plot) == "plotext chart"


def test_render_plotext_text_fits_requested_dimensions(sample_file):
    model = build_browser_model(sample_file)
    plot = _find(model.root, "Durations")

    output = render_plotext_text(plot, width=48, height=12)
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    lines = [ansi_escape.sub("", line) for line in output.splitlines()]

    assert len(lines) <= 12
    assert max(map(len, lines)) <= 48


def test_plotext_rendering_is_serialized(sample_file, monkeypatch):
    model = build_browser_model(sample_file)
    plot = _find(model.root, "Durations")
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()

    def fake_plot(*args, **kwargs):
        with state_lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.02)
        with state_lock:
            state["active"] -= 1

    monkeypatch.setattr("scope_profiler.tui.render_plot", fake_plot)
    threads = [
        threading.Thread(target=render_plotext_text, args=(plot,)) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["maximum"] == 1


def test_render_plotext_text_reports_invalid_region_filter(sample_file, monkeypatch):
    model = build_browser_model(sample_file)
    plot = _find(model.root, "Durations")

    def fake_plot(results, **kwargs):
        raise re.error("nothing to repeat")

    monkeypatch.setattr("scope_profiler.tui.render_plot", fake_plot)

    with pytest.raises(ValueError, match="Invalid region filter"):
        render_plotext_text(plot, settings={"exclude": "*print_report"})


def test_tui_help_does_not_require_textual(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["tui", "--help"])

    assert exc_info.value.code == 0
    assert "scope-profiler tui" in capsys.readouterr().out


def test_tui_is_listed_in_top_level_help(capsys):
    with pytest.raises(SystemExit):
        cli_main(["--help"])

    assert "tui" in capsys.readouterr().out


def test_changing_a_plot_setting_reschedules_the_plotext_refresh(sample_file):
    """Regression: ``set_timer()`` accepts no arguments for its callback.

    Changing any of the plot settings while a plotext-capable plot is
    selected crashed the whole app with a TypeError, because the selected
    node was passed to ``set_timer`` as a third positional argument.
    """
    import asyncio

    from scope_profiler.tui import _build_textual_app_class

    model = build_browser_model(sample_file)
    durations = _find(model.root, "Durations")
    assert durations is not None
    app = _build_textual_app_class()(model)

    async def scenario():
        async with app.run_test() as pilot:
            app.selected_browser_node = durations
            app._schedule_plotext_refresh()
            await pilot.pause()
            first = app._plotext_refresh_timer
            assert first is not None
            # A second change replaces the pending refresh rather than adding one.
            app._schedule_plotext_refresh()
            await pilot.pause()
            assert app._plotext_refresh_timer is not first

    asyncio.run(scenario())


@pytest.mark.parametrize("target", ["Run", "Durations"])
def test_textual_navigation_renders_stable_detail_views(sample_file, target):
    import asyncio

    from scope_profiler.tui import _build_textual_app_class

    app = _build_textual_app_class()(build_browser_model(sample_file))

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            tree = app.query_one("#nav")
            tree.select_node(_find_tree_node(tree.root, target))
            await pilot.pause()
            detail = str(app.query_one("#detail").render())
            assert "Traceback" not in detail
            assert detail.strip()

    asyncio.run(scenario())


def test_navigation_pane_scrolls_when_tree_is_tall(tmp_path):
    import asyncio

    from scope_profiler.tui import _build_textual_app_class

    path = tmp_path / "many_regions.h5"
    _write_sample_h5(
        path,
        {
            0: {
                f"region-{index:02d}": ([index * NS], [(index + 1) * NS])
                for index in range(40)
            }
        },
    )
    app = _build_textual_app_class()(build_browser_model(path))

    async def scenario():
        async with app.run_test(size=(120, 20)) as pilot:
            nav = app.query_one("#nav")
            nav.root.expand_all()
            await pilot.pause()
            assert nav.max_scroll_y > 0

    asyncio.run(scenario())


def test_textual_navigation_renders_line_profile(line_profile_file):
    import asyncio

    from scope_profiler.tui import _build_textual_app_class

    model = build_browser_model(line_profile_file)
    line_profile_rank = _find(_find(model.root, "Line Profile"), "Rank 0")
    app = _build_textual_app_class()(model)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            tree = app.query_one("#nav")
            tree.select_node(_find_tree_data(tree.root, line_profile_rank))
            await pilot.pause()
            detail = str(app.query_one("#detail").render())
            assert "Rank 0 | solve | solve" in detail
            assert "    total = 0" in detail

    asyncio.run(scenario())


def test_detail_view_scrolls_long_line_profile_content(line_profile_file):
    import asyncio

    from scope_profiler.tui import _build_textual_app_class

    app = _build_textual_app_class()(build_browser_model(line_profile_file))

    async def scenario():
        async with app.run_test(size=(80, 20)) as pilot:
            detail = app.query_one("#detail")
            detail.update("\n".join(f"line {index}" for index in range(100)))
            await pilot.pause()

            viewport = app.query_one("#detail-scroll")
            assert viewport.max_scroll_y > 0
            viewport.scroll_to(y=viewport.max_scroll_y, animate=False)
            await pilot.pause()
            assert viewport.scroll_y > 0

    asyncio.run(scenario())


def test_detail_rendering_failure_is_contained(sample_file, monkeypatch):
    from scope_profiler.tui import _build_textual_app_class

    app = _build_textual_app_class()(build_browser_model(sample_file))
    node = _find(app.model.root, "Run")
    monkeypatch.setattr(
        "scope_profiler.tui.node_detail_text",
        lambda _node: (_ for _ in ()).throw(TypeError("bad detail")),
    )

    detail = app._detail(node)

    assert "Unable to render Run" in detail.plain
    assert "TypeError: bad detail" in detail.plain


def test_stale_plot_refresh_does_not_replace_current_detail(sample_file, monkeypatch):
    import asyncio

    from scope_profiler.tui import _build_textual_app_class

    model = build_browser_model(sample_file)
    durations = _find(model.root, "Durations")
    overview = _find(model.root, "Overview")
    app = _build_textual_app_class()(model)

    async def scenario():
        async with app.run_test() as pilot:
            detail = app.query_one("#detail")
            detail.update("current view")

            def stale_render(*args, **kwargs):
                app.selected_browser_node = overview
                return "stale chart"

            monkeypatch.setattr("scope_profiler.tui.render_plotext_text", stale_render)
            app.selected_browser_node = durations
            app._show_plotext_in_detail(durations)
            await pilot.pause()
            assert str(detail.render()) == "current view"

    asyncio.run(scenario())
