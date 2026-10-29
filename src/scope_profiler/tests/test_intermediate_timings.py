"""``print_timings`` and ``current_timings`` during a running session."""

import io

import pytest

import scope_profiler as sp
from scope_profiler import ProfileManager


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


@pytest.mark.parametrize("aggregation_mode", [False, True])
def test_current_timings_reports_the_running_session(aggregation_mode):
    with sp.session(
        deactivate_file_output=True, aggregation_mode=aggregation_mode, verbose=False
    ):
        for _ in range(3):
            with sp.region("step"):
                with sp.region("solver.solve"):
                    pass
        with sp.region("solver.io"):
            pass
        timings = sp.current_timings()
        assert list(timings) == ["step", "solver.solve", "solver.io"]
        step = timings["step"]
        assert step["num_calls"] == 3
        assert step["min"] <= step["mean"] <= step["max"]
        assert step["total"] == pytest.approx(3 * step["mean"])
        assert step["total"] >= timings["solver.solve"]["total"]
        assert list(sp.current_timings("solver.*")) == ["solver.solve", "solver.io"]
        assert list(sp.current_timings(["step", "solver.io"])) == ["step", "solver.io"]


def test_current_timings_leaves_out_open_calls():
    with sp.session(deactivate_file_output=True, verbose=False):
        with sp.region("outer"):
            assert sp.current_timings() == {}


def test_print_timings_prints_label_and_sorted_rows():
    stream = io.StringIO()
    with sp.session(deactivate_file_output=True, verbose=False):
        with sp.region("b"):
            sum(range(10_000))
        with sp.region("a"):
            pass
        returned = sp.print_timings("timestep 7", sort="name", file=stream)
        assert set(returned) == {"a", "b"}
        sp.print_timings(regions="missing", file=stream)
        sp.print_timings(sort=None, file=stream)
        sp.print_timings(file=stream)
    text = stream.getvalue()
    assert "Timings (timestep 7)" in text
    assert text.index("│ a") < text.index("│ b")
    assert "Timings: no regions recorded yet" in text
    for header in ("Region", "Calls", "Total", "Mean", "Min", "Max"):
        assert header in text


def test_format_seconds_picks_a_readable_unit():
    from scope_profiler.summary import _format_seconds

    assert _format_seconds(2.5) == "2.500 s"
    assert _format_seconds(2.5e-3) == "2.500 ms"
    assert _format_seconds(2.5e-6) == "2.500 µs"
    assert _format_seconds(2.5e-9) == "2 ns"
