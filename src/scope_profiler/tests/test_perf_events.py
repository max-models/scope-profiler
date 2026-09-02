"""Unit tests for the dependency-free and optional perf-event backends."""

from io import StringIO
from types import SimpleNamespace

import pytest

import scope_profiler.perf_events as perf
from scope_profiler import ProfileManager, region_profiler
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import perf_event_tables, print_perf_event_tables


def test_validate_events_normalizes_and_rejects_invalid_names():
    assert perf.validate_events(["Cycles", "instructions"]) == (
        "cycles",
        "instructions",
    )
    with pytest.raises(ValueError, match="Unknown perf event"):
        perf.validate_events(["made-up-event"])
    with pytest.raises(ValueError, match="unique"):
        perf.validate_events(["cycles", "cycles"])
    assert perf.validate_events("cycles") == ("cycles",)
    with pytest.raises(ValueError, match="non-empty"):
        perf.validate_events([])


def test_validate_events_rejects_non_linux_and_unknown_architecture(monkeypatch):
    monkeypatch.setattr(perf.platform, "system", lambda: "Darwin")
    with pytest.raises(perf.PerfEventError, match="only on Linux"):
        perf.validate_events(["cycles"])
    monkeypatch.setattr(perf.platform, "system", lambda: "Linux")
    monkeypatch.setattr(perf.platform, "machine", lambda: "sparc64")
    with pytest.raises(perf.PerfEventError, match="not implemented"):
        perf.validate_events(["cycles"])


def test_direct_group_opens_reads_and_closes_counters(monkeypatch):
    monkeypatch.setattr(perf, "_make_py_perf_measure", lambda events: None)
    opened = iter((10, 11))
    ioctls = []
    closed = []
    monkeypatch.setattr(perf, "_open_event", lambda *args: next(opened))
    monkeypatch.setattr(
        perf, "_ioctl", lambda fd, request: ioctls.append((fd, request))
    )
    monkeypatch.setattr(
        perf.os, "read", lambda fd, size: (fd * 10).to_bytes(8, "little")
    )
    monkeypatch.setattr(perf.os, "close", closed.append)

    group = perf.PerfEventGroup(("cycles", "instructions"))
    group.start()
    assert group.stop() == {"cycles": 100, "instructions": 110}
    assert closed == [11, 10]
    assert len(ioctls) == 6  # reset/enable/disable for both counters


def test_optional_group_reports_partial_read_and_disables():
    class Measure:
        def __init__(self):
            self.disabled = False

        def enable(self):
            pass

        def read(self):
            return SimpleNamespace(
                measurements=[10, 20], time_enabled_ns=10, time_running_ns=5
            )

        def disable(self):
            self.disabled = True

    measure = Measure()
    group = perf.PerfEventGroup(("cycles", "instructions"))
    group._measure = measure
    with pytest.raises(perf.PerfEventError, match="multiplexed"):
        group.stop()
    assert measure.disabled


def test_optional_group_returns_complete_read(monkeypatch):
    class Measure:
        def enable(self):
            pass

        def read(self):
            return SimpleNamespace(
                measurements=[10, 20], time_enabled_ns=5, time_running_ns=5
            )

        def disable(self):
            pass

    monkeypatch.setattr(perf, "_make_py_perf_measure", lambda events: Measure())
    group = perf.PerfEventGroup(("cycles", "instructions"))
    group.start()
    assert group.stop() == {"cycles": 10, "instructions": 20}


def test_group_cleans_up_if_direct_open_or_optional_enable_fails(monkeypatch):
    monkeypatch.setattr(perf, "_make_py_perf_measure", lambda events: None)
    monkeypatch.setattr(perf, "_open_event", lambda *args: 10)
    monkeypatch.setattr(
        perf, "_ioctl", lambda *args: (_ for _ in ()).throw(OSError("bad"))
    )
    closed = []
    monkeypatch.setattr(perf.os, "close", closed.append)
    with pytest.raises(OSError):
        perf.PerfEventGroup(("cycles",)).start()
    assert closed == [10]

    class BrokenMeasure:
        def enable(self):
            raise RuntimeError("Permission denied")

    monkeypatch.setattr(perf, "_make_py_perf_measure", lambda events: BrokenMeasure())
    with pytest.raises(perf.PerfEventError, match="perf_event_paranoid"):
        perf.PerfEventGroup(("cycles",)).start()


def test_optional_measure_factory_and_low_level_error_paths(monkeypatch):
    assert perf._make_py_perf_measure(("page-faults",)) is None
    fake_module = SimpleNamespace(
        Hardware=SimpleNamespace(CPU_CYCLES="cycles"),
        Measure=lambda events: ("measure", events),
    )
    monkeypatch.setitem(__import__("sys").modules, "py_perf_event", fake_module)
    assert perf._make_py_perf_measure(("cycles",)) == ("measure", ["cycles"])

    class BrokenMeasure:
        def __init__(self, events):
            raise RuntimeError("broken")

    fake_module.Measure = BrokenMeasure
    with pytest.raises(perf.PerfEventError, match="broken"):
        perf._make_py_perf_measure(("cycles",))

    monkeypatch.setattr(
        perf.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(syscall=lambda *args: -1),
    )
    monkeypatch.setattr(perf.ctypes, "get_errno", lambda: 13)
    with pytest.raises(perf.PerfEventError, match="perf_event_paranoid"):
        perf._open_event(0, 0)


def test_low_level_ioctl_and_optional_read_errors(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(ioctl=lambda *args: (_ for _ in ()).throw(OSError("nope"))),
    )
    with pytest.raises(perf.PerfEventError, match="ioctl failed"):
        perf._ioctl(1, 2)

    class BrokenMeasure:
        def read(self):
            raise RuntimeError("read failed")

        def disable(self):
            self.disabled = True

    group = perf.PerfEventGroup(("cycles",))
    group._measure = BrokenMeasure()
    with pytest.raises(perf.PerfEventError, match="read failed"):
        group.stop()
    assert group._measure is None


def test_open_success_and_close_optional_measure(monkeypatch):
    monkeypatch.setattr(
        perf.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(syscall=lambda *args: 17),
    )
    assert perf._open_event(0, 0) == 17

    class Measure:
        disabled = False

        def disable(self):
            self.disabled = True

    measure = Measure()
    group = perf.PerfEventGroup(("cycles",))
    group._measure = measure
    group.close()
    assert measure.disabled and group._measure is None


def test_optional_permission_error_has_kernel_policy_hint():
    error = perf._py_perf_error(
        "create", RuntimeError("Permission denied (os error 13)")
    )
    assert "perf_event_paranoid" in str(error)


def test_profile_manager_records_perf_event_totals(monkeypatch):
    class FakeGroup:
        def __init__(self, events):
            self.events = events

        def start(self):
            pass

        def stop(self):
            return {event: 11 for event in self.events}

    monkeypatch.setattr(region_profiler, "PerfEventGroup", FakeGroup)
    ProfileManager.setup(
        deactivate_file_output=True,
        perf_events=["cycles", "instructions"],
    )
    with ProfileManager.profile_region("work"):
        pass

    @ProfileManager.profile("decorated")
    def decorated():
        return 42

    assert decorated() == 42
    results = ProfileManager.finalize(verbose=False, return_results=True)
    totals = results.get_perf_events(0)
    assert totals["work"].values == {"cycles": 11, "instructions": 11}
    assert totals["decorated"].calls == 1


def test_perf_event_summary_tables_filter_and_format_counts():
    results = ProfilingResults(
        {},
        perf_events={
            0: {
                "keep": perf.PerfEventTotals(2, {"cycles": 1234, "instructions": 9}),
                "skip": perf.PerfEventTotals(1, {"cycles": 8}),
            }
        },
    )
    tables = perf_event_tables(results, include="keep")
    assert tables[0]["events"] == ["cycles", "instructions"]
    assert tables[0]["rows"] == [("keep", 2, 1234, 9)]
    stream = StringIO()
    print_perf_event_tables(results, include="keep", stream=stream)
    assert "Perf events (rank 0)" in stream.getvalue()
    assert "1234" in stream.getvalue()
