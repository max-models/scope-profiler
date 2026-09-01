"""The region classes behind each optional profiling mode.

Each mode swaps in a different ``BaseProfileRegion`` subclass, and each one
repeats the same slot-reservation dance around a different third-party call --
an NVTX range, a LIKWID marker, a CUDA event, a line profiler. The timing
contract has to hold identically in all of them, including under recursion,
under ``pause()``, and when the profiled call raises.

LIKWID and CUDA need hardware; NVTX and line_profiler do not but should not be
required either. All four are therefore driven through stand-in modules that
record the calls, which is what these tests assert on.
"""

import sys
import types

import numpy as np
import pytest

from scope_profiler.profile_config import ProfilingConfig
from scope_profiler.region_profiler import (
    CUDATimingNVTXProfileRegion,
    CUDATimingProfileRegion,
    DisabledProfileRegion,
    FullProfileRegion,
    LineProfilerRegion,
    NVTXProfileRegion,
    TimeOnlyProfileRegion,
    _function_for_frame,
    _import_line_profiler,
    _import_nvtx,
    _import_pylikwid,
)


class _RecordingNVTX(types.ModuleType):
    """Stands in for the nvtx module, recording the range stack."""

    def __init__(self):
        super().__init__("nvtx")
        self.calls = []
        self.depth = 0
        self.max_depth = 0

    def push_range(self, name):
        self.calls.append(("push", name))
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)

    def pop_range(self):
        self.calls.append(("pop", None))
        self.depth -= 1


class _RecordingLikwid(types.ModuleType):
    """Stands in for pylikwid's marker API."""

    def __init__(self):
        super().__init__("pylikwid")
        self.calls = []

    def markerstartregion(self, name):
        self.calls.append(("start", name))

    def markerstopregion(self, name):
        self.calls.append(("stop", name))


class _FakeCUDAEvent:
    def __init__(self, index):
        self.index = index

    def synchronize(self):
        pass


class _RecordingGPUBackend:
    """A GPU timing backend that hands out numbered events."""

    name = "fake"

    def __init__(self):
        self.events = 0

    def record_event(self):
        self.events += 1
        return _FakeCUDAEvent(self.events)

    def elapsed_time_ns(self, start_event, end_event):
        # A stable, checkable number: 1000 ns per event recorded in between.
        return 1000 * (end_event.index - start_event.index)


@pytest.fixture
def nvtx(monkeypatch):
    module = _RecordingNVTX()
    monkeypatch.setitem(sys.modules, "nvtx", module)
    return module


@pytest.fixture
def likwid(monkeypatch):
    module = _RecordingLikwid()
    monkeypatch.setitem(sys.modules, "pylikwid", module)
    return module


@pytest.fixture
def config():
    return ProfilingConfig(deactivate_file_output=True, buffer_limit=2)


@pytest.fixture
def gpu_config(config, monkeypatch):
    backend = _RecordingGPUBackend()
    monkeypatch.setattr(
        "scope_profiler.region_profiler.resolve_gpu_timing_backend",
        lambda requested: backend,
    )
    return config, backend


def _module_level_target():
    """Visible in its own module globals, so it is found rather than rebuilt."""
    return sys._getframe()


def _durations(region):
    return region.get_durations_numpy()


# --------------------------------------------------------------------------
# NVTX
# --------------------------------------------------------------------------


def test_nvtx_context_manager_brackets_the_timed_region(nvtx, config):
    region = NVTXProfileRegion("solve", config)

    with region:
        pass

    assert nvtx.calls == [("push", "solve"), ("pop", None)]
    assert region.num_calls == 1
    assert _durations(region)[0] >= 0


def test_nvtx_decorator_brackets_the_call(nvtx, config):
    region = NVTXProfileRegion("solve", config)

    @region.wrap
    def work():
        return 42

    assert work() == 42
    assert nvtx.calls == [("push", "solve"), ("pop", None)]
    assert region.num_calls == 1


def test_nvtx_range_is_popped_when_the_body_raises(nvtx, config):
    region = NVTXProfileRegion("solve", config)

    with pytest.raises(ZeroDivisionError), region:
        # Raising inside the region is the point of the test.
        1 / 0  # noqa: B018

    assert nvtx.depth == 0
    assert nvtx.calls[-1] == ("pop", None)


def test_nvtx_range_is_popped_when_the_decorated_call_raises(nvtx, config):
    region = NVTXProfileRegion("solve", config)

    @region.wrap
    def work():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        work()

    assert nvtx.depth == 0


def test_nvtx_ranges_nest_under_recursion(nvtx, config):
    region = NVTXProfileRegion("recurse", config)

    def recurse(depth):
        with region:
            if depth:
                recurse(depth - 1)

    recurse(2)

    assert nvtx.depth == 0
    assert nvtx.max_depth == 3
    assert region.num_calls == 3
    assert (_durations(region) >= 0).all()


def test_a_paused_nvtx_region_emits_no_range_and_records_nothing(nvtx, config):
    config._paused = True
    region = NVTXProfileRegion("solve", config)

    with region:
        pass

    @region.wrap
    def work():
        return 1

    work()

    assert nvtx.calls == []
    assert region.num_calls == 0


def test_nvtx_reports_a_missing_install(monkeypatch, config):
    monkeypatch.setitem(sys.modules, "nvtx", None)
    with pytest.raises(ImportError, match="nvtx is not installed"):
        NVTXProfileRegion("solve", config)


def test_import_nvtx_returns_the_module(nvtx):
    assert _import_nvtx() is nvtx


# --------------------------------------------------------------------------
# LIKWID
# --------------------------------------------------------------------------


def test_likwid_markers_bracket_the_timed_region(likwid, config):
    region = FullProfileRegion("solve", config)

    with region:
        pass

    assert likwid.calls == [("start", "solve"), ("stop", "solve")]
    assert region.num_calls == 1


def test_likwid_decorator_brackets_the_call(likwid, config):
    region = FullProfileRegion("solve", config)

    @region.wrap
    def work():
        return "done"

    assert work() == "done"
    assert likwid.calls == [("start", "solve"), ("stop", "solve")]


def test_likwid_marker_is_stopped_when_the_decorated_call_raises(likwid, config):
    region = FullProfileRegion("solve", config)

    @region.wrap
    def work():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        work()

    assert likwid.calls == [("start", "solve"), ("stop", "solve")]
    # The slot is still filled in: a failed call is a recorded call.
    assert region.num_calls == 1
    assert _durations(region)[0] >= 0


def test_a_paused_likwid_region_emits_no_markers(likwid, config):
    config._paused = True
    region = FullProfileRegion("solve", config)

    with region:
        pass

    assert likwid.calls == []
    assert region.num_calls == 0


def test_likwid_region_grows_its_buffer_like_any_other(likwid, config):
    region = FullProfileRegion("solve", config)

    for _ in range(5):
        with region:
            pass

    assert region.num_calls == 5
    assert region.capacity >= 5
    assert len(_durations(region)) == 5


def test_import_pylikwid_returns_the_module(likwid):
    assert _import_pylikwid() is likwid


# --------------------------------------------------------------------------
# CUDA events, and CUDA + NVTX together
# --------------------------------------------------------------------------


def test_cuda_region_records_one_event_pair_per_call(gpu_config):
    config, backend = gpu_config
    region = CUDATimingProfileRegion("kernel", config)

    with region:
        pass

    assert backend.events == 2
    assert region.get_gpu_durations_numpy().tolist() == [1000]


def test_cuda_decorator_records_an_event_pair(gpu_config):
    config, backend = gpu_config
    region = CUDATimingProfileRegion("kernel", config)

    @region.wrap
    def work():
        return 1

    work()

    assert backend.events == 2
    assert region.get_gpu_durations_numpy().tolist() == [1000]


def test_cuda_region_grows_its_event_lists_with_the_buffers(gpu_config):
    config, _ = gpu_config
    region = CUDATimingProfileRegion("kernel", config)

    for _ in range(5):
        with region:
            pass

    assert region.num_calls == 5
    assert len(region.get_gpu_durations_numpy()) == 5
    assert len(region._gpu_start_events) >= 5


def test_a_paused_cuda_region_records_no_events(gpu_config):
    config, backend = gpu_config
    config._paused = True
    region = CUDATimingProfileRegion("kernel", config)

    with region:
        pass

    assert backend.events == 0
    assert region.num_calls == 0


def test_a_call_with_no_recorded_events_keeps_its_slot(gpu_config):
    """A slot whose events are missing is skipped, not treated as zero-length."""
    config, _ = gpu_config
    region = CUDATimingProfileRegion("kernel", config)

    with region:
        pass
    region._gpu_end_events[0] = None
    region.gpu_durations[0] = -1

    assert region.get_gpu_durations_numpy().tolist() == [-1]


def test_cuda_nvtx_region_does_both(nvtx, gpu_config):
    config, backend = gpu_config
    region = CUDATimingNVTXProfileRegion("kernel", config)

    with region:
        pass

    assert nvtx.calls == [("push", "kernel"), ("pop", None)]
    assert backend.events == 2
    assert region.num_calls == 1


def test_cuda_nvtx_decorator_does_both(nvtx, gpu_config):
    config, backend = gpu_config
    region = CUDATimingNVTXProfileRegion("kernel", config)

    @region.wrap
    def work():
        return 3

    assert work() == 3
    assert nvtx.calls == [("push", "kernel"), ("pop", None)]
    assert backend.events == 2


def test_cuda_nvtx_pops_its_range_when_the_call_raises(nvtx, gpu_config):
    config, _ = gpu_config
    region = CUDATimingNVTXProfileRegion("kernel", config)

    @region.wrap
    def work():
        raise KeyError("boom")

    with pytest.raises(KeyError):
        work()

    assert nvtx.depth == 0


def test_a_paused_cuda_nvtx_region_does_neither(nvtx, gpu_config):
    config, backend = gpu_config
    config._paused = True
    region = CUDATimingNVTXProfileRegion("kernel", config)

    with region:
        pass

    assert nvtx.calls == []
    assert backend.events == 0
    assert region.num_calls == 0


# --------------------------------------------------------------------------
# line_profiler
# --------------------------------------------------------------------------


def test_line_profiler_region_registers_the_decorated_function(config):
    region = LineProfilerRegion("work", config)

    @region.wrap
    def work(total):
        for _ in range(3):
            total += 1
        return total

    assert work(0) == 3
    assert region.num_calls == 1
    stats = region._line_profiler.get_stats()
    # line_profiler keys by qualname, so a nested function is
    # "<test>.<locals>.work" rather than a bare "work".
    assert any(key[2].endswith("work") for key in stats.timings)


def test_a_paused_line_profiler_region_records_nothing(config):
    config._paused = True
    region = LineProfilerRegion("work", config)

    @region.wrap
    def work():
        return 1

    work()
    assert region.num_calls == 0


def test_entering_by_frame_registers_the_running_function(config):
    region = LineProfilerRegion("block", config)

    def work():
        frame = sys._getframe()
        region.enter_frame(frame)
        try:
            return sum(range(10))
        finally:
            region.__exit__(None, None, None)

    assert work() == 45
    assert region.num_calls == 1
    assert region._registered_codes


def test_entering_by_frame_registers_a_code_object_once(config):
    region = LineProfilerRegion("block", config)

    def work():
        region.enter_frame(sys._getframe())
        region.__exit__(None, None, None)

    work()
    work()

    assert region.num_calls == 2
    assert len(region._registered_codes) == 1


def test_entering_timing_only_skips_line_registration(config):
    region = LineProfilerRegion("block", config)

    region.enter_timing_only()
    region.__exit__(None, None, None)

    assert region.num_calls == 1
    assert region._registered_codes == set()


def test_a_paused_frame_entry_records_nothing(config):
    config._paused = True
    region = LineProfilerRegion("block", config)

    region.enter_frame(sys._getframe())
    region.__exit__(None, None, None)
    region.enter_timing_only()
    region.__exit__(None, None, None)

    assert region.num_calls == 0


def test_manually_recorded_line_timings_are_accumulated(config):
    region = LineProfilerRegion("block", config)
    frame = sys._getframe()
    # One fixed line number, so the two samples accumulate on the same line
    # instead of on whichever line the call was written.
    lineno = 42

    region.record_line_timing(frame, lineno, 100)
    region.record_line_timing(frame, lineno, 400)
    # Non-positive samples are dropped rather than recorded as zero-hit lines.
    region.record_line_timing(frame, lineno, 0)

    records = region.manual_line_records()
    assert len(records) == 1
    record = records[0]
    assert record["function"] == "test_manually_recorded_line_timings_are_accumulated"
    assert record["line_numbers"].tolist() == [lineno]
    assert record["hits"].tolist() == [2]
    assert record["times"].tolist() == [500.0]
    assert record["unit"] == 1e-9


def test_line_profiler_reports_a_missing_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "line_profiler", None)
    with pytest.raises(ImportError, match="line_profiler is not"):
        _import_line_profiler()


def test_a_module_level_function_is_recovered_by_identity():
    """The preferred path: the running function is visible in its own globals."""
    frame = _module_level_target()
    assert _function_for_frame(frame) is _module_level_target


def test_a_local_function_is_reconstructed_from_its_code():
    """A function only its caller can see is rebuilt, not found."""

    def target():
        return sys._getframe()

    frame = target()
    recovered = _function_for_frame(frame)

    assert recovered is not target
    assert recovered.__code__ is target.__code__


def test_a_closure_is_reconstructed_from_its_frame():
    """A nested function is rebuilt with matching cells so it can be registered."""
    captured = 7

    def outer():
        def inner():
            return captured, sys._getframe()

        return inner()

    _, frame = outer()
    recovered = _function_for_frame(frame)

    assert recovered is not None
    assert recovered.__code__ is frame.f_code


# --------------------------------------------------------------------------
# The plain modes, for comparison
# --------------------------------------------------------------------------


def test_a_disabled_region_records_nothing_and_still_runs_the_code(config):
    region = DisabledProfileRegion("solve", config)

    @region.wrap
    def work():
        return "ran"

    with region:
        pass

    assert work() == "ran"
    assert region.num_calls == 0
    assert region.get_durations_numpy().size == 0


def test_a_time_only_region_records_a_slot_per_call(config):
    region = TimeOnlyProfileRegion("solve", config)

    for _ in range(3):
        with region:
            pass

    assert region.num_calls == 3
    assert (np.diff(region.get_start_times_numpy()) >= 0).all()
