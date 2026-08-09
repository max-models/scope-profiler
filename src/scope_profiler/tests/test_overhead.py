"""Instrumentation cost of a profiled region, in absolute time per call.

What matters to a user is what one ``with ProfileManager.profile_region(...)``
or one decorated call *adds* to their program, so every budget here is an
absolute per-call cost in nanoseconds, measured against the same loop without
instrumentation. Ratios against a workload are deliberately avoided: they only
say how big the workload was.

The budgets in ``BUDGET_NS`` are regression guards, not benchmarks. They sit
roughly an order of magnitude above what a current laptop measures, so a slow
or loaded CI machine still passes while a change that makes the hot path
allocate, take a lock, or touch the filesystem per call does not. Each
measurement takes the *minimum* over several repeats, which is the robust
estimator for a cost: noise can only ever add time.

Every measurement is printed, plus a summary table once the module finishes.
pytest captures stdout, so pass ``-s`` to see it::

    pytest -s src/scope_profiler/tests/test_overhead.py
"""

import gc
import math
from contextlib import contextmanager
from time import perf_counter_ns

import numpy as np
import pytest

from scope_profiler import ProfileManager
from scope_profiler.region_profiler import DisabledProfileRegion, TimeOnlyProfileRegion

# Per-call budgets in nanoseconds, by profiling mode. Measured on an idle
# laptop (2026): ~100 ns disabled, ~780 ns with timestamps.
BUDGET_NS = {
    "disabled": 2_000,
    "time": 6_000,
}

# Extra budget for resolving a region by name on every call instead of
# hoisting the region object out of the loop (measured: ~45 ns, one dict get).
LOOKUP_BUDGET_NS = 2_000

# A region records one int64 start and one int64 end per call.
BYTES_PER_EVENT = 16

ITERATIONS = 20_000
REPEATS = 5

MODES = {
    "disabled": (dict(deactivate_profiling=True), DisabledProfileRegion),
    "time": (dict(), TimeOnlyProfileRegion),
}


@contextmanager
def _no_gc():
    """Keep a collection cycle from landing inside a measurement."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _ns_per_iteration(body, iterations=ITERATIONS, repeats=REPEATS):
    """Fastest observed cost of one ``body`` iteration, in nanoseconds."""
    best = math.inf
    with _no_gc():
        for _ in range(repeats):
            start = perf_counter_ns()
            body(iterations)
            end = perf_counter_ns()
            best = min(best, (end - start) / iterations)
    return best


def _overhead_ns(instrumented, baseline, iterations=ITERATIONS, repeats=REPEATS):
    """Nanoseconds ``instrumented`` adds per iteration over ``baseline``.

    Both callables take an iteration count and must do the same work apart
    from the instrumentation, so the difference is the profiler's cost.
    """
    cost = _ns_per_iteration(instrumented, iterations, repeats)
    reference = _ns_per_iteration(baseline, iterations, repeats)
    return cost - reference


# Everything reported by _report(), for the summary table printed at the end
# of the module: (label, measured ns, budget ns).
_MEASUREMENTS: list[tuple[str, float, int]] = []


def _report(label, measured_ns, budget_ns):
    """Print one measurement and keep it for the closing summary.

    pytest captures stdout, so run with ``-s`` to watch these live::

        pytest -s src/scope_profiler/tests/test_overhead.py
    """
    _MEASUREMENTS.append((label, measured_ns, budget_ns))
    headroom = budget_ns / measured_ns if measured_ns > 0 else math.inf
    print(
        f"  {label:<34s} {measured_ns:8.0f} ns   "
        f"(budget {budget_ns:>6d} ns, {headroom:5.1f}x headroom)"
    )


@pytest.fixture(scope="module", autouse=True)
def overhead_summary():
    """Print every measurement in one table once the module has finished."""
    yield
    if not _MEASUREMENTS:
        return
    width = max(len(label) for label, _, _ in _MEASUREMENTS)
    print("\n\nInstrumentation overhead, nanoseconds per call")
    print("-" * (width + 46))
    for label, measured_ns, budget_ns in _MEASUREMENTS:
        headroom = measured_ns and budget_ns / measured_ns
        print(
            f"{label:<{width}s}  {measured_ns:8.0f} ns   "
            f"budget {budget_ns:>6d} ns   {headroom:5.1f}x headroom"
        )
    print("-" * (width + 46))


def _empty_loop(iterations):
    for _ in range(iterations):
        pass


def _noop():
    pass


def _plain_call_loop(iterations, func=_noop):
    for _ in range(iterations):
        func()


@pytest.fixture
def configure(tmp_path):
    """Configure the profiler for one measurement, then reset global state.

    ``flush_to_disk=False`` keeps the run out of the filesystem entirely: this
    module measures the instrumentation, not the writer.
    """

    def _configure(**kwargs):
        kwargs.setdefault("deactivate_file_output", True)
        kwargs.setdefault("buffer_limit", 1 << 20)
        kwargs.setdefault("file_path", str(tmp_path / "overhead.h5"))
        ProfileManager.setup(**kwargs)
        return ProfileManager.get_config()

    yield _configure
    ProfileManager._reset()


@pytest.mark.parametrize("mode", list(MODES))
def test_context_manager_overhead_per_call(configure, mode):
    """One ``with region:`` costs less than its mode's budget."""
    settings, region_cls = MODES[mode]
    configure(**settings)
    region = ProfileManager.profile_region("bench")
    assert isinstance(region, region_cls)

    def instrumented(iterations, region=region):
        for _ in range(iterations):
            with region:
                pass

    overhead = _overhead_ns(instrumented, _empty_loop)
    _report(f"context manager [{mode}]", overhead, BUDGET_NS[mode])

    assert overhead < BUDGET_NS[mode], (
        f"{mode}: context manager costs {overhead:.0f} ns/call, "
        f"budget {BUDGET_NS[mode]} ns"
    )


@pytest.mark.parametrize("mode", list(MODES))
def test_decorator_overhead_per_call(configure, mode):
    """A decorated call costs less than its mode's budget over a plain call."""
    settings, _ = MODES[mode]
    configure(**settings)

    @ProfileManager.profile("decorated")
    def decorated():
        pass

    def instrumented(iterations, func=decorated):
        for _ in range(iterations):
            func()

    overhead = _overhead_ns(instrumented, _plain_call_loop)
    _report(f"decorator [{mode}]", overhead, BUDGET_NS[mode])

    assert overhead < BUDGET_NS[mode], (
        f"{mode}: decorated call costs {overhead:.0f} ns/call, "
        f"budget {BUDGET_NS[mode]} ns"
    )


@pytest.mark.parametrize("mode", list(MODES))
def test_region_lookup_by_name_stays_cheap(configure, mode):
    """Resolving the region by name per call adds only a dict lookup.

    Hot loops usually hoist the region object out, but the documented form
    calls ``ProfileManager.profile_region(name)`` inline. That path must not
    construct anything for an existing region.
    """
    settings, _ = MODES[mode]
    configure(**settings)
    region = ProfileManager.profile_region("bench")

    def by_name(iterations):
        profile_region = ProfileManager.profile_region
        for _ in range(iterations):
            with profile_region("bench"):
                pass

    def hoisted(iterations, region=region):
        for _ in range(iterations):
            with region:
                pass

    overhead = _overhead_ns(by_name, hoisted)
    _report(f"lookup by name [{mode}]", overhead, LOOKUP_BUDGET_NS)

    assert overhead < LOOKUP_BUDGET_NS, (
        f"{mode}: looking the region up by name adds {overhead:.0f} ns/call, "
        f"budget {LOOKUP_BUDGET_NS} ns"
    )
    # The lookup must reuse the region, not build a second one per call.
    assert ProfileManager.get_region("bench") is region


def test_nested_regions_cost_scales_with_depth(configure):
    """Three nested regions cost about three entries, not more.

    Nesting is the common shape in real code, and the per-scope pointer stack
    is the part that could quietly turn superlinear.
    """
    configure()
    outer = ProfileManager.profile_region("outer")
    middle = ProfileManager.profile_region("middle")
    inner = ProfileManager.profile_region("inner")

    def nested(iterations, outer=outer, middle=middle, inner=inner):
        for _ in range(iterations):
            with outer:
                with middle:
                    with inner:
                        pass

    overhead = _overhead_ns(nested, _empty_loop)
    budget = 3 * BUDGET_NS["time"]
    _report("three nested regions [time]", overhead, budget)

    assert (
        overhead < budget
    ), f"three nested regions cost {overhead:.0f} ns/iteration, budget {budget} ns"
    assert outer.num_calls == middle.num_calls == inner.num_calls


def test_buffer_growth_stays_amortized(configure):
    """A region starting with a tiny buffer still costs its per-call budget.

    ``_grow`` doubles the timestamp buffers, so the copying it does has to
    amortize to nothing across many calls. A linear-growth regression would
    blow this budget long before it ran out of memory.
    """
    calls = 50_000
    configure(buffer_limit=8)
    region = ProfileManager.profile_region("grows")

    def instrumented(iterations, region=region):
        for _ in range(iterations):
            with region:
                pass

    # One pass only: growth happens once per capacity doubling, so repeating
    # the measurement would time an already-grown buffer.
    overhead = _overhead_ns(instrumented, _empty_loop, iterations=calls, repeats=1)
    _report("buffer growth, amortized [time]", overhead, BUDGET_NS["time"])

    assert overhead < BUDGET_NS["time"], (
        f"growing from buffer_limit=8 costs {overhead:.0f} ns/call amortized, "
        f"budget {BUDGET_NS['time']} ns"
    )
    # Growth must not lose or duplicate events.
    assert region.num_calls == calls
    assert region.ptr == calls
    assert len(region.get_durations_numpy()) == calls
    assert np.all(region.end_times[:calls] >= region.start_times[:calls])


def test_recording_is_two_int64_per_call(configure):
    """Memory per recorded call is exactly one start and one end timestamp."""
    calls = 10_000
    configure(buffer_limit=1024)
    region = ProfileManager.profile_region("memory")

    for _ in range(calls):
        with region:
            pass

    recorded_bytes = region.start_times.nbytes + region.end_times.nbytes
    print(
        f"  {'memory for ' + str(calls) + ' calls':<34s} "
        f"{recorded_bytes / 1024:8.1f} KiB  "
        f"({recorded_bytes / calls:.0f} B/call allocated, "
        f"{BYTES_PER_EVENT} B/call recorded)"
    )
    # Capacity doubles, so the buffers hold at most twice the recorded calls.
    assert recorded_bytes <= 2 * calls * BYTES_PER_EVENT
    assert region.start_times.dtype == np.int64
    assert region.end_times.dtype == np.int64


def test_disabled_profiling_allocates_nothing_per_region(configure):
    """With profiling off there are no buffers and no wrapper at all."""
    configure(deactivate_profiling=True)
    region = ProfileManager.profile_region("off")

    assert isinstance(region, DisabledProfileRegion)
    # No per-region allocation: the shared read-only stand-in is handed out.
    assert region.start_times.nbytes == 0
    assert region.end_times.nbytes == 0
    assert region.capacity == 0

    def func():
        pass

    # The decorator form returns the function itself, so a disabled build
    # pays nothing inside the wrapper.
    assert region.wrap(func) is func

    with region:
        pass
    assert region.num_calls == 0


def test_measured_duration_matches_wall_clock(configure):
    """A region reports the wall-clock time of its body, not the time to
    record it.

    The instrumentation cost has to fall *outside* the recorded interval;
    if it leaked in, every duration in every report would be inflated.
    """
    calls = 200
    busy_ns = 200_000  # ~0.2 ms of real work per call
    configure()
    region = ProfileManager.profile_region("busy")

    def spin(duration_ns):
        deadline = perf_counter_ns() + duration_ns
        while perf_counter_ns() < deadline:
            pass

    outer_start = perf_counter_ns()
    for _ in range(calls):
        with region:
            spin(busy_ns)
    outer_end = perf_counter_ns()

    recorded_ns = int(np.sum(region.end_times[:calls] - region.start_times[:calls]))
    wall_ns = outer_end - outer_start
    # The time the loop spent outside the recorded intervals is the
    # instrumentation, measured here on a body that does real work.
    _report(
        "unrecorded gap, busy body [time]",
        (wall_ns - recorded_ns) / calls,
        BUDGET_NS["time"],
    )

    # Recorded time cannot exceed the wall clock, and the gap is the
    # instrumentation: it stays inside the budget for these calls.
    assert recorded_ns <= wall_ns
    assert wall_ns - recorded_ns < calls * BUDGET_NS["time"]

    per_call_ns = recorded_ns / calls
    assert per_call_ns >= busy_ns
