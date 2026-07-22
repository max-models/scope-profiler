import socket
from time import sleep

import h5py
import pytest

import scope_profiler.tests.examples as examples
from scope_profiler import ProfileManager
from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.region_profiler import (
    DisabledProfileRegion,
    FullProfileRegion,
    LikwidOnlyProfileRegion,
    LineProfilerRegion,
    NCallsOnlyProfileRegion,
    TimeOnlyProfileRegion,
    TimeOnlyProfileRegionNoFlush,
)


@pytest.mark.parametrize("time_trace", [True, False])
@pytest.mark.parametrize("use_likwid", [False])
@pytest.mark.parametrize("num_loops", [10, 50, 100])
@pytest.mark.parametrize("profiling_activated", [True, False])
def test_profile_manager(
    time_trace: bool,
    use_likwid: bool,
    num_loops: int,
    profiling_activated: bool,
):
    ProfileManager.setup(
        use_likwid=use_likwid,
        time_trace=time_trace,
        profiling_activated=profiling_activated,
        flush_to_disk=True,
    )

    examples.loop(
        label="loop1",
        num_loops=num_loops,
    )

    examples.loop(
        label="loop2",
        num_loops=num_loops * 2,
    )

    @ProfileManager.profile("test_decorator_labeled")
    def test_decorator():
        return

    @ProfileManager.profile
    def test_decorator_unlabeled():
        return

    for i in range(num_loops):
        test_decorator()
        test_decorator_unlabeled()

    with ProfileManager.profile_region("main"):
        pass

    ProfileManager.finalize()

    regions = ProfileManager.get_all_regions()

    print(
        f"{profiling_activated = } {time_trace = } {ProfileManager._config.profiling_activated = }"
    )

    if profiling_activated:
        assert regions["loop1"].num_calls == num_loops
        assert regions["loop2"].num_calls == num_loops * 2
        assert regions["test_decorator_labeled"].num_calls == num_loops
        assert regions["test_decorator_unlabeled"].num_calls == num_loops
        assert regions["main"].num_calls == 1
    else:
        assert regions["loop1"].num_calls == 0
        assert regions["loop2"].num_calls == 0
        assert regions["test_decorator_labeled"].num_calls == 0
        assert regions["test_decorator_unlabeled"].num_calls == 0
        assert regions["main"].num_calls == 0


def test_all_region_types():
    # Disabled region
    ProfileManager.setup(
        use_likwid=False,
        time_trace=False,
        profiling_activated=False,
        flush_to_disk=False,
    )

    with ProfileManager.profile_region("disabled_region"):
        pass

    region = ProfileManager.get_region("disabled_region")
    assert isinstance(region, DisabledProfileRegion)
    assert region.num_calls == 0

    # NCallsOnly region
    ProfileManager.setup(
        use_likwid=False,
        time_trace=False,
        profiling_activated=True,
        flush_to_disk=False,
    )

    with ProfileManager.profile_region("ncalls_region"):
        pass

    region = ProfileManager.get_region("ncalls_region")
    assert isinstance(region, NCallsOnlyProfileRegion)
    assert region.num_calls == 1
    assert region.get_durations_numpy().size == 0

    # Time-only region
    ProfileManager._region_cls = TimeOnlyProfileRegion
    with ProfileManager.profile_region("time_only_region"):
        sleep(0.001)

    region = ProfileManager.get_region("time_only_region")
    assert isinstance(region, TimeOnlyProfileRegion)
    assert region.num_calls == 1
    assert region.ptr == 1
    durations = region.get_durations_numpy()
    assert durations[0] > 0

    # Time-only region without flush
    ProfileManager._region_cls = TimeOnlyProfileRegionNoFlush
    with ProfileManager.profile_region("time_only_noflush"):
        sleep(0.001)

    region = ProfileManager.get_region("time_only_noflush")
    assert isinstance(region, TimeOnlyProfileRegionNoFlush)
    assert region.num_calls == 1
    assert region.ptr == 1
    durations = region.get_durations_numpy()
    assert durations[0] > 0

    # LIKWID-only region (mocked if pylikwid not installed)
    try:
        ProfileManager._region_cls = LikwidOnlyProfileRegion
        with ProfileManager.profile_region("likwid_only"):
            pass
        region = ProfileManager.get_region("likwid_only")
        assert isinstance(region, LikwidOnlyProfileRegion)
        assert region.num_calls == 1
    except ModuleNotFoundError:
        print("pylikwid not installed, skipping LIKWID-only test")

    # Full region (time + LIKWID)
    try:
        ProfileManager._region_cls = FullProfileRegion
        with ProfileManager.profile_region("full_region"):
            sleep(0.001)
        region = ProfileManager.get_region("full_region")
        assert isinstance(region, FullProfileRegion)
        assert region.num_calls == 1
        durations = region.get_durations_numpy()
        assert durations.size == 1
        assert durations[0] > 0
    except ModuleNotFoundError:
        print("pylikwid not installed, skipping FullProfileRegion test")

    # Finalize (should flush everything)
    ProfileManager.finalize(verbose=False)


def test_line_profiler_decorator():
    ProfileManager.setup(
        use_line_profiler=True,
        time_trace=True,
        flush_to_disk=True,
    )

    @ProfileManager.profile("lp_func")
    def work(n=1000):
        s = 0
        for i in range(n):
            s += i
        return s

    for _ in range(5):
        work()

    region = ProfileManager.get_region("lp_func")
    assert isinstance(region, LineProfilerRegion)
    assert region.num_calls == 5
    assert region.ptr == 5
    durations = region.get_durations_numpy()
    assert durations.size == 5
    assert all(d > 0 for d in durations)

    # Verify line_profiler captured stats
    stats = region.get_stats()
    assert len(stats.timings) > 0

    ProfileManager.finalize(verbose=False)


def test_line_profiler_context_manager():
    ProfileManager.setup(
        use_line_profiler=True,
        time_trace=True,
        flush_to_disk=True,
    )

    def work(n=500):
        s = 0
        for i in range(n):
            s += i
        return s

    with ProfileManager.profile_region("lp_ctx", functions=[work]):
        work()

    region = ProfileManager.get_region("lp_ctx")
    assert isinstance(region, LineProfilerRegion)
    assert region.num_calls == 1
    assert region.ptr == 1
    durations = region.get_durations_numpy()
    assert durations[0] > 0

    # Verify line_profiler captured stats for the registered function
    stats = region.get_stats()
    assert len(stats.timings) > 0

    ProfileManager.finalize(verbose=False)


def test_recursive_decorator_profiles_nested_calls():
    ProfileManager.setup(
        use_likwid=False,
        time_trace=False,
        flush_to_disk=False,
    )

    def helper_leaf(x):
        return x + 1

    def helper_mid(x):
        return helper_leaf(x) * 2

    @ProfileManager.profile("entry_recursive", recursive=True)
    def entry():
        total = 0
        for i in range(3):
            total += helper_mid(i)
        return total

    assert entry() == 12

    regions = ProfileManager.get_all_regions()
    leaf_name = f"{__name__}.test_recursive_decorator_profiles_nested_calls.<locals>.helper_leaf"
    mid_name = (
        f"{__name__}.test_recursive_decorator_profiles_nested_calls.<locals>.helper_mid"
    )

    assert regions["entry_recursive"].num_calls == 1
    assert regions[mid_name].num_calls == 3
    assert regions[leaf_name].num_calls == 3

    ProfileManager.finalize(verbose=False)


def test_self_recursive_region_decorator():
    """A single region re-entered by recursion must not corrupt its buffer."""
    ProfileManager.setup(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=False,
    )

    @ProfileManager.profile("fib_decorator")
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    assert fib(8) == 21

    region = ProfileManager.get_region("fib_decorator")
    assert region.num_calls == region.ptr
    starts = region.start_times[: region.ptr]
    ends = region.end_times[: region.ptr]
    assert (ends >= starts).all()

    ProfileManager.finalize(verbose=False)


def test_self_recursive_region_context_manager():
    """A single region re-entered by recursion must not corrupt its buffer."""
    ProfileManager.setup(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=False,
    )

    def fib(n):
        with ProfileManager.profile_region("fib_context"):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)

    assert fib(8) == 21

    region = ProfileManager.get_region("fib_context")
    assert region.num_calls == region.ptr
    starts = region.start_times[: region.ptr]
    ends = region.end_times[: region.ptr]
    assert (ends >= starts).all()

    ProfileManager.finalize(verbose=False)


def test_recursive_profile_setup_default_and_override():
    ProfileManager.setup(
        use_likwid=False,
        time_trace=False,
        flush_to_disk=False,
        recursive_profile=True,
    )

    def recurse(n):
        if n <= 1:
            return 1
        return recurse(n - 1) + 1

    def helper():
        return 42

    @ProfileManager.profile("root_default_recursive")
    def root():
        return recurse(4)

    @ProfileManager.profile("root_non_recursive", recursive=False)
    def root_non_recursive():
        return helper()

    assert root() == 4
    assert root_non_recursive() == 42

    recurse_name = (
        f"{__name__}.test_recursive_profile_setup_default_and_override.<locals>.recurse"
    )
    helper_name = (
        f"{__name__}.test_recursive_profile_setup_default_and_override.<locals>.helper"
    )
    regions = ProfileManager.get_all_regions()

    assert regions["root_default_recursive"].num_calls == 1
    assert regions[recurse_name].num_calls == 4
    assert regions["root_non_recursive"].num_calls == 1
    assert helper_name not in regions

    ProfileManager.finalize(verbose=False)


def test_finalize_writes_global_metadata(tmp_path):
    file_path = tmp_path / "profiling_metadata.h5"
    ProfileManager.setup(file_path=str(file_path))

    with ProfileManager.profile_region("region"):
        pass

    ProfileManager.finalize(verbose=False)

    expected_keys = {
        "timestamp",
        "hostname",
        "platform",
        "python_version",
        "scope_profiler_version",
        "working_directory",
        "omp_num_threads",
        "user",
    }

    with h5py.File(file_path, "r") as f:
        assert "metadata" in f
        assert "rank0" in f
        # Metadata is global (gathered from rank 0 only), not duplicated
        # per rank.
        assert "metadata" not in f["rank0"]

        attrs = dict(f["metadata"].attrs)
        assert expected_keys <= attrs.keys()
        assert attrs["hostname"] == socket.gethostname()
        assert attrs["omp_num_threads"] >= 1

    reader = ProfilingH5Reader(file_path)
    assert reader.metadata == attrs


if __name__ == "__main__":
    # test_readme()
    # test_all_region_types()
    test_line_profiler_context_manager()
