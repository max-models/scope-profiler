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


def test_frame_region_name_without_co_qualname():
    """Python 3.10 code objects have no co_qualname; naming must still work.

    The unit-test matrix runs 3.11+, where the attribute always exists, so the
    3.10 path is exercised with a stand-in frame.
    """

    class CodeWithoutQualname:
        co_name = "my_function"

    class FrameWithoutQualname:
        f_globals = {"__name__": "my_module"}
        f_code = CodeWithoutQualname()

    name = ProfileManager._frame_region_name(FrameWithoutQualname())
    assert name == "my_module.my_function"


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


def test_finalize_prints_the_shared_summary_table(tmp_path, capsys):
    """finalize() renders the same table as print_summary(), not its own."""
    file_path = tmp_path / "summary.h5"
    ProfileManager.setup(file_path=str(file_path))

    with ProfileManager.profile_region("outer"):
        for _ in range(2):
            with ProfileManager.profile_region("inner"):
                sleep(0.001)

    ProfileManager.finalize()
    printed = capsys.readouterr().out

    # Same header, columns and TOTAL row as ProfilingH5Reader.print_summary().
    reader = ProfilingH5Reader(file_path)
    reader.print_summary(title=f"{file_path}  (1 rank(s))")
    assert printed == capsys.readouterr().out

    assert "region" in printed and "std [s]" in printed
    assert "outer" in printed and "inner" in printed
    assert "TOTAL" in printed
    # The old per-region block format is gone.
    assert "Total Calls" not in printed


def test_finalize_quiet(tmp_path, capsys):
    file_path = tmp_path / "quiet.h5"
    ProfileManager.setup(file_path=str(file_path))
    with ProfileManager.profile_region("region"):
        pass
    ProfileManager.finalize(verbose=False)

    assert capsys.readouterr().out == ""


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
        "uname",
        "chip_information",
        "python_version",
        "scope_profiler_version",
        "working_directory",
        "omp_num_threads",
        "mpi_size",
        "total_cores",
        "user",
        "modules",
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
        assert attrs["mpi_size"] == 1
        assert attrs["total_cores"] == attrs["mpi_size"] * attrs["omp_num_threads"]

    reader = ProfilingH5Reader(file_path)
    # The reader exposes the same fields, decoded into plain Python types
    # (list-valued attributes come back from h5py as numpy arrays).
    assert reader.metadata.keys() == attrs.keys()
    assert reader.metadata["hostname"] == attrs["hostname"]
    assert isinstance(reader.metadata["modules"], list)


@pytest.mark.parametrize("flush_to_disk", [True, False])
def test_ncalls_only_persists_call_counts(tmp_path, flush_to_disk):
    """time_trace=False must persist call counts, not just hold them in memory."""
    file_path = tmp_path / f"profiling_ncalls_{flush_to_disk}.h5"
    ProfileManager.setup(
        time_trace=False, flush_to_disk=flush_to_disk, file_path=str(file_path)
    )

    for _ in range(5):
        with ProfileManager.profile_region("ctx_region"):
            pass

    @ProfileManager.profile("decorated_region")
    def decorated():
        pass

    for _ in range(3):
        decorated()

    ProfileManager.finalize(verbose=False)

    with h5py.File(file_path, "r") as f:
        regions = f["rank0"]["regions"]
        assert regions["ctx_region"].attrs["num_calls"] == 5
        assert regions["decorated_region"].attrs["num_calls"] == 3
        # No timing was requested, so no timestamps are stored.
        assert "start_times" not in regions["ctx_region"]

    reader = ProfilingH5Reader(file_path)
    assert reader.get_region("ctx_region")[0].num_calls == 5
    assert reader.get_region("decorated_region")[0].num_calls == 3
    # Duration-derived stats stay well-defined despite the absence of timings.
    assert reader.get_region("ctx_region")[0].total_duration == 0.0
    assert len(reader.get_region("ctx_region")[0].durations) == 0


def test_time_trace_region_reports_timestamp_count(tmp_path):
    """Regions that do record timing keep deriving num_calls from timestamps."""
    file_path = tmp_path / "profiling_timed.h5"
    ProfileManager.setup(file_path=str(file_path))

    for _ in range(4):
        with ProfileManager.profile_region("timed_region"):
            sleep(0.001)

    ProfileManager.finalize(verbose=False)

    region = ProfilingH5Reader(file_path).get_region("timed_region")[0]
    assert region.num_calls == 4
    assert len(region.durations) == 4
    assert region.min_duration > 0


if __name__ == "__main__":
    # test_readme()
    # test_all_region_types()
    test_line_profiler_context_manager()
