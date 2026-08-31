import socket
import sys
from time import perf_counter_ns, sleep
from types import SimpleNamespace

import h5py
import pytest

from scope_profiler import ProfileManager, ProfilingResults, read_h5
from scope_profiler.region_profiler import (
    CUDATimingNVTXProfileRegion,
    CUDATimingProfileRegion,
    DisabledProfileRegion,
    FullProfileRegion,
    LineProfilerRegion,
    NVTXProfileRegion,
    TimeOnlyProfileRegion,
)
from scope_profiler.tests import examples


class FakeGPUTimingBackend:
    name = "fake"

    def __init__(self):
        self._next = 0
        self.events = []

    def record_event(self):
        self._next += 10
        self.events.append(self._next)
        return self._next

    def elapsed_time_ns(self, start_event, end_event):
        return end_event - start_event


@pytest.mark.parametrize("use_likwid", [False])
@pytest.mark.parametrize("num_loops", [10, 50, 100])
@pytest.mark.parametrize("deactivate_profiling", [False, True])
def test_profile_manager(
    use_likwid: bool,
    num_loops: int,
    deactivate_profiling: bool,
):
    ProfileManager.setup(
        use_likwid=use_likwid,
        deactivate_profiling=deactivate_profiling,
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

    print(f"{deactivate_profiling = } {ProfileManager._config.deactivate_profiling = }")

    if deactivate_profiling:
        assert regions["loop1"].num_calls == 0
        assert regions["loop2"].num_calls == 0
        assert regions["test_decorator_labeled"].num_calls == 0
        assert regions["test_decorator_unlabeled"].num_calls == 0
        assert regions["main"].num_calls == 0
    else:
        assert regions["loop1"].num_calls == num_loops
        assert regions["loop2"].num_calls == num_loops * 2
        assert regions["test_decorator_labeled"].num_calls == num_loops
        assert regions["test_decorator_unlabeled"].num_calls == num_loops
        assert regions["main"].num_calls == 1


def test_all_region_types():
    # Disabled region
    ProfileManager.setup(
        use_likwid=False,
        deactivate_profiling=True,
        deactivate_file_output=True,
    )

    with ProfileManager.profile_region("disabled_region"):
        pass

    region = ProfileManager.get_region("disabled_region")
    assert isinstance(region, DisabledProfileRegion)
    assert region.num_calls == 0

    # Time-only region: the default once profiling is on
    ProfileManager.setup(
        use_likwid=False,
        deactivate_profiling=False,
        deactivate_file_output=True,
    )
    assert ProfileManager._region_cls is TimeOnlyProfileRegion
    with ProfileManager.profile_region("time_only_region"):
        sleep(0.001)

    region = ProfileManager.get_region("time_only_region")
    assert isinstance(region, TimeOnlyProfileRegion)
    assert region.num_calls == 1
    assert region.ptr == 1
    durations = region.get_durations_numpy()
    assert durations[0] > 0

    # NVTX region: CPU timing plus an NVTX range.
    calls = []
    fake_nvtx = SimpleNamespace(
        push_range=lambda name: calls.append(("push", name)),
        pop_range=lambda: calls.append(("pop",)),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "nvtx", fake_nvtx)
    try:
        ProfileManager.setup(
            use_nvtx=True,
            deactivate_file_output=True,
        )
        assert ProfileManager._region_cls is NVTXProfileRegion
        with ProfileManager.profile_region("nvtx_region"):
            sleep(0.001)
        assert calls == [("push", "nvtx_region"), ("pop",)]
    finally:
        monkeypatch.undo()

    # CUDA-event timing region: CPU timing plus device elapsed duration.
    backend = FakeGPUTimingBackend()
    ProfileManager.setup(
        use_gpu_timing=True,
        gpu_timing_backend=backend,
        deactivate_file_output=True,
    )
    assert ProfileManager._region_cls is CUDATimingProfileRegion
    with ProfileManager.profile_region("gpu_region"):
        sleep(0.001)
    region = ProfileManager.get_region("gpu_region")
    assert isinstance(region, CUDATimingProfileRegion)
    assert region.get_gpu_durations_numpy().tolist() == [10]

    # CUDA-event timing composes with NVTX annotations.
    calls = []
    fake_nvtx = SimpleNamespace(
        push_range=lambda name: calls.append(("push", name)),
        pop_range=lambda: calls.append(("pop",)),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "nvtx", fake_nvtx)
    try:
        backend = FakeGPUTimingBackend()
        ProfileManager.setup(
            use_gpu_timing=True,
            gpu_timing_backend=backend,
            use_nvtx=True,
            deactivate_file_output=True,
        )
        assert ProfileManager._region_cls is CUDATimingNVTXProfileRegion
        with ProfileManager.profile_region("gpu_nvtx_region"):
            sleep(0.001)
        assert calls == [("push", "gpu_nvtx_region"), ("pop",)]
        assert ProfileManager.get_region(
            "gpu_nvtx_region"
        ).get_gpu_durations_numpy().tolist() == [10]
    finally:
        monkeypatch.undo()

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
    pytest.importorskip("line_profiler")
    ProfileManager.setup(
        use_line_profiler=True,
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
    pytest.importorskip("line_profiler")
    ProfileManager.setup(
        use_line_profiler=True,
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


def test_line_profiler_context_manager_profiles_caller_without_functions():
    pytest.importorskip("line_profiler")
    ProfileManager.setup(use_line_profiler=True)

    def work(n=500):
        total = 0
        with ProfileManager.profile_region("lp_scope_only"):
            for i in range(n):
                total += i
        return total

    work()

    region = ProfileManager.get_region("lp_scope_only")
    stats = region.get_stats()
    assert len(stats.timings) > 0
    assert any("work" in str(key) for key in stats.timings)

    ProfileManager.finalize(verbose=False)


def test_recursive_line_profiler_records_traced_frame_lines(tmp_path, monkeypatch):
    added_functions = []

    class FakeStats:
        unit = 1e-9
        timings = {}

    class FakeLineProfiler:
        def add_function(self, func):
            added_functions.append(func.__name__)

        def enable_by_count(self):
            pass

        def disable_by_count(self):
            pass

        def get_stats(self):
            return FakeStats()

    monkeypatch.setattr(
        "scope_profiler.region_profiler._import_line_profiler",
        lambda: FakeLineProfiler,
    )
    script = tmp_path / "script.py"
    script.write_text(
        "def target():\n"
        "    total = 0\n"
        "    for i in range(3):\n"
        "        total += i\n"
        "    return total\n"
        "\n"
        "target()\n",
        encoding="utf-8",
    )

    try:
        ProfileManager.setup(use_line_profiler=True, deactivate_file_output=True)
        ProfileManager.run_script(str(script))
        records = ProfileManager._snapshot_line_profile()
    finally:
        ProfileManager._reset()

    assert "tracer" not in added_functions
    assert "run_script" not in added_functions
    assert any(record["function"] == "target" for record in records)
    target = next(record for record in records if record["function"] == "target")
    assert target["line_numbers"].size > 0
    assert target["hits"].size == target["line_numbers"].size
    assert target["times"].size == target["line_numbers"].size


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
        deactivate_file_output=True,
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
        deactivate_file_output=True,
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
        deactivate_file_output=True,
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
        deactivate_file_output=True,
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

    # Same header, columns and TOTAL row as ProfilingResults.print_summary().
    results = read_h5(file_path)
    results.print_summary()
    assert printed == capsys.readouterr().out

    header = next(line for line in printed.splitlines() if "region" in line)
    assert "region" in header and "avg [s]" in header
    assert "min [s]" not in header and "std [s]" not in header
    assert "outer" in printed and "inner" in printed
    assert "TOTAL" in printed
    # The old per-region block format is gone.
    assert "Total Calls" not in printed


def test_read_results_opens_the_finalized_file(tmp_path):
    """Results can be post-processed in the script that produced them."""
    file_path = tmp_path / "results.h5"
    ProfileManager.setup(file_path=str(file_path))

    for _ in range(3):
        with ProfileManager.profile_region("step"):
            sleep(0.001)

    ProfileManager.finalize(verbose=False)
    results = ProfileManager.read_results()

    assert isinstance(results, ProfilingResults)
    assert results.file_path == file_path
    assert results["step"].num_calls == 3
    assert len(results.events(include="step")) == 3


def test_setup_registers_the_run_start_time(tmp_path):
    """The run's start time is persisted and becomes the timeline origin."""
    file_path = tmp_path / "start_time.h5"
    before = perf_counter_ns()
    ProfileManager.setup(file_path=str(file_path))
    after = perf_counter_ns()

    sleep(0.01)  # un-instrumented work, invisible to the regions
    with ProfileManager.profile_region("step"):
        sleep(0.001)

    ProfileManager.finalize(verbose=False)
    results = ProfileManager.read_results()

    recorded = results.metadata["start_time_ns"]
    assert before <= recorded <= after
    assert results.run_start_time == pytest.approx(recorded / 1e9)
    assert results.time_origin == results.run_start_time

    # The sleep before the first region is now visible as a startup gap.
    startup = results.minimum_start_time - results.run_start_time
    assert startup >= 0.01
    assert results.events()[0]["start"] == pytest.approx(startup)


def test_finalize_registers_the_finalize_time(tmp_path):
    """finalize() records its own call time, and total_time spans setup->finalize."""
    file_path = tmp_path / "finalize_time.h5"
    ProfileManager.setup(file_path=str(file_path))

    sleep(0.01)  # un-instrumented startup, invisible to time_span
    with ProfileManager.profile_region("step"):
        sleep(0.001)
    sleep(0.01)  # un-instrumented teardown, also invisible to time_span

    before = perf_counter_ns()
    ProfileManager.finalize(verbose=False)
    after = perf_counter_ns()
    results = ProfileManager.read_results()

    recorded = results.metadata["finalize_time_ns"]
    assert before <= recorded <= after
    assert results.finalize_time == pytest.approx(recorded / 1e9)

    assert results.total_time == pytest.approx(
        results.finalize_time - results.run_start_time
    )
    # The un-instrumented startup and teardown make total_time the larger span.
    assert results.total_time > results.time_span
    assert results.total_time >= 0.02


def test_finalize_return_results_also_carries_total_time(tmp_path):
    """The in-memory path (deactivate_file_output=True) gets it too."""
    ProfileManager.setup(deactivate_file_output=True)
    with ProfileManager.profile_region("step"):
        sleep(0.001)

    results = ProfileManager.finalize(return_results=True, verbose=False)

    assert results.run_start_time is not None
    assert results.finalize_time is not None
    assert results.total_time == pytest.approx(
        results.finalize_time - results.run_start_time
    )


def test_read_results_before_finalize_raises(tmp_path):
    ProfileManager.setup(file_path=str(tmp_path / "missing.h5"))

    with pytest.raises(FileNotFoundError):
        ProfileManager.read_results()

    ProfileManager.finalize(verbose=False)


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
        assert "events" in f
        assert "region_table" in f
        # Metadata is global (gathered from rank 0 only), not duplicated
        # per rank.
        assert all("metadata" not in f[name] for name in f if name.startswith("rank"))

        attrs = dict(f["metadata"].attrs)
        assert expected_keys <= attrs.keys()
        assert attrs["hostname"] == socket.gethostname()
        assert attrs["omp_num_threads"] >= 1
        assert attrs["mpi_size"] == 1
        assert attrs["total_cores"] == attrs["mpi_size"] * attrs["omp_num_threads"]

    results = read_h5(file_path)
    # read_h5 exposes the same fields, decoded into plain Python types
    # (list-valued attributes come back from h5py as numpy arrays).
    assert results.metadata.keys() == attrs.keys()
    assert results.metadata["hostname"] == attrs["hostname"]
    assert isinstance(results.metadata["modules"], list)


def test_deactivate_file_output_writes_nothing(tmp_path):
    """With file output off, not even the metadata file appears on disk."""
    file_path = tmp_path / "never_written.h5"
    ProfileManager.setup(deactivate_file_output=True, file_path=str(file_path))

    for _ in range(5):
        with ProfileManager.profile_region("ctx_region"):
            sleep(0.001)

    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert not file_path.exists()
    assert list(tmp_path.iterdir()) == []
    # The run is still fully available in memory.
    assert results.get_region("ctx_region")[0].num_calls == 5
    assert results.get_region("ctx_region")[0].total_duration > 0


def test_run_without_any_region_writes_metadata_only(tmp_path):
    """A rank that recorded nothing gets no group, not an empty one.

    The rank groups in a file are exactly the ranks with something to report,
    which is what lets `check_mpi_launch.py` assert on the full rank list.
    """
    file_path = tmp_path / "nothing_profiled.h5"
    ProfileManager.setup(file_path=str(file_path))

    ProfileManager.finalize(verbose=False)

    with h5py.File(file_path, "r") as f:
        assert sorted(f) == [
            "events",
            "metadata",
            "rank_region_index",
            "region_table",
        ]

    results = read_h5(file_path)
    assert results.region_names == []
    assert results.num_ranks == 0


def test_region_reports_timestamp_count(tmp_path):
    """num_calls is derived from the recorded timestamps."""
    file_path = tmp_path / "profiling_timed.h5"
    ProfileManager.setup(file_path=str(file_path))

    for _ in range(4):
        with ProfileManager.profile_region("timed_region"):
            sleep(0.001)

    ProfileManager.finalize(verbose=False)

    region = read_h5(file_path).get_region("timed_region")[0]
    assert region.num_calls == 4
    assert len(region.durations) == 4
    assert region.min_duration > 0


if __name__ == "__main__":
    # test_readme()
    # test_all_region_types()
    test_line_profiler_context_manager()
