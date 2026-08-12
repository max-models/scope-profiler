"""The Fortran region API, compiled and run for real.

Every test here builds the shipped ``scope_profiler.f90`` with the system Fortran
compiler, runs a program against it, and checks what came out. That is the
only way to catch the things that actually go wrong in this layer: a clock id
the platform rejects, a struct layout that does not match, a trace the reader
cannot parse. The whole module skips when no compiler is available.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.native_trace import (
    FORMAT_VERSION,
    FORTRAN_DIR,
    MAGIC,
    TraceFormatError,
    convert_traces,
    find_traces,
    fortran_source_path,
    load_traces,
    read_trace,
)

MODULE_SOURCE = fortran_source_path()

#: First compiler on PATH; gfortran everywhere, the vendor ones on clusters.
COMPILERS = ("gfortran", "ifx", "ifort", "flang", "nvfortran")


def find_compiler() -> str | None:
    """Return a Fortran compiler from PATH, or None."""
    for name in COMPILERS:
        path = shutil.which(name)
        if path:
            return path
    return None


COMPILER = find_compiler()

pytestmark = [
    pytest.mark.skipif(COMPILER is None, reason="no Fortran compiler on PATH"),
    pytest.mark.skipif(
        not MODULE_SOURCE.exists(), reason=f"{MODULE_SOURCE} not found (installed?)"
    ),
]


def build(tmp_path: Path, program: str, name: str = "prog") -> Path:
    """Compile ``program`` against the module and return the executable."""
    source = tmp_path / f"{name}.f90"
    source.write_text(program)
    executable = tmp_path / name

    result = subprocess.run(
        [COMPILER, "-O1", "-o", str(executable), str(MODULE_SOURCE), str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"compilation failed\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    return executable


def run(executable: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run a built program in ``tmp_path`` and return the completed process."""
    result = subprocess.run(
        [str(executable)], cwd=tmp_path, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, f"run failed:\n{result.stderr}"
    return result


BASIC_PROGRAM = """
program basic
   use scope_profiler
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   implicit none
   integer :: i, outer, inner
   real(kind=8) :: acc

   call sp_init("trace")
   outer = sp_region("outer")
   inner = sp_region("inner")

   call sp_begin(outer)
   do i = 1, 5
      call sp_begin(inner)
      acc = work(2000)
      call sp_end(inner)
   end do
   call sp_end(outer)

   write (output_unit, "(a,i0)") "inner calls: ", sp_num_calls(inner)
   write (output_unit, "(f0.4)") acc
   call sp_finalize()

contains

   function work(n) result(acc)
      integer, intent(in) :: n
      real(kind=8) :: acc
      integer :: i
      acc = 0.0d0
      do i = 1, n
         acc = acc + sqrt(real(i, kind=8))
      end do
   end function work

end program basic
"""


def test_module_compiles_without_warnings(tmp_path):
    """The module builds clean at -Wall -Wextra, in strict Fortran 2008."""
    result = subprocess.run(
        [
            COMPILER,
            "-c",
            "-Wall",
            "-Wextra",
            "-O2",
            "-o",
            str(tmp_path / "scope_profiler.o"),
            str(MODULE_SOURCE),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    if "gfortran" in COMPILER:
        assert "Warning" not in result.stderr, result.stderr


def test_trace_records_regions_and_durations(tmp_path):
    """The end-to-end shape: names, call counts and positive durations."""
    executable = build(tmp_path, BASIC_PROGRAM)
    output = run(executable, tmp_path).stdout
    assert "inner calls: 5" in output

    rank, regions = read_trace(tmp_path / "trace_rank00000.spt")

    assert rank == 0
    assert sorted(regions) == ["inner", "outer"]

    starts, ends = regions["inner"]
    assert len(starts) == len(ends) == 5
    assert starts.dtype == np.int64
    durations = ends - starts
    assert (durations > 0).all(), "every timed region must take measurable time"
    # The inner regions all fall inside the single outer one.
    outer_start, outer_end = (arr[0] for arr in regions["outer"])
    assert (starts >= outer_start).all() and (ends <= outer_end).all()


def test_timestamps_share_pythons_clock(tmp_path):
    """Fortran and Python regions must land on one timeline.

    The whole point of resolving the clock at run time is that the Fortran
    timestamps are comparable with ``time.perf_counter_ns()``. If the module
    ever picks a different clock, this catches it: the run's timestamps would
    no longer bracket the moment we observed it from Python.
    """
    executable = build(tmp_path, BASIC_PROGRAM)

    before = time.perf_counter_ns()
    run(executable, tmp_path)
    after = time.perf_counter_ns()

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    first = int(regions["outer"][0][0])
    last = int(regions["outer"][1][0])

    assert before <= first <= after, (
        "Fortran timestamps are not on Python's perf_counter_ns clock: "
        f"region started at {first}, but the process ran between "
        f"{before} and {after}"
    )
    assert first < last <= after


RECURSION_PROGRAM = """
program recursion
   use scope_profiler
   implicit none
   integer :: value

   call sp_init("trace")
   value = fib(10)
   call sp_finalize()

contains

   recursive function fib(n) result(value)
      integer, intent(in) :: n
      integer :: value
      integer :: id

      id = sp_region("fib")
      call sp_begin(id)
      if (n < 2) then
         value = n
      else
         value = fib(n - 1) + fib(n - 2)
      end if
      call sp_end(id)
   end function fib

end program recursion
"""


def test_recursive_region_keeps_every_call_intact(tmp_path):
    """A region re-entered by recursion must not corrupt its own buffer."""
    executable = build(tmp_path, RECURSION_PROGRAM)
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    starts, ends = regions["fib"]

    # fib(10) enters the region once per invocation: 177 for n = 10.
    assert len(starts) == 177
    assert (ends >= starts).all(), "recursion mispaired a start with an end"
    # The outermost call encloses every other one.
    assert starts.min() == starts[0]
    assert ends.max() == ends[0]


GROWTH_PROGRAM = """
program growth
   use scope_profiler
   implicit none
   integer :: i, id

   call sp_init("trace")
   id = sp_region("hot")
   do i = 1, 5000
      call sp_begin(id)
      call sp_end(id)
   end do
   call sp_finalize()
end program growth
"""


def test_buffers_grow_past_the_initial_capacity(tmp_path):
    """More calls than the initial 1024 slots: the buffers must double."""
    executable = build(tmp_path, GROWTH_PROGRAM)
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    starts, ends = regions["hot"]

    assert len(starts) == 5000
    assert (ends >= starts).all()
    # Growth must preserve order: reallocation copies to the same indices.
    assert (np.diff(starts) >= 0).all()


MULTI_RANK_PROGRAM = """
program multi_rank
   use scope_profiler
   implicit none
   integer :: id, rank, i
   character(len=32) :: argument
   real(kind=8) :: acc

   call get_command_argument(1, argument)
   read (argument, *) rank

   call sp_init("trace", rank=rank)
   id = sp_region("step")
   do i = 1, 3 + rank
      call sp_begin(id)
      acc = sum([(sqrt(real(i, kind=8)), i=1, 1000)])
      call sp_end(id)
   end do
   call sp_finalize()
   if (acc < 0.0d0) print *, acc
end program multi_rank
"""


def test_ranks_merge_into_one_result_set(tmp_path):
    """Each rank writes its own trace; the importer merges them by rank."""
    executable = build(tmp_path, MULTI_RANK_PROGRAM, name="multi")
    for rank in range(4):
        subprocess.run(
            [str(executable), str(rank)], cwd=tmp_path, check=True, timeout=300
        )

    assert len(find_traces(tmp_path)) == 4

    results = load_traces(tmp_path, label="four ranks")

    assert results.num_ranks == 4
    step = results["step"]
    assert list(step.regions) == [0, 1, 2, 3], "ranks must be ordered by rank"
    # Rank r runs 3 + r iterations.
    assert [step.regions[r].num_calls for r in range(4)] == [3, 4, 5, 6]
    assert step.num_calls == 3 + 4 + 5 + 6
    assert results.label == "four ranks"


def test_converted_file_is_a_normal_profiling_file(tmp_path):
    """The HDF5 the importer writes is indistinguishable from a Python run's."""
    executable = build(tmp_path, MULTI_RANK_PROGRAM, name="multi")
    for rank in range(2):
        subprocess.run(
            [str(executable), str(rank)], cwd=tmp_path, check=True, timeout=300
        )

    output = convert_traces(tmp_path, tmp_path / "converted.h5", label="fortran")
    from_disk = read_h5(output)
    in_memory = load_traces(tmp_path, label="fortran")

    assert from_disk.summary() == in_memory.summary()
    assert from_disk.num_ranks == 2
    assert from_disk.metadata["source"] == "native"
    assert from_disk.label == "fortran"
    # The timeline origin survives, so relative timestamps mean something.
    assert from_disk.run_start_time is not None
    assert from_disk.events()[0]["start"] >= 0


def test_cli_import_fortran(tmp_path, capsys):
    """``scope-profiler import-native`` writes a file pproc can read."""
    from scope_profiler.__main__ import main

    executable = build(tmp_path, BASIC_PROGRAM)
    run(executable, tmp_path)

    output = tmp_path / "cli.h5"
    main(["import-native", str(tmp_path), "-o", str(output)])

    printed = capsys.readouterr().out
    assert "outer" in printed and "inner" in printed
    assert read_h5(output)["inner"].num_calls == 5


def test_unfinished_region_is_reported_and_dropped(tmp_path):
    """A region still open at sp_finalize must not reach the trace half-written."""
    program = """
program unfinished
   use scope_profiler
   implicit none
   integer :: closed, open_region

   call sp_init("trace")
   closed = sp_region("closed")
   open_region = sp_region("never_closed")

   call sp_begin(closed)
   call sp_end(closed)

   call sp_begin(open_region)
   call sp_finalize()
end program unfinished
"""
    executable = build(tmp_path, program, name="unfinished")
    result = subprocess.run(
        [str(executable)], cwd=tmp_path, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0
    assert "still open at sp_finalize" in result.stderr

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    assert "closed" in regions
    assert "never_closed" not in regions, "an unterminated region was written out"


def test_python_and_fortran_regions_share_one_timeline(tmp_path):
    """The point of matching clocks: one timeline across both languages.

    A Python region wraps the Fortran process, so every Fortran timestamp must
    fall strictly inside the Python region's start/end. Nothing else in the
    suite proves the two APIs are directly comparable.
    """
    from scope_profiler import ProfileManager

    executable = build(tmp_path, BASIC_PROGRAM)

    ProfileManager.setup(deactivate_file_output=True)
    with ProfileManager.profile_region("before"):
        time.sleep(0.005)
    with ProfileManager.profile_region("fortran_kernel"):
        run(executable, tmp_path)
    with ProfileManager.profile_region("after"):
        time.sleep(0.005)
    python_results = ProfileManager.finalize(verbose=False, return_results=True)

    fortran = load_traces(tmp_path)

    kernel = python_results["fortran_kernel"][0]
    kernel_start = int(kernel.start_times_ns[0])
    kernel_end = int(kernel.end_times_ns[0])

    for name in fortran.region_names:
        starts = fortran[name][0].start_times_ns
        ends = fortran[name][0].end_times_ns
        assert starts.min() >= kernel_start, (
            f"Fortran region {name!r} started before the Python region that "
            f"launched it: the two clocks do not share an epoch"
        )
        assert ends.max() <= kernel_end, (
            f"Fortran region {name!r} ended after the Python region that "
            f"launched it: the two clocks do not share an epoch"
        )

    # ...and the surrounding Python regions really do bracket it.
    before_end = int(python_results["before"][0].end_times_ns[0])
    after_start = int(python_results["after"][0].start_times_ns[0])
    assert before_end <= kernel_start < kernel_end <= after_start

    ProfileManager._reset()


def test_reader_rejects_files_that_are_not_traces(tmp_path):
    junk = tmp_path / "junk.spt"
    junk.write_bytes(b"not a trace at all, really")
    with pytest.raises(TraceFormatError, match="not a scope-profiler"):
        read_trace(junk)

    short = tmp_path / "short.spt"
    short.write_bytes(MAGIC)
    with pytest.raises(TraceFormatError, match="too short"):
        read_trace(short)


def test_reader_rejects_an_unknown_format_version(tmp_path):
    import struct

    path = tmp_path / "future.spt"
    path.write_bytes(MAGIC + struct.pack("<iiq", FORMAT_VERSION + 99, 0, 0))
    with pytest.raises(TraceFormatError, match="unsupported trace format version"):
        read_trace(path)


def test_reader_rejects_a_truncated_trace(tmp_path):
    executable = build(tmp_path, BASIC_PROGRAM)
    run(executable, tmp_path)

    path = tmp_path / "trace_rank00000.spt"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    with pytest.raises(TraceFormatError):
        read_trace(path)


def test_duplicate_ranks_are_refused(tmp_path):
    """Two files claiming one rank is a mistake worth naming, not merging."""
    executable = build(tmp_path, BASIC_PROGRAM)
    run(executable, tmp_path)

    original = tmp_path / "trace_rank00000.spt"
    copy = tmp_path / "copy_rank00000.spt"
    copy.write_bytes(original.read_bytes())

    with pytest.raises(TraceFormatError, match="both claim rank 0"):
        load_traces(tmp_path)


def test_find_traces_reports_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .spt trace files"):
        find_traces(tmp_path)
    with pytest.raises(FileNotFoundError, match="no such file"):
        find_traces(tmp_path / "nope.spt")


def test_the_shipped_example_builds_and_runs(tmp_path):
    """The example in fortran/ has to keep working, since it is the docs."""
    example = FORTRAN_DIR / "example.f90"
    if not example.exists():
        pytest.skip("example.f90 not installed")

    executable = tmp_path / "example"
    build_result = subprocess.run(
        [COMPILER, "-O1", "-o", str(executable), str(MODULE_SOURCE), str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert build_result.returncode == 0, build_result.stderr

    run(executable, tmp_path)
    results = load_traces(tmp_path)
    assert {"solve", "assemble", "checkpoint", "fibonacci", "fib_call"} <= set(
        results.region_names
    )
    assert results["solve"].num_calls == 20


def test_the_examples_directory_still_builds_and_runs(tmp_path):
    """examples/fortran is documentation; keep it from rotting silently.

    Builds the standalone example into a temporary directory (so the repo is
    left alone) and checks its trace holds the regions the README shows.
    """
    example_dir = Path(__file__).resolve().parents[3] / "examples" / "fortran"
    if not (example_dir / "Makefile").exists() or shutil.which("make") is None:
        pytest.skip("examples/fortran not present, or no make available")

    # The Makefile asks the interpreter where scope_profiler.f90 lives, and
    # `scope-profiler import-native` has to be the same code under test --
    # so point both at this source tree rather than whatever is installed.
    source_root = str(Path(__file__).resolve().parents[2])
    env = {
        **os.environ,
        "MAKEFLAGS": "",
        "PYTHONPATH": os.pathsep.join(
            [source_root, os.environ.get("PYTHONPATH", "")]
        ).strip(os.pathsep),
    }
    result = subprocess.run(
        [
            "make",
            "run-standalone",
            f"BUILD_DIR={tmp_path}",
            f"FC={COMPILER}",
            f"PYTHON={sys.executable}",
        ],
        cwd=example_dir,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    results = read_h5(tmp_path / "profiling_data.h5")
    # The regions kernels.f90 records inside itself, plus the driver's own.
    assert {
        "fortran:setup",
        "fortran:timestep",
        "fortran:stencil",
        "fortran:residual",
        "fortran:checkpoint",
    } == set(results.region_names)
    assert results["fortran:timestep"].num_calls == 20
    assert results["fortran:stencil"].num_calls == 100


def test_makefile_builds_the_example(tmp_path):
    """The shipped Makefile is the first thing a user runs; keep it working."""
    makefile = FORTRAN_DIR / "Makefile"
    if not makefile.exists() or shutil.which("make") is None:
        pytest.skip("no Makefile or make available")

    result = subprocess.run(
        ["make", "-f", str(makefile), f"BUILD_DIR={tmp_path}", f"FC={COMPILER}"],
        cwd=FORTRAN_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "MAKEFLAGS": ""},
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert (tmp_path / "example").exists()
