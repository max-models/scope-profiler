"""The C region API, compiled and run for real.

Same approach as the Fortran tests: build the shipped source with the system C
compiler, run a program against it, and check what came out. Compiling is the
only way to catch what actually breaks here -- a feature macro that hides the
right clock, a struct written with the wrong width, a trace the reader cannot
parse. The module skips when no C compiler is available.
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
    C_DIR,
    TraceFormatError,
    c_include_dir,
    c_source_path,
    find_traces,
    load_traces,
    read_trace,
)

SOURCE = c_source_path()

#: cc first: it is whatever the platform considers the system compiler.
COMPILERS = ("cc", "gcc", "clang", "icx", "icc")


def find_compiler() -> str | None:
    """Return a C compiler from PATH, or None."""
    for name in COMPILERS:
        path = shutil.which(name)
        if path:
            return path
    return None


COMPILER = find_compiler()

pytestmark = [
    pytest.mark.skipif(COMPILER is None, reason="no C compiler on PATH"),
    pytest.mark.skipif(not SOURCE.exists(), reason=f"{SOURCE} not found (installed?)"),
]


def build(tmp_path: Path, program: str, name: str = "prog", extra=()) -> Path:
    """Compile ``program`` against the API and return the executable."""
    source = tmp_path / f"{name}.c"
    source.write_text(program)
    executable = tmp_path / name

    result = subprocess.run(
        [
            COMPILER,
            "-std=c99",
            "-O1",
            f"-I{c_include_dir()}",
            str(source),
            str(SOURCE),
            "-lm",
            "-o",
            str(executable),
            *extra,
        ],
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


def run(executable: Path, tmp_path: Path, *args) -> subprocess.CompletedProcess:
    """Run a built program in ``tmp_path`` and return the completed process."""
    result = subprocess.run(
        [str(executable), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"run failed:\n{result.stderr}"
    return result


BASIC_PROGRAM = """
#include "scope_profiler.h"
#include <math.h>
#include <stdio.h>

static double work(int n)
{
    double acc = 0.0;
    int i;
    for (i = 1; i <= n; ++i) acc += sqrt((double)i);
    return acc;
}

int main(void)
{
    int i, outer, inner;
    double acc = 0.0;

    sp_init("trace", 0);
    outer = sp_region("outer");
    inner = sp_region("inner");

    sp_begin(outer);
    for (i = 0; i < 5; ++i) {
        sp_begin(inner);
        acc += work(2000);
        sp_end(inner);
    }
    sp_end(outer);

    printf("inner calls: %lld\\n", (long long)sp_num_calls(inner));
    printf("%.4f\\n", acc);
    sp_finalize();
    printf("after finalize: %lld\\n", (long long)sp_num_calls(inner));
    return 0;
}
"""


def test_compiles_clean_in_strict_c99(tmp_path):
    """No warnings at -Wall -Wextra -pedantic, in strict C99."""
    result = subprocess.run(
        [
            COMPILER,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-O2",
            f"-I{c_include_dir()}",
            "-c",
            str(SOURCE),
            "-o",
            str(tmp_path / "scope_profiler.o"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "warning" not in result.stderr.lower(), result.stderr


def test_the_header_is_usable_from_cxx(tmp_path):
    """C++ codes are the other half of the audience; extern "C" must hold."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")

    source = tmp_path / "main.cpp"
    source.write_text(
        '#include "scope_profiler.h"\n'
        "int main() {\n"
        '    sp_init("trace", 0);\n'
        '    int id = sp_region("from_cxx");\n'
        "    sp_begin(id);\n"
        "    sp_end(id);\n"
        "    return sp_finalize();\n"
        "}\n"
    )
    object_file = tmp_path / "sp.o"
    subprocess.run(
        [
            COMPILER,
            "-std=c99",
            "-O1",
            f"-I{c_include_dir()}",
            "-c",
            str(SOURCE),
            "-o",
            str(object_file),
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
    result = subprocess.run(
        [
            cxx,
            f"-I{c_include_dir()}",
            str(source),
            str(object_file),
            "-o",
            str(tmp_path / "cxx_prog"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    run(tmp_path / "cxx_prog", tmp_path)
    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    assert "from_cxx" in regions


def test_trace_records_regions_and_durations(tmp_path):
    executable = build(tmp_path, BASIC_PROGRAM)
    output = run(executable, tmp_path).stdout

    assert "inner calls: 5" in output
    # Call counts outlive the run, as in the Fortran and Python APIs.
    assert "after finalize: 5" in output

    rank, regions = read_trace(tmp_path / "trace_rank00000.spt")

    assert rank == 0
    assert sorted(regions) == ["inner", "outer"]

    starts, ends = regions["inner"]
    assert len(starts) == len(ends) == 5
    assert starts.dtype == np.int64
    assert ((ends - starts) > 0).all()

    outer_start, outer_end = (arr[0] for arr in regions["outer"])
    assert (starts >= outer_start).all() and (ends <= outer_end).all()


def test_timestamps_share_pythons_clock(tmp_path):
    """C and Python regions must land on one timeline.

    This is the test that catches a wrong clock. On macOS, defining
    _POSIX_C_SOURCE hides CLOCK_UPTIME_RAW and the code falls back to
    CLOCK_MONOTONIC -- which is microsecond-granular *and* starts from a
    different epoch, so these bounds would fail by hundreds of seconds.
    """
    executable = build(tmp_path, BASIC_PROGRAM)

    before = time.perf_counter_ns()
    run(executable, tmp_path)
    after = time.perf_counter_ns()

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    first = int(regions["outer"][0][0])
    last = int(regions["outer"][1][0])

    assert before <= first <= after, (
        "C timestamps are not on Python's perf_counter_ns clock: region "
        f"started at {first}, but the process ran between {before} and {after}"
    )
    assert first < last <= after


def test_the_clock_has_nanosecond_resolution(tmp_path):
    """Not just the right epoch: the right precision.

    A clock that ticks in microseconds would report most short regions as
    exactly 0 ns, and every duration as a multiple of 1000.
    """
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    int i, id;
    sp_init("trace", 0);
    id = sp_region("tick");
    for (i = 0; i < 200; ++i) {
        sp_begin(id);
        sp_end(id);
    }
    sp_finalize();
    return 0;
}
"""
    executable = build(tmp_path, program, name="ticks")
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    starts, _ = regions["tick"]
    gaps = np.diff(starts)
    gaps = gaps[gaps > 0]

    assert gaps.size, "the clock never advanced between calls"
    assert (gaps % 1000 != 0).any(), (
        "every observed gap is a whole number of microseconds; the clock is "
        "microsecond-granular, not nanosecond"
    )


RECURSION_PROGRAM = """
#include "scope_profiler.h"

static int fib(int n)
{
    int id = sp_region("fib");
    int value;
    sp_begin(id);
    value = n < 2 ? n : fib(n - 1) + fib(n - 2);
    sp_end(id);
    return value;
}

int main(void)
{
    sp_init("trace", 0);
    fib(10);
    sp_finalize();
    return 0;
}
"""


def test_recursive_region_keeps_every_call_intact(tmp_path):
    executable = build(tmp_path, RECURSION_PROGRAM, name="recursion")
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    starts, ends = regions["fib"]

    assert len(starts) == 177  # fib(10) enters the region 177 times
    assert (ends >= starts).all(), "recursion mispaired a start with an end"
    assert starts.min() == starts[0]
    assert ends.max() == ends[0]


def test_buffers_grow_past_the_initial_capacity(tmp_path):
    program = """
#include "scope_profiler.h"

int main(void)
{
    int i, id;
    sp_init("trace", 0);
    id = sp_region("hot");
    for (i = 0; i < 5000; ++i) {
        sp_begin(id);
        sp_end(id);
    }
    sp_finalize();
    return 0;
}
"""
    executable = build(tmp_path, program, name="growth")
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    starts, ends = regions["hot"]

    assert len(starts) == 5000
    assert (ends >= starts).all()
    assert (np.diff(starts) >= 0).all()


MULTI_RANK_PROGRAM = """
#include "scope_profiler.h"
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int rank = argc > 1 ? atoi(argv[1]) : 0;
    int i, j, id;
    double acc = 0.0;

    sp_init("trace", rank);
    id = sp_region("step");
    for (i = 0; i < 3 + rank; ++i) {
        sp_begin(id);
        for (j = 1; j <= 1000; ++j) acc += sqrt((double)j);
        sp_end(id);
    }
    sp_finalize();
    return acc < 0.0 ? 1 : 0;
}
"""


def test_ranks_merge_into_one_result_set(tmp_path):
    executable = build(tmp_path, MULTI_RANK_PROGRAM, name="multi")
    for rank in range(4):
        run(executable, tmp_path, str(rank))

    assert len(find_traces(tmp_path)) == 4

    results = load_traces(tmp_path, label="four ranks")

    assert results.num_ranks == 4
    step = results["step"]
    assert list(step.regions) == [0, 1, 2, 3]
    assert [step.regions[r].num_calls for r in range(4)] == [3, 4, 5, 6]


def test_converted_file_is_a_normal_profiling_file(tmp_path):
    from scope_profiler.native_trace import convert_traces

    executable = build(tmp_path, MULTI_RANK_PROGRAM, name="multi")
    for rank in range(2):
        run(executable, tmp_path, str(rank))

    output = convert_traces(tmp_path, tmp_path / "converted.h5", label="c run")
    from_disk = read_h5(output)

    assert from_disk.summary() == load_traces(tmp_path, label="c run").summary()
    assert from_disk.num_ranks == 2
    assert from_disk.metadata["source"] == "native"
    assert from_disk.run_start_time is not None


def test_c_and_fortran_traces_merge_into_one_run(tmp_path):
    """The formats are shared, so a mixed C/Fortran program is just one run."""
    from .test_fortran_api import COMPILER as FC
    from .test_fortran_api import build as build_fortran

    if FC is None:
        pytest.skip("no Fortran compiler on PATH")

    c_executable = build(tmp_path, BASIC_PROGRAM, name="c_side")
    run(c_executable, tmp_path)
    # Rename so the two do not collide on rank 0.
    (tmp_path / "trace_rank00000.spt").rename(tmp_path / "c_rank00000.spt")

    fortran_program = """
program f_side
   use scope_profiler
   implicit none
   integer :: id, i
   call sp_init("f", rank=1)
   id = sp_region("from_fortran")
   do i = 1, 4
      call sp_begin(id)
      call sp_end(id)
   end do
   call sp_finalize()
end program f_side
"""
    fortran_executable = build_fortran(tmp_path, fortran_program, name="f_side")
    subprocess.run([str(fortran_executable)], cwd=tmp_path, check=True, timeout=300)

    results = load_traces(tmp_path)

    assert results.num_ranks == 2
    assert {"inner", "outer", "from_fortran"} == set(results.region_names)
    assert results["from_fortran"].num_calls == 4


def test_unfinished_region_is_reported_and_dropped(tmp_path):
    program = """
#include "scope_profiler.h"

int main(void)
{
    int closed, open_region;
    sp_init("trace", 0);
    closed = sp_region("closed");
    open_region = sp_region("never_closed");

    sp_begin(closed);
    sp_end(closed);

    sp_begin(open_region);
    sp_finalize();
    return 0;
}
"""
    executable = build(tmp_path, program, name="unfinished")
    result = subprocess.run(
        [str(executable)], cwd=tmp_path, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0
    assert "still open at sp_finalize" in result.stderr

    _, regions = read_trace(tmp_path / "trace_rank00000.spt")
    assert "closed" in regions
    assert "never_closed" not in regions


def test_calls_before_init_are_harmless(tmp_path):
    """Instrumentation must be safe to leave in an unprofiled build."""
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    int id = sp_region("never_started");   /* SP_INVALID_REGION */
    sp_begin(id);
    sp_end(id);
    printf("id=%d active=%d calls=%lld\\n",
           id, sp_is_active(), (long long)sp_num_calls(id));
    return sp_finalize();
}
"""
    executable = build(tmp_path, program, name="uninitialized")
    output = run(executable, tmp_path).stdout

    assert "id=-1 active=0 calls=0" in output
    assert not list(tmp_path.glob("*.spt")), "no trace should be written"


def test_the_shipped_example_builds_and_runs(tmp_path):
    """The example in c/ is documentation; keep it working."""
    example = C_DIR / "example.c"
    if not example.exists():
        pytest.skip("example.c not installed")

    executable = tmp_path / "example"
    result = subprocess.run(
        [
            COMPILER,
            "-std=c99",
            "-O1",
            f"-I{c_include_dir()}",
            str(example),
            str(SOURCE),
            "-lm",
            "-o",
            str(executable),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    run(executable, tmp_path)
    results = load_traces(tmp_path)
    assert {"solve", "assemble", "checkpoint", "fibonacci", "fib_call"} <= set(
        results.region_names
    )
    assert results["solve"].num_calls == 20


def test_makefile_builds_the_example(tmp_path):
    if not (C_DIR / "Makefile").exists() or shutil.which("make") is None:
        pytest.skip("no Makefile or make available")

    result = subprocess.run(
        [
            "make",
            "-f",
            str(C_DIR / "Makefile"),
            f"BUILD_DIR={tmp_path}",
            f"CC={COMPILER}",
        ],
        cwd=C_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "MAKEFLAGS": ""},
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert (tmp_path / "example").exists()


def test_reader_rejects_a_truncated_c_trace(tmp_path):
    executable = build(tmp_path, BASIC_PROGRAM)
    run(executable, tmp_path)

    path = tmp_path / "trace_rank00000.spt"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    with pytest.raises(TraceFormatError):
        read_trace(path)


def test_the_examples_directory_still_builds_and_runs(tmp_path):
    """examples/c is documentation; keep it from rotting silently."""
    example_dir = Path(__file__).resolve().parents[3] / "examples" / "c"
    if not (example_dir / "Makefile").exists() or shutil.which("make") is None:
        pytest.skip("examples/c not present, or no make available")

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
            f"CC={COMPILER}",
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
    assert {
        "c:setup",
        "c:timestep",
        "c:stencil",
        "c:residual",
        "c:checkpoint",
    } == set(results.region_names)
    assert results["c:timestep"].num_calls == 20
    assert results["c:stencil"].num_calls == 100
