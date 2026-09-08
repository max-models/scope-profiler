"""The C region API, compiled and run for real.

Same approach as the Fortran tests: build the shipped source with the system C
compiler, run a program against it, and check what came out. Compiling is the
only way to catch what actually breaks here -- a feature macro that hides the
right clock, a struct written with the wrong width, a trace the reader cannot
parse. The module skips when no C compiler is available.
"""

import os
import re
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

#: c++ first: it is whatever the platform considers the system compiler.
CXX_COMPILERS = ("c++", "g++", "clang++")


def find_cxx_compiler() -> str | None:
    """Return a C++ compiler from PATH, or None."""
    for name in CXX_COMPILERS:
        path = shutil.which(name)
        if path:
            return path
    return None


CXX_COMPILER = find_cxx_compiler()

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
        check=False,
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
        check=False,
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
        check=False,
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
        "}\n",
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
        check=False,
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
        [str(executable)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
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
        check=False,
    )
    assert result.returncode == 0, result.stderr

    run(executable, tmp_path)
    results = load_traces(tmp_path)
    assert {"solve", "assemble", "checkpoint", "fibonacci", "fib_call"} <= set(
        results.region_names,
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
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert (tmp_path / "example").exists()


@pytest.mark.skipif(CXX_COMPILER is None, reason="no C++ compiler on PATH")
def test_makefile_hpp_check_target(tmp_path):
    if not (C_DIR / "Makefile").exists() or shutil.which("make") is None:
        pytest.skip("no Makefile or make available")

    result = subprocess.run(
        [
            "make",
            "-f",
            str(C_DIR / "Makefile"),
            "hpp-check",
            f"BUILD_DIR={tmp_path}",
            f"CXX={CXX_COMPILER}",
        ],
        cwd=C_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "MAKEFLAGS": ""},
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert (tmp_path / "hpp_check.o").exists()


def test_explicit_contexts_are_independent(tmp_path):
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    sp_profiler *a = sp_create("a", 0);
    sp_profiler *b = sp_create("b", 1);
    int ra = sp_profiler_region(a, "solve");
    int rb = sp_profiler_region(b, "solve");
    sp_region_stats stats;

    sp_profiler_begin(a, ra);
    sp_profiler_begin(b, rb);
    sp_profiler_end(b, rb);
    sp_profiler_begin(b, rb);
    sp_profiler_end(b, rb);
    sp_profiler_end(a, ra);

    sp_profiler_get_region_stats(a, ra, &stats);
    printf("a calls: %lld\\n", (long long)stats.calls);
    sp_profiler_get_region_stats(b, rb, &stats);
    printf("b calls: %lld\\n", (long long)stats.calls);

    sp_profiler_finalize(a);
    sp_profiler_finalize(b);
    sp_destroy(a);
    sp_destroy(b);
    return 0;
}
"""
    executable = build(tmp_path, program, name="contexts")
    output = run(executable, tmp_path).stdout

    assert "a calls: 1" in output
    assert "b calls: 2" in output

    _, a_regions = read_trace(tmp_path / "a_rank00000.spt")
    _, b_regions = read_trace(tmp_path / "b_rank00001.spt")
    assert len(a_regions["solve"][0]) == 1
    assert len(b_regions["solve"][0]) == 2


def test_region_at_records_source_location(tmp_path):
    program = """
#include "scope_profiler.h"

int main(void)
{
    sp_profiler *p = sp_create("src", 0);
    int solve = sp_profiler_region_at(p, "solve", "solver.c", 42);
    /* a second registration for the same name must not overwrite the source */
    int again = sp_profiler_region_at(p, "solve", "elsewhere.c", 99);

    sp_profiler_begin(p, solve);
    sp_profiler_end(p, again);
    sp_profiler_finalize(p);
    sp_destroy(p);
    return 0;
}
"""
    executable = build(tmp_path, program, name="region_at")
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "src_rank00000.spt")
    assert regions["solve"].source_file == "solver.c"
    assert regions["solve"].source_lineno == 42

    results = load_traces(tmp_path)
    assert results["solve"].source_file == "solver.c"
    assert results["solve"].source_lineno == 42


def test_region_at_backfills_source_onto_a_plain_handle(tmp_path):
    program = """
#include "scope_profiler.h"

int main(void)
{
    sp_profiler *p = sp_create("backfill", 0);
    int first = sp_profiler_region(p, "solve");            /* no source yet */
    int backfilled = sp_profiler_region_at(p, "solve", "solver.c", 7);
    int again = sp_profiler_region(p, "solve");             /* unaffected */

    sp_profiler_begin(p, first);
    sp_profiler_end(p, backfilled);
    sp_profiler_begin(p, again);
    sp_profiler_end(p, again);

    sp_profiler_finalize(p);
    sp_destroy(p);
    return first == backfilled && backfilled == again ? 0 : 1;
}
"""
    executable = build(tmp_path, program, name="backfill")
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "backfill_rank00000.spt")
    assert regions["solve"].source_file == "solver.c"
    assert regions["solve"].source_lineno == 7
    assert len(regions["solve"][0]) == 2


def test_region_at_macros_capture_the_call_site(tmp_path):
    program = """
#include "scope_profiler.h"

int main(void)
{
    sp_profiler *p = sp_create("macro", 0);
    int ctx = SP_PROFILER_REGION_AT(p, "ctx_region");
    sp_profiler_begin(p, ctx);
    sp_profiler_end(p, ctx);
    sp_profiler_finalize(p);
    sp_destroy(p);

    sp_init("macro_default", 0);
    {
        int def = SP_REGION_AT("default_region");
        sp_begin(def);
        sp_end(def);
    }
    sp_finalize();
    return 0;
}
"""
    executable = build(tmp_path, program, name="region_at_macro")
    run(executable, tmp_path)

    _, ctx_regions = read_trace(tmp_path / "macro_rank00000.spt")
    assert ctx_regions["ctx_region"].source_file.endswith("region_at_macro.c")
    assert ctx_regions["ctx_region"].source_lineno > 0

    _, default_regions = read_trace(tmp_path / "macro_default_rank00000.spt")
    assert default_regions["default_region"].source_file.endswith("region_at_macro.c")
    assert default_regions["default_region"].source_lineno > 0


def build_cxx(tmp_path: Path, program: str, name: str = "prog") -> Path:
    """Compile ``program`` (C++) against scope_profiler.hpp/.c."""
    source = tmp_path / f"{name}.cpp"
    source.write_text(program)
    object_file = tmp_path / "scope_profiler.o"
    executable = tmp_path / name

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
            CXX_COMPILER,
            "-std=c++11",
            "-Wall",
            "-Wextra",
            "-O1",
            f"-I{c_include_dir()}",
            str(source),
            str(object_file),
            "-o",
            str(executable),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"compilation failed\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    return executable


@pytest.mark.skipif(CXX_COMPILER is None, reason="no C++ compiler on PATH")
def test_cxx_scope_raii_wrapper(tmp_path):
    program = """
#include "scope_profiler.hpp"
#include <cstdio>
#include <stdexcept>
#include <utility>

static void throws_inside_scope(sp_profiler *p, int region)
{
    sp::Scope s(p, region);
    throw std::runtime_error("boom");
}

int main()
{
    sp_profiler *p = sp_create("cxx", 0);
    int solve = sp_profiler_region(p, "solve");

    { sp::Scope s(p, solve); }                 // normal exit

    try {                                       // exception unwind
        throws_inside_scope(p, solve);
    } catch (const std::runtime_error &) {
    }

    {                                            // move construction
        sp::Scope a(p, solve);
        sp::Scope b(std::move(a));
    }

    sp_region_stats stats;
    sp_profiler_get_region_stats(p, solve, &stats);
    std::printf("calls: %lld\\n", (long long)stats.calls);

    sp_profiler_finalize(p);
    sp_destroy(p);
    return 0;
}
"""
    executable = build_cxx(tmp_path, program, name="cxx_scope")
    output = run(executable, tmp_path).stdout

    assert "calls: 3" in output


@pytest.mark.skipif(CXX_COMPILER is None, reason="no C++ compiler on PATH")
def test_profile_macros_record_scope_function_and_source(tmp_path):
    program = """
#include "scope_profiler.hpp"

static void solve()
{
    SP_PROFILE_FUNCTION();
    { SP_PROFILE_SCOPE("inner"); }
}

int main()
{
    sp_init("macros", 0);
    solve();
    return sp_finalize();
}
"""
    executable = build_cxx(tmp_path, program, name="profile_macros")
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "macros_rank00000.spt")
    assert "inner" in regions
    function_name = next(name for name in regions if "solve" in name)
    assert regions[function_name].source_file.endswith("profile_macros.cpp")
    assert regions["inner"].source_lineno > 0


@pytest.mark.skipif(CXX_COMPILER is None, reason="no C++ compiler on PATH")
def test_disabled_profile_macros_emit_no_code_or_references(tmp_path):
    """Arguments disappear too: disabled instrumentation has no side effects."""
    source = tmp_path / "disabled.cpp"
    source.write_text(
        """
#define SP_DISABLE_PROFILING
#include "scope_profiler.hpp"

const char *expensive_name(); // deliberately has no definition
int main()
{
    SP_PROFILE_SCOPE(expensive_name());
    SP_PROFILE_SCOPE("disabled-region-that-must-not-survive");
    SP_PROFILE_FUNCTION();
    return 0;
}
""",
    )
    executable = tmp_path / "disabled"
    result = subprocess.run(
        [
            CXX_COMPILER,
            "-std=c++11",
            "-O2",
            f"-I{c_include_dir()}",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    run(executable, tmp_path)
    assert b"disabled-region-that-must-not-survive" not in executable.read_bytes()


@pytest.mark.skipif(CXX_COMPILER is None, reason="no C++ compiler on PATH")
def test_header_only_backend_is_shared_across_translation_units(tmp_path):
    """C++17 inline state links once and records scopes entered in another TU."""
    (tmp_path / "worker.cpp").write_text(
        """
#define SP_HEADER_ONLY
#include "scope_profiler.hpp"
void worker() { SP_PROFILE_SCOPE("worker"); }
""",
    )
    (tmp_path / "main.cpp").write_text(
        """
#define SP_HEADER_ONLY
#include "scope_profiler.hpp"
void worker();
int main()
{
    sp_init("header_only", 0);
    worker();
    return sp_finalize();
}
""",
    )
    executable = tmp_path / "header_only"
    result = subprocess.run(
        [
            CXX_COMPILER,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-O1",
            f"-I{c_include_dir()}",
            str(tmp_path / "main.cpp"),
            str(tmp_path / "worker.cpp"),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    run(executable, tmp_path)
    _, regions = read_trace(tmp_path / "header_only_rank00000.spt")
    assert len(regions["worker"][0]) == 1


@pytest.mark.skipif(CXX_COMPILER is None, reason="no C++ compiler on PATH")
def test_native_likwid_scopes_drive_marker_api(tmp_path):
    """A stand-in header exercises LIKWID integration without special hardware."""
    (tmp_path / "likwid-marker.h").write_text(
        """
#ifndef FAKE_LIKWID_MARKER_H
#define FAKE_LIKWID_MARKER_H
void fake_likwid_init(void);
void fake_likwid_thread_init(void);
void fake_likwid_start(const char *);
void fake_likwid_stop(const char *);
void fake_likwid_close(void);
#define LIKWID_MARKER_INIT fake_likwid_init()
#define LIKWID_MARKER_THREADINIT fake_likwid_thread_init()
#define LIKWID_MARKER_START(tag) fake_likwid_start(tag)
#define LIKWID_MARKER_STOP(tag) fake_likwid_stop(tag)
#define LIKWID_MARKER_CLOSE fake_likwid_close()
#endif
""",
    )
    source = tmp_path / "likwid_scope.cpp"
    source.write_text(
        """
#define SP_USE_LIKWID
#include "scope_profiler.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>

static int initialized, thread_initialized, started, stopped, closed;
void fake_likwid_init(void) { ++initialized; }
void fake_likwid_thread_init(void) { ++thread_initialized; }
void fake_likwid_start(const char *tag) { if (!std::strcmp(tag, "solve")) ++started; }
void fake_likwid_stop(const char *tag) { if (!std::strcmp(tag, "solve")) ++stopped; }
void fake_likwid_close(void) { ++closed; }

int main()
{
    {
        sp::LikwidSession counters;
        sp_init("likwid", 0);
        try {
            SP_PROFILE_SCOPE("solve");
            throw std::runtime_error("leave by exception");
        } catch (const std::runtime_error &) {}
        sp_finalize();
    }
    std::printf("%d %d %d %d %d\\n",
                initialized, thread_initialized, started, stopped, closed);
    return 0;
}
""",
    )
    object_file = tmp_path / "scope_profiler.o"
    subprocess.run(
        [
            COMPILER,
            "-std=c99",
            f"-I{c_include_dir()}",
            "-c",
            str(SOURCE),
            "-o",
            str(object_file),
        ],
        check=True,
    )
    executable = tmp_path / "likwid_scope"
    result = subprocess.run(
        [
            CXX_COMPILER,
            "-std=c++11",
            "-Wall",
            "-Wextra",
            f"-I{tmp_path}",
            f"-I{c_include_dir()}",
            str(source),
            str(object_file),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert run(executable, tmp_path).stdout.strip() == "1 1 1 1 1"
    _, regions = read_trace(tmp_path / "likwid_rank00000.spt")
    assert len(regions["solve"][0]) == 1


@pytest.mark.skipif(CXX_COMPILER is None, reason="no C++ compiler on PATH")
def test_native_mpi_wrappers_record_operations_and_metadata(tmp_path):
    """A stand-in MPI header exercises wrappers without an MPI installation."""
    (tmp_path / "mpi.h").write_text(
        """
#ifndef FAKE_MPI_H
#define FAKE_MPI_H
typedef int MPI_Comm;
typedef int MPI_Fint;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Request;
typedef struct { int source; } MPI_Status;
#define MPI_SUCCESS 0
#define MPI_COMM_WORLD 7
#define MPI_REQUEST_NULL 0
#define MPI_STATUS_IGNORE ((MPI_Status *)0)
int MPI_Type_size(MPI_Datatype, int *);
MPI_Fint MPI_Comm_c2f(MPI_Comm);
int MPI_Send(const void *, int, MPI_Datatype, int, int, MPI_Comm);
int MPI_Recv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
int MPI_Isend(const void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int MPI_Irecv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int MPI_Wait(MPI_Request *, MPI_Status *);
int MPI_Barrier(MPI_Comm);
int MPI_Bcast(void *, int, MPI_Datatype, int, MPI_Comm);
int MPI_Reduce(const void *, void *, int, MPI_Datatype, MPI_Op, int, MPI_Comm);
int MPI_Allreduce(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
#endif
""",
    )
    source = tmp_path / "mpi_wrappers.cpp"
    source.write_text(
        """
#include "scope_profiler_mpi.hpp"

static int calls;
int MPI_Type_size(MPI_Datatype datatype, int *size) { *size = datatype; return 0; }
MPI_Fint MPI_Comm_c2f(MPI_Comm communicator) { return communicator; }
int MPI_Send(const void *, int, MPI_Datatype, int, int, MPI_Comm) { ++calls; return 0; }
int MPI_Recv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *) { ++calls; return 0; }
int MPI_Isend(const void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *r)
{ ++calls; *r = 1; return 0; }
int MPI_Irecv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *r)
{ ++calls; *r = 2; return 0; }
int MPI_Wait(MPI_Request *r, MPI_Status *) { ++calls; *r = MPI_REQUEST_NULL; return 0; }
int MPI_Barrier(MPI_Comm) { ++calls; return 0; }
int MPI_Bcast(void *, int, MPI_Datatype, int, MPI_Comm) { ++calls; return 0; }
int MPI_Reduce(const void *, void *, int, MPI_Datatype, MPI_Op, int, MPI_Comm)
{ ++calls; return 0; }
int MPI_Allreduce(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm)
{ ++calls; return 0; }

int main()
{
    int value = 1, result = 0;
    sp_init("mpi", 0);
    sp::mpi::send(&value, 4, 8, 2, 9, MPI_COMM_WORLD);
    sp::mpi::recv(&value, 4, 8, 3, 10, MPI_COMM_WORLD);
    sp::mpi::Request sent = sp::mpi::isend(&value, 4, 8, 2, 11, MPI_COMM_WORLD);
    if (sent.error() != MPI_SUCCESS || !sent.active()) return 2;
    sp::mpi::wait(sent);
    sp::mpi::Request received = sp::mpi::irecv(&value, 4, 8, 3, 12, MPI_COMM_WORLD);
    sp::mpi::wait(received);
    sp::mpi::barrier(MPI_COMM_WORLD);
    sp::mpi::bcast(&value, 4, 8, 1, MPI_COMM_WORLD);
    sp::mpi::reduce(&value, &result, 4, 8, 1, 1, MPI_COMM_WORLD);
    sp::mpi::allreduce(&value, &result, 4, 8, 1, MPI_COMM_WORLD);
    sp_finalize();
    return calls == 10 ? 0 : 3;
}
""",
    )
    object_file = tmp_path / "scope_profiler.o"
    subprocess.run(
        [
            COMPILER,
            "-std=c99",
            f"-I{c_include_dir()}",
            "-c",
            str(SOURCE),
            "-o",
            str(object_file),
        ],
        check=True,
    )
    executable = tmp_path / "mpi_wrappers"
    result = subprocess.run(
        [
            CXX_COMPILER,
            "-std=c++11",
            "-Wall",
            "-Wextra",
            "-pedantic",
            f"-I{tmp_path}",
            f"-I{c_include_dir()}",
            str(source),
            str(object_file),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    run(executable, tmp_path)

    _, regions = read_trace(tmp_path / "mpi_rank00000.spt")
    assert any(
        name == ("mpi:send kind=point-to-point bytes=32 peer=2 root=-1 " "tag=9 comm=7")
        for name in regions
    )
    assert any("mpi:barrier kind=collective bytes=0" in name for name in regions)
    assert any(
        "mpi:wait kind=wait" in name and "request=isend" in name for name in regions
    )

    disabled_source = tmp_path / "mpi_disabled.cpp"
    disabled_source.write_text(
        """
#define SP_DISABLE_PROFILING
#include "scope_profiler_mpi.hpp"
static int type_size_calls;
int MPI_Type_size(MPI_Datatype datatype, int *size)
{ ++type_size_calls; *size = datatype; return 0; }
MPI_Fint MPI_Comm_c2f(MPI_Comm communicator) { return communicator; }
int MPI_Send(const void *, int, MPI_Datatype, int, int, MPI_Comm) { return 23; }
int main()
{
    int value = 1;
    int status = sp::mpi::send(&value, 4, 8, 2, 9, MPI_COMM_WORLD);
    return status == 23 && type_size_calls == 0 ? 0 : 1;
}
""",
    )
    disabled = tmp_path / "mpi_disabled"
    result = subprocess.run(
        [
            CXX_COMPILER,
            "-std=c++11",
            "-O2",
            f"-I{tmp_path}",
            f"-I{c_include_dir()}",
            str(disabled_source),
            "-o",
            str(disabled),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    run(disabled, tmp_path)
    assert b"mpi:send" not in disabled.read_bytes()


def test_cmake_project_version_matches_python_package():
    """The independently installed CMake package carries the release version."""
    repository = Path(__file__).resolve().parents[3]
    pyproject = (repository / "pyproject.toml").read_text()
    cmake = (repository / "CMakeLists.txt").read_text()
    python_version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    cmake_version = re.search(
        r"project\(scope-profiler VERSION ([^ )]+)", cmake
    )
    assert python_version is not None
    assert cmake_version is not None
    assert cmake_version.group(1) == python_version.group(1)


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake is not installed")
def test_cmake_install_exports_working_cpp_targets(tmp_path):
    """Installed and FetchContent configs expose both C++ recorder variants."""
    repository = Path(__file__).resolve().parents[3]
    build_dir = tmp_path / "scope-build"
    prefix = tmp_path / "prefix"
    consumer_build = tmp_path / "consumer-build"

    subprocess.run(
        [
            "cmake",
            "-S",
            str(repository),
            "-B",
            str(build_dir),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "install"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "cmake",
            "-S",
            str(repository / "examples" / "cmake"),
            "-B",
            str(consumer_build),
            f"-DCMAKE_PREFIX_PATH={prefix}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(consumer_build)],
        check=True,
        capture_output=True,
        text=True,
    )
    for executable_name in (
        "profiled-app",
        "profiled-app-compiled",
        "profiled-native",
    ):
        executable = consumer_build / executable_name
        run(executable, consumer_build)
        _, regions = read_trace(consumer_build / "cmake-profile_rank00000.spt")
        assert any("solve" in name for name in regions)

    fetch_build = tmp_path / "fetch-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(repository / "examples" / "cmake"),
            "-B",
            str(fetch_build),
            "-DSCOPE_PROFILER_USE_FETCHCONTENT=ON",
            f"-DSCOPE_PROFILER_SOURCE_DIR={repository}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(fetch_build)],
        check=True,
        capture_output=True,
        text=True,
    )
    for executable_name in (
        "profiled-app",
        "profiled-app-compiled",
        "profiled-native",
    ):
        run(fetch_build / executable_name, fetch_build)
        _, fetched_regions = read_trace(
            fetch_build / "cmake-profile_rank00000.spt"
        )
        assert any("solve" in name for name in fetched_regions)


def test_scope_token_checked_ordering(tmp_path):
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    sp_profiler *p = sp_create("scope", 0);
    int solve = sp_profiler_region(p, "solve");
    sp_scope outer, inner;

    outer = sp_profiler_scope_begin(p, solve);
    inner = sp_profiler_scope_begin(p, solve); /* recursive re-entry */

    printf("end outer while inner open: %d\\n", sp_scope_end(&outer));
    printf("end inner: %d\\n", sp_scope_end(&inner));
    printf("end outer now on top: %d\\n", sp_scope_end(&outer));
    printf("end outer again: %d\\n", sp_scope_end(&outer));

    sp_profiler_finalize(p);
    sp_destroy(p);
    return 0;
}
"""
    executable = build(tmp_path, program, name="scope_token")
    output = run(executable, tmp_path).stdout

    assert "end outer while inner open: 5" in output  # SP_ERR_UNMATCHED_END
    assert "end inner: 0" in output  # SP_OK
    assert "end outer now on top: 0" in output  # SP_OK -- retry after inner closed
    assert "end outer again: 5" in output  # inert now


def test_end_last_ends_the_most_recently_opened_call(tmp_path):
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    sp_profiler *p = sp_create("last", 0);
    int outer = sp_profiler_region(p, "outer");
    int inner = sp_profiler_region(p, "inner");
    sp_region_stats stats;

    sp_profiler_begin(p, outer);
    sp_profiler_begin(p, inner);
    printf("end_last (should close inner): %d\\n", sp_profiler_end_last(p));

    sp_profiler_get_region_stats(p, inner, &stats);
    printf("inner calls: %lld\\n", (long long)stats.calls);
    sp_profiler_get_region_stats(p, outer, &stats);
    printf("outer calls (still open): %lld\\n", (long long)stats.calls);

    printf("end_last (should close outer): %d\\n", sp_profiler_end_last(p));
    printf("end_last with nothing open: %d\\n", sp_profiler_end_last(p));

    sp_profiler_finalize(p);
    sp_destroy(p);
    return 0;
}
"""
    executable = build(tmp_path, program, name="end_last")
    output = run(executable, tmp_path).stdout

    assert "end_last (should close inner): 0" in output
    assert "inner calls: 1" in output
    assert "outer calls (still open): 1" in output
    assert "end_last (should close outer): 0" in output
    assert "end_last with nothing open: 5" in output  # SP_ERR_UNMATCHED_END


def test_reset_discards_calls_but_keeps_regions(tmp_path):
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    sp_profiler *p = sp_create("reset", 0);
    int solve = sp_profiler_region(p, "solve");
    sp_region_stats stats;

    sp_profiler_begin(p, solve);
    printf("reset while open: %d\\n", sp_profiler_reset(p));
    sp_profiler_end(p, solve);

    printf("reset once closed: %d\\n", sp_profiler_reset(p));
    printf("regions after reset: %d\\n", sp_profiler_num_regions(p));

    sp_profiler_get_region_stats(p, solve, &stats);
    printf("calls after reset: %lld\\n", (long long)stats.calls);

    /* the handle is still valid after reset */
    sp_profiler_begin(p, solve);
    sp_profiler_end(p, solve);
    sp_profiler_get_region_stats(p, solve, &stats);
    printf("calls after reuse: %lld\\n", (long long)stats.calls);

    sp_profiler_finalize(p);
    sp_destroy(p);
    return 0;
}
"""
    executable = build(tmp_path, program, name="reset")
    output = run(executable, tmp_path).stdout

    assert "reset while open: 6" in output  # SP_ERR_OPEN_SCOPES
    assert "reset once closed: 0" in output  # SP_OK
    assert "regions after reset: 1" in output
    assert "calls after reset: 0" in output
    assert "calls after reuse: 1" in output


def test_stats_track_total_min_and_max(tmp_path):
    program = """
#define _POSIX_C_SOURCE 200809L /* struct timespec/nanosleep under -std=c99 on glibc */
#include "scope_profiler.h"
#include <stdio.h>
#include <time.h>

static void sleep_ns(long ns)
{
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = ns;
    nanosleep(&ts, NULL);
}

int main(void)
{
    sp_profiler *p = sp_create("stats", 0);
    int solve = sp_profiler_region(p, "solve");
    sp_region_stats stats;

    sp_profiler_begin(p, solve);
    sleep_ns(1000000); /* ~1ms: the short call */
    sp_profiler_end(p, solve);

    sp_profiler_begin(p, solve);
    sleep_ns(5000000); /* ~5ms: the long call */
    sp_profiler_end(p, solve);

    sp_profiler_get_region_stats(p, solve, &stats);
    printf("calls=%lld total>0:%d min<=max:%d min>0:%d\\n",
           (long long)stats.calls,
           stats.total_ns > 0,
           stats.min_ns <= stats.max_ns,
           stats.min_ns > 0);

    sp_profiler_finalize(p);
    sp_destroy(p);
    return 0;
}
"""
    executable = build(tmp_path, program, name="stats")
    output = run(executable, tmp_path).stdout

    assert "calls=2 total>0:1 min<=max:1 min>0:1" in output


def test_flush_writes_without_stopping_profiling(tmp_path):
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    sp_profiler *p = sp_create("flush", 0);
    int solve = sp_profiler_region(p, "solve");

    sp_profiler_begin(p, solve);
    sp_profiler_end(p, solve);
    printf("flush 1: %d\\n", sp_profiler_flush(p));
    printf("active after flush: %d\\n", sp_profiler_is_active(p));

    sp_profiler_begin(p, solve); /* left open on purpose */
    printf("flush 2 (one call open): %d\\n", sp_profiler_flush(p));
    sp_profiler_end(p, solve);

    printf("finalize: %d\\n", sp_profiler_finalize(p));
    sp_destroy(p);
    return 0;
}
"""
    executable = build(tmp_path, program, name="flush")
    output = run(executable, tmp_path).stdout

    assert "flush 1: 0" in output
    assert "active after flush: 1" in output
    assert "flush 2 (one call open): 0" in output
    assert "finalize: 0" in output

    _, regions = read_trace(tmp_path / "flush_rank00000.spt")
    assert len(regions["solve"][0]) == 2


def test_error_introspection_and_output_path(tmp_path):
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    sp_profiler *p = sp_create("errors", 3);
    int solve = sp_profiler_region(p, "solve");

    printf("path: %s\\n", sp_profiler_output_path(p));

    sp_profiler_end(p, solve); /* no matching begin */
    printf("last error: %s\\n", sp_error_string(sp_profiler_last_error(p)));

    printf("null profiler active: %d\\n", sp_profiler_is_active(NULL));
    printf("null profiler region: %d\\n", sp_profiler_region(NULL, "x"));
    printf("null profiler error: %s\\n", sp_error_string(sp_profiler_last_error(NULL)));

    sp_profiler_finalize(p);
    sp_destroy(p);
    return 0;
}
"""
    executable = build(tmp_path, program, name="errors")
    output = run(executable, tmp_path).stdout

    assert "path: errors_rank00003.spt" in output
    assert "last error: no open call to end" in output
    assert "null profiler active: 0" in output
    assert "null profiler region: -1" in output
    assert "null profiler error: ok" in output


def test_output_format_falls_back_to_a_trace_without_hdf5(tmp_path):
    """A build without the HDF5 backend refuses the format and keeps writing .spt.

    Instrumentation that always asks for HDF5 has to stay portable to a build
    that cannot provide it, so the request is reported and declined rather
    than either failing the run or silently producing nothing.
    """
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    sp_profiler *p = sp_create("fallback", 0);
    int solve = sp_profiler_region(p, "solve");

    printf("available: %d\\n", sp_hdf5_available());
    printf("default: %d\\n", (int)sp_default_output_format());
    printf("asked: %s\\n",
           sp_error_string(
               (sp_status)sp_profiler_set_output_format(p, SP_OUTPUT_HDF5)));
    printf("format: %d\\n", (int)sp_profiler_output_format(p));
    printf("path: %s\\n", sp_profiler_output_path(p));

    sp_profiler_begin(p, solve);
    sp_profiler_end(p, solve);
    sp_profiler_finalize(p);
    sp_destroy(p);
    return 0;
}
"""
    executable = build(tmp_path, program, name="fallback")
    output = run(executable, tmp_path).stdout

    assert "available: 0" in output
    assert "default: 1" in output  # SP_OUTPUT_TRACE
    assert "asked: output format not available in this build" in output
    assert "format: 1" in output
    assert "path: fallback_rank00000.spt" in output

    _, regions = read_trace(tmp_path / "fallback_rank00000.spt")
    assert len(regions["solve"][0]) == 1


def test_selecting_the_trace_format_explicitly_renames_the_output(tmp_path):
    """Asking for the always-available format works in any build."""
    program = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    int solve;

    sp_init("chosen", 1);
    printf("selected: %d\\n", sp_set_output_format(SP_OUTPUT_TRACE));
    printf("path: %s\\n", sp_output_path());
    solve = sp_region("solve");
    sp_begin(solve);
    sp_end(solve);
    sp_finalize();
    return 0;
}
"""
    executable = build(tmp_path, program, name="chosen")
    output = run(executable, tmp_path).stdout

    assert "selected: 0" in output  # SP_OK
    assert "path: chosen_rank00001.spt" in output
    rank, regions = read_trace(tmp_path / "chosen_rank00001.spt")
    assert rank == 1
    assert len(regions["solve"][0]) == 1


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
            [source_root, os.environ.get("PYTHONPATH", "")],
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
        check=False,
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
