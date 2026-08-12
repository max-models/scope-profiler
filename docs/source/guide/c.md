# Profiling C and C++ code

scope-profiler ships a C region API with the same model as the Python one. A C
program marks regions, writes a small trace file per rank, and
`scope-profiler import-native` turns those traces into the usual HDF5 output —
so a C run gets the same summaries, charts, exporters and `pproc` workflow as a
Python one.

It is one C99 file plus a header: no dependencies beyond libc, no HDF5, no MPI,
nothing to link. The header is `extern "C"`, so C++ codes can include it
directly.

The trace format is shared with {doc}`the Fortran API <fortran>`, so a program
built from both lands in a single profile.

## Getting the source

It ships inside the installed package:

```bash
cc -c $(python -c "import scope_profiler.native_trace as t; print(t.c_source_path())") \
   -I$(python -c "import scope_profiler.native_trace as t; print(t.c_include_dir())")
```

or from a checkout, `src/scope_profiler/c/`. A `Makefile` and a runnable
`example.c` sit next to it.

## Marking regions

```c
#include "scope_profiler.h"

int main(void)
{
    int step, solve, assemble;

    sp_init("profile", 0);              /* prefix, MPI rank */

    solve = sp_region("solve");         /* resolve names once... */
    assemble = sp_region("assemble");

    for (step = 0; step < nsteps; ++step) {
        sp_begin(assemble);             /* ...then use the handles */
        assemble_matrix();
        sp_end(assemble);

        sp_begin(solve);
        solve_system();
        sp_end(solve);
    }

    sp_finalize();                      /* writes profile_rank00000.spt */
    return 0;
}
```

Regions may nest, and a region may re-enter itself recursively — each entry
reserves its own slot, exactly as in the Python API.

## The API

| Call | Purpose |
| --- | --- |
| `int sp_init(const char *prefix, int rank)` | Start profiling. Returns non-zero if no monotonic clock is available. |
| `int sp_region(const char *name)` | Handle for a region name, created on first use. |
| `void sp_begin(int region)` / `void sp_end(int region)` | Enter and leave a region. |
| `int64_t sp_num_calls(int region)` | How often a region was entered; readable after `sp_finalize()`. |
| `int64_t sp_now_ns(void)` | The raw clock, in nanoseconds. |
| `int sp_is_active(void)` | Whether profiling is running. |
| `int sp_finalize(void)` | Write `<prefix>_rank<NNNNN>.spt` and stop. Returns non-zero on write failure. |

Calls made before `sp_init()` are harmless: `sp_region()` returns
`SP_INVALID_REGION` and `sp_begin`/`sp_end` ignore it. Instrumentation can stay
in a build that never profiles, with no `#ifdef` at the call sites.

## Under MPI

Pass each rank's id so the traces do not collide:

```c
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
sp_init("profile", rank);
```

Every rank writes its own file; nothing is communicated, so `sp_finalize()` is
not collective and a rank that dies takes only its own trace with it:

```bash
mpirun -n 128 ./simulation
scope-profiler import-native . -o profiling_data.h5
scope-profiler pproc profiling_data.h5 -o figures
```

## Python calling C

The case this is really for. Compile the kernels and the profiler into one
shared library and load it with `ctypes` — no binding generator, no build
backend:

```bash
cc -std=c99 -O2 -fPIC -shared kernels.c $(python -c \
    "import scope_profiler.native_trace as t; print(t.c_source_path())") \
   -I$(python -c "import scope_profiler.native_trace as t; print(t.c_include_dir())") \
   -o libkernels.so
```

```python
import ctypes
from scope_profiler import ProfileManager

kernels = ctypes.CDLL("./libkernels.so")
kernels.kernels_jacobi_solve.argtypes = [ctypes.c_int, ctypes.c_int]
kernels.kernels_jacobi_solve.restype = ctypes.c_double   # see the trap below

ProfileManager.setup(file_path="profiling_data.h5")
kernels.kernels_start_profiling(b"trace", 0)

for step in range(nsteps):
    with ProfileManager.profile_region("python:step"):
        kernels.kernels_jacobi_solve(n, sweeps)          # records its own regions

kernels.kernels_stop_profiling()                          # writes the trace
ProfileManager.finalize(native_traces=".")                # ...and folds it in
```

Because both APIs read the same clock, the C regions nest inside the Python
region that called them and `call_stack()` sees a single tree:

```
  python:setup                   11.970 ms   [python]
  python:timestep                 0.331 ms   [python]
    python:call_solver              0.320 ms   [python]
      c:stencil                       0.023 ms   [c]
      c:residual                      0.040 ms   [c]
```

Two rules:

- **Call the C `sp_finalize()` first.** Its trace has to exist by the time
  `finalize()` reads it.
- **Give the two sides distinct region names.** A name recorded by both raises,
  rather than silently double-counting a wrapper and the region inside it. A
  `python:` / `c:` prefix is the simplest convention.

### One ctypes trap

`ctypes` assumes a function returns `int` unless told otherwise, so a `double`
return comes back as garbage without `restype`. The regions are still recorded
correctly, which makes it easy to miss. Declare `argtypes` and `restype` for
everything you call.

## One timeline with Python

The API reads the same OS clock CPython's `time.perf_counter_ns()` uses —
`CLOCK_MONOTONIC` on Linux, `CLOCK_UPTIME_RAW` on macOS — resolved by probing
at run time.

The macOS case is worth knowing about if you ever vendor this file: defining
`_POSIX_C_SOURCE` there *hides* `CLOCK_UPTIME_RAW`, and the fallback
`CLOCK_MONOTONIC` is both microsecond-granular and on a different epoch. The
implementation therefore defines `_DARWIN_C_SOURCE` on Apple platforms and
`_POSIX_C_SOURCE` elsewhere — for opposite reasons on each.

## The trace format

Identical to the Fortran API's; see {doc}`fortran` for the layout. 16 bytes per
recorded call, the same as the Python side.

## Limitations

- **Not thread safe.** A region must be entered and left by the same thread.
  OpenMP threads inside a region are fine — wrap the whole parallel construct
  in one `sp_begin`/`sp_end` from the master thread.
- **No LIKWID counters.** Hardware counters are collected only by the Python
  API; a native trace carries timings alone.
- **Recursion depth** per region is capped at `SP_MAX_DEPTH` (64); deeper
  nesting is reported on stderr and left untimed rather than corrupting the
  buffer.
- **A region still open at `sp_finalize()`** is reported on stderr and dropped,
  rather than written with a missing end time.
