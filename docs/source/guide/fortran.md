# Profiling Fortran code

scope-profiler ships a Fortran module with the same region model as the Python
API. A Fortran program marks regions, writes a small trace file per rank, and
`scope-profiler import-fortran` turns those traces into the usual HDF5 output —
so a Fortran run gets the same summaries, charts, exporters and `pproc`
workflow as a Python one.

The module is deliberately undemanding: plain Fortran 2008 with
`iso_c_binding`, one file, no preprocessor flags, no HDF5, no MPI, nothing to
link beyond libc.

## Getting the source

It ships inside the installed package:

```bash
gfortran -c $(python -c "import scope_profiler.fortran_trace as t; print(t.module_source_path())")
```

or from a checkout, `src/scope_profiler/fortran/scope_profiler.f90`. A
`Makefile` and a runnable `example.f90` sit next to it.

## Marking regions

```fortran
program simulation
   use scope_profiler
   implicit none
   integer :: step, solve, assemble

   call sp_init("profile")            ! output prefix

   solve = sp_region("solve")         ! resolve names once...
   assemble = sp_region("assemble")

   do step = 1, 1000
      call sp_begin(assemble)         ! ...then use the handles
      call assemble_matrix()
      call sp_end(assemble)

      call sp_begin(solve)
      call solve_system()
      call sp_end(solve)
   end do

   call sp_finalize()                 ! writes profile_rank00000.spt
end program simulation
```

`sp_begin_name("solve")` / `sp_end_name("solve")` take the name directly, which
reads better in cold code; they look the name up on every call, so prefer the
handle form in hot loops.

Regions may nest, and a region may re-enter itself recursively — each entry
reserves its own slot, exactly as in the Python API.

## The API

| Call | Purpose |
| --- | --- |
| `sp_init(prefix [, rank])` | Start profiling. `rank` (default 0) gives each MPI rank its own trace file. |
| `sp_region(name)` | Handle for a region name, created on first use. |
| `sp_begin(id)` / `sp_end(id)` | Enter and leave a region. |
| `sp_begin_name(name)` / `sp_end_name(name)` | The same, looking the name up each time. |
| `sp_num_calls(id)` | How often a region was entered; readable after `sp_finalize()`. |
| `sp_now_ns()` | The raw clock, in nanoseconds. |
| `sp_is_active()` | Whether profiling is running. |
| `sp_finalize()` | Write `<prefix>_rank<NNNNN>.spt` and stop. |

## Under MPI

Pass each rank's id so the traces do not collide:

```fortran
call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)
call sp_init("profile", rank=my_rank)
```

Every rank writes its own file; nothing is communicated, so `sp_finalize()` is
not collective and a rank that dies takes only its own trace with it. The
importer merges whatever it finds:

```bash
mpirun -n 128 ./simulation
scope-profiler import-fortran . -o profiling_data.h5
scope-profiler pproc profiling_data.h5 -o figures
```

## From Python

```python
from scope_profiler.fortran_trace import load_traces, convert_traces

results = load_traces("run_dir")            # -> ProfilingResults, as usual
results.print_summary()
results.to_dataframe()

convert_traces("run_dir", "profiling_data.h5", label="128 ranks")
```

`load_traces` returns the very same `ProfilingResults` a Python run produces,
so every method, `plot_*` function and exporter works on it unchanged.

## One timeline with Python

The module reads the same OS clock CPython's `time.perf_counter_ns()` uses —
`CLOCK_MONOTONIC` on Linux, `CLOCK_UPTIME_RAW` on macOS — and picks it by
probing at run time rather than by compile-time platform macros, which
gfortran does not reliably define.

That means Fortran and Python regions recorded in the same process tree share
an epoch: a Python driver's timestamps and its Fortran kernels' timestamps are
directly comparable, and can be read on one timeline.

## The trace format

`sp_finalize()` writes `<prefix>_rank<NNNNN>.spt`, a small binary file:

```
char[8]   "SCOPEPRF"
int32     format version (1)
int32     rank
int64     number of regions
per region:
    int32     length of the name in bytes
    char[]    name
    int64     number of calls
    int64[]   start timestamps, nanoseconds
    int64[]   end timestamps, nanoseconds
```

Native endianness; the reader detects and handles both. 16 bytes per recorded
call, the same as the Python side.

## Limitations

- **Not thread safe.** A region must be entered and left by the same thread.
  OpenMP threads inside a region are fine — wrap the whole parallel construct
  in one `sp_begin`/`sp_end` from the master thread.
- **No LIKWID counters.** Hardware counters are collected only by the Python
  API; a Fortran trace carries timings alone.
- **Names are truncated** at `SP_MAX_NAME` (128) characters.
- **Recursion depth** per region is capped at `SP_MAX_DEPTH` (64); deeper
  nesting is reported on stderr and left untimed rather than corrupting the
  buffer.
- **A region still open at `sp_finalize()`** is reported on stderr and dropped,
  rather than written with a missing end time.
