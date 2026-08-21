

# C examples

One C kernel library, driven two ways.

`kernels.c` is a small Jacobi smoother that marks regions **inside
itself** — the shape a real kernel library takes, where the caller
neither knows nor cares what is instrumented in there. The same file is
then used by:

- **`standalone.c`** — a pure C program. No Python at run time; the
  trace is converted afterwards with `scope-profiler import-native`.
- **`run_mixed.py`** — a Python driver that loads the very same kernels
  with `ctypes`, so both languages record into **one** profile with one
  call stack.

Everything here needs is a C compiler. The `ctypes` bridge needs no
binding generator and no build backend — unlike the Fortran examples,
which need f2py.

## Standalone C

``` bash
make run-standalone
```

Builds and runs the program, then imports its trace:

      region        ranks  calls    total [s]      avg [s]      min [s]      max [s]
      c:timestep        1     20   0.00458696  0.000229348  0.000117375  0.000371625
      c:residual        1    100   0.00369654  3.69654e-05   1.8708e-05   6.2875e-05
      c:checkpoint      1      5   0.00159525   0.00031905   4.9167e-05  0.000577209
      c:stencil         1    100  0.000734292  7.34292e-06    3.416e-06   2.3542e-05
      c:setup           1      1   5.0209e-05   5.0209e-05   5.0209e-05   5.0209e-05

From there it is an ordinary profile —
`scope-profiler plot default build/profiling_data.h5 -o figures` and the
rest.

## Python driving C

``` bash
make run-mixed
```

Builds the kernels and the profiler into one shared library and runs the
Python driver, which loads it with `ctypes`. Both sides record;
`finalize(native_traces=...)` folds the C trace into the Python run’s
output, and one file comes out.

Because both APIs read the same clock, the call stack is a single tree
across the language boundary:

      python:setup                   11.970 ms   [python]
      python:timestep                 0.331 ms   [python]
        python:call_solver              0.320 ms   [python]
          c:stencil                       0.023 ms   [c]
          c:residual                      0.040 ms   [c]

It also writes `figures/gantt.png` and a speedscope trace, where the
`c:` bars sit visibly inside the `python:` ones.

## Two rules for mixing

1.  **Call `kernels_stop_profiling()` before the profiling session
    exits** — the trace has to exist by the time Python reads it.
2.  **Give the two sides distinct region names.** A name recorded by
    both raises rather than silently double-counting a wrapper and the
    region inside it. Hence the `python:` / `c:` prefixes.

## One ctypes trap

`ctypes` assumes every function returns `int` unless told otherwise. The
solver returns a `double`, so without

``` python
kernels.kernels_jacobi_solve.restype = ctypes.c_double
```

it comes back as garbage — the regions are still recorded correctly,
which makes it easy to miss. Declare `argtypes` and `restype` for
everything.

## Under MPI

Nothing changes structurally: pass the rank to `sp_init` (here via
`kernels_start_profiling`) so each process writes its own trace, and
each rank folds in the one matching itself.

``` c
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
kernels_start_profiling("trace", rank);
```

`standalone.c` takes the rank as an optional argument for exactly this
reason. To build it against MPI, point `CC` at your wrapper:

``` bash
make standalone CC=mpicc
mpirun -n 4 ./build/standalone      # add MPI_Init/MPI_Finalize to the program
scope-profiler import-native build -o profiling_data.h5
```

## Mixing C and Fortran

The two native APIs write the same trace format, so a program built from
both lands in one profile with no extra work — see `examples/fortran/`
for the Fortran side.

## Cleaning up

``` bash
make clean
```
