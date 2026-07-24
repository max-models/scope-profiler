# Tutorials & examples

Step-by-step tutorials and self-contained example scripts.

## Notebooks

Four notebooks that build on each other. Each one is self-contained --- it
generates its own profiling data in a temporary directory, so they can be run
in any order and in any environment.

1. **Getting started** --- configuring the profiler, marking regions with a
   context manager and a decorator, finalizing, and a first look at the output
   file.
2. **Post-processing** --- the analysis API in depth: the reader as a mapping of
   regions, `MPIRegion` vs `Region`, summaries, DataFrames, filtering, metadata
   and JSON export.
3. **Visualizing results** --- Gantt, flame, duration and speedup charts, plus
   the equivalent `scope-profiler pproc` invocations.
4. **Profiling modes** --- what each `setup()` option records, recursive and
   line-by-line profiling, the zero-instrumentation CLI, MPI and LIKWID.

```{toctree}
:maxdepth: 1
:glob:

tutorials/*
```

Notebooks 3 and 4 need the plotting extra:

```bash
pip install "scope-profiler[pproc]"
```

## Example scripts

The `examples/` directory in the repository contains ready-to-run
scripts:

- **`ex_line_profiling.py`** --- demonstrates line-by-line profiling
  with `use_line_profiler=True`.
- **`ex_recursive_profiling.py`** --- profiles nested function calls from
  one decorated entrypoint using `recursive_profile=True`.
- **`ex_cli_profiling.py`** --- an uninstrumented script, profiled with
  `scope-profiler run` (no decorators or setup calls needed).
- **`benchmark_overhead.py`** --- measures per-call overhead of every
  profiling mode and produces a bar chart.

Run any example with:

```bash
python examples/<script>.py
```
