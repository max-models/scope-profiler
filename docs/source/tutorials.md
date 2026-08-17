# Tutorials & examples

Step-by-step tutorials and self-contained example scripts.

## Notebooks

Six notebooks that build on each other. Each one is self-contained --- it
generates its own profiling data in a temporary directory, so they can be run
in any order and in any environment.

1. **Getting started** --- configuring the profiler, marking regions with a
   context manager and a decorator, finalizing, and a first look at the output
   file.
2. **Post-processing** --- the analysis API in depth: `ProfilingResults` as a mapping of
   regions, `MPIRegion` vs `Region`, summaries, DataFrames, filtering, metadata
   and JSON export.
3. **Visualizing results** --- Gantt, flame, duration and speedup charts, plus
   the equivalent `scope-profiler pproc` invocations.
4. **Profiling modes** --- what each `setup()` option records, recursive and
   line-by-line profiling, the zero-instrumentation CLI, MPI and LIKWID.
5. **Custom analysis** --- one level below the summaries: `events()` and the
   per-call DataFrame, the timeline anchors, `call_stack()` and the self-time
   it makes possible.
6. **Building your own plots** --- five charts drawn with plain matplotlib on
   top of that data: a timeline, a duration histogram, drift over the run, a
   total-vs-self breakdown, a hand-rolled flame graph and a run comparison.

```{toctree}
:maxdepth: 1
:glob:

tutorials/*
```

Notebooks 3, 4 and 6 need the plotting extra:

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
- **`ex_region_source.py`** --- reads back the source code a region was
  defined with, both from Python and via `scope-profiler inspect --source`.
- **`benchmark_overhead.py`** --- measures per-call overhead of every
  profiling mode and produces a bar chart.

Run any example with:

```bash
python examples/<script>.py
```
