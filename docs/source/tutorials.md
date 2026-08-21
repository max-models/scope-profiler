

# Interactive Jupyter tutorials

These notebooks are the fastest way to learn `scope-profiler`. They are
listed individually in the documentation navigation so you can jump
directly to the topic you need.

## Choose a tutorial

Six notebooks that build on each other. Each one is self-contained — it
generates its own profiling data in a temporary directory, so they can
be run in any order and in any environment.

1.  [Getting started](tutorials/01_getting_started.ipynb) — configure
    the profiler, mark regions, finalize, and inspect the first output
    file.
2.  [Post-processing profiling data](tutorials/02_postprocessing.ipynb)
    — use `ProfilingResults`, summaries, DataFrames, filtering,
    metadata, and JSON.
3.  [Visualizing results](tutorials/03_visualization.ipynb) — create
    Gantt, flame, duration, and speedup charts.
4.  [Profiling modes and
    configuration](tutorials/04_profiling_modes.ipynb) — recursive,
    line-by-line, CLI, MPI, and LIKWID profiling.
5.  [Custom analysis with the Python
    API](tutorials/05_custom_analysis.ipynb) — inspect individual calls,
    timelines, call stacks, and self-time.
6.  [Building your own plots](tutorials/06_custom_plots.ipynb) — draw
    custom visualizations from events and call-stack data.

Notebooks 3, 4 and 6 need the post-processing extra:

``` bash
pip install "scope-profiler[pproc]"
```

## Example scripts

The `examples/` directory in the repository contains ready-to-run
scripts:

- **`ex_line_profiling.py`** — demonstrates line-by-line profiling with
  `use_line_profiler=True`.
- **`ex_recursive_profiling.py`** — profiles nested function calls from
  one decorated entrypoint using `recursive_profile=True`.
- **`ex_cli_profiling.py`** — an uninstrumented script, profiled with
  `scope-profiler run` (no decorators or setup calls needed).
- **`ex_region_source.py`** — reads back the source code a region was
  defined with, both from Python and via
  `scope-profiler inspect --source`.
- **`benchmark_overhead.py`** — measures per-call overhead of every
  profiling mode and produces a bar chart.

Run any example with:

``` bash
python examples/<script>.py
```
