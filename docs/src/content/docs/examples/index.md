---
title: Examples
sidebar:
  label: Overview
  order: 0
---

> **Install for all notebooks:**
> `pip install "scope-profiler[notebook,pproc,line-profiler]"`. Notebooks 1
> and 4 use only the base install; notebook 2 also needs pandas (`pproc`),
> notebooks 3, 6 and 8 need `pproc`, and notebook 7 needs all three extras.

These notebooks are the fastest way to learn `scope-profiler`. They are listed
individually in the documentation navigation so you can jump directly to the
topic you need.

## Notebooks

Eight notebooks that build on each other. Each one is self-contained — it
generates its own profiling data in a temporary directory, so they can be run
in any order and in any environment.

1. [Getting started](/scope-profiler/guide/01-getting-started/) — configure the
   profiler, mark regions, print timings during the run, finalize, and inspect
   the first output file.
2. [Post-processing profiling data](/scope-profiler/guide/02-postprocessing/) — use
   `ProfilingResults`, summaries, DataFrames, filtering, metadata, and JSON.
3. [Visualizing results](/scope-profiler/guide/03-visualization/) — create Gantt,
   flame, duration, and speedup charts.
4. [Profiling modes and configuration](/scope-profiler/guide/04-profiling-modes/) —
   recursive, line-by-line, CLI, MPI, and LIKWID profiling.
5. [Custom analysis with the Python API](/scope-profiler/guide/05-custom-analysis/) —
   inspect individual calls, timelines, call stacks, and self-time.
6. [Building your own plots](/scope-profiler/guide/06-custom-plots/) — draw custom
   visualizations from events and call-stack data.
7. [Notebook magics](/scope-profiler/guide/07-notebook-magics/) — profile a cell
   with `%%scope`, `%%scope_line` and `%%scope_recursive`, then compare,
   analyse and export runs with `%scope_compare`, `%scope_df`,
   `%scope_load` and `%scope_export`.
8. [Reports and comparison](/scope-profiler/guide/08-reports-and-comparison/) — summarize
   runs with `inspect`, compare them with `diff`, write HTML reports, gate
   regressions with `check` and publish a catalogue with `site`.

The single command above installs everything. For a smaller environment, use
only the extras named beside the notebooks you plan to run.

```bash
pip install "scope-profiler[notebook,pproc,line-profiler]"
```

## Scripts

[Scripts and workflows](/scope-profiler/examples/scripts/) lists the ready-to-run scripts in
the repository's `examples/` directory and the AI optimization workflow.
