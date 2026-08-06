# Post-processing with the CLI

`scope-profiler pproc` turns one or more `profiling_data.h5` files into
charts, an aggregate statistics JSON, and — on request — exports for external
profile viewers. It covers the same ground as the plotting functions described
in {doc}`/guide/hdf5_and_python_api`, without writing any code.

This page walks through the command with a concrete example. For the complete
list of flags, see {doc}`/cli`.

## The example run

Every figure below comes from the same mock solver: a `setup` phase followed
by three `timestep`s, each containing `assemble`, `solve` and
`halo_exchange`, with `io` on every second step. It was run at 1, 2 and 4 MPI
ranks, producing `run_1.h5`, `run_2.h5` and `run_4.h5`:

```python
with ProfileManager.profile_region("setup"):
    ...

for step in range(3):
    with ProfileManager.profile_region("timestep"):
        with ProfileManager.profile_region("assemble"):
            ...
        with ProfileManager.profile_region("solve"):
            ...
        with ProfileManager.profile_region("halo_exchange"):
            ...
        if step % 2 == 1:
            with ProfileManager.profile_region("io"):
                ...
```

The complete script, including the commands used to render every figure on
this page, is `examples/generate_cli_docs_figures.py`.

## Basic usage

Point `pproc` at a file and give it an output directory:

```bash
scope-profiler pproc run_2.h5 -o figures
```

```text
Plotting Gantt chart for ranks: [0, 1]
Plotting flame graph for: run_2 (rank 0)
Plotting duration comparison (total) for files: run_2
Plotting duration over time for files: run_2
Outputs saved to:
  figures/gantt_plot.png
  figures/flame_plot.png
  figures/durations_plot.png
  figures/duration_timeseries_plot.png
  figures/region_statistics.json
```

Without `-o/--output` nothing is written; use `--show` to open the charts
interactively instead. The two can be combined.

## Text summary

`--summary` prints the numbers instead of drawing them, which is often all
you need over ssh:

```bash
scope-profiler pproc run_2.h5 --summary
```

```text
run_2.h5  (2 rank(s))
  region        ranks  calls  total [s]   avg [s]    min [s]   max [s]    std [s]
  -------------------------------------------------------------------------------
  main              1      1    0.16637   0.16637    0.16637   0.16637          0
  matmul            1      3   0.125949  0.041983  0.0083437  0.109189  0.0475221
  memory_bound      1      1   0.035121  0.035121   0.035121  0.035121          0
  -------------------------------------------------------------------------------
  TOTAL                    5    0.32744
```

This is the same table `ProfileManager.finalize()` and
{doc}`scope-profiler inspect <../cli>` render. `--summary-sort` reorders it
(`total`, `calls`, `avg`, `max` or `name`), and `--include`/`--exclude`/
`--ranks` narrow it as they do for the charts.

On its own `--summary` produces no plots --- the summary is the whole job.
Combine it with `--show` or `-o/--output` to get both.

If the run recorded LIKWID hardware counters, they follow as a separate table
per rank and event group; see {doc}`likwid`.

## Gantt chart

`gantt_plot.png` places one lane per (region, rank) pair, with a bar for every
recorded call. It is the chart to reach for when the question is *when* things
happened — startup cost, gaps between steps, ranks drifting apart:

```{image} /_static/figures/cli/gantt_plot.png
:alt: Gantt chart of the mock solver at two MPI ranks
:width: 100%
```

Both ranks appear by default, one lane per rank. Calls of the same region
share a color, and nested regions such as `assemble` and `solve` get their own
lanes rather than being drawn inside `timestep` — the flame graph below is the
view that shows the nesting.

## Flame graph

`flame_plot.png` answers *where the time went* instead. The call stack is
reconstructed from timestamp containment — a region whose interval lies
inside another's is drawn one level above it:

```{image} /_static/figures/cli/flame_plot.png
:alt: Flame graph of the mock solver, rank 0
:width: 100%
```

Unlike the Gantt chart, the flame graph is drawn per rank and defaults to
rank 0 only; pass `--ranks` to render more (one panel each).

## Duration bar charts

One bar chart is written per duration statistic. By default only `total` time
is shown; `--metrics` selects which statistics to include:

```bash
scope-profiler pproc run_2.h5 -o figures --metrics avg total
```

```{image} /_static/figures/cli/durations_plot_avg.png
:alt: Average duration per call, by region
:width: 100%
```

```{image} /_static/figures/cli/durations_plot_total.png
:alt: Total duration, by region
:width: 100%
```

The two views tell different stories: `setup` and `solve` cost about the same
per call, but `solve` runs once per timestep and so accounts for three times
as much total time. With several input files, bars are grouped per region so
runs can be compared side by side.

## Duration over time

`duration_timeseries_plot.png` plots each region's per-call duration against
wall-clock time, with a band spanning the minimum and maximum across the
selected ranks. Rank imbalance shows up as a widening band, and slow drift
(cache growth, memory pressure, throttling) as a trend:

```{image} /_static/figures/cli/duration_timeseries_plot.png
:alt: Region duration over wall-clock time
:width: 100%
```

Here the middle `timestep` stands out, because that is the step that also
writes `io`. Regions called only once — `setup` and `io` in this run — are a
single point and stay invisible between the lines.

## Comparing several runs

Passing several files compares them. Each chart then either stacks one panel
per file (Gantt, flame) or groups the files together (durations), and a
speedup plot is added:

```bash
scope-profiler pproc run_1.h5 run_2.h5 run_4.h5 -o figures_scaling
```

```{image} /_static/figures/cli/speedup_plot.png
:alt: Per-region speedup at 1, 2 and 4 MPI ranks
:width: 100%
```

Each line is one region's speedup relative to the smallest run, derived from
average per-call durations, against the dashed ideal-scaling line. The mock
solver behaves as designed: `solve` and `assemble` scale well, `io` is serial
and flat, and `halo_exchange` gets *slower* with more ranks, dragging
`timestep` below the ideal line.

The x-axis is the MPI rank count by default. `--x-field` switches it to
`omp_num_threads` or `total_cores` (both read from the run metadata), or to
any other metadata field — in which case the files stay in the order given on
the command line and no ideal-scaling line is drawn:

```bash
scope-profiler pproc omp_*.h5 -o figures_scaling --x-field omp_num_threads
```

Files can also be selected with wildcards. Quote the pattern to let
scope-profiler expand it rather than the shell:

```bash
scope-profiler pproc "runs/run_*.h5" -o figures_scaling
```

## Selecting which plots to generate

By default all five plots are generated. `--plots` (short: `-p`) restricts
the run to any subset:

```bash
# only the total-time bar chart and the speedup comparison
scope-profiler pproc run_1.h5 run_2.h5 run_4.h5 -o figures \
    --plots durations speedup
```

Available plot names: `gantt`, `flame`, `durations`, `timeseries`, `speedup`.
Omitting `--plots` is equivalent to passing all five. Filtering applies before
any chart is drawn, so skipping charts you do not need also speeds up the
run.

## Filtering regions and ranks

`--include` and `--exclude` take regular expressions matched against region
names, and `--ranks` selects ranks — as individual values, dash ranges, or a
mix (`0,2,4-7` expands to 0, 2, 4, 5, 6, 7):

```bash
scope-profiler pproc run_2.h5 -o figures_filtered \
    --include solve assemble \
    --ranks 0
```

```{image} /_static/figures/cli/gantt_plot_filtered.png
:alt: Gantt chart restricted to the solve and assemble regions on rank 0
:width: 100%
```

Filtering applies to every output of the run, including
`region_statistics.json` and the exports below. On a large run it is also the
quickest way to make the charts readable again.

## Region statistics JSON

Whenever `-o/--output` is given, `region_statistics.json` is written
alongside the figures. It holds the numbers behind the charts: per-file,
per-region aggregates, the same statistics per rank, and the region names
common to all input files.

```json
{
  "units": { "durations": "seconds" },
  "filters": { "include": null, "exclude": null, "ranks": null },
  "common_regions": ["setup", "timestep", "assemble", "solve", "halo_exchange", "io"],
  "files": [
    {
      "label": "run_2",
      "file_path": "/scratch/run/run_2.h5",
      "num_ranks": 2,
      "region_statistics": {
        "setup": {
          "count": 2,
          "average_duration_seconds": 0.039551083,
          "min_duration_seconds": 0.0395205,
          "max_duration_seconds": 0.039581666,
          "std_duration_seconds": 3.0583e-05,
          "total_duration_seconds": 0.079102166,
          "per_rank": {
            "0": { "count": 1, "average_duration_seconds": 0.0395205, "...": "..." },
            "1": { "count": 1, "average_duration_seconds": 0.039581666, "...": "..." }
          }
        }
      }
    }
  ]
}
```

## Interactive HTML charts

`--backend plotly` renders the same charts as self-contained interactive
`.html` files instead of PNGs, with zooming and hover labels — useful for
long runs where a static Gantt chart becomes a smear. With `--show`, the
pages open in a browser:

```bash
scope-profiler pproc run_2.h5 -o figures_html --backend plotly
```

The matplotlib backend (the default) writes `.png` and needs no extra
dependency; the Plotly backend writes `.html` and likewise needs nothing
beyond Plotly itself.

## Exporting the data behind the charts

`--export-data` writes the exact series each chart was drawn from, so the
figures can be reproduced — or re-styled elsewhere — without the HDF5 files:

```bash
scope-profiler pproc run_2.h5 -o figures --export-data
```

```text
gantt_data.csv                    file, rank, region, start_seconds, end_seconds
flame_data.csv                    file, rank, region, depth, start_seconds, end_seconds
durations_data.csv                file, region, metric, value_seconds
duration_timeseries_data.csv      file, region, call_index, time_seconds,
                                  mean/min/max_duration_seconds, num_ranks
speedup_data.csv                  only when several files are passed
```

```text
file,rank,region,start_seconds,end_seconds
run_2,0,setup,0.000202333,0.039720250
run_2,1,setup,0.0,0.038490875
run_2,0,timestep,0.039737417,0.106206750
```

`--export-data-format json` writes JSON instead, including a `colors` map
matching the colors used in the plots, so a re-rendered chart keeps the same
region colors. If only the data is wanted, `--skip-plot-images` skips
rendering the images altogether:

```bash
scope-profiler pproc run_2.h5 -o data --export-data --export-data-format json \
    --skip-plot-images
```

## Exporting to external profile viewers

Two flags export the run in formats other tools understand:

```bash
# cProfile/pstats format, one file per exported rank
scope-profiler pproc run_2.h5 -o figures --export-prof --skip-plot-images
snakeviz figures/profile_rank0.prof

# speedscope, one file holding one profile per exported rank
scope-profiler pproc run_2.h5 -o figures --export-speedscope --skip-plot-images
npx speedscope figures/profile.speedscope.json
```

Both reconstruct the call graph from region nesting the same way the flame
graph does, and both export only the ranks selected with `--ranks` (rank 0 by
default). {doc}`/cli` documents what that reconstruction implies for
partially overlapping regions and recursion.

## Files without timing data

A file profiled with `time_trace=False` records call counts but no
timestamps, so there is nothing to plot. Rather than failing inside the
plotting code, `pproc` reports the counts and stops:

```text
No timing data found — these files were profiled with time_trace=False,
which records call counts only.

ncalls_only.h5:
  setup: 1 calls
  timestep: 3 calls
```

## Reproducing the figures on this page

```bash
python examples/generate_cli_docs_figures.py
```

The script runs the mock solver at 1, 2 and 4 ranks (via `mpirun`, if
available), invokes the `pproc` commands shown above, and copies the
resulting PNGs into `figures/cli/`, which the docs build picks up as
`_static/figures/cli/`. Pass `--keep DIR` to also keep the HDF5 files and the
raw CLI output around.
