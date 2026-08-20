# CLI reference

All subcommands live under the single `scope-profiler` executable (also
runnable as `python -m scope_profiler`).

## `scope-profiler tui`

Open an interactive browser for a profiling HDF5 file. The **Plots** section
reuses the plotting functions from `scope-profiler plot`:

```bash
scope-profiler tui profiling_data.h5
```

Select a plot and press `g` to display it with Matplotlib, or `s` to save it
as a PNG. Saved plots go to `<file-stem>_plots/` by default; use
`--plot-output DIR` to choose another directory.

When a plot is selected, its settings panel shows the regions matched by the
include/exclude filters as you type. It also supports rank ranges, colormap,
log scale, duration metrics, duration sorting and top-N selection, histogram
bin count, and imbalance metric. Press **Apply settings** before displaying or
saving the plot.

Available plot types include Gantt, flame, durations, timeseries, histogram,
imbalance, and any LIKWID metrics recorded in the file. Install the optional
TUI/plotting dependencies with `pip install "scope-profiler[pproc]"`.

## `scope-profiler run`

Profile a script's function calls without modifying it, similar to
`python -m cProfile`. By default only the script's own code is instrumented
(the standard library and installed packages are skipped) to keep overhead
low; pass `--all` to trace everything.

```text
usage: scope-profiler run [-h] [-o OUTFILE] [-q] [--all] [--line-profile]
                          [--buffer-limit BUFFER_LIMIT]
                          script ...
```

| Flag                | Description                                                          |
| -------------------- | --------------------------------------------------------------------- |
| `-o`, `--outfile`    | Path to the merged HDF5 output file (default: `profiling_data.h5`)   |
| `-q`, `--quiet`      | Suppress the per-region summary printed after the run                |
| `--all`              | Also instrument standard-library/installed-package calls (default: only the script's own code) |
| `--line-profile`     | Also collect line-by-line timings via `line_profiler` (requires `scope-profiler[line-profiler]`) |
| `--buffer-limit`     | Initial buffer capacity per region; grows as needed (default: 1024)     |

```bash
scope-profiler run my_script.py [script args...]
```

## `scope-profiler inspect`

Print what is inside a profiling file --- the run metadata in full, and an
overall statistics line per region --- without producing any plots. Useful
for checking which environment a run came from, and where its time went, at a
glance.

```text
usage: scope-profiler inspect [-h] [--include INCLUDE [INCLUDE ...]]
                              [--exclude EXCLUDE [EXCLUDE ...]]
                              [--ranks RANKS [RANKS ...]]
                              [--sort {total,calls,avg,min,max,first,last,std,p50,p95,p99,imbalance,name}] [--full]
                              [--columns {region,ranks,calls,total,avg,min,max,first,last,std,p50,p95,p99,imbalance} [...]]
                              [--source NAME [NAME ...]]
                              [--metadata-only | --regions-only]
                              files [files ...]
```

| Flag              | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `--include`       | Only report regions matching these regex patterns                       |
| `--exclude`       | Skip regions matching these regex patterns                              |
| `--ranks`         | Restrict region statistics to these ranks, e.g. `0 2` or `0-3`          |
| `--sort`          | Order regions by any region-table statistic, or by `name` alphabetically |
| `--columns`       | Restrict the region table to selected columns; default is `region ranks calls total avg` |
| `--full`          | Print long metadata values (`PATH`, `LD_LIBRARY_PATH`, ...) in full     |
| `--source`        | Print the captured call-site source (the `with` block or decorated function that defines it) of these regions |
| `--export-metadata` | Also write the metadata of every inspected file to this JSON file     |
| `-q`, `--quiet`   | Suppress the printed summary (useful with `--export-metadata`)          |
| `--metadata-only` | Print only the metadata section                                         |
| `--regions-only`  | Print only the region statistics                                        |

```bash
scope-profiler inspect profiling_data.h5
scope-profiler inspect 'run_*.h5' --regions-only --sort calls
scope-profiler inspect profiling_data.h5 --regions-only --columns region ranks calls total avg
scope-profiler inspect profiling_data.h5 --export-metadata metadata.json --quiet
scope-profiler inspect profiling_data.h5 --source solve assemble
```

Example output:

```text
==============================================================================
profiling_data.h5
2 rank(s), 4 region(s), 0.18 MiB, 0.0951538 s wall clock
==============================================================================

Metadata
  Run
    timestamp              : 2026-07-26T18:57:49
    user                   : mlindqvi
    hostname               : lrdn1234
    ...
  System
    chip_information       : AMD EPYC 9654 96-Core Processor
  Parallelism
    mpi_size               : 2
    omp_num_threads        : 8
    total_cores            : 16
  Slurm
    SLURM_JOB_ID           : 9988776
    ...
  Modules (4)
    profile/base
    gcc/12.3.0
    openmpi/4.1.6--gcc--12.3.0
    python/3.11.7

Regions (4)
  region    ranks  calls  total [s]    avg [s]
  ---------------------------------------------
  timestep      2      8   0.139235  0.0174044
  solve         2      8  0.0991292  0.0123911
  setup         2      2  0.0473326  0.0236663
  assemble      2      8  0.0399496  0.0049937
  ---------------------------------------------
  TOTAL               26   0.325647
```

Region durations are in seconds, aggregated over the selected ranks. See
{doc}`/guide/hdf5_and_python_api` for what each metadata field means.

### Showing a region's source

A region can remember the `with` block or decorated function it was defined
with -- captured once when the region is first created, but only if the run
was profiled with `capture_region_source=True` (off by default; see
{doc}`/guide/configuration` for its cost and {doc}`/guide/hdf5_and_python_api`
for the Python API). `--source` prints it, after the region table:

```bash
scope-profiler inspect profiling_data.h5 --source solve assemble
```

```text
Source (2)
  solve  (kernels.py:42)
        with ProfileManager.profile_region("solve"):
            return solver.step(state)
  assemble  (kernels.py:18)
    @ProfileManager.profile("assemble")
    def assemble(size):
        return build_matrix(size)
```

A name with nothing captured (`capture_region_source` was left at its
default, the file predates this feature, or the region was only ever created
by the recursive tracer) prints "source not captured" instead of failing the
whole command; an unknown name prints the list of regions the file actually
has.

### Exporting metadata to JSON

`--export-metadata` writes the metadata of every inspected file to one JSON
document. Values are never clipped there, regardless of `--full`:

```bash
scope-profiler inspect profiling_data.h5 --export-metadata metadata.json --quiet
```

```json
{
  "files": [
    {
      "file_path": "/scratch/run/profiling_data.h5",
      "num_ranks": 2,
      "metadata": {
        "chip_information": "AMD EPYC 9654 96-Core Processor",
        "modules": ["profile/base", "gcc/12.3.0", "python/3.11.7"],
        "SLURM_JOB_ID": "9988776",
        "...": "..."
      }
    }
  ]
}
```

Several files (or a glob) produce one entry each, which makes the export
convenient for comparing the environments of a set of runs. The same is
available from Python:

```python
from scope_profiler.inspection import collect_file_metadata, write_metadata_json

payload = write_metadata_json("profiling_data.h5", "metadata.json")
payload = collect_file_metadata(["run_1.h5", "run_2.h5"])  # no file written
```

## `scope-profiler diff`

Compare region statistics between two merged HDF5 profiling files, region by
region, so a regression (or an improvement) between two runs -- two commits,
two configs, two job sizes -- shows up as a single table.

```text
usage: scope-profiler diff [-h] [--include INCLUDE [INCLUDE ...]]
                           [--exclude EXCLUDE [EXCLUDE ...]]
                           [--ranks RANKS [RANKS ...]]
                           [--metric {total,avg,min,max,p50,p95,p99,imbalance,calls}]
                           [--sort {delta,pct,name}] [--threshold PCT]
                           file_a file_b
```

| Flag              | Description                                                              |
| ----------------- | ------------------------------------------------------------------------- |
| `--include`       | Only compare regions matching these regex patterns                       |
| `--exclude`       | Skip regions matching these regex patterns                               |
| `--ranks`         | Restrict the statistics to these ranks, e.g. `0 2` or `0-3`              |
| `--metric`        | Statistic to compare: `total` (default), `avg`, `min`, `max`, `p50`, `p95`, `p99`, `imbalance` or `calls` |
| `--sort`          | Order regions by descending `|delta|` (default), descending `|delta %|`, or `name` |
| `--threshold`     | Only show regions whose absolute percent change is at least this many percent |

```bash
scope-profiler diff baseline.h5 candidate.h5
scope-profiler diff baseline.h5 candidate.h5 --metric avg --threshold 5
```

## `scope-profiler check`

Use `check` in CI to fail when a candidate exceeds a performance budget:

```bash
scope-profiler check baseline.h5 candidate.h5 --max-regression 5
```

The command prints the comparison table and returns exit code `1` when any
region is more than 5% slower. Use `--fail-on-new` to treat newly appearing
regions as failures, and `--metric p95` or `--metric imbalance` to enforce a
tail-latency or MPI-balance budget.

Example output:

```text
==============================================================================
a: baseline - baseline.h5  (2 rank(s))
b: candidate - candidate.h5  (2 rank(s))
==============================================================================

Regions (3)
  region     total [s] (a)  total [s] (b)  delta  delta [%]
  -----------------------------------------------------------
  solve            0.0991         0.1487  +0.05        +50%
  teardown              -         0.0123  +0.01           -
  setup            0.0473         0.0473  +0.00          +0%
  -----------------------------------------------------------
  Only in b: teardown
```

A region present in only one file still gets a `delta` (treating the missing
side as 0 calls), but a percent change is only reported when the file it is
missing from is `b` -- dropping out from a nonzero baseline in `a` is a
well-defined -100%, while a region appearing fresh in `b` has no baseline to
divide by, and is listed under "Only in b" instead. `--threshold` never drops
those regions, since there is nothing to compare against.

## `scope-profiler plot`

Post-process one or more HDF5 profiling files and render a named plot or plot
preset. For text/JSON summaries (including LIKWID hardware counters), see
`scope-profiler inspect` above instead. For machine-readable exports without
rendering charts, see `scope-profiler export` below.

```text
usage: scope-profiler plot [-h]
                           {list,default,all,quick,gantt,flame,durations,
                            timeseries,speedup,histogram,imbalance,likwid}
                           ...
```

`scope-profiler plot list` prints the available plot kinds and presets.

### Positional arguments

| Argument | Description                                      |
| -------- | ------------------------------------------------ |
| `kind`   | One of `default`, `all`, `quick`, or a plot kind |
| `files`  | Path(s) or glob patterns for `profiling_data.h5` files |

`default` renders `gantt` and the total `durations` plot. `quick` renders
`durations` and `speedup`.
`all` renders every plot except `likwid`; pass `--metric` to include `likwid`.

### Selecting data

| Flag              | Description                                      |
| ----------------- | ------------------------------------------------ |
| `--label`          | Override a file's display label in the outputs (repeat once per file, in order) |
| `-i`, `--include` | Region names to include (regex patterns)         |
| `-e`, `--exclude` | Region names to exclude (regex patterns)         |
| `-r`, `--ranks`   | Ranks to include; supports ranges (e.g. `0-3,5`) |

### Choosing and rendering plots

| Flag              | Description                                      |
| ----------------- | ------------------------------------------------ |
| `--show`          | Display the plot interactively (default: off)    |
| `-o`, `--output`  | Directory to save generated outputs; for a single plot kind this may be a target `.png` or `.html` file |
| `--backend`       | Renderer: `matplotlib` (default, writes `.png`) or `plotly` (writes interactive `.html`) |
| `--cmap`          | Matplotlib colormap used to color regions/files in all plots (default: `tab20`) |

### Plot-specific options

| Flag                 | Plot(s)    | Description                                      |
| -------------------- | ---------- | ------------------------------------------------- |
| `--metrics`          | durations  | Duration statistics to draw as bar columns: any of `avg`, `min`, `max`, `total` (default: `total`) |
| `--sort-by`           | durations, likwid | Order the bar chart's regions by this statistic, descending (`name` sorts alphabetically). Default: order of first appearance |
| `--top-n`             | durations, likwid | Keep only the top N regions after `--sort-by` |
| `--combine-regions`   | durations  | Merge several regions into one bar: `NAME=PATTERN1,PATTERN2` (repeat once per group). Region names are matched against the comma-separated regexes like `--include`; a region matched by more than one group is claimed by whichever group is listed first |
| `--log-scale`         | durations, timeseries, histogram, imbalance, likwid | Logarithmic y-axis |
| `--bins`             | histogram  | Number of duration bins (default: 30) |
| `--metric`           | imbalance  | Per-call duration statistic plotted per rank: any of `avg`, `min`, `max`, `total` (default: `total`) |
| `--metric`           | likwid     | Name of the LIKWID derived metric or raw event to plot, e.g. `CPI`, `MFlops/s` |
| `--x`                | speedup    | X-axis: `num_ranks` (default), `omp_num_threads`, `total_cores`, or any other metadata field |

When `-o/--output` is supplied, the CLI saves one `<name>_plot.png` per plot
selected by the plot kind or preset (`durations_plot.png` becomes one
`durations_plot_<metric>.png` per metric when `--metrics` requests
several, e.g. `durations_plot_avg.png`, `durations_plot_total.png`), plus
`region_statistics.json`. `speedup` is skipped unless multiple files are
passed. With `--backend plotly` the plots are written as `.html` instead of
`.png`.

For multiple files, the JSON includes per-file region statistics and the set of
common regions across all inputs.

## `scope-profiler export`

Export profiling data without rendering plot images.

```text
usage: scope-profiler export [-h] {prof,speedscope,plot-data} ...
```

### Plot data

`plot-data` writes the raw data behind selected charts as CSV or JSON, so plots
can be reconstructed later without the original HDF5 files:

```bash
scope-profiler export plot-data profiling_data.h5 -o data --format json
scope-profiler export plot-data run_1.h5 run_2.h5 -o data \
    --plots durations speedup --format json
```

With JSON, each data file includes a `colors` map matching the plot colors.
`plot-data` supports the same data-selection flags as `plot`, plus `--plots`,
`--format`, `--metrics`, `--bins`, `--imbalance-metric`, `--likwid-metric`,
and `--x`.

### Exporting to `.prof` for snakeviz

`export prof` writes the profile in the format `cProfile` uses, so regions
can be browsed with any pstats-based viewer:

```bash
scope-profiler export prof profiling_data.h5 -o figures
snakeviz figures/profile_rank0.prof
```

One file is written per exported rank (`profile_rank0.prof`, …; only the ranks
selected with `-r/--ranks` are exported, default rank 0), prefixed with the
input file's stem when several HDF5 files are passed. Regions become
"functions", `cumtime` is a region's total wall time and `tottime` is that
minus the time spent in its nested regions, with a synthetic
`<file rank N>` frame as the root of the tree.

Since regions carry no call graph, the caller/callee relations are
reconstructed from timestamp containment, exactly as the flame chart does. So:

- Regions that only partially overlap (async work, threads) are attributed to
  whichever region enclosed their start, and the enclosing region's `tottime`
  is clamped at zero rather than going negative.
- A region called from several places is merged into one entry, as pstats is
  keyed by function rather than by call path; recursion is reported as
  `2/1`-style call counts, like `cProfile`.
- LIKWID counters have no place in the `.prof` format and are left out.

### Exporting to speedscope

`export speedscope` writes a [speedscope](https://www.speedscope.app) JSON
file. Where `.prof` keeps aggregates per region, speedscope keeps every
individual call, so the timeline shows the run as it actually happened —
closest in spirit to the Gantt and flame charts, but interactive:

```bash
scope-profiler export speedscope profiling_data.h5 -o figures
```

Then open `figures/profile.speedscope.json` at <https://www.speedscope.app>
(the file never leaves the browser), or run `npx speedscope
figures/profile.speedscope.json`. Its three views are all useful here: "Time
Order" is the run's timeline, "Left Heavy" aggregates identical call paths
(the flame graph), and "Sandwich" ranks regions by self and total time.

One file is written per input HDF5 file, holding one profile per exported rank
(only the ranks selected with `-r/--ranks`, default rank 0) — the format
carries several profiles per file, and speedscope switches between them from
the dropdown in its top bar. Every profile in a file shares one time origin, so
ranks stay aligned with each other. The stem of the input file is added to the
name when several HDF5 files are passed.

The call graph is reconstructed from timestamp containment, as for `.prof`,
with one extra consequence: speedscope replays the events as a stack machine,
so a region that starts inside another but ends after it is clipped to its
parent instead of overhanging it. The same caveats otherwise apply — regions
called from several places and recursion behave as described above.

### Examples

**Save plots for a single file:**

```bash
scope-profiler plot profiling_data.h5 -o figures/
```

Omitting the plot kind selects the default preset (`gantt` and total
`durations`). The explicit equivalent is `scope-profiler plot default
profiling_data.h5 -o figures/`.

**Compare multiple files:**

```bash
scope-profiler plot default run_1.h5 run_2.h5 run_4.h5 -o figures/
```

**Select files via wildcard patterns:**

```bash
scope-profiler plot default files/*.h5 -o figures/
scope-profiler plot default "files/file_*.h5" -o figures/
```

**Display interactively with region filtering:**

```bash
scope-profiler plot profiling_data.h5 --show \
    --include "solver.*" "rhs.*" \
    --exclude "io"
```

**Select specific MPI ranks:**

```bash
scope-profiler plot default profiling_data.h5 --show --ranks 0-3 8
```

The `--ranks` flag accepts comma-separated values and dash ranges that
can be combined: `0,2,4-7` expands to ranks 0, 2, 4, 5, 6, 7.

**Only export average and total duration plots:**

```bash
scope-profiler plot durations profiling_data.h5 -o figures/ --metrics avg total
```

**Only generate the duration bar chart and speedup plot:**

```bash
scope-profiler plot quick run_1.h5 run_2.h5 run_4.h5 -o figures/
```
