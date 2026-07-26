# CLI reference

All subcommands live under the single `scope-profiler` executable (also
runnable as `python -m scope_profiler`).

## `scope-profiler run`

Profile a script's function calls without modifying it, similar to
`python -m cProfile`. By default only the script's own code is instrumented
(the standard library and installed packages are skipped) to keep overhead
low; pass `--all` to trace everything.

```text
usage: scope-profiler run [-h] [-o OUTFILE] [-q] [--all]
                          [--buffer-limit BUFFER_LIMIT]
                          script ...
```

| Flag                | Description                                                          |
| -------------------- | --------------------------------------------------------------------- |
| `-o`, `--outfile`    | Path to the merged HDF5 output file (default: `profiling_data.h5`)   |
| `-q`, `--quiet`      | Suppress the per-region summary printed after the run                |
| `--all`              | Also instrument standard-library/installed-package calls (default: only the script's own code) |
| `--buffer-limit`     | Max buffered calls per region before flushing to disk (default: 100000) |

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
                              [--sort {total,calls,avg,max,name}] [--full]
                              [--metadata-only | --regions-only]
                              files [files ...]
```

| Flag              | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `--include`       | Only report regions matching these regex patterns                       |
| `--exclude`       | Skip regions matching these regex patterns                              |
| `--ranks`         | Restrict region statistics to these ranks, e.g. `0 2` or `0-3`          |
| `--sort`          | Order regions by `total` (default), `calls`, `avg`, `max` or `name`     |
| `--full`          | Print long metadata values (`PATH`, `LD_LIBRARY_PATH`, ...) in full     |
| `--metadata-only` | Print only the metadata section                                         |
| `--regions-only`  | Print only the region statistics                                        |

```bash
scope-profiler inspect profiling_data.h5
scope-profiler inspect 'run_*.h5' --regions-only --sort calls
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
  region    ranks  calls  total [s]    avg [s]     min [s]    max [s]      std [s]
  --------------------------------------------------------------------------------
  timestep      2      8   0.139235  0.0174044   0.0165256  0.0176325  0.000338551
  solve         2      8  0.0991292  0.0123911   0.0115046  0.0125382  0.000335212
  setup         2      2  0.0473326  0.0236663   0.0222938  0.0250388   0.00137254
  assemble      2      8  0.0399496  0.0049937  0.00484729  0.0050345   5.5882e-05
  --------------------------------------------------------------------------------
  TOTAL               26   0.325647
```

Region durations are in seconds, aggregated over the selected ranks. See
{doc}`/guide/hdf5_and_visualization` for what each metadata field means.

## `scope-profiler pproc`

Post-process one or more HDF5 profiling files, generate plots, and export
aggregate region statistics to JSON.

```text
usage: scope-profiler pproc [-h] [--show] [-o OUTPUT]
                            [--include [INCLUDE ...]]
                            [--exclude [EXCLUDE ...]]
                            [--ranks [RANKS ...]]
                            [--metrics [{avg,min,max,total} ...]]
                            [--cmap CMAP]
                            [--export-data]
                            files [files ...]
```

### Positional arguments

| Argument | Description                          |
| -------- | ------------------------------------ |
| `files`  | Path(s) to `profiling_data.h5` files |

### Optional arguments

| Flag              | Description                                      |
| ----------------- | ------------------------------------------------ |
| `--show`          | Display the plot interactively (default: off)    |
| `-o`, `--output`  | Directory to save generated outputs              |
| `-i`, `--include` | Region names to include (regex patterns)         |
| `-e`, `--exclude` | Region names to exclude (regex patterns)         |
| `-r`, `--ranks`   | Ranks to include; supports ranges (e.g. `0-3,5`) |
| `-m`, `--metrics` | Duration statistics to plot: any of `avg`, `min`, `max`, `total` (default: all four) |
| `--cmap`          | Matplotlib colormap used to color regions/files in all plots (default: `tab20`) |
| `--export-data`   | Also write the exact data behind each plot as CSV (requires `-o/--output`) |

When `-o/--output` is supplied, the CLI saves:
1. `gantt_plot.png`
2. one `durations_plot_<metric>.png` per selected metric (e.g.
   `durations_plot_avg.png`, `durations_plot_min.png`, `durations_plot_max.png`,
   `durations_plot_total.png`)
3. `speedup_plot.png` (only when multiple files are passed)
4. `region_statistics.json`

Adding `--export-data` also writes the raw data behind each chart as CSV,
so plots can be reconstructed later without the original HDF5 files:
`gantt_data.csv` (file, rank, region, start/end seconds), `flame_data.csv`
(file, rank, region, depth, start/end seconds), `durations_data.csv` (file,
region, metric, value), and `speedup_data.csv` (region, rank count, speedup;
only when multiple files are passed).

For multiple files, the JSON includes per-file region statistics and the set of
common regions across all inputs.

### Examples

**Save plots for a single file:**

```bash
scope-profiler pproc profiling_data.h5 -o figures/
```

**Compare multiple files:**

```bash
scope-profiler pproc run_1.h5 run_2.h5 run_4.h5 -o figures/
```

**Select files via wildcard patterns:**

```bash
scope-profiler pproc files/*.h5 -o figures/
scope-profiler pproc "files/file_*.h5" -o figures/
```

**Display interactively with region filtering:**

```bash
scope-profiler pproc profiling_data.h5 --show \
    --include "solver.*" "rhs.*" \
    --exclude "io"
```

**Select specific MPI ranks:**

```bash
scope-profiler pproc profiling_data.h5 --show --ranks 0-3 8
```

The `--ranks` flag accepts comma-separated values and dash ranges that
can be combined: `0,2,4-7` expands to ranks 0, 2, 4, 5, 6, 7.

**Only export average and total duration plots:**

```bash
scope-profiler pproc profiling_data.h5 -o figures/ --metrics avg total
```
