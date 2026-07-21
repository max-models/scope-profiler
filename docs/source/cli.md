# CLI reference

Both subcommands live under the single `scope-profiler` executable (also
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

When `-o/--output` is supplied, the CLI saves:
1. `gantt_plot.png`
2. one `durations_plot_<metric>.png` per selected metric (e.g.
   `durations_plot_avg.png`, `durations_plot_min.png`, `durations_plot_max.png`,
   `durations_plot_total.png`)
3. `speedup_plot.png` (only when multiple files are passed)
4. `region_statistics.json`

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
