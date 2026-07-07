# CLI reference

## `scope-profiler-pproc`

Post-process one or more HDF5 profiling files, generate a Gantt chart, and export
aggregate region statistics to JSON.

```text
usage: scope-profiler-pproc [-h] [--show] [-o OUTPUT]
                            [--include [INCLUDE ...]]
                            [--exclude [EXCLUDE ...]]
                            [--ranks [RANKS ...]]
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

When `-o/--output` is supplied, the CLI saves:
1. `gantt_plot.png`
2. `region_statistics.json`

For multiple files, the JSON includes per-file region statistics and the set of
common regions across all inputs.

### Examples

**Save plots for a single file:**

```bash
scope-profiler-pproc profiling_data.h5 -o figures/
```

**Compare multiple files:**

```bash
scope-profiler-pproc run_1.h5 run_2.h5 run_4.h5 -o figures/
```

**Select files via wildcard patterns:**

```bash
scope-profiler-pproc files/*.h5 -o figures/
scope-profiler-pproc "files/file_*.h5" -o figures/
```

**Display interactively with region filtering:**

```bash
scope-profiler-pproc profiling_data.h5 --show \
    --include "solver.*" "rhs.*" \
    --exclude "io"
```

**Select specific MPI ranks:**

```bash
scope-profiler-pproc profiling_data.h5 --show --ranks 0-3 8
```

The `--ranks` flag accepts comma-separated values and dash ranges that
can be combined: `0,2,4-7` expands to ranks 0, 2, 4, 5, 6, 7.
