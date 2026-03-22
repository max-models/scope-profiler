# CLI reference

## `scope-profiler-pproc`

Post-process an HDF5 profiling file and generate Gantt charts.

```text
usage: scope-profiler-pproc [-h] [--show] [-o OUTPUT]
                            [--include [INCLUDE ...]]
                            [--exclude [EXCLUDE ...]]
                            [--ranks [RANKS ...]]
                            file
```

### Positional arguments

| Argument | Description                          |
| -------- | ------------------------------------ |
| `file`   | Path to the `profiling_data.h5` file |

### Optional arguments

| Flag              | Description                                      |
| ----------------- | ------------------------------------------------ |
| `--show`          | Display the plot interactively (default: off)    |
| `-o`, `--output`  | Directory to save the generated plots            |
| `-i`, `--include` | Region names to include (regex patterns)         |
| `-e`, `--exclude` | Region names to exclude (regex patterns)         |
| `-r`, `--ranks`   | Ranks to include; supports ranges (e.g. `0-3,5`) |

### Examples

**Save a Gantt chart:**

```bash
scope-profiler-pproc profiling_data.h5 -o figures/
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
