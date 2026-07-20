# HDF5 output & visualization

When `flush_to_disk=True` (the default), scope-profiler writes timing
data into HDF5 files and merges them on `finalize()`.

## HDF5 file structure

The merged output file (default: `profiling_data.h5`) has the following
layout:

```text
profiling_data.h5
├── rank0/
│   └── regions/
│       ├── region_a/
│       │   ├── start_times   (int64, nanoseconds)
│       │   └── end_times     (int64, nanoseconds)
│       └── region_b/
│           ├── start_times
│           └── end_times
├── rank1/
│   └── regions/
│       └── ...
└── ...
```

- Each MPI rank gets its own top-level group (`rank0`, `rank1`, ...).
  For serial runs there is only `rank0`.
- Timestamps are stored as **int64 nanoseconds** from
  `time.perf_counter_ns()`.

## Reading data with `ProfilingH5Reader`

```python
from scope_profiler.h5reader import ProfilingH5Reader

reader = ProfilingH5Reader("profiling_data.h5")

# Number of MPI ranks in the file
print(reader.num_ranks)

# Get all regions (sorted by first start time)
for region in reader.get_regions():
    r0 = region[0]  # Region data for rank 0
    print(f"{region.name}: {r0.num_calls} calls, "
          f"avg {r0.average_duration/1e9:.6f} s")
```

### Filtering regions

`get_regions()` accepts `include` and `exclude` patterns (Python regex):

```python
# Only regions whose name starts with "solver"
reader.get_regions(include="solver.*")

# Everything except IO regions
reader.get_regions(exclude="io.*")
```

## Gantt chart and statistics CLI

The `scope-profiler-pproc` command generates a Gantt chart and exports region
statistics JSON from one or more HDF5 files:

```bash
# Save to a directory
scope-profiler-pproc profiling_data.h5 -o figures/

# Compare multiple scaling runs
scope-profiler-pproc run_1.h5 run_2.h5 run_4.h5 -o figures/

# Display interactively
scope-profiler-pproc profiling_data.h5 --show

# Filter regions and ranks
scope-profiler-pproc profiling_data.h5 --show \
    --include solver rhs \
    --exclude io \
    --ranks 0-3
```

See {doc}`/cli` for the full option reference.

## Gantt chart from Python

```python
from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_gantt

reader = ProfilingH5Reader("profiling_data.h5")

plot_gantt(
    profiling_data=reader,
    include=["solver.*", "rhs.*"],
    exclude=["io"],
    ranks=[0, 1],
    filepath="gantt.png",
    show=True,
)
```

The chart displays one horizontal lane per (region, rank) combination,
with bars spanning each recorded start-to-end interval. When multiple files
are provided, each file gets its own stacked subplot in the exported chart.

When `-o/--output` is used, the CLI also writes `region_statistics.json` with:
1. per-file, per-region aggregate statistics (`count`, `average`, `min`, `max`, `std`, `total`)
2. per-rank statistics for each region
3. common region names across all input files

## Comparison bar charts from Python

```python
from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_durations

readers = [
    ProfilingH5Reader("run_a.h5"),
    ProfilingH5Reader("run_b.h5"),
]

saved_paths = plot_durations(
    readers,
    filepath="durations.png",
    show=True,
)
```

Each bar chart compares matching regions across files, with bars grouped by
file when several files are provided. `plot_durations` renders a separate
figure per requested statistic — by default `avg`, `min`, `max`, and `total`
duration per call. Use the `metrics` argument to select a subset:

```python
plot_durations(
    readers,
    metrics=["avg", "total"],
    filepath="durations.png",
    show=True,
)
```

When `filepath` is given and more than one metric is plotted, the metric name
is inserted before the file extension, e.g. `durations_avg.png`,
`durations_total.png`. `plot_durations` returns the list of filepaths it
wrote (empty if `filepath` is `None`).

## Speedup graph from Python

```python
from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_speedup

readers = [
    ProfilingH5Reader("run_1.h5"),
    ProfilingH5Reader("run_2.h5"),
    ProfilingH5Reader("run_4.h5"),
]

plot_speedup(
    readers,
    filepath="speedup.png",
    show=True,
)
```

The speedup plot shows one line per scope, with MPI rank count on the x-axis
and speedup on the y-axis, derived from average per-call durations for each
matching scope. The dashed reference line shows optimal scaling relative to
the smallest rank count present in the inputs.
