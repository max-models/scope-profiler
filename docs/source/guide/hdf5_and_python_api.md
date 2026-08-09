# HDF5 output & post-processing from Python

When `flush_to_disk=True` (the default), scope-profiler writes timing
data into HDF5 files and merges them on `finalize()`.

This page covers the file layout and the Python API for reading and plotting
it. The same charts are available from the command line without writing any
code --- see {doc}`/guide/postprocessing_cli`.

## HDF5 file structure

The merged output file (default: `profiling_data.h5`) has the following
layout:

```text
profiling_data.h5
├── metadata/                  (attributes describing the run)
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
- Runs with `use_likwid=True` additionally get a `rank<N>/likwid/` group
  holding the hardware counters and derived metrics of every marker region.
  See {doc}`likwid`.

## Run metadata

Every file records the environment it was produced in, as attributes on the
`metadata` group. This is what lets you tell two otherwise identical runs
apart months later.

Derived fields use lower-case names:

| Field | Description |
| --- | --- |
| `timestamp` | ISO-8601 time the run started |
| `start_time_ns` | run start on the `perf_counter_ns` clock, the origin of the relative timeline |
| `user`, `hostname` | who ran it, and where |
| `platform`, `uname` | OS description, and the full `uname` tuple |
| `chip_information` | CPU model (from `/proc/cpuinfo` or `sysctl`) |
| `python_version` | interpreter version |
| `scope_profiler_version` | version of this package |
| `working_directory` | directory the run started in |
| `omp_num_threads`, `mpi_size`, `total_cores` | parallelism, usable as scaling-plot x-axes |
| `modules` | loaded environment modules, as a **list of strings** |

Captured environment variables keep their own upper-case names and are
present only when set:

- `PATH`, `LD_LIBRARY_PATH`, `VIRTUAL_ENV`
- `LOADEDMODULES`, `MODULEPATH`, `MODULESHOME`, `MODULES_CMD`,
  `MODULES_RUN_QUARANTINE`
- `PYTHON_HOME`, `PYTHON_INC`, `PYTHON_INCLUDE`, `PYTHON_LIB`
- every `SLURM_*` / `SLURMD_*` variable the batch system exported, so a run
  can be traced back to its job

```python
from scope_profiler.h5reader import ProfilingH5Reader

metadata = ProfilingH5Reader("profiling_data.h5").metadata

print(metadata["chip_information"])   # 'AMD EPYC 9654 96-Core Processor'
print(metadata["modules"])            # ['profile/base', 'gcc/12.3.0', ...]
print(metadata.get("SLURM_JOB_ID"))   # '1234567', or None outside a job
```

Values longer than 60 000 characters are truncated with a trailing
`...[truncated]`, since HDF5 attributes cannot exceed 64 KB.

Metadata is collected on every rank but only rank 0's copy is stored, so it
describes the run as a whole. Per-task values such as `SLURM_PROCID` reflect
rank 0.

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
          f"avg {r0.average_duration:.6f} s")
```

Durations and timestamps on `Region` and `MPIRegion` are reported in
**seconds**, converted from the nanoseconds stored in the file.

### Filtering regions

`get_regions()` accepts `include` and `exclude` patterns (Python regex):

```python
# Only regions whose name starts with "solver"
reader.get_regions(include="solver.*")

# Everything except IO regions
reader.get_regions(exclude="io.*")
```

### Post-processing in the script that produced the data

`ProfileManager.read_results()` opens the file the current configuration just
wrote, so a run can analyse itself without repeating the path:

```python
ProfileManager.finalize()
reader = ProfileManager.read_results()
reader.print_summary()
```

Under MPI only rank 0 writes the merged file, so guard the call accordingly.

### Getting the results without touching disk

`finalize(return_results=True)` hands back the run's data directly from the
in-memory buffers, with no file to write and read back:

```python
results = ProfileManager.finalize(return_results=True)

results.print_summary()
df = results.to_dataframe()
plot_gantt(results)
```

The returned `ProfilingResults` is exactly what `ProfilingH5Reader` is — the
reader is that class loaded from a file — so every method on this page, every
`plot_*` function and every exporter accepts it.

This works with `flush_to_disk=False`, where no timing data is written at all.
Under MPI the per-rank data is gathered on rank 0, which is collective: every
rank must pass `return_results=True`, and only rank 0 gets the results back
(the others get `None`), mirroring the merged file.

## Building your own plots and analyses

The built-in charts cover the common cases; when you want something else,
work from the raw calls rather than the aggregates.

### One row per call

`events()` returns the long-form ("tidy") view: one entry per recorded call,
with `name`, `rank`, `call_index`, `start`, `end` and `duration` in seconds.
Timestamps are measured from the first region entry in the file, so the
timeline starts at zero and is directly plottable — pass `relative=False` for
the raw monotonic-clock values.

```python
import matplotlib.pyplot as plt

reader = ProfilingH5Reader("profiling_data.h5")

for event in reader.events(include="solver.*", ranks=0):
    plt.barh(event["name"], event["duration"], left=event["start"])
```

`to_events_dataframe()` returns the same data as a pandas DataFrame, which is
usually the shortest path to a custom chart:

```python
events = reader.to_events_dataframe()

# Which region has the most variable calls?
events.groupby("name")["duration"].std().sort_values(ascending=False)

# Per-rank load imbalance in one line
events.pivot_table(index="rank", columns="name", values="duration", aggfunc="sum")

# Distribution of a single region's call durations
events.query("name == 'solve'")["duration"].hist(bins=50)
```

The same filters apply as everywhere else: `include`/`exclude` regexes and
`ranks`. Regions profiled with `time_trace=False` record only a call count
and so contribute no events.

Individual `Region` and `MPIRegion` objects expose the same view for a single
region (`reader["solve"].events()`), and `Region` also offers the stored
integer nanoseconds via `start_times_ns`, `end_times_ns` and `durations_ns`
for anyone who wants to avoid the float conversion.

### Useful timeline anchors

`reader.minimum_start_time`, `reader.maximum_end_time` and `reader.time_span`
bound the profiled window in seconds — handy for normalising axes or
computing what fraction of the run a region accounts for:

```python
frame = reader.to_dataframe()
frame["fraction_of_run"] = frame["total_duration"] / reader.time_span
```

`reader.run_start_time` is when the run itself started, as registered by
`ProfileManager.setup()`, and `reader.startup_time` is the gap from there to
the first region — the time the instrumentation never saw:

```python
print(f"{reader.startup_time:.3f} s elapsed before the first region was entered")
```

### Which zero the timeline uses

`events()` and `call_stack()` measure from `reader.time_origin`: the
registered start time when the file has one, and the first region entry
otherwise. Two ways to override it:

```python
reader.events(relative=False)                        # raw clock timestamps
reader.events(origin=reader.minimum_start_time)      # zero on the first region
```

The `plot_*` functions are the exception: they frame the x axis on the first
region entry, so that a long gap between `setup()` and the first region does
not fill a chart with empty space. The second line above reproduces exactly
what a chart's axis shows.

Files that carry no start time — anything written before `setup()` began
recording one — need no special handling anywhere: `run_start_time` is `None`,
`time_origin` falls back to the first region entry, `startup_time` is `0.0`,
and every reader method, export and chart behaves exactly as it did before.

### Walking the reconstructed call stack

`call_stack()` recovers the nesting the flame graph draws, as plain dicts you
can render however you like. Each call carries its `depth` and the index of
its `parent` in the returned list:

```python
from scope_profiler import call_stack_children, call_stack_roots

calls = reader.call_stack(rank=0)

for call in calls:
    print(f"{'  ' * call['depth']}{call['name']}: {call['duration']:.6f} s")

# Or walk it as a tree
children = call_stack_children(calls)
for root in call_stack_roots(calls):
    print(calls[root]["name"], "has", len(children[root]), "direct children")
```

Calls are identified by position rather than by name, because a region that
is called repeatedly — or recursively — contributes several entries under one
name.

```{note}
Everything below has a command-line equivalent that needs no code:
`scope-profiler pproc profiling_data.h5 -o figures/` writes the same charts
plus a statistics JSON. See {doc}`/guide/postprocessing_cli` for a worked
walkthrough with example figures, and {doc}`/cli` for the flag reference.
```

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

## Flame graph from Python

```python
from scope_profiler.plotting_scripts import plot_flame

plot_flame(reader, ranks=[0], filepath="flame.png", show=True)
```

The call stack is reconstructed from timestamp containment: a region whose
interval falls inside another's becomes its child. Unlike the Gantt chart,
the flame graph draws one panel per rank, defaulting to rank 0 only.

## Duration over time from Python

```python
from scope_profiler.plotting_scripts import plot_duration_timeseries

plot_duration_timeseries(reader, filepath="duration_timeseries.png", show=True)
```

One line per region tracks the mean duration of each call over wall-clock
time, shaded between the minimum and maximum across the selected ranks, so
rank imbalance and drift over the run become visible.

## Statistics JSON from Python

```python
from scope_profiler.plotting_scripts import (
    collect_region_statistics,
    write_region_statistics_json,
)

stats = collect_region_statistics(readers)                    # dict, nothing written
stats = write_region_statistics_json(readers, "stats.json")   # same dict, and a file
```

Both return per-file, per-region aggregates (`count`, `average`, `min`,
`max`, `std`, `total`, all in seconds), per-rank statistics for each region,
and the region names common to all inputs. This is the same document
`scope-profiler pproc -o ...` writes as `region_statistics.json`.

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
