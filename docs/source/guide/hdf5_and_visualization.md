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

## Gantt chart CLI

The `scope-profiler-pproc` command generates a Gantt chart from an HDF5
file:

```bash
# Save to a directory
scope-profiler-pproc profiling_data.h5 -o figures/

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
with bars spanning each recorded start-to-end interval.
