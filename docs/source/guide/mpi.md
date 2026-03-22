# MPI support

scope-profiler is MPI-aware out of the box. When `mpi4py` is installed,
the profiler automatically detects `MPI.COMM_WORLD` and handles
per-rank data collection and merging.

## Installation

```bash
pip install "scope-profiler[mpi]"
```

## How it works

1. **Setup** --- `ProfilingConfig` reads `COMM_WORLD` to determine rank
   and size. Rank 0 creates a shared temporary directory and broadcasts
   the path to all ranks.

2. **Recording** --- each rank writes its own per-rank HDF5 file
   (`rank_<N>.h5`) inside the shared temporary directory.

3. **Finalize** --- `ProfileManager.finalize()` calls `MPI.Barrier()`,
   then rank 0 merges all per-rank files into a single output file with
   the structure `rank<N>/regions/<name>/{start_times,end_times}`.

## Example

The code is identical to the serial case --- no MPI-specific API calls
are needed:

```python
# mpi_example.py
from scope_profiler import ProfileManager

ProfileManager.setup(
    time_trace=True,
    flush_to_disk=True,
)

@ProfileManager.profile("compute")
def compute():
    s = 0
    for i in range(100_000):
        s += i
    return s

compute()
ProfileManager.finalize()
```

Run with MPI:

```bash
mpirun -n 4 python mpi_example.py
```

The output `profiling_data.h5` will contain groups `rank0` through
`rank3`, each with their own timing data.

## Visualizing MPI results

The Gantt chart CLI and Python API support rank selection:

```bash
# Show all ranks
scope-profiler-pproc profiling_data.h5 --show

# Show only ranks 0 and 2
scope-profiler-pproc profiling_data.h5 --show --ranks 0 2

# Range syntax
scope-profiler-pproc profiling_data.h5 --show --ranks 0-3
```

From Python:

```python
from scope_profiler.h5reader import ProfilingH5Reader

reader = ProfilingH5Reader("profiling_data.h5")
region = reader.get_region("compute")

# Access per-rank data
for rank_id, rank_region in region.regions.items():
    print(f"Rank {rank_id}: avg = {rank_region.average_duration/1e9:.6f} s")
```

## Without MPI

If `mpi4py` is not installed, scope-profiler silently falls back to
single-rank mode. No code changes are needed --- the API is identical.
