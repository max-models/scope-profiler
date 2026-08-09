# MPI support

scope-profiler is MPI-aware out of the box. When the process is launched
under an MPI launcher and `mpi4py` is installed, the profiler
automatically detects `MPI.COMM_WORLD` and handles per-rank data
collection and merging.

## Installation

```bash
pip install "scope-profiler[mpi]"
```

## How it works

0. **Launcher detection** --- before anything MPI-related happens,
   scope-profiler checks whether this process was started by
   `mpirun`/`mpiexec`/`srun` or an equivalent launcher, by looking for the
   per-rank environment variables those launchers export
   (`OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `PMIX_RANK`, ...). If none is
   present, `mpi4py` is never imported and no MPI call is ever made --- a
   plain `python script.py` run pays nothing for MPI support, not even
   `MPI_Init`. The one exception is an application that already imported
   `mpi4py` and initialized MPI itself; then the existing communicator is
   used.

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
scope-profiler pproc profiling_data.h5 --show

# Show only ranks 0 and 2
scope-profiler pproc profiling_data.h5 --show --ranks 0 2

# Range syntax
scope-profiler pproc profiling_data.h5 --show --ranks 0-3
```

From Python:

```python
from scope_profiler import read_h5

results = read_h5("profiling_data.h5")
region = results["compute"]

# Aggregated over every rank (durations in seconds)
print(region.num_calls, region.total_duration, region.average_duration)

# Per-rank breakdowns, for spotting load imbalance
print(region.average_durations())   # {rank: seconds}
print(region.max_durations())
print(region.num_calls_per_rank())

# Or the raw per-rank data
for rank_id in region.ranks:
    print(f"Rank {rank_id}: avg = {region[rank_id].average_duration:.6f} s")
```

## Without MPI

If the run was not started by an MPI launcher, or `mpi4py` is not
installed, scope-profiler silently falls back to single-rank mode. No
code changes are needed --- the API is identical.

## Overriding the detection

If a launcher is not recognized (or you want to profile an MPI-enabled
build as if it were serial), the decision can be forced:

```python
ProfileManager.setup(use_mpi=True)   # always use MPI.COMM_WORLD
ProfileManager.setup(use_mpi=False)  # never touch MPI
```

or from the environment, without touching the code:

```bash
SCOPE_PROFILER_MPI=1 ./my_launcher python my_script.py
```

`use_mpi=True` raises `ImportError` if `mpi4py` is missing, since the
request cannot be honoured.
