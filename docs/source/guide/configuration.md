

# Configuration

All profiling behaviour is controlled through `ProfileManager.setup()`.
This must be called **once** before any regions are created. The
configuration is global: every call to `profile()` or `profile_region()`
— even from other modules — uses the active settings.

## `ProfileManager.setup()` parameters

| Parameter | Type | Default | Description |
|----|----|----|----|
| `deactivate_profiling` | `bool` | `False` | Master switch. When `True`, all regions become no-ops with near-zero cost. |
| `use_likwid` | `bool` | `False` | Wrap regions with LIKWID marker API calls for hardware counter collection. Requires `pylikwid`. |
| `use_line_profiler` | `bool` | `False` | Enable line-by-line profiling via `line_profiler`. See {doc}`line_profiler`. |
| `use_nvtx` | `bool` | `False` | Add NVTX ranges for NVIDIA Nsight tools; requires `scope-profiler[nvtx]`. |
| `recursive_profile` | `bool` | `False` | Enable recursive nested-call profiling for all decorated functions by default. |
| `deactivate_file_output` | `bool` | `False` | When `True`, write no HDF5 file at all; the run stays in memory. See below. |
| `buffer_limit` | `int` | `1024` | Initial per-region buffer capacity. Buffers grow on demand, so this is a starting size, not a cap. |
| `file_path` | `str` | `"profiling_data.h5"` | Output path for the merged HDF5 file written by `finalize()`. |
| `label` | `str` | `None` | Short name for the run, used by post-processing wherever a run has to be named. See below. |
| `capture_region_source` | `bool` | `False` | Record where each region is defined (see {doc}`hdf5_and_python_api`). Off by default; see below for its cost and how to turn it on. |

## Enabling `capture_region_source`

Off by default, because its cost – while cheap for a typical file – is
not always. Capturing a region’s source parses its defining file’s AST
and walks it once per distinct file, the first time any of its regions
is created – not once per region, and not on any later call. Measured
cost tracks that file’s **total size**, not the size or number of the
regions it defines:

| File | Cost, 1 rank | Cost per rank, 64 ranks (shared, oversubscribed node) |
|----|----|----|
| Typical, a few hundred lines | \< 1 ms | a few ms |
| ~10,000 lines, ~1,000 regions (one spanning 3,000 lines) | ~0.3 s | ~2.9 s |

The per-region text itself is cheap to extract even when huge (~0.1 ms
for a 3,000-line block) – the file-wide parse dominates. Every rank pays
this independently and concurrently, so on a job with more ranks than
idle cores it compounds under contention; at rank counts within the idle
core count (1–8 ranks, above), it stays flat. For a typical, modestly
sized codebase this is negligible either way – turn it on with:

``` python
ProfileManager.setup(capture_region_source=True)
```

## Naming a run with `label`

Post-processing names a run after its output file: `run_a.h5` becomes
`run_a` in chart legends, summary headings and the JSON statistics.
`label` overrides that with something you choose:

``` python
ProfileManager.setup(file_path="run_a.h5", label="128 ranks")
```

The label is stored as metadata in the output file, so it survives into
every later step — `scope-profiler plot`, `scope-profiler inspect`, the
plotting functions and the exporters all pick it up with no extra flags.
It is especially worth setting for scaling studies, where a legend
reading `128 ranks` beats one reading `run_a`.

Reading it back, `results.label` is the label or `None`, while
`results.display_label` is the label or the file stem — what
post-processing actually prints.

## Profiling modes

Every active region records nanosecond timestamps; the remaining flags
decide what it records *on top* of them. This **strategy dispatch**
picks the region class once, at `setup()`, so there are no runtime
conditionals in the hot path:

| Flags                  | Region class            | What it records           |
|------------------------|-------------------------|---------------------------|
| `deactivate_profiling` | `DisabledProfileRegion` | Nothing (profiling off)   |
| *(defaults)*           | `TimeOnlyProfileRegion` | Timestamps                |
| `use_likwid`           | `FullProfileRegion`     | Timestamps + LIKWID       |
| `use_line_profiler`    | `LineProfilerRegion`    | Timestamps + line-by-line |
| `use_nvtx`             | `NVTXProfileRegion`     | Timestamps + NVTX ranges  |

`use_line_profiler=True` takes precedence over `use_likwid=True`.

`use_nvtx=True` adds NVTX annotations while retaining the normal CPU
timing records. NVTX does not measure GPU kernel duration itself; use
NVIDIA Nsight Systems/Compute or CUDA events for device-side timings.
`use_likwid=True` and `use_nvtx=True` are currently separate modes, with
LIKWID taking precedence.

`deactivate_file_output` is not part of this dispatch: recording is
identical either way, and the flag only decides whether `finalize()`
writes the buffers out. With `deactivate_file_output=True`, use
`finalize(return_results=True)` to get the recorded data back — see [the
Python API
guide](hdf5_and_python_api.md#getting-the-results-without-touching-disk).

## What is no longer configurable

Two things used to be options and are now decided for you, because there
was only ever one sensible answer:

- **The run’s start time** is the moment `setup()` is called. It is
  stored as the `start_time_ns` metadata field and is the origin of the
  relative timeline in post-processing.
- **MPI** is used exactly when the process was started by an MPI
  launcher (`mpirun`, `mpiexec`, `srun`, …), so a plain
  `python script.py` never imports `mpi4py`. Set `SCOPE_PROFILER_MPI=0`
  or `=1` in the environment to overrule the detection.

## Toggling profiling at runtime

Because the configuration is a singleton, you can leave all
instrumentation in place and simply flip the master switch:

``` python
import os
from scope_profiler import ProfileManager

ProfileManager.setup(
    deactivate_profiling=os.environ.get("DISABLE_PROFILING", "0") == "1",
)
```

## Recursive profiling of decorated entrypoints

Set `recursive_profile=True` to record Python function calls made inside
decorated functions:

``` python
ProfileManager.setup(recursive_profile=True)

@ProfileManager.profile("entry")
def entry():
    compute_step()
```

You can override this per function with
`@ProfileManager.profile(..., recursive=False)` or
`@ProfileManager.profile(..., recursive=True)`.

When `deactivate_profiling=True`, every region is a
`DisabledProfileRegion` whose `__enter__` / `__exit__` / `wrap` are
trivial no-ops, adding only the cost of a Python function call (~0.1
µs).

## Re-configuring

Calling `setup()` again resets all existing regions and applies the new
configuration:

``` python
ProfileManager.setup(file_path="run_a.h5")
# ... profile some code ...
ProfileManager.finalize()

# Start a fresh session with different settings
ProfileManager.setup(file_path="run_b.h5", use_line_profiler=True)
# ...
ProfileManager.finalize()
```
