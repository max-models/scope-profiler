# Configuration

All profiling behaviour is controlled through `ProfileManager.setup()`.
This must be called **once** before any regions are created. The
configuration is global: every call to `profile()` or `profile_region()`
--- even from other modules --- uses the active settings.

## `ProfileManager.setup()` parameters

| Parameter             | Type   | Default               | Description                                                                                     |
| --------------------- | ------ | --------------------- | ----------------------------------------------------------------------------------------------- |
| `profiling_activated` | `bool` | `True`                | Master switch. When `False`, all regions become no-ops with near-zero cost.                     |
| `use_likwid`          | `bool` | `False`               | Wrap regions with LIKWID marker API calls for hardware counter collection. Requires `pylikwid`. |
| `use_line_profiler`   | `bool` | `False`               | Enable line-by-line profiling via `line_profiler`. See {doc}`line_profiler`.                    |
| `time_trace`          | `bool` | `True`                | Record nanosecond start/end timestamps for every call.                                          |
| `flush_to_disk`       | `bool` | `True`                | Periodically flush timing buffers to per-rank HDF5 files.                                       |
| `buffer_limit`        | `int`  | `100_000`             | Number of events buffered in memory before an automatic flush.                                  |
| `file_path`           | `str`  | `"profiling_data.h5"` | Output path for the merged HDF5 file written by `finalize()`.                                   |

## Profiling modes

The combination of flags determines which internal region class is used.
This **strategy dispatch** avoids runtime conditionals in the hot path:

| `time_trace` | `use_likwid` | `flush_to_disk` | Region class                   | What it records                  |
| :----------: | :----------: | :-------------: | ------------------------------ | -------------------------------- |
|      --      |      --      |       --        | `DisabledProfileRegion`        | Nothing (profiling off)          |
|      no      |      no      |       --        | `NCallsOnlyProfileRegion`      | Call count only                  |
|     yes      |      no      |       no        | `TimeOnlyProfileRegionNoFlush` | Timestamps (in-memory)           |
|     yes      |      no      |       yes       | `TimeOnlyProfileRegion`        | Timestamps + HDF5                |
|      no      |     yes      |       --        | `LikwidOnlyProfileRegion`      | LIKWID markers only              |
|     yes      |     yes      |       no        | `FullProfileRegionNoFlush`     | Timestamps + LIKWID (in-memory)  |
|     yes      |     yes      |       yes       | `FullProfileRegion`            | Timestamps + LIKWID + HDF5       |
|      --      |      --      |       --        | `LineProfilerRegion`           | Timestamps + HDF5 + line-by-line |

When `use_line_profiler=True` it takes precedence over the other
combinations.

## Toggling profiling at runtime

Because the configuration is a singleton, you can leave all instrumentation
in place and simply flip the master switch:

```python
import os
from scope_profiler import ProfileManager

ProfileManager.setup(
    profiling_activated=os.environ.get("ENABLE_PROFILING", "0") == "1",
)
```

When `profiling_activated=False`, every region is a `DisabledProfileRegion`
whose `__enter__` / `__exit__` / `wrap` are trivial no-ops, adding only
the cost of a Python function call (~45 ns).

## Re-configuring

Calling `setup()` again resets all existing regions and applies the new
configuration:

```python
ProfileManager.setup(time_trace=True)
# ... profile some code ...
ProfileManager.finalize()

# Start a fresh session with different settings
ProfileManager.setup(time_trace=False)
# ...
ProfileManager.finalize()
```
