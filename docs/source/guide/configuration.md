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
| `recursive_profile`   | `bool` | `False`               | Enable recursive nested-call profiling for all decorated functions by default.                    |
| `time_trace`          | `bool` | `True`                | Record nanosecond start/end timestamps for every call.                                          |
| `flush_to_disk`       | `bool` | `True`                | Write the recorded timings to per-rank HDF5 files at `finalize()`. When `False`, results stay in memory. |
| `buffer_limit`        | `int`  | `1024`                | Initial per-region buffer capacity. Buffers grow on demand, so this is a starting size, not a cap. |
| `file_path`           | `str`  | `"profiling_data.h5"` | Output path for the merged HDF5 file written by `finalize()`.                                   |

## Profiling modes

The combination of flags determines which internal region class is used.
This **strategy dispatch** avoids runtime conditionals in the hot path:

| `time_trace` | `use_likwid` | Region class              | What it records                  |
| :----------: | :----------: | ------------------------- | -------------------------------- |
|      --      |      --      | `DisabledProfileRegion`   | Nothing (profiling off)          |
|      no      |      no      | `NCallsOnlyProfileRegion` | Call count only                  |
|     yes      |      no      | `TimeOnlyProfileRegion`   | Timestamps                       |
|      no      |     yes      | `LikwidOnlyProfileRegion` | LIKWID markers only              |
|     yes      |     yes      | `FullProfileRegion`       | Timestamps + LIKWID              |
|      --      |      --      | `LineProfilerRegion`      | Timestamps + line-by-line        |

When `use_line_profiler=True` it takes precedence over the other
combinations.

`flush_to_disk` is not part of this dispatch: recording is identical either
way, and the flag only decides whether `finalize()` writes the buffers out.

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

## Recursive profiling of decorated entrypoints

Set `recursive_profile=True` to record Python function calls made inside
decorated functions:

```python
ProfileManager.setup(recursive_profile=True)

@ProfileManager.profile("entry")
def entry():
    compute_step()
```

You can override this per function with
`@ProfileManager.profile(..., recursive=False)` or
`@ProfileManager.profile(..., recursive=True)`.

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
