# scope-profiler

**A lightweight, low-overhead profiling framework for Python and HPC applications.**

scope-profiler lets you instrument code regions with decorators or context managers,
collect nanosecond-resolution timing traces, and optionally integrate
[LIKWID](https://github.com/RRZE-HPC/likwid) hardware performance counters or
[line_profiler](https://github.com/pyutils/line_profiler) line-by-line analysis ---
all through a single, unified API.

## Key features

- **Two instrumentation styles** --- `@ProfileManager.profile` decorator and
  `with ProfileManager.profile_region()` context manager.
- **Near-zero overhead** --- the default timing mode adds ~700 ns per call;
  profiling can be toggled off at startup with no code changes.
- **HDF5 time traces** --- start/end timestamps are flushed to disk and merged
  automatically, ready for post-hoc analysis and Gantt charts.
- **MPI-aware** --- per-rank data is collected and merged transparently via
  `mpi4py`.
- **LIKWID integration** --- hardware counter regions are opened/closed
  alongside timing, with no extra boilerplate.
- **Line profiler integration** --- enable `line_profiler` per-line stats on
  any decorated function with a single flag.
- **CLI post-processing** --- `scope-profiler-pproc` reads HDF5 output and
  generates Gantt charts with region/rank filtering.

## Quick example

```python
from scope_profiler import ProfileManager

ProfileManager.setup(time_trace=True, flush_to_disk=True)

@ProfileManager.profile("compute")
def compute():
    return sum(i * i for i in range(100_000))

compute()
ProfileManager.finalize()
```

## Documentation

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

guide/configuration
guide/regions
guide/hdf5_and_visualization
guide/mpi
guide/line_profiler
guide/overhead
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
cli
```

```{toctree}
:maxdepth: 2
:caption: Tutorials & examples

tutorials
```
