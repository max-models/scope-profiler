# API reference

## ProfileManager

The main entry point for all profiling operations. All methods are
class methods on a singleton --- there is no need to instantiate the
class.

```{eval-rst}
.. autoclass:: scope_profiler.profile_manager.ProfileManager
   :members: setup, profile, profile_region, finalize, read_results, get_region, get_all_regions, get_config, set_config
   :undoc-members:
```

## ProfilingConfig

Singleton that holds the global profiling configuration. Normally you
interact with it through `ProfileManager.setup()`, but you can also
construct one directly for advanced use cases.

```{eval-rst}
.. autoclass:: scope_profiler.profile_config.ProfilingConfig
   :members:
   :undoc-members:
```

## Region classes

### BaseProfileRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.BaseProfileRegion
   :members:
   :undoc-members:
```

### DisabledProfileRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.DisabledProfileRegion
   :members:
   :undoc-members:
```

### TimeOnlyProfileRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.TimeOnlyProfileRegion
   :members:
   :undoc-members:
```

### FullProfileRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.FullProfileRegion
   :members:
   :undoc-members:
```

### LineProfilerRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.LineProfilerRegion
   :members:
   :undoc-members:
```

## Post-processing

Everything in this section is importable from the package root:

```python
from scope_profiler import (
    build_call_stack,
    read_h5,
    plot_flame,
    plot_gantt,
    write_region_statistics_json,
)
```

All durations and timestamps exposed here are in **seconds**; the HDF5 file
stores nanoseconds.

### ProfilingResults

The post-processing API itself, and the only type the analysis layer has.
`ProfilingResults.from_h5()` builds one from a merged file, and
`ProfileManager.finalize(return_results=True)` builds the same thing straight
from memory.

```{eval-rst}
.. autoclass:: scope_profiler.results.ProfilingResults
   :members:
   :undoc-members:
```

### read_h5

The module-level spelling of `ProfilingResults.from_h5()`; the two are
interchangeable.

```{eval-rst}
.. autofunction:: scope_profiler.h5reader.read_h5
```

### Native (C and Fortran) traces

Reading what the C and Fortran region APIs write -- they share one format; see
{doc}`guide/c` and {doc}`guide/fortran`.

```{eval-rst}
.. autofunction:: scope_profiler.native_trace.load_traces
.. autofunction:: scope_profiler.native_trace.convert_traces
.. autofunction:: scope_profiler.native_trace.read_trace
.. autofunction:: scope_profiler.native_trace.find_traces
.. autofunction:: scope_profiler.native_trace.write_results
.. autofunction:: scope_profiler.native_trace.fortran_source_path
.. autofunction:: scope_profiler.native_trace.c_source_path
.. autofunction:: scope_profiler.native_trace.c_include_dir
```

### Combining runs

```{eval-rst}
.. autofunction:: scope_profiler.results.merge_results
```

### Region

```{eval-rst}
.. autoclass:: scope_profiler.region.Region
   :members:
   :undoc-members:
```

### MPIRegion

```{eval-rst}
.. autoclass:: scope_profiler.mpi_region.MPIRegion
   :members:
   :undoc-members:
```

### Call stack reconstruction

Regions record no call graph, so nesting is recovered from timestamp
containment. `ProfilingResults.call_stack()` is the usual entry point; the
functions below operate on its result and let you walk the reconstructed tree
when building your own nested visualisation.

```{eval-rst}
.. autofunction:: scope_profiler.call_stack.build_call_stack
.. autofunction:: scope_profiler.call_stack.call_stack_roots
.. autofunction:: scope_profiler.call_stack.call_stack_children
```

### Plotting

```{eval-rst}
.. autofunction:: scope_profiler.plotting_scripts.plot_gantt
.. autofunction:: scope_profiler.plotting_scripts.plot_flame
.. autofunction:: scope_profiler.plotting_scripts.plot_durations
.. autofunction:: scope_profiler.plotting_scripts.plot_duration_timeseries
.. autofunction:: scope_profiler.plotting_scripts.plot_speedup
```

### Statistics export

```{eval-rst}
.. autofunction:: scope_profiler.plotting_scripts.collect_region_statistics
.. autofunction:: scope_profiler.plotting_scripts.write_region_statistics_json
```

### Exporting to other tools

```{eval-rst}
.. autofunction:: scope_profiler.speedscope_export.export_speedscope
.. autofunction:: scope_profiler.prof_export.export_prof

## LIKWID

See {doc}`guide/likwid` for the workflow and the HDF5 layout.

### LikwidRegionResult

```{eval-rst}
.. autoclass:: scope_profiler.likwid_data.LikwidRegionResult
   :members:
   :undoc-members:
```

### Summary tables

```{eval-rst}
.. autofunction:: scope_profiler.summary.likwid_tables
.. autofunction:: scope_profiler.summary.print_likwid_table
.. autofunction:: scope_profiler.summary.print_likwid_tables
```

### Collection and storage

```{eval-rst}
.. autofunction:: scope_profiler.likwid_data.collect_marker_results_isolated
.. autofunction:: scope_profiler.likwid_data.parse_marker_file
.. autofunction:: scope_profiler.likwid_data.collect_marker_results
.. autofunction:: scope_profiler.likwid_data.collect_region_snapshots
.. autofunction:: scope_profiler.likwid_data.snapshots_to_results
.. autofunction:: scope_profiler.likwid_data.write_likwid_results
.. autofunction:: scope_profiler.likwid_data.markers_available
.. autofunction:: scope_profiler.likwid_data.likwid_environment
```
