# API reference

## ProfileManager

The main entry point for all profiling operations. All methods are
class methods on a singleton --- there is no need to instantiate the
class.

```{eval-rst}
.. autoclass:: scope_profiler.profile_manager.ProfileManager
   :members: setup, profile, profile_region, finalize, get_region, get_all_regions, get_config, set_config
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

### NCallsOnlyProfileRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.NCallsOnlyProfileRegion
   :members:
   :undoc-members:
```

### TimeOnlyProfileRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.TimeOnlyProfileRegion
   :members:
   :undoc-members:
```

### TimeOnlyProfileRegionNoFlush

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.TimeOnlyProfileRegionNoFlush
   :members:
   :undoc-members:
```

### LikwidOnlyProfileRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.LikwidOnlyProfileRegion
   :members:
   :undoc-members:
```

### FullProfileRegion

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.FullProfileRegion
   :members:
   :undoc-members:
```

### FullProfileRegionNoFlush

```{eval-rst}
.. autoclass:: scope_profiler.region_profiler.FullProfileRegionNoFlush
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

### ProfilingH5Reader

```{eval-rst}
.. autoclass:: scope_profiler.h5reader.ProfilingH5Reader
   :members:
   :undoc-members:
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

### plot_gantt

```{eval-rst}
.. autofunction:: scope_profiler.plotting_scripts.plot_gantt
```
