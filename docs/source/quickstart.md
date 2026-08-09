# Quickstart

This page walks through the core workflow: **setup**, **instrument**,
**finalize**, and **inspect**.

## 1. Setup

Call `ProfileManager.setup()` once at the start of your program to configure
the profiling system. All regions created afterwards --- even in other
modules --- share this configuration.

```python
from scope_profiler import ProfileManager

ProfileManager.setup(
    time_trace=True,       # record start/end timestamps
    flush_to_disk=True,    # write HDF5 time-trace data
    recursive_profile=False,  # profile nested Python calls from decorators
)
```

## 2. Instrument your code

### Decorator

Use `@ProfileManager.profile` to wrap an entire function:

```python
@ProfileManager.profile("matrix_multiply")
def matrix_multiply(a, b):
    return a @ b
```

The decorator also works without an explicit name --- it uses the function
name by default:

```python
@ProfileManager.profile
def matrix_multiply(a, b):
    return a @ b
```

To include nested Python calls made by a decorated function:

```python
ProfileManager.setup(recursive_profile=True)

@ProfileManager.profile("solver_step")
def solver_step():
    return advance_state()  # nested calls are recorded automatically
```

### Context manager

Use `ProfileManager.profile_region()` for finer-grained control:

```python
for step in range(num_steps):
    with ProfileManager.profile_region("time_step"):
        evolve(state, dt)

    with ProfileManager.profile_region("io"):
        write_checkpoint(state)
```

The two styles can be mixed freely.

## 3. Finalize

Call `finalize()` when profiling is done. This writes all buffered data,
merges per-rank HDF5 files, and prints a summary:

```python
ProfileManager.finalize()
```

Output:

```text
profiling_data.h5  (1 rank(s))
  region           ranks  calls   total [s]     avg [s]     min [s]     max [s]     std [s]
  ------------------------------------------------------------------------------------------
  matrix_multiply      1    100    0.523189    0.005231    0.004912    0.006104    0.000287
  time_step            1   1000         ...         ...         ...         ...         ...
  ------------------------------------------------------------------------------------------
  TOTAL                     1100         ...
```

The same table is available from `ProfilingResults.print_summary()` and from
`scope-profiler inspect`; pass `verbose=False` to `finalize()` to suppress it.

## 4. Inspect the data

After finalization the timing data is saved to `profiling_data.h5` (default).
Use the built-in CLI to generate a Gantt chart:

```bash
scope-profiler pproc profiling_data.h5 --show
```

See {doc}`/guide/postprocessing_cli` for the other charts and exports it can
produce. Or load the data programmatically:

```python
from scope_profiler import read_h5

results = read_h5("profiling_data.h5")

# The quickest look: a summary table of every region.
results.print_summary()

# Or region by region (durations are in seconds).
for region in results:
    print(f"{region.name}: {region.num_calls} calls, "
          f"avg {region.average_duration:.4f} s")
```

## Complete example

```python
from scope_profiler import ProfileManager

ProfileManager.setup(
    time_trace=True,
    flush_to_disk=True,
)

@ProfileManager.profile("main")
def main():
    x = 0
    for i in range(10):
        with ProfileManager.profile_region("iteration"):
            x += 1

main()
ProfileManager.finalize()
```

```bash
python example.py
```

```text
profiling_data.h5  (1 rank(s))
  region     ranks  calls    total [s]      avg [s]      min [s]      max [s]  std [s]
  ------------------------------------------------------------------------------------
  main           1      1   0.00150371   0.00150371   0.00150371   0.00150371        0
  iteration      1     10    3.832e-06    3.832e-07     2.08e-07     8.75e-07      ...
  ------------------------------------------------------------------------------------
  TOTAL                11   0.00150754
```
