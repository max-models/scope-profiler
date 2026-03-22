# Quickstart

This page walks through the core workflow: **setup**, **instrument**,
**finalize**, and **inspect**.

## 1. Setup

Call `ProfileManager.setup()` once at the start of your program to configure
the profiling system.  All regions created afterwards --- even in other
modules --- share this configuration.

```python
from scope_profiler import ProfileManager

ProfileManager.setup(
    time_trace=True,       # record start/end timestamps
    flush_to_disk=True,    # write HDF5 time-trace data
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

Call `finalize()` when profiling is done.  This flushes all buffered data,
merges per-rank HDF5 files, and prints a summary:

```python
ProfileManager.finalize()
```

Output:

```text
Region: matrix_multiply
  Total Calls : 100
  Total Time  : 0.523189 s
  Avg Time    : 0.005231 s
  Min Time    : 0.004912 s
  Max Time    : 0.006104 s
  Std Dev     : 0.000287 s
----------------------------------------
Region: time_step
  Total Calls : 1000
  ...
```

## 4. Inspect the data

After finalization the timing data is saved to `profiling_data.h5` (default).
Use the built-in CLI to generate a Gantt chart:

```bash
scope-profiler-pproc profiling_data.h5 --show
```

Or load the data programmatically:

```python
from scope_profiler.h5reader import ProfilingH5Reader

reader = ProfilingH5Reader("profiling_data.h5")
for region in reader.get_regions():
    stats = region[0].get_summary()  # rank 0
    print(f"{region.name}: {stats['num_calls']} calls, "
          f"avg {stats['average_duration']/1e9:.4f} s")
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
Region: main
  Total Calls : 1
  Total Time  : 0.001503709 s
  ...
----------------------------------------
Region: iteration
  Total Calls : 10
  Total Time  : 3.832e-06 s
  ...
----------------------------------------
```
