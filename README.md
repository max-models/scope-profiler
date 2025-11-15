# scope-profiler

This module provides a unified profiling system for Python applications, with optional integration of [LIKWID](https://github.com/RRZE-HPC/likwid) markers using the [pylikwid](https://github.com/RRZE-HPC/pylikwid) marker API for hardware performance counters.

It allows you to:

- Configure profiling globally via a singleton ProfilingConfig.
- Collect timing data via context-managed profiling regions.
- Use a clean decorator syntax to profile functions.
- Optionally record time traces in HDF5 files.
- Automatically initialize and close LIKWID markers only when needed.
- Print aggregated summaries of all profiling regions.

## Install

Install from [PyPI](https://pypi.org/project/scope-profiler/):

```
pip install scope-profiler
```

## Usage

To set up the configuration, create an instance of `ProfilingConfig` and add it to the `ProfileManager`, this should be done once at application startup and will persist until the program exits or is explicitly finalized (see below). Note that the config applies to any profiling contexts created (even in other files) after it has been initialized.

```python
from scope_profiler import ProfileManager

# Setup global profiling configuration
ProfileManager.setup(
    use_likwid=False,
    time_trace=True,
    flush_to_disk=True,
)

# Profile the main() function with a decorator
@ProfileManager.profile("main")
def main():
    x = 0
    for i in range(10):
        # Profile each iteration with a context manager
        with ProfileManager.profile_region(region_name="iteration"):
            x += 1

# Call main
main()

# Finalize profiler
ProfileManager.finalize()
```

Execution:

```bash
❯ python test.py
Region: main
  Total Calls : 1
  Total Time  : 0.001503709 s
  Avg Time    : 0.001503709 s
  Min Time    : 0.001503709 s
  Max Time    : 0.001503709 s
  Std Dev     : 0.0 s
----------------------------------------
Region: iteration
  Total Calls : 10
  Total Time  : 3.832e-06 s
  Avg Time    : 3.832e-07 s
  Min Time    : 2.08e-07 s
  Max Time    : 8.75e-07 s
  Std Dev     : 2.2431888016838885e-07 s
----------------------------------------
```