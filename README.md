# scope-profiler

This module provides a unified profiling system for Python applications, with optional integration of [LIKWID](https://github.com/RRZE-HPC/likwid) markers using the [pylikwid](https://github.com/RRZE-HPC/pylikwid) marker API for hardware performance counters.

It allows you to:

- Configure profiling globally via a singleton ProfilingConfig.
- Collect timing data via context-managed profiling regions.
- Use a clean decorator syntax to profile functions.
- Optionally record time traces in HDF5 files.
- Automatically initialize and close LIKWID markers only when needed.
- Print aggregated summaries of all profiling regions.

Documentation: https://max-models.github.io/scope-profiler/

## Install

Install from [PyPI](https://pypi.org/project/scope-profiler/):

```
pip install scope-profiler
```

## Usage

To set up the configuration, create an instance of `ProfilingConfig`, this should be done once at application startup and will persist until the program exits or is explicitly finalized (see below). Note that the config applies to any profiling contexts created (even in other files) after it has been initialized.

```python
from scope_profiler.profiling import (
    ProfilingConfig,
    ProfileManager,
)

# Setup global profiling configuration
config = ProfilingConfig(
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

# Print summary of profiling results
ProfileManager.print_summary()

# Finalize profiler
ProfileManager.finalize()
```

Execution:

```bash
❯ python test.py
Profiling Summary:
========================================
Region: main
  Number of Calls: 1
  Total Duration: 0.000315 seconds
  Average Duration: 0.000315 seconds
  Min Duration: 0.000315 seconds
  Max Duration: 0.000315 seconds
  Std Deviation: 0.000000 seconds
----------------------------------------
Region: iteration
  Number of Calls: 10
  Total Duration: 0.000007 seconds
  Average Duration: 0.000001 seconds
  Min Duration: 0.000000 seconds
  Max Duration: 0.000003 seconds
  Std Deviation: 0.000001 seconds
----------------------------------------
```
