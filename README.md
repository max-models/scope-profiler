# scope-profiler - Python Profiling System with Optional LIKWID Integration

This module provides a unified profiling system for Python applications, with optional integration of [LIKWID](https://github.com/RRZE-HPC/likwid) markers using the [pylikwid](https://github.com/RRZE-HPC/pylikwid) marker API for hardware performance counters. It allows you to:
- Configure profiling globally via a singleton ProfilingConfig.
- Collect timing data via context-managed profiling regions.
- Use a clean decorator syntax to profile functions.
- Optionally record time traces in HDF5 files.
- Automatically initialize and close LIKWID markers only when needed.
- Print aggregated summaries of all profiling regions.

Documentation: https://max-models.github.io/scope-profiler/

## Install

```
pip install scope-profiler
```

## Usage

```python
from scope_profiler.profiling import (
    ProfilingConfig,
    ProfileManager,
)

config = ProfilingConfig(
    use_likwid=False,
    time_trace=True,
    flush_to_disk=True,
)

@ProfileManager.profile("main")
def main():
    x = 0
    for i in range(10):
        with ProfileManager.profile_region(region_name="iteration"):
            x += 1

main()    

ProfileManager.print_summary()

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
