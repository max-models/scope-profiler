"""
Line-by-line profiling with scope-profiler
==========================================

This example shows how to use the ``line_profiler`` integration to get
per-line timing breakdowns of profiled functions.

Requirements::

    pip install scope-profiler[line-profiler]

Run::

    python examples/ex_line_profiling.py

The output will show:
  1. Per-region timing summary (total / avg / min / max / std).
  2. A line-by-line table for every ``@ProfileManager.profile``-decorated
     function, showing hits, time, and % time for each source line.
"""

import math
import random

from scope_profiler import ProfileManager

# Enable line_profiler (time_trace and flush_to_disk default to True)
ProfileManager.setup(use_line_profiler=True)


@ProfileManager.profile("compute")
def compute(N=50_000):
    """Some mixed math to illustrate per-line costs."""
    s = 0.0
    for _ in range(N):
        x = random.random()
        s += math.sin(x) * math.sqrt(x + 1.0)
    return s


@ProfileManager.profile("allocate")
def allocate(N=100_000):
    """List comprehension vs. append to show line-level differences."""
    a = [i * i for i in range(N)]
    b = []
    for i in range(N):
        b.append(i * i)
    return a, b


# Call the profiled functions
compute()
allocate()

# finalize() flushes timing data to profiling_data.h5 and prints
# both the region summaries and the line_profiler tables.
ProfileManager.finalize()
