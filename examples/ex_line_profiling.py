"""
Line-by-line profiling with scope-profiler
==========================================

This example shows how to use the ``line_profiler`` integration to get
per-line timing breakdowns of profiled functions.

Run::

    python examples/ex_line_profiling.py

The output will show:
  1. Per-region timing summary (total / avg / min / max / std).
  2. A line-by-line table for every profiled function, showing hits,
     time, and % time for each source line.

Two usage patterns are shown:

Decorator form (simplest)
  Decorate the function with ``@ProfileManager.profile``. The function is
  registered with line_profiler automatically.

Context manager form
  Pass the function(s) you want line-profiled via the ``functions`` keyword.
  This is useful when you cannot or do not want to modify the function
  definition::

      with ProfileManager.profile_region("region", functions=[my_func]):
          my_func()
"""

import math
import random

from scope_profiler import ProfileManager

# Enable line_profiler (time_trace and flush_to_disk default to True)
ProfileManager.setup(use_line_profiler=True)

# ---------------------------------------------------------------------------
# Pattern 1: decorator form
# ---------------------------------------------------------------------------


@ProfileManager.profile("compute")
def compute(N=50_000):
    """Some mixed math to illustrate per-line costs."""
    s = 0.0
    for _ in range(N):
        x = random.random()
        s += math.sin(x) * math.sqrt(x + 1.0)
    return s


# ---------------------------------------------------------------------------
# Pattern 2: context manager form with functions=
# ---------------------------------------------------------------------------


def allocate(N=100_000):
    """List comprehension vs. append to show line-level differences."""
    a = [i * i for i in range(N)]
    b = []
    for i in range(N):
        b.append(i * i)
    return a, b


compute()

# Pass functions=[allocate] so line_profiler knows which function to trace.
with ProfileManager.profile_region("allocate", functions=[allocate]):
    allocate()

# finalize() flushes timing data to profiling_data.h5 and prints
# both the region summaries and the line_profiler tables.
ProfileManager.finalize()
