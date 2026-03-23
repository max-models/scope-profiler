"""
Lazy setup: decorate before calling setup()
============================================

In real applications the profiling configuration is often not known at
import time — it may come from CLI arguments, environment variables, or a
config file that is read after the module containing the class has been
imported.

This example shows that ``@ProfileManager.profile`` and
``with ProfileManager.profile_region(...)`` both work correctly even when
``setup()`` is called *after* the class is defined and the decorated methods
are already bound.

Run::

    python examples/ex_lazy_setup.py
"""

import math
import random

from scope_profiler import ProfileManager

# -----------------------------------------------------------------------
# Class defined at import time — setup() has NOT been called yet.
# The decorators install a lightweight lazy wrapper that binds to the
# actual profiling region on the first call after setup() runs.
# -----------------------------------------------------------------------


class Solver:
    def __init__(self, n=300):
        self.n = n
        self.data = [random.random() for _ in range(n)]

    @ProfileManager.profile("smooth")
    def smooth(self, iterations=5):
        """Jacobi smoother — hot inner loop."""
        for _ in range(iterations):
            new = list(self.data)
            for i in range(1, self.n - 1):
                new[i] = 0.5 * (self.data[i - 1] + self.data[i + 1])
            self.data = new

    @ProfileManager.profile("norm")
    def norm(self):
        """Compute L2 norm."""
        total = 0.0
        for x in self.data:
            total += x * x
        return math.sqrt(total)


# -----------------------------------------------------------------------
# setup() called here — after the class is defined.
# Both decorator and with-block regions now use LineProfilerRegion.
# -----------------------------------------------------------------------

ProfileManager.setup(use_line_profiler=True)

solver = Solver(n=500)

# Decorator path — re-binds to the new LineProfilerRegion on first call
for _ in range(10):
    solver.smooth()
    solver.norm()

# with-block path — profile_region() is called at runtime so it always
# picks up the current config.  Timing is recorded for the whole block;
# line-by-line output requires decorated functions (not inline code).
with ProfileManager.profile_region("postprocess"):
    result = []
    for x in solver.data:
        result.append(math.sin(x) * math.sqrt(abs(x) + 1.0))

ProfileManager.finalize()
