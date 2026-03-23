"""
Profiling class methods with scope-profiler
============================================

This example shows how to attach ``@ProfileManager.profile`` decorators to
individual methods of a class so that each method gets its own named region
and line-by-line timing table.

Important: call ``ProfileManager.setup()`` *before* the class is defined so
that the decorators capture the correct profiling mode.

Run::

    python examples/ex_class_profiling.py
"""

import math
import random

from scope_profiler import ProfileManager

# Must be called before the class definition so decorators use LineProfilerRegion
ProfileManager.setup(use_line_profiler=True)


class FluidSimulation:
    """Toy fluid simulation with three profiled stages."""

    def __init__(self, n=200):
        self.n = n
        self.u = [random.random() for _ in range(n)]
        self.v = [random.random() for _ in range(n)]

    @ProfileManager.profile("advect")
    def advect(self, dt=0.01):
        """Advect velocity field with a simple upwind scheme."""
        n = self.n
        u_new = [0.0] * n
        for i in range(1, n):
            u_new[i] = self.u[i] - dt * self.u[i] * (self.u[i] - self.u[i - 1])
        self.u = u_new

    @ProfileManager.profile("diffuse")
    def diffuse(self, nu=0.01, dt=0.01):
        """Diffuse velocity field with an explicit Laplacian."""
        n = self.n
        u_new = list(self.u)
        for i in range(1, n - 1):
            u_new[i] = self.u[i] + nu * dt * (
                self.u[i + 1] - 2 * self.u[i] + self.u[i - 1]
            )
        self.u = u_new

    @ProfileManager.profile("compute_energy")
    def compute_energy(self):
        """Compute total kinetic energy."""
        energy = 0.0
        for i in range(self.n):
            energy += 0.5 * (self.u[i] ** 2 + self.v[i] ** 2)
        return energy

    @ProfileManager.profile("apply_forcing")
    def apply_forcing(self, dt=0.01):
        """Apply a sinusoidal body force."""
        for i in range(self.n):
            x = i / self.n
            self.u[i] += dt * math.sin(2 * math.pi * x)
            self.v[i] += dt * math.cos(2 * math.pi * x)

    def step(self, dt=0.01):
        self.advect(dt)
        self.diffuse(dt=dt)
        self.apply_forcing(dt)
        return self.compute_energy()


sim = FluidSimulation(n=500)

for _ in range(20):
    sim.step()

# finalize() prints per-region summaries and then the line-by-line tables
ProfileManager.finalize()
