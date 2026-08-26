"""Record and plot a call graph using explicit call and parent ids.

Run::

    python examples/ex_callgraph.py
    scope-profiler plot callgraph profiling_data.h5 -o callgraph.png
"""

from pathlib import Path

from scope_profiler import ProfileManager, read_h5
from scope_profiler.plotting_scripts import plot_callgraph


@ProfileManager.profile("pic_simulation")
def run_pic_simulation(steps=1, particles=32):
    """A compact particle-in-cell-like workload with nested phases."""
    positions = [index / particles for index in range(particles)]
    velocities = [0.1 * ((index % 7) - 3) for index in range(particles)]
    fields = initialize_fields(32)
    for step in range(steps):
        charge = deposit_charge(positions, len(fields))
        fields = solve_fields(charge)
        velocities = gather_fields_and_push(positions, velocities, fields)
        diagnostics(step, positions, velocities, fields)
    return positions, velocities


@ProfileManager.profile("initialize_fields")
def initialize_fields(grid_size):
    return [0.0] * grid_size


@ProfileManager.profile("deposit_charge")
def deposit_charge(positions, grid_size):
    charge = [0.0] * grid_size
    for position in positions:
        charge[min(grid_size - 1, int(position * grid_size))] += 1.0
    return charge


@ProfileManager.profile("solve_fields")
def solve_fields(charge):
    potential = [0.0] * len(charge)
    for _ in range(3):
        potential = smooth_field(charge, potential)
    return apply_field_boundary(potential)


@ProfileManager.profile("smooth_field")
def smooth_field(charge, potential):
    return [
        0.5 * (charge[index] + potential[index])
        for index in range(len(charge))
    ]


@ProfileManager.profile("apply_field_boundary")
def apply_field_boundary(potential):
    if potential:
        potential[0] = potential[-1] = 0.0
    return potential


@ProfileManager.profile("gather_fields_and_push")
def gather_fields_and_push(positions, velocities, fields):
    for index, position in enumerate(positions):
        field = fields[min(len(fields) - 1, int(position * len(fields)))]
        velocities[index] += 0.01 * field
        positions[index] = (position + velocities[index] * 0.01) % 1.0
    return velocities


@ProfileManager.profile("diagnostics")
def diagnostics(step, positions, velocities, fields):
    # Representative reduction/output preparation, kept in memory for the example.
    return collect_moments(positions, velocities), field_energy(fields), step


@ProfileManager.profile("collect_moments")
def collect_moments(positions, velocities):
    return sum(positions), sum(velocities)


@ProfileManager.profile("field_energy")
def field_energy(fields):
    return sum(value * value for value in fields)


output = Path("profiling_data.h5")
with ProfileManager.session(
    file_path=str(output), use_likwid=False, return_results=True
) as run:
    run_pic_simulation()

print("call graph nodes:", len(run.results.call_graph()))
plot_callgraph(run.results, filepath="callgraph.png", verbose=False)

# The same explicit graph is available after reloading the profile.
print("reloaded root:", read_h5(output).call_graph()[0])
