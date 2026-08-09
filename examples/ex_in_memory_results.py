"""
Post-processing straight from memory, serial or under MPI
=========================================================

``ProfileManager.finalize(return_results=True)`` hands back the run's data as a
``ProfilingResults`` --- the same post-processing API ``ProfilingH5Reader``
gives you, but built from the in-memory buffers instead of by reading a file
back. Here it is used with ``flush_to_disk=False``, so no timing data is
written to disk at all (the output file is left holding just the run metadata)
and the summary and figures are produced from memory.

The same script runs unchanged on any number of ranks. There is deliberately
no ``if rank == 0`` anywhere below: under MPI the per-rank timings are gathered
on rank 0, and the other ranks get an empty result set for which
``print_summary()``, the ``plot_*`` functions and the exporters do nothing. So
the summary is printed once and each figure is written once, by rank 0.

The one thing to keep in mind: ``finalize(return_results=True)`` is collective.
Every rank must call it, which happens naturally as long as the call is not
itself hidden behind a rank guard.

Run::

    python examples/ex_in_memory_results.py
    mpirun -n 4 python examples/ex_in_memory_results.py
"""

import math
from pathlib import Path

from scope_profiler import ProfileManager, plot_durations, plot_gantt

OUTPUT_DIR = Path("figures")


@ProfileManager.profile("assemble")
def assemble(size):
    """Stand-in for the expensive setup phase of a solver."""
    return [math.sin(i) * math.cos(i) for i in range(size)]


@ProfileManager.profile("solve")
def solve(values):
    """Stand-in for one solver iteration."""
    return sum(math.sqrt(abs(value)) + math.log1p(abs(value)) for value in values)


def simulate(num_iterations=5, size=20_000):
    """Run a few iterations of the toy 'solver', profiling each phase."""
    with ProfileManager.profile_region("simulation"):
        values = assemble(size)
        for _ in range(num_iterations):
            with ProfileManager.profile_region("iteration"):
                solve(values)
    return values


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # flush_to_disk=False: no timing data goes to disk, the results come back
    # from memory instead. Drop it (or set it to True) to get a full HDF5 file
    # as well; everything below works the same either way.
    ProfileManager.setup(
        flush_to_disk=False, file_path=str(OUTPUT_DIR / "in_memory_example.h5")
    )

    simulate()

    # Collective: every rank calls it, rank 0 ends up holding the whole run.
    # verbose=False because we print the table ourselves, just below.
    results = ProfileManager.finalize(verbose=False, return_results=True)

    # Prints once, on rank 0.
    results.print_summary(title=f"In-memory results ({results.num_ranks} rank(s))")

    # The full post-processing API is available without a file: aggregates,
    # per-call events, the reconstructed call stack, pandas frames. On the
    # ranks that hold nothing these simply iterate over nothing.
    for region in results.get_regions(include="iteration|solve"):
        print(
            f"  {region.name:<12} {region.num_calls:>3} calls, "
            f"{region.total_duration:.4f} s total"
        )

    events = results.events(include="iteration")
    for event in sorted(events, key=lambda event: -event["duration"])[:3]:
        print(
            f"  slow iteration: rank {event['rank']}, "
            f"call {event['call_index']}, {event['duration']:.4f} s"
        )

    # Each figure is written once, by rank 0. No guard needed: the other ranks
    # hold a non-root result set, and these calls return immediately for it.
    plot_gantt(results, filepath=str(OUTPUT_DIR / "in_memory_gantt.png"))
    plot_durations(results, filepath=str(OUTPUT_DIR / "in_memory_durations.png"))


if __name__ == "__main__":
    main()
