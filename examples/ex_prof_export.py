"""
Reading a run with pstats and SnakeViz
======================================

``.prof`` is the format ``cProfile`` writes and ``pstats``, SnakeViz, gprof2dot
and friends read. Exporting to it buys the whole ecosystem of viewers that
already exist for Python profiles, at the cost of what does not fit the format:
a ``.prof`` file holds one rank, and has nowhere to put LIKWID counters.

Regions carry no call graph of their own, so caller/callee relations are
reconstructed from timestamp containment --- the same reconstruction the flame
chart uses. Two shapes come out of that, and both are shown below:

1. ``call_paths=True`` (the default) keeps each distinct path as its own entry,
   named ``parent > child``, so a helper called from two phases stays two nodes
   in the tree instead of one merged bar.
2. ``call_paths=False`` merges every call of a region into a single entry, the
   way pstats keys a real ``cProfile`` run, and reports recursion as
   ``2/1``-style call counts.

There are three ways to get a ``.prof``, all shown here:

1. ``to_pstats(results)`` --- a :class:`pstats.Stats` in memory, no file.
2. ``export_prof(results, ...)`` --- write one file per rank.
3. ``scope-profiler export prof run.h5 -o figures [--no-call-paths]`` --- from
   the command line, without writing Python.

Run::

    python examples/ex_prof_export.py
    snakeviz figures/prof_example_rank0.prof
"""

import math
from pathlib import Path

from scope_profiler import ProfileManager, export_prof, load_prof, to_pstats

OUTPUT_DIR = Path("figures")


@ProfileManager.profile("assemble")
def assemble(size):
    """Stand-in for the expensive setup phase of a solver."""
    return [math.sin(i) * math.cos(i) for i in range(size)]


def reduce_values(values):
    """Called from two different phases, to show what ``call_paths`` does."""
    with ProfileManager.profile_region("reduce"):
        return sum(math.sqrt(abs(value)) for value in values)


def simulate(num_iterations=5, size=20_000):
    """Run a few iterations of the toy 'solver', profiling each phase."""
    with ProfileManager.profile_region("simulation"):
        with ProfileManager.profile_region("setup"):
            values = assemble(size)
            reduce_values(values)
        for _ in range(num_iterations):
            with ProfileManager.profile_region("iteration"):
                reduce_values(values)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with ProfileManager.session(
        file_path=str(OUTPUT_DIR / "prof_example.h5"),
        verbose=False,
        return_results=True,
    ) as run:
        simulate()

    results = run.results

    # 1. In memory. `to_pstats` returns one real pstats.Stats per (run, rank),
    # so the whole pstats API works without a file ever being written.
    stats = to_pstats(results)[("prof_example", 0)]
    print("Top regions by cumulative time:\n")
    stats.sort_stats("cumulative").print_stats(6)

    # 'reduce' is called from both phases, so with the default call_paths it
    # is two entries -- one per calling context -- rather than one merged bar.
    contexts = sorted(key[2] for key in stats.stats if key[2].endswith("reduce"))
    print("Separate entries for the two calling contexts:")
    for name in contexts:
        print(f"  {name}")
    assert len(contexts) == 2, contexts

    # The compact shape instead: one entry per region, whatever called it.
    aggregated = to_pstats(results, call_paths=False)[("prof_example", 0)]
    merged = [key[2] for key in aggregated.stats if key[2] == "reduce"]
    assert merged == ["reduce"], merged
    print("\nWith call_paths=False, the same calls merge into:", merged[0])

    # 2. Write the files. One per exported rank, since .prof has no notion of
    # ranks -- pass ranks=[...] for an MPI run.
    written = export_prof(results, OUTPUT_DIR / "prof_example.prof", verbose=False)

    # 3. Read one back. `load_prof` is just pstats.Stats(path), and gives the
    # same numbers the in-memory Stats holds.
    from_disk = load_prof(written[0])
    assert from_disk.stats == to_pstats(results)[("prof_example", 0)].stats

    print()
    for path in written:
        size = path.stat().st_size / 1024
        print(f"  {path!s:<40} {size:7.1f} KiB")

    print("\n  Open it with any pstats viewer:")
    print(f"  snakeviz {written[0]}")
    print("\n  Or export from the command line, without writing Python:")
    print("  scope-profiler export prof figures/prof_example.h5 -o figures")
    print(
        "  scope-profiler export prof figures/prof_example.h5 -o figures"
        " --no-call-paths"
    )


if __name__ == "__main__":
    main()
