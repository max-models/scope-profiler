"""
A Python driver over Fortran kernels, profiled as one run
=========================================================

The case the Fortran API exists for. ``kernels.f90`` marks regions inside
itself; this script marks regions around the calls into it. Both sides read the
same clock, so the Fortran regions nest *inside* the Python ones and the whole
thing comes out as a single profile with one call stack.

The pieces:

- ``kernels.start_profiling(prefix, rank)`` starts the Fortran side, and
  ``kernels.stop_profiling()`` writes its trace. That has to happen **before**
  ``ProfileManager.finalize()``, which reads it.
- ``finalize(fortran_traces=...)`` folds the trace into this run's own output,
  so one ``profiling_data.h5`` holds everything.
- Region names are prefixed ``python:`` and ``fortran:``. A name recorded by
  both sides is refused rather than silently double-counted.

Build the extension first, then run::

    make mixed
    python run_mixed.py

Under MPI the only change is passing the rank to ``start_profiling``: each rank
folds in its own trace, and the usual single output file comes out the end.
"""

import sys
from pathlib import Path

from scope_profiler import ProfileManager

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "profiling_data.h5"

GRID = 20000
STEPS = 20
SWEEPS = 5


def load_kernels():
    """Import the f2py extension, with a useful message if it is not built."""
    sys.path.insert(0, str(HERE))
    try:
        import kernels
    except ImportError:
        sys.exit(
            "The Fortran extension is not built.\n"
            "Run `make mixed` in this directory first "
            "(needs a Fortran compiler plus meson and ninja)."
        )
    # f2py exposes the Fortran module as an attribute of the extension.
    return kernels.kernels


def rank_and_size():
    """This process's MPI rank, without requiring MPI to be installed."""
    config = ProfileManager.get_config()
    return config._rank, config._size


def main():
    kernels = load_kernels()

    ProfileManager.setup(file_path=str(OUTPUT), label="mixed python/fortran")
    rank, size = rank_and_size()

    # Start the Fortran side, telling it which rank it is so the traces of a
    # parallel run do not collide.
    kernels.start_profiling(str(HERE / "trace"), rank)

    with ProfileManager.profile_region("python:setup"):
        # Nothing but Python here, to show a region with no Fortran under it.
        total = sum(i * i for i in range(200_000))

    for step in range(STEPS):
        with ProfileManager.profile_region("python:timestep"):
            # Everything the Fortran kernel records lands under this region.
            with ProfileManager.profile_region("python:call_solver"):
                residual = kernels.jacobi_solve(GRID, SWEEPS)

            if step % 5 == 4:
                with ProfileManager.profile_region("python:checkpoint"):
                    kernels.checkpoint(200000)

    # Write the Fortran trace *before* finalizing, then fold it in.
    kernels.stop_profiling()
    results = ProfileManager.finalize(
        verbose=False, return_results=True, fortran_traces=HERE
    )

    results.print_summary(title=f"one profile, two languages ({size} rank(s))")

    print(f"\nfinal residual: {residual:.5e}  (checksum {total})")
    print(f"wrote {OUTPUT}")

    show_call_stack(results)
    write_figures(results)


def show_call_stack(results):
    """The point of the shared clock: one tree across both languages."""
    if not results.is_root:
        return

    print("\nreconstructed call stack (rank 0, first few entries):")
    for call in results.call_stack(rank=0)[:8]:
        language = call["name"].split(":", 1)[0]
        marker = "  " * call["depth"]
        print(
            f"  {marker}{call['name']:<28} {call['duration'] * 1e3:8.3f} ms"
            f"   [{language}]"
        )


def write_figures(results):
    """Charts and exports, if the optional plotting stack is installed."""
    if not results.is_root:
        return

    try:
        from scope_profiler import export_speedscope, plot_gantt
    except ImportError:  # pragma: no cover - optional dependency
        print("\n(install scope-profiler[pproc] for charts)")
        return

    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    try:
        plot_gantt(results, filepath=str(figures / "gantt.png"), verbose=False)
        export_speedscope(
            results, str(figures / "mixed.speedscope.json"), verbose=False
        )
    except Exception as exc:  # pragma: no cover - plotting stack is optional
        print(f"\n(charts skipped: {exc})")
        return

    print(f"\nwrote {figures}/gantt.png and {figures}/mixed.speedscope.json")
    print("the Gantt chart shows the fortran: bars nested inside the python: ones")


if __name__ == "__main__":
    main()
