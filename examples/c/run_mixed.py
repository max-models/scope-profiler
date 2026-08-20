"""
A Python driver over C kernels, profiled as one run
===================================================

The case the C API exists for. ``kernels.c`` marks regions inside itself; this
script marks regions around the calls into it. Both sides read the same clock,
so the C regions nest *inside* the Python ones and the whole thing comes out as
a single profile with one call stack.

The bridge here is ``ctypes``: the kernels and the profiler are compiled into
one shared library, which Python loads directly. No binding generator, no build
backend — just a ``.so`` and the standard library.

The pieces:

- ``kernels_start_profiling(prefix, rank)`` starts the C side, and
  ``kernels_stop_profiling()`` writes its trace. That has to happen **before**
  ``ProfileManager.finalize()``, which reads it.
- ``finalize(native_traces=...)`` folds the trace into this run's own output,
  so one ``profiling_data.h5`` holds everything.
- Region names are prefixed ``python:`` and ``c:``. A name recorded by both
  sides is refused rather than silently double-counted.

Build the library first, then run::

    make mixed
    python run_mixed.py build/libkernels.so
"""

import ctypes
import sys
from pathlib import Path

from scope_profiler import ProfileManager

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "profiling_data.h5"

GRID = 20000
STEPS = 20
SWEEPS = 5


def load_kernels(path):
    """Load the shared library and declare the signatures ctypes needs.

    ctypes assumes ``int`` for everything it is not told about, so the
    ``double`` return of the solver has to be declared or it comes back as
    garbage.
    """
    library = Path(path)
    if not library.exists():
        sys.exit(
            f"{library} does not exist.\n"
            "Run `make mixed` in this directory first (needs a C compiler)."
        )

    kernels = ctypes.CDLL(str(library))

    kernels.kernels_start_profiling.argtypes = [ctypes.c_char_p, ctypes.c_int]
    kernels.kernels_start_profiling.restype = None

    kernels.kernels_stop_profiling.argtypes = []
    kernels.kernels_stop_profiling.restype = None

    kernels.kernels_jacobi_solve.argtypes = [ctypes.c_int, ctypes.c_int]
    kernels.kernels_jacobi_solve.restype = ctypes.c_double

    kernels.kernels_checkpoint.argtypes = [ctypes.c_int]
    kernels.kernels_checkpoint.restype = None

    return kernels


def main():
    library = sys.argv[1] if len(sys.argv) > 1 else HERE / "build" / "libkernels.so"
    kernels = load_kernels(library)

    ProfileManager.setup(file_path=str(OUTPUT), label="mixed python/c")
    config = ProfileManager.get_config()
    rank, size = config._rank, config._size

    # Start the C side, telling it which rank it is so the traces of a
    # parallel run do not collide.
    kernels.kernels_start_profiling(str(HERE / "trace").encode(), rank)

    with ProfileManager.profile_region("python:setup"):
        # Nothing but Python here, to show a region with no C under it.
        total = sum(i * i for i in range(200_000))

    for step in range(STEPS):
        with ProfileManager.profile_region("python:timestep"):
            # Everything the C kernel records lands under this region.
            with ProfileManager.profile_region("python:call_solver"):
                residual = kernels.kernels_jacobi_solve(GRID, SWEEPS)

            if step % 5 == 4:
                with ProfileManager.profile_region("python:checkpoint"):
                    kernels.kernels_checkpoint(200000)

    # Write the C trace *before* finalizing, then fold it in.
    kernels.kernels_stop_profiling()
    results = ProfileManager.finalize(
        verbose=False, return_results=True, native_traces=HERE
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
        print("\n(install scope-profiler[plot] for charts)")
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
    print("the Gantt chart shows the c: bars nested inside the python: ones")


if __name__ == "__main__":
    main()
