"""
Two runs to compare in an HTML report
=====================================

This example profiles the same toy solver twice and writes one profiling file
per case, ready for ``scope-profiler report``:

``baseline``
    A pure-Python kernel, a preconditioner rebuilt every step, and - under
    MPI - work split unevenly, so higher ranks do more of it.

``optimized``
    A NumPy kernel, a preconditioner built once and reused (a new region,
    while the old one is gone), balanced work, and an extra checksum in the
    output step: a small regression, so the comparison has one to show.

Each case alone gets a full report (hotspots, region table,
charts, and load balance across ranks under MPI); both together get a
comparison report, which also builds and links the two full reports. The
commands are printed at the end.

Run::

    python examples/ex_html_report.py
    mpirun -n 4 python examples/ex_html_report.py   # adds load-balance views
"""

import math
from pathlib import Path

import numpy as np

from scope_profiler import ProfileManager

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:  # serial run without mpi4py
    COMM = None

RANK = COMM.rank if COMM is not None else 0
SIZE = COMM.size if COMM is not None else 1

OUTPUT_DIR = Path("report_example")
NUM_STEPS = 8
SIZE_PER_RANK = 40_000


def work_size(balanced):
    """Elements this rank handles: even when balanced, skewed when not."""
    if balanced or SIZE == 1:
        return SIZE_PER_RANK
    # Ranks 0..N-1 get 0.5x..1.5x of the average share.
    return int(SIZE_PER_RANK * (0.5 + RANK / (SIZE - 1)))


def assemble(n):
    with ProfileManager.region("assemble"):
        values = [math.sin(i) * math.cos(i) for i in range(n)]
        with ProfileManager.region("assemble:boundary"):
            values[0] = values[-1] = 0.0
        return values


def kernel_python(values):
    with ProfileManager.region("kernel"):
        return sum(math.sqrt(abs(v)) + math.log1p(abs(v)) for v in values)


def kernel_numpy(values):
    with ProfileManager.region("kernel"):
        array = np.asarray(values)
        return float(np.sum(np.sqrt(np.abs(array)) + np.log1p(np.abs(array))))


def build_preconditioner(n):
    return [1.0 / (1.0 + (i % 17)) for i in range(n)]


def write_output(values, checksum):
    with ProfileManager.region("output"):
        text = "\n".join(f"{value:.12e}" for value in values[:5_000])
        if checksum:
            # The optimized case's regression: verify part of what it writes.
            sum(ord(char) for char in text[:40_000])
        return len(text)


def simulate(optimized):
    n = work_size(balanced=optimized)
    kernel = kernel_numpy if optimized else kernel_python
    cached = None
    for _ in range(NUM_STEPS):
        with ProfileManager.region("timestep"):
            values = assemble(n)
            with ProfileManager.region("solve"):
                if optimized:
                    if cached is None:
                        with ProfileManager.region("precond:build_once"):
                            cached = build_preconditioner(n)
                else:
                    with ProfileManager.region("precond"):
                        build_preconditioner(n)
                kernel(values)
            write_output(values, checksum=optimized)
    if COMM is not None and SIZE > 1:
        # Ranks with less work wait here for the others.
        with ProfileManager.region("wait"):
            COMM.Barrier()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    paths = {}
    for case, optimized in (("baseline", False), ("optimized", True)):
        paths[case] = OUTPUT_DIR / f"{case}.h5"
        # The label names the run in the report instead of the file stem.
        with ProfileManager.session(
            file_path=str(paths[case]), label=case, verbose=False
        ):
            simulate(optimized)

    if RANK == 0:
        baseline, optimized = paths["baseline"], paths["optimized"]
        print(f"Wrote {baseline} and {optimized}\n")
        print("Compare the two runs (also writes and links a report per run):")
        print(
            f"  scope-profiler report {baseline} {optimized} "
            f"-o {OUTPUT_DIR / 'comparison.html'} --show\n"
        )
        print("Or open one run on its own:")
        print(
            f"  scope-profiler report {baseline} -o {OUTPUT_DIR / 'baseline.html'} --show"
        )


if __name__ == "__main__":
    main()
