"""
A strong-scaling comparison in an HTML report
=============================================

This example runs the same toy solver on 1, 2 and 4 MPI ranks and writes one
profiling file per run, ready for a comparison report. With runs of different
rank counts, ``scope-profiler report`` opens with a speedup chart and compares
each region's time per rank.

The solver has three kinds of region, so the speedup chart has something to
tell apart:

``compute``
    A fixed amount of work divided over the ranks: it scales.

``output``
    Done by rank 0 alone while the others wait: it does not scale at all.

``halo_exchange``
    A stand-in for communication whose cost grows with the rank count: it
    gets slower as ranks are added.

The script launches the runs itself through ``mpiexec``/``mpirun`` (it needs
an MPI installation and mpi4py), then prints the report command.

Run::

    python examples/ex_html_report_scaling.py                 # 1, 2 and 4 ranks
    python examples/ex_html_report_scaling.py --ranks 1 2 4 8
"""

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path("report_example_scaling")
NUM_STEPS = 8
# Elements in the whole domain, divided over the ranks at every step.
DOMAIN_SIZE = 400_000


def busy(seconds):
    """Keep the CPU busy for ``seconds``: a stand-in for real work."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def solve(comm):
    from scope_profiler import ProfileManager

    # This rank's share of the domain.
    start = DOMAIN_SIZE * comm.rank // comm.size
    stop = DOMAIN_SIZE * (comm.rank + 1) // comm.size
    for _ in range(NUM_STEPS):
        with ProfileManager.region("timestep"):
            with ProfileManager.region("compute"):
                sum(math.sin(i) * math.cos(i) for i in range(start, stop))
            with ProfileManager.region("halo_exchange"):
                busy(0.0005 * comm.size)
                comm.Barrier()
            with ProfileManager.region("output"):
                if comm.rank == 0:
                    busy(0.004)
                comm.Barrier()


def worker(path, label):
    """One run: the solver on however many ranks this process was given."""
    from mpi4py import MPI

    from scope_profiler import ProfileManager

    with ProfileManager.session(file_path=path, label=label, verbose=False):
        solve(MPI.COMM_WORLD)


def launch(rank_counts):
    """Run this script once per rank count, each under the MPI launcher."""
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        sys.exit("This example needs an MPI launcher (mpiexec or mpirun) and mpi4py.")
    version = subprocess.run(
        [launcher, "--version"], capture_output=True, text=True, check=False
    )
    # Open MPI refuses more ranks than cores unless told otherwise.
    extra = ["--oversubscribe"] if "Open MPI" in version.stdout else []

    OUTPUT_DIR.mkdir(exist_ok=True)
    paths = []
    for count in rank_counts:
        path = OUTPUT_DIR / f"ranks_{count}.h5"
        label = f"{count} rank" if count == 1 else f"{count} ranks"
        command = [launcher, *extra, "-n", str(count), sys.executable, __file__]
        subprocess.run(command + ["--worker", str(path), label], check=True)
        print(f"Wrote {path} ({label})")
        paths.append(path)

    files = " ".join(str(path) for path in paths)
    print("\nCompare the runs (also writes and links a report per run):")
    print(f"  scope-profiler report {files} -o {OUTPUT_DIR / 'scaling.html'} --show")


def main():
    parser = argparse.ArgumentParser(
        description="Profile a toy solver on several MPI rank counts."
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="rank counts to run (default: 1 2 4)",
    )
    # Internal: what each launched run executes.
    parser.add_argument(
        "--worker", nargs=2, metavar=("PATH", "LABEL"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        launch(args.ranks)


if __name__ == "__main__":
    main()
