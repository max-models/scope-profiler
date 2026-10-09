"""
A scaling comparison in an HTML report
======================================

This example runs the same toy solver on 1, 2 and 4 MPI ranks and writes one
profiling file per run, ready for a comparison report. With runs of different
rank counts, ``scope-profiler report`` opens with scaling charts and compares
each region's time per rank.

By default the runs are a *strong-scaling* study: one fixed domain is divided
over the ranks, which the report's speedup chart (``--scaling strong``) reads.
With ``--weak`` the domain grows with the rank count instead, so every rank
has the same share - a *weak-scaling* study, read by the report's
weak-scaling efficiency chart (``--scaling weak``). The profiles cannot tell
the two apart; the report shows both charts unless told which one applies.

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

The speedup chart's x-axis defaults to the rank count here. ``--speedup-x``
picks another - ``nodes``, ``threads`` (OpenMP) or ``cores`` (ranks times
threads) - and the report has a button for every axis the runs differ in.

Run::

    python examples/ex_html_report_scaling.py                 # 1, 2 and 4 ranks
    python examples/ex_html_report_scaling.py --ranks 1 2 4 8
    python examples/ex_html_report_scaling.py --weak          # weak scaling
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
# Elements in the whole domain, divided over the ranks at every step; in a
# weak-scaling run, the elements on each rank.
DOMAIN_SIZE = 400_000


def busy(seconds):
    """Keep the CPU busy for ``seconds``: a stand-in for real work."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def solve(comm, weak=False):
    from scope_profiler import ProfileManager

    # This rank's share of the domain, which grows with the ranks when weak.
    domain = DOMAIN_SIZE * comm.size if weak else DOMAIN_SIZE
    start = domain * comm.rank // comm.size
    stop = domain * (comm.rank + 1) // comm.size
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


def worker(path, label, weak=False):
    """One run: the solver on however many ranks this process was given."""
    from mpi4py import MPI

    from scope_profiler import ProfileManager

    with ProfileManager.session(file_path=path, label=label, verbose=False):
        solve(MPI.COMM_WORLD, weak=weak)


def launch(rank_counts, weak=False):
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
        path = OUTPUT_DIR / f"{'weak' if weak else 'ranks'}_{count}.h5"
        label = f"{count} rank" if count == 1 else f"{count} ranks"
        command = [launcher, *extra, "-n", str(count), sys.executable, __file__]
        command += ["--worker", str(path), label] + (["--weak"] if weak else [])
        subprocess.run(command, check=True)
        print(f"Wrote {path} ({label})")
        paths.append(path)

    files = " ".join(str(path) for path in paths)
    print("\nCompare the runs (also writes and links a report per run):")
    study = "weak" if weak else "strong"
    print(
        f"  scope-profiler report {files} -o {OUTPUT_DIR / 'scaling.html'} "
        f"--scaling {study} --show"
    )
    print(
        "Choose the scaling charts' x-axis with "
        "--speedup-x {auto,ranks,nodes,threads,cores}.",
    )


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
    parser.add_argument(
        "--weak",
        action="store_true",
        help="grow the domain with the ranks: a weak-scaling study",
    )
    # Internal: what each launched run executes.
    parser.add_argument(
        "--worker", nargs=2, metavar=("PATH", "LABEL"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker, weak=args.weak)
    else:
        launch(args.ranks, weak=args.weak)


if __name__ == "__main__":
    main()
