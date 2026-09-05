#!/usr/bin/env python3
"""
Generate the figures shown on the "Plotting with the CLI" docs page
====================================================================

Runs a small mock solver (setup + a timestep loop with nested regions) at
1, 2 and 4 MPI ranks, then post-processes the resulting HDF5 files with
``scope-profiler plot`` --- exactly the commands quoted on the docs page ---
and copies the generated figures into ``figures/cli`` at the repo root. The
docs build copies that directory into ``docs/source/_static/figures``, where
the page references it as ``/_static/figures/cli/...``.

Run::

    python examples/generate_cli_docs_figures.py

If ``mpirun`` or ``mpi4py`` is unavailable the runs fall back to a single rank
each, and the speedup figure is skipped.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The docs build (docs/source/conf.py) copies figures/ into the Sphinx
# _static directory, so figures live here rather than under docs/.
DOCS_FIGURES = os.path.join(REPO_ROOT, "figures", "cli")

RANK_COUNTS = (1, 2, 4)

# Figures copied into the docs, generated from the 2-rank run.
SINGLE_RUN_FIGURES = (
    "gantt_plot.png",
    "flame_plot.png",
    "durations_plot.png",
    "duration_timeseries_plot.png",
)


def run_workload(h5_path: str) -> None:
    """Mock solver: work per rank shrinks with the rank count, imperfectly."""
    from scope_profiler import ProfileManager

    with ProfileManager.session(file_path=h5_path, verbose=False):
        try:
            from mpi4py import MPI

            size = MPI.COMM_WORLD.Get_size()
        except ImportError:
            size = 1

        # Strong scaling with a serial fraction, so the speedup plot bends away
        # from the ideal line instead of sitting on top of it.
        def scaled(seconds: float, serial_fraction: float = 0.15) -> float:
            return seconds * (serial_fraction + (1.0 - serial_fraction) / size)

        with ProfileManager.profile_region("setup"):
            time.sleep(scaled(0.060))

        for step in range(3):
            with ProfileManager.profile_region("timestep"):
                with ProfileManager.profile_region("assemble"):
                    time.sleep(scaled(0.025))
                with ProfileManager.profile_region("solve"):
                    time.sleep(scaled(0.060))
                with ProfileManager.profile_region("halo_exchange"):
                    # Communication grows with the rank count.
                    time.sleep(0.004 * size)
                if step % 2 == 1:
                    with ProfileManager.profile_region("io"):
                        time.sleep(0.012)


def generate_h5_files(work_dir: str) -> tuple[list[str], bool]:
    """Produce one HDF5 file per rank count, with mpirun when available."""
    mpirun = shutil.which("mpirun")
    mpi_available = (
        mpirun is not None and importlib.util.find_spec("mpi4py") is not None
    )
    paths = []

    # HDF5 file locking can fail on NFS-mounted filesystems (errno 11).
    env = {**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"}

    for num_ranks in RANK_COUNTS:
        h5_path = os.path.join(work_dir, f"run_{num_ranks}.h5")
        cmd = [sys.executable, os.path.abspath(__file__), "--workload", h5_path]
        if mpi_available and num_ranks > 1:
            cmd = [mpirun, "-n", str(num_ranks), *cmd]
        elif num_ranks > 1:
            print(
                "mpirun/mpi4py not available - running the "
                f"{num_ranks}-rank case serially"
            )
        subprocess.run(cmd, check=True, cwd=work_dir, env=env)
        paths.append(h5_path)

    return paths, mpi_available


def plot(args: list[str], work_dir: str) -> None:
    """Invoke the plot CLI the same way the docs page does."""
    print("$ scope-profiler plot " + " ".join(args))
    subprocess.run(
        [sys.executable, "-m", "scope_profiler", "plot", *args],
        check=True,
        cwd=work_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        metavar="H5_PATH",
        help="Internal: run the mock solver and write this HDF5 file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DOCS_FIGURES,
        help=f"Directory to copy the figures into (default: {DOCS_FIGURES})",
    )
    parser.add_argument(
        "--keep",
        metavar="DIR",
        help="Keep the HDF5 files and raw CLI output in this directory",
    )
    args = parser.parse_args()

    if args.workload:
        run_workload(args.workload)
        return

    os.makedirs(args.output, exist_ok=True)
    work_dir = args.keep or tempfile.mkdtemp(prefix="scope-profiler-docs-")
    os.makedirs(work_dir, exist_ok=True)

    try:
        _, mpi_available = generate_h5_files(work_dir)
        figures_dir = os.path.join(work_dir, "figures")

        # The default preset is intentionally small: Gantt plus total duration.
        plot(["default", "run_2.h5", "-o", "figures"], work_dir)

        # Generate the additional single-run figures referenced by the guide.
        plot(
            ["flame_chart", "run_2.h5", "-o", "figures/flame_plot.png"],
            work_dir,
        )
        plot(
            [
                "timeseries",
                "run_2.h5",
                "-o",
                "figures/duration_timeseries_plot.png",
            ],
            work_dir,
        )

        # Also produce an avg plot for the docs illustration of --metrics.
        plot(
            [
                "durations",
                "run_2.h5",
                "-o",
                "figures_avg",
                "--metrics",
                "avg",
            ],
            work_dir,
        )

        # Several runs: comparison preset, then the explicit speedup figure.
        plot(
            ["default", "run_1.h5", "run_2.h5", "run_4.h5", "-o", "figures_scaling"],
            work_dir,
        )
        if mpi_available:
            plot(
                [
                    "speedup",
                    "run_1.h5",
                    "run_2.h5",
                    "run_4.h5",
                    "-o",
                    "figures_scaling/speedup_plot.png",
                ],
                work_dir,
            )

        copied = []
        for name in SINGLE_RUN_FIGURES:
            src = os.path.join(figures_dir, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.output, name))
                copied.append(name)

        # Copy the total durations chart under the _total suffix for the docs
        # page which references it as durations_plot_total.png.
        total_src = os.path.join(figures_dir, "durations_plot.png")
        if os.path.exists(total_src):
            shutil.copy(
                total_src, os.path.join(args.output, "durations_plot_total.png")
            )
            copied.append("durations_plot_total.png")

        # The avg chart was generated in a separate output dir to avoid
        # overwriting the total chart.
        avg_src = os.path.join(work_dir, "figures_avg", "durations_plot.png")
        if os.path.exists(avg_src):
            shutil.copy(avg_src, os.path.join(args.output, "durations_plot_avg.png"))
            copied.append("durations_plot_avg.png")

        speedup_src = os.path.join(work_dir, "figures_scaling", "speedup_plot.png")
        if os.path.exists(speedup_src):
            shutil.copy(speedup_src, os.path.join(args.output, "speedup_plot.png"))
            copied.append("speedup_plot.png")

        # Region/rank filtering, shown on the docs page next to the full chart.
        plot(
            [
                "gantt",
                "run_2.h5",
                "-o",
                "figures_filtered",
                "--include",
                "solve",
                "assemble",
                "--ranks",
                "0",
            ],
            work_dir,
        )
        filtered_src = os.path.join(work_dir, "figures_filtered", "gantt_plot.png")
        if os.path.exists(filtered_src):
            shutil.copy(
                filtered_src, os.path.join(args.output, "gantt_plot_filtered.png")
            )
            copied.append("gantt_plot_filtered.png")

        print(f"\nFigures copied to {args.output}:")
        for name in copied:
            print(f"  {name}")
    finally:
        if not args.keep:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
