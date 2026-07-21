#!/usr/bin/env python3
"""
Generate the example plots shown in the README
================================================

Runs a small illustrative workload (a mock PDE timestep loop with nested
and self-recursive regions) through scope-profiler and renders the Gantt
chart, flame graph, and duration bar chart used as README examples.

Run::

    python examples/generate_readme_figures.py          # save figures only
    python examples/generate_readme_figures.py --show   # also display them
"""

import argparse
import os
import tempfile
import time

from scope_profiler import ProfileManager
from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_durations, plot_flame, plot_gantt


@ProfileManager.profile("assemble_matrix")
def assemble_matrix(n):
    time.sleep(0.008)


@ProfileManager.profile("apply_bc")
def apply_bc(n):
    time.sleep(0.002)


@ProfileManager.profile("solve")
def solve(n):
    assemble_matrix(n)
    apply_bc(n)


@ProfileManager.profile("refine_mesh")
def refine_mesh(depth):
    """Self-recursive region: each level gets its own timing slot."""
    if depth == 0:
        return
    time.sleep(0.001)
    refine_mesh(depth - 1)


@ProfileManager.profile("setup")
def setup():
    time.sleep(0.003)
    refine_mesh(4)


@ProfileManager.profile("main")
def run_workload():
    setup()
    for _ in range(3):
        solve(100)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the example plots used in the README."
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the plots interactively"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="figures",
        help="Directory to save the figures (default: figures/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = os.path.join(tmp_dir, "readme_example.h5")
        ProfileManager.setup(use_likwid=False, time_trace=True, file_path=h5_path)
        run_workload()
        ProfileManager.finalize(verbose=False)

        reader = ProfilingH5Reader(h5_path)

        gantt_path = os.path.join(args.output, "gantt_plot.png")
        flame_path = os.path.join(args.output, "flame_plot.png")
        durations_path = os.path.join(args.output, "durations_plot.png")

        plot_gantt(reader, filepath=gantt_path, show=args.show, verbose=False)
        plot_flame(reader, filepath=flame_path, show=args.show, verbose=False)
        plot_durations(
            reader,
            filepath=durations_path,
            metrics="avg",
            show=args.show,
            verbose=False,
        )

    print(
        "Figures saved to:\n  " + "\n  ".join([gantt_path, flame_path, durations_path])
    )


if __name__ == "__main__":
    main()
