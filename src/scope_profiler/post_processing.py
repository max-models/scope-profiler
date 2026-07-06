"""CLI entry point for post-processing HDF5 profiling data."""

import argparse
import os

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_durations, plot_gantt


def parse_ranks(spec: str, verbose: bool = False) -> list[int]:
    """Parse a rank specification string into a list of integers.

    Supports comma-separated values and ranges (e.g., '1-3,5').
    """
    ranks = []
    for part in spec.split(","):
        if verbose:
            print(f"Parsing rank part: {part}")
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            ranks.extend(range(start, end + 1))
        else:
            ranks.append(int(part))
    if verbose:
        print(f"Parsed ranks: {ranks}")
    return ranks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read and summarize profiling HDF5 data."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths to profiling_data.h5 files",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (default: do not show plots)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Directory or file prefix to save plots instead of displaying them",
    )
    parser.add_argument(
        "--include",
        "-i",
        nargs="*",
        type=str,
        default=None,
        help="List of region names to include in the plots (optional)",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="*",
        type=str,
        default=None,
        help="List of region names to exclude from the plots (optional)",
    )
    parser.add_argument(
        "--ranks",
        "-r",
        nargs="*",
        type=str,
        default=None,
        help=(
            "List of ranks to include in the plots (optional). "
            "Supports comma-separated values and ranges (e.g., 1-3,5)."
        ),
    )
    return parser


def main(argv: list[str] | None = None):
    """Main function for reading and summarizing profiling HDF5 data."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.ranks:
        ranks = []
        for spec in args.ranks:
            ranks.extend(parse_ranks(spec))
        args.ranks = sorted(set(ranks))

    readers = [ProfilingH5Reader(file_path) for file_path in args.files]

    gantt_path = None
    durations_path = None
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        gantt_path = os.path.join(args.output, "gantt_plot.png")
        durations_path = os.path.join(args.output, "durations_plot.png")

    plot_gantt(
        profiling_data=readers,
        filepath=gantt_path,
        show=args.show,
        include=args.include,
        exclude=args.exclude,
        ranks=args.ranks,
    )

    plot_durations(
        profiling_data=readers,
        filepath=durations_path,
        show=args.show,
        include=args.include,
        exclude=args.exclude,
        ranks=args.ranks,
    )

    if args.output and not args.show:
        saved = [path for path in (gantt_path, durations_path) if path]
        print("Plots saved to:\n  " + "\n  ".join(saved))


if __name__ == "__main__":
    main()
