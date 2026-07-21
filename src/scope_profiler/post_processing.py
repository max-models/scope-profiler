"""CLI entry point for post-processing HDF5 profiling data."""

import argparse
import glob
import os

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import (
    DEFAULT_CMAP,
    plot_durations,
    plot_flame,
    plot_gantt,
    plot_speedup,
    write_region_statistics_json,
)


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
        prog="scope-profiler pproc",
        description="Read and summarize profiling HDF5 data.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths or glob patterns for profiling_data.h5 files",
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
        help=(
            "Directory where outputs are saved "
            "(gantt_plot.png, flame_plot.png, durations_plot.png, "
            "optional speedup_plot.png, and region_statistics.json)"
        ),
    )
    parser.add_argument(
        "--include",
        "-i",
        nargs="*",
        type=str,
        default=None,
        help="List of region names to include in the outputs (optional)",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="*",
        type=str,
        default=None,
        help="List of region names to exclude from the outputs (optional)",
    )
    parser.add_argument(
        "--ranks",
        "-r",
        nargs="*",
        type=str,
        default=None,
        help=(
            "List of ranks to include in the outputs (optional). "
            "Supports comma-separated values and ranges (e.g., 1-3,5)."
        ),
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="*",
        type=str,
        choices=["avg", "min", "max", "total"],
        default=None,
        help=(
            "Which duration statistics to include in the durations bar plot "
            "(default: all of avg, min, max, total)."
        ),
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
        help=(
            "Name of the matplotlib colormap used to color regions/files in "
            f"all plots (default: {DEFAULT_CMAP!r}). See "
            "https://matplotlib.org/stable/users/explain/colors/colormaps.html"
        ),
    )
    return parser


def expand_file_patterns(
    file_args: list[str], parser: argparse.ArgumentParser
) -> list[str]:
    """Expand CLI file arguments that contain shell-style wildcard patterns."""
    expanded_files: list[str] = []

    for file_arg in file_args:
        if glob.has_magic(file_arg):
            matches = sorted(
                match
                for match in glob.glob(file_arg, recursive=True)
                if os.path.isfile(match)
            )
            if not matches:
                parser.error(f"No files matched pattern: {file_arg}")
            expanded_files.extend(matches)
        else:
            expanded_files.append(file_arg)

    if not expanded_files:
        parser.error("No input files provided.")

    # Keep first occurrence order in case overlapping patterns are supplied.
    return list(dict.fromkeys(expanded_files))


def main(argv: list[str] | None = None):
    """Main function for reading and summarizing profiling HDF5 data."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.files = expand_file_patterns(args.files, parser)

    if args.ranks:
        ranks = []
        for spec in args.ranks:
            ranks.extend(parse_ranks(spec))
        args.ranks = sorted(set(ranks))

    readers = [ProfilingH5Reader(file_path) for file_path in args.files]

    gantt_path = None
    flame_path = None
    durations_path = None
    speedup_path = None
    statistics_path = None
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        gantt_path = os.path.join(args.output, "gantt_plot.png")
        flame_path = os.path.join(args.output, "flame_plot.png")
        durations_path = os.path.join(args.output, "durations_plot.png")
        if len(readers) > 1:
            speedup_path = os.path.join(args.output, "speedup_plot.png")
        statistics_path = os.path.join(args.output, "region_statistics.json")

    plot_gantt(
        profiling_data=readers,
        filepath=gantt_path,
        show=args.show,
        include=args.include,
        exclude=args.exclude,
        ranks=args.ranks,
        cmap=args.cmap,
    )

    plot_flame(
        profiling_data=readers,
        filepath=flame_path,
        show=args.show,
        include=args.include,
        exclude=args.exclude,
        ranks=args.ranks,
        cmap=args.cmap,
    )

    durations_paths = plot_durations(
        profiling_data=readers,
        filepath=durations_path,
        show=args.show,
        include=args.include,
        exclude=args.exclude,
        ranks=args.ranks,
        metrics=args.metrics,
        cmap=args.cmap,
    )

    if len(readers) > 1:
        plot_speedup(
            profiling_data=readers,
            ranks=args.ranks,
            filepath=speedup_path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            cmap=args.cmap,
        )

    if statistics_path:
        write_region_statistics_json(
            profiling_data=readers,
            filepath=statistics_path,
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
        )

    if args.output and not args.show:
        saved = [
            path
            for path in (
                gantt_path,
                flame_path,
                *durations_paths,
                speedup_path,
                statistics_path,
            )
            if path
        ]
        print("Outputs saved to:\n  " + "\n  ".join(saved))


if __name__ == "__main__":
    main()
