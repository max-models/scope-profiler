"""CLI entry point for post-processing HDF5 profiling data."""

import argparse
import glob
import os

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_gantt, write_region_statistics_json


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
        help="Directory where output files are saved (gantt_plot.png and region_statistics.json)",
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
    statistics_path = None
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        gantt_path = os.path.join(args.output, "gantt_plot.png")
        statistics_path = os.path.join(args.output, "region_statistics.json")

    plot_gantt(
        profiling_data=readers,
        filepath=gantt_path,
        show=args.show,
        include=args.include,
        exclude=args.exclude,
        ranks=args.ranks,
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
        saved = [path for path in (gantt_path, statistics_path) if path]
        print("Outputs saved to:\n  " + "\n  ".join(saved))


if __name__ == "__main__":
    main()
