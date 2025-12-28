import argparse
import os

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_gantt  # , plot_durations


def main():
    """Main function for reading and summarizing profiling HDF5 data."""
    parser = argparse.ArgumentParser(
        description="Read and summarize profiling HDF5 data."
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the profiling_data.h5 file",
    )

    # parser.add_argument(
    #     "--region",
    #     type=str,
    #     help="Region name to inspect (optional)",
    # )
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
    args = parser.parse_args()

    reader = ProfilingH5Reader(args.file)

    # Prepare output filepaths if requested
    gantt_path = durations_path = None
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        gantt_path = os.path.join(args.output, "gantt_plot.png")
        durations_path = os.path.join(args.output, "durations_plot.png")

    # Call the plotting functions with the appropriate arguments
    plot_gantt(
        profiling_data=reader,
        filepath=gantt_path,
        show=args.show,
        include=args.include,
        exclude=args.exclude,
    )
    # plot_durations(profiling_data=reader, regions=regions, filepath=durations_path, show=args.show,)

    # If saving only (no show), print confirmation
    if args.output and not args.show:
        print(f"Plots saved to:\n  {gantt_path}\n  {durations_path}")


if __name__ == "__main__":
    main()
