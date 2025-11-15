import argparse
import os

from scope_profiler.h5reader import ProfilingH5Reader


def main():
    """Main function for reading and summarizing profiling HDF5 data."""
    parser = argparse.ArgumentParser(
        description="Read and summarize profiling HDF5 data."
    )
    parser.add_argument("file", type=str, help="Path to the profiling_data.h5 file")
    parser.add_argument("--region", type=str, help="Region name to inspect (optional)")
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
    args = parser.parse_args()

    reader = ProfilingH5Reader(args.file)

    # Handle optional region selection
    if args.region:
        regions = [args.region]
        print(f"\nRegion: {args.region}")
        print(reader.get_region(args.region))
    else:
        regions = None
        print(reader)

    # Prepare output filepaths if requested
    gantt_path = durations_path = None
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        gantt_path = os.path.join(args.output, "gantt_plot.png")
        durations_path = os.path.join(args.output, "durations_plot.png")

    # Call the plotting functions with the appropriate arguments
    reader.plot_gantt(regions=regions, filepath=gantt_path, show=args.show)
    reader.plot_durations(regions=regions, filepath=durations_path, show=args.show)

    # If saving only (no show), print confirmation
    if args.output and not args.show:
        print(f"Plots saved to:\n  {gantt_path}\n  {durations_path}")


if __name__ == "__main__":
    main()
