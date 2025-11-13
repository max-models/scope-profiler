import argparse

from scope_profiler.h5reader import ProfilingH5Reader


def main():
    """Main function for reading and summarizing profiling HDF5 data."""
    parser = argparse.ArgumentParser(
        description="Read and summarize profiling HDF5 data."
    )
    parser.add_argument("file", type=str, help="Path to the profiling_data.h5 file")
    parser.add_argument("--region", type=str, help="Region name to inspect (optional)")
    args = parser.parse_args()

    reader = ProfilingH5Reader(args.file)

    if args.region:
        region = reader.get_region(args.region)
        print(f"\nRegion: {args.region}")
        print(region)
        reader.plot_gantt([args.region])
        reader.plot_durations([args.region])
    else:
        print(reader)
        reader.plot_gantt()
        reader.plot_durations()


if __name__ == "__main__":
    main()
