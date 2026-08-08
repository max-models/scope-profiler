"""CLI entry point for post-processing HDF5 profiling data."""

import argparse
import glob
import os

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import (
    DEFAULT_CMAP,
    plot_duration_timeseries,
    plot_durations,
    plot_flame,
    plot_gantt,
    plot_speedup,
    write_region_statistics_json,
)
from scope_profiler.prof_export import export_prof
from scope_profiler.speedscope_export import export_speedscope
from scope_profiler.summary import (
    SORT_KEYS,
    print_likwid_tables,
    print_region_table,
    region_rows,
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


def print_summary(
    reader,
    include=None,
    exclude=None,
    ranks=None,
    sort: str = "total",
    stream=None,
) -> None:
    """Print one file's region statistics, and its LIKWID counters if any.

    The region table is the same one ``ProfileManager.finalize()`` and
    ``scope-profiler inspect`` render. LIKWID results, when the run recorded
    them, follow as one additional table per rank and event group; files
    without LIKWID data simply end after the region table.

    Parameters
    ----------
    reader : ProfilingH5Reader
        File to summarize.
    include, exclude : list of str or str, optional
        Regex patterns selecting which regions to report.
    ranks : list of int, optional
        Restrict the statistics to these ranks (default: all).
    sort : str, optional
        Region table ordering; one of
        :data:`~scope_profiler.summary.SORT_KEYS`.
    stream : file-like, optional
        Where to write (default: stdout).
    """
    rows = region_rows(reader, include=include, exclude=exclude, ranks=ranks, sort=sort)
    print_region_table(
        rows,
        title=f"{reader.file_path}  ({reader.num_ranks} rank(s))",
        stream=stream,
    )
    print_likwid_tables(
        reader, include=include, exclude=exclude, ranks=ranks, stream=stream
    )


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
            "duration_timeseries_plot.png, optional speedup_plot.png, "
            "and region_statistics.json)"
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
        "--plots",
        "-p",
        nargs="*",
        type=str,
        choices=["gantt", "flame", "durations", "timeseries", "speedup"],
        default=None,
        help=(
            "Which plots to generate (default: all). "
            "Choices: gantt, flame, durations, timeseries, speedup. "
            "Example: --plots gantt durations"
        ),
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="*",
        type=str,
        choices=["avg", "min", "max", "total"],
        default=["total"],
        help=(
            "Which duration statistics to include in the durations bar plot "
            "(default: total). "
            "Example: --metrics avg min max total"
        ),
    )
    parser.add_argument(
        "--x-field",
        type=str,
        default="num_ranks",
        help=(
            "What to plot on the x-axis of the speedup plot (default: "
            "'num_ranks'). One of 'num_ranks', 'omp_num_threads', "
            "'total_cores' (num_ranks * omp_num_threads) — these are ordered "
            "numerically with an ideal-scaling line — or any other metadata "
            "field name, in which case files are kept in the order given on "
            "the command line and no ideal-scaling line is drawn."
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
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help=(
            "Renderer used for the plots (default: matplotlib). 'matplotlib' "
            "writes static .png files; 'plotly' writes interactive .html "
            "files instead, and makes --show open them in a browser."
        ),
    )
    parser.add_argument(
        "--export-data",
        action="store_true",
        help=(
            "Also write the exact data behind each plot as a data file "
            "(gantt_data, flame_data, durations_data, "
            "duration_timeseries_data, optional speedup_data; "
            "see --export-data-format for the file extension/content), so "
            "charts can be reconstructed later without the original HDF5 "
            "files. Requires -o/--output."
        ),
    )
    parser.add_argument(
        "--export-data-format",
        choices=["csv", "json"],
        default="csv",
        help=(
            "File format used by --export-data (default: csv). 'json' also "
            "includes a 'colors' map matching the colors used in each plot, "
            "so the chart can be re-rendered (e.g. with Plotly) with "
            "consistent colors."
        ),
    )
    parser.add_argument(
        "--export-prof",
        action="store_true",
        help=(
            "Also write one profile_rank<N>.prof file per exported rank in the "
            "cProfile/pstats format, so the run can be browsed with external "
            "tools (`snakeviz profile_rank0.prof`, `python -m pstats ...`). "
            "The call graph is reconstructed from region nesting; only ranks "
            "selected with --ranks are exported (default: rank 0). Requires "
            "-o/--output."
        ),
    )
    parser.add_argument(
        "--export-speedscope",
        action="store_true",
        help=(
            "Also write a profile.speedscope.json file holding one profile per "
            "exported rank, viewable at https://www.speedscope.app (or with "
            "`npx speedscope profile.speedscope.json`). Unlike --export-prof "
            "this keeps every individual call, so the timeline shows the run "
            "as it happened. The call graph is reconstructed from region "
            "nesting; only ranks selected with --ranks are exported (default: "
            "rank 0). Requires -o/--output."
        ),
    )
    parser.add_argument(
        "--skip-plot-images",
        action="store_true",
        help=(
            "Do not render/save the PNG plot images, only the outputs from "
            "--export-data/--export-prof/--export-speedscope. Useful when "
            "charts are rendered entirely client-side (e.g. with Plotly) from "
            "the exported data. Requires one of those export options."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print the per-region statistics table for each file, plus a "
            "separate LIKWID hardware counter table per rank and event group "
            "when the run recorded any. Honours --include/--exclude/--ranks. "
            "On its own this prints the summary and produces no plots; "
            "combine it with --show or -o/--output to get both."
        ),
    )
    parser.add_argument(
        "--summary-sort",
        choices=SORT_KEYS,
        default="total",
        help="Order the --summary region table by this column (default: total)",
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

    if args.export_data and not args.output:
        parser.error("--export-data requires -o/--output.")

    if args.export_prof and not args.output:
        parser.error("--export-prof requires -o/--output.")

    if args.export_speedscope and not args.output:
        parser.error("--export-speedscope requires -o/--output.")

    exports_requested = args.export_data or args.export_prof or args.export_speedscope
    if args.skip_plot_images and not exports_requested:
        parser.error(
            "--skip-plot-images requires --export-data, --export-prof or "
            "--export-speedscope."
        )

    _ALL_PLOTS = {"gantt", "flame", "durations", "timeseries", "speedup"}
    selected_plots: set[str] = set(args.plots) if args.plots is not None else _ALL_PLOTS

    if args.ranks:
        ranks = []
        for spec in args.ranks:
            ranks.extend(parse_ranks(spec))
        args.ranks = sorted(set(ranks))

    readers = [ProfilingH5Reader(file_path) for file_path in args.files]

    if args.summary:
        # Before the time_trace check below: a count-only file still has a
        # perfectly good summary table, and so does a LIKWID-only one.
        for index, reader in enumerate(readers):
            if index:
                print()
            print_summary(
                reader,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                sort=args.summary_sort,
            )
        # Asking only for a summary means the summary is the whole job;
        # rendering charts nobody requested would just cost time.
        if not (args.show or args.output):
            return

    # Files profiled with time_trace=False hold call counts but no timestamps,
    # so every chart here would be empty. Report the counts and stop, rather
    # than failing deep inside the plotting code.
    if not any(
        len(region[rank].durations)
        for reader in readers
        for region in reader.get_regions()
        for rank in region.regions
    ):
        print(
            "No timing data found — these files were profiled with "
            "time_trace=False, which records call counts only.\n"
        )
        for reader in readers:
            print(f"{reader.file_path}:")
            for region in reader.get_regions():
                total = sum(r.num_calls for r in region.regions.values())
                print(f"  {region.name}: {total} calls")
        return

    gantt_path = None
    flame_path = None
    durations_path = None
    timeseries_path = None
    speedup_path = None
    statistics_path = None
    gantt_data_path = None
    flame_data_path = None
    durations_data_path = None
    timeseries_data_path = None
    speedup_data_path = None
    prof_path = None
    prof_paths: list = []
    speedscope_path = None
    speedscope_paths: list = []
    durations_paths: list = []
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        if not args.skip_plot_images:
            # Plotly's native output is a self-contained interactive page;
            # writing .png from it would additionally require kaleido.
            ext = "html" if args.backend == "plotly" else "png"
            if "gantt" in selected_plots:
                gantt_path = os.path.join(args.output, f"gantt_plot.{ext}")
            if "flame" in selected_plots:
                flame_path = os.path.join(args.output, f"flame_plot.{ext}")
            if "durations" in selected_plots:
                durations_path = os.path.join(args.output, f"durations_plot.{ext}")
            if "timeseries" in selected_plots:
                timeseries_path = os.path.join(
                    args.output, f"duration_timeseries_plot.{ext}"
                )
            if len(readers) > 1 and "speedup" in selected_plots:
                speedup_path = os.path.join(args.output, f"speedup_plot.{ext}")
        statistics_path = os.path.join(args.output, "region_statistics.json")
        if args.export_data:
            data_ext = args.export_data_format
            if "gantt" in selected_plots:
                gantt_data_path = os.path.join(args.output, f"gantt_data.{data_ext}")
            if "flame" in selected_plots:
                flame_data_path = os.path.join(args.output, f"flame_data.{data_ext}")
            if "durations" in selected_plots:
                durations_data_path = os.path.join(
                    args.output, f"durations_data.{data_ext}"
                )
            if "timeseries" in selected_plots:
                timeseries_data_path = os.path.join(
                    args.output, f"duration_timeseries_data.{data_ext}"
                )
            if len(readers) > 1 and "speedup" in selected_plots:
                speedup_data_path = os.path.join(
                    args.output, f"speedup_data.{data_ext}"
                )
        if args.export_prof:
            prof_path = os.path.join(args.output, "profile.prof")
        if args.export_speedscope:
            speedscope_path = os.path.join(args.output, "profile.speedscope.json")

    # --skip-plot-images still needs the plotting functions to produce the
    # --export-data files, but an export-only run should not touch the
    # plotting backend at all.
    render_plots = not args.skip_plot_images or args.export_data

    if args.export_prof:
        prof_paths = export_prof(
            profiling_data=readers,
            filepath=prof_path,
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
            verbose=False,
        )

    if args.export_speedscope:
        speedscope_paths = export_speedscope(
            profiling_data=readers,
            filepath=speedscope_path,
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
            verbose=False,
        )

    if render_plots:
        if "gantt" in selected_plots:
            plot_gantt(
                profiling_data=readers,
                filepath=gantt_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                cmap=args.cmap,
                data_filepath=gantt_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if "flame" in selected_plots:
            plot_flame(
                profiling_data=readers,
                filepath=flame_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                cmap=args.cmap,
                data_filepath=flame_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if "durations" in selected_plots:
            durations_paths = plot_durations(
                profiling_data=readers,
                filepath=durations_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                metrics=args.metrics,
                cmap=args.cmap,
                data_filepath=durations_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if "timeseries" in selected_plots:
            plot_duration_timeseries(
                profiling_data=readers,
                filepath=timeseries_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                cmap=args.cmap,
                data_filepath=timeseries_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if len(readers) > 1 and "speedup" in selected_plots:
            plot_speedup(
                profiling_data=readers,
                x_field=args.x_field,
                ranks=args.ranks,
                filepath=speedup_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                cmap=args.cmap,
                data_filepath=speedup_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
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
            str(path)
            for path in (
                gantt_path,
                flame_path,
                *durations_paths,
                timeseries_path,
                speedup_path,
                statistics_path,
                gantt_data_path,
                flame_data_path,
                durations_data_path,
                timeseries_data_path,
                speedup_data_path,
                *prof_paths,
                *speedscope_paths,
            )
            if path
        ]
        print("Outputs saved to:\n  " + "\n  ".join(saved))
        if prof_paths:
            print(f"\nView a .prof file with: snakeviz {prof_paths[0]}")
        if speedscope_paths:
            print(
                f"\nView {speedscope_paths[0]} at https://www.speedscope.app "
                "(or: npx speedscope <file>)"
            )


if __name__ == "__main__":
    main()
