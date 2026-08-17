"""CLI entry point for post-processing HDF5 profiling data."""

import argparse
import glob
import os

from scope_profiler.h5reader import read_h5
from scope_profiler.plotting_scripts import (
    DEFAULT_CMAP,
    plot_duration_histogram,
    plot_duration_timeseries,
    plot_durations,
    plot_flame,
    plot_gantt,
    plot_imbalance,
    plot_likwid,
    plot_speedup,
    write_region_statistics_json,
)
from scope_profiler.prof_export import export_prof
from scope_profiler.speedscope_export import export_speedscope

# Single source of truth for --plots: name -> (one-line description, is a
# default plot). Everything else derives from this -- the argparse choices,
# the --plots help text, and the default set used when --plots is omitted --
# so the three can never drift out of sync.
_PLOT_CATALOG: dict[str, tuple[str, bool]] = {
    "gantt": ("per-rank timeline of every call", True),
    "flame": ("reconstructed call-stack flame graph", True),
    "durations": ("bar chart of duration statistics per region", True),
    "timeseries": ("duration per call over wall-clock time", True),
    "speedup": ("scaling across multiple files (2+ files only)", True),
    "histogram": ("call-duration distribution per region", False),
    "imbalance": ("per-rank duration comparison, to spot stragglers", False),
    "likwid": ("one LIKWID hardware-counter metric (needs --likwid-metric)", False),
}
_DEFAULT_PLOTS = frozenset(
    name for name, (_, is_default) in _PLOT_CATALOG.items() if is_default
)


def _plots_help() -> str:
    """Build the --plots help text from :data:`_PLOT_CATALOG`."""
    lines = [f"{name}: {desc}" for name, (desc, _) in _PLOT_CATALOG.items()]
    default_names = ", ".join(sorted(_DEFAULT_PLOTS))
    opt_in_names = ", ".join(
        name for name in _PLOT_CATALOG if name not in _DEFAULT_PLOTS
    )
    return (
        "Which plots to generate. "
        + " | ".join(lines)
        + f". Default (no --plots given): {default_names}. "
        f"Opt-in only, pass explicitly to get them: {opt_in_names}. "
        "Example: --plots gantt durations"
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


def parse_region_groups(
    specs: list[str] | None, parser: argparse.ArgumentParser
) -> dict[str, list[str]] | None:
    """Parse ``--combine-regions`` specs into a ``{name: [patterns]}`` dict.

    Each spec has the form ``NAME=PATTERN1,PATTERN2``. Repeating the same
    NAME across specs is not supported -- pass every pattern for a group in
    one spec instead.
    """
    if not specs:
        return None
    groups: dict[str, list[str]] = {}
    for spec in specs:
        name, sep, patterns = spec.partition("=")
        if not sep or not name or not patterns:
            parser.error(
                "--combine-regions expects 'NAME=PATTERN1,PATTERN2' " f"(got {spec!r})"
            )
        if name in groups:
            parser.error(f"--combine-regions group {name!r} given more than once.")
        groups[name] = [pattern for pattern in patterns.split(",") if pattern]
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scope-profiler plot",
        description="Render plots and export data from HDF5 profiling files.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths or glob patterns for profiling_data.h5 files",
    )

    selecting = parser.add_argument_group(
        "Selecting data",
        "Which files, regions and ranks feed every plot and export below.",
    )
    selecting.add_argument(
        "--label",
        action="append",
        default=None,
        metavar="LABEL",
        help=(
            "Name for a file in the outputs, overriding the label its run was "
            "given with setup(label=...) (and the file stem for runs without "
            "one). Repeat once per file, in the order the files are listed: "
            "--label '128 ranks' --label '256 ranks'"
        ),
    )
    selecting.add_argument(
        "--include",
        "-i",
        nargs="*",
        type=str,
        default=None,
        help="Region names to include (optional; regex patterns)",
    )
    selecting.add_argument(
        "--exclude",
        "-e",
        nargs="*",
        type=str,
        default=None,
        help="Region names to exclude (optional; regex patterns)",
    )
    selecting.add_argument(
        "--ranks",
        "-r",
        nargs="*",
        type=str,
        default=None,
        help=(
            "Ranks to include (optional; default: all). Supports "
            "comma-separated values and ranges (e.g. 1-3,5)."
        ),
    )

    plots = parser.add_argument_group(
        "Choosing and rendering plots",
        "What to draw, and where.",
    )
    plots.add_argument(
        "--plots",
        "-p",
        nargs="*",
        type=str,
        choices=list(_PLOT_CATALOG),
        default=None,
        help=_plots_help(),
    )
    plots.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (default: do not show plots)",
    )
    plots.add_argument(
        "-o",
        "--output",
        type=str,
        help=(
            "Directory where outputs are saved: one <name>_plot.png (or "
            ".html for --backend plotly) per plot selected by --plots, plus "
            "region_statistics.json."
        ),
    )
    plots.add_argument(
        "--backend",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help=(
            "Renderer used for the plots (default: matplotlib). 'matplotlib' "
            "writes static .png files; 'plotly' writes interactive .html "
            "files instead, and makes --show open them in a browser."
        ),
    )
    plots.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
        help=(
            "Name of the matplotlib colormap used to color regions/files in "
            f"all plots (default: {DEFAULT_CMAP!r}). See "
            "https://matplotlib.org/stable/users/explain/colors/colormaps.html"
        ),
    )

    tuning = parser.add_argument_group(
        "Tuning individual plots",
        "Each option only affects the plot(s) named in its own description; "
        "picking a plot that ignores an option is harmless.",
    )
    tuning.add_argument(
        "--duration-metrics",
        nargs="*",
        type=str,
        choices=["avg", "min", "max", "total"],
        default=["total"],
        metavar="{avg,min,max,total}",
        help=(
            "[durations] Which duration statistics to draw as bar-chart "
            "columns (default: total). Example: --duration-metrics avg max"
        ),
    )
    tuning.add_argument(
        "--sort-by",
        choices=["name", "avg", "min", "max", "total"],
        default=None,
        help=(
            "[durations, likwid] Order the bar chart's regions by this "
            "statistic, descending ('name' sorts alphabetically instead). "
            "Default: keep regions in the order they first appeared."
        ),
    )
    tuning.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "[durations, likwid] Keep only the top N regions (ranked by "
            "--sort-by, or by descending total duration if --sort-by is not "
            "given). Useful when a run has many regions."
        ),
    )
    tuning.add_argument(
        "--combine-regions",
        nargs="*",
        type=str,
        default=None,
        metavar="NAME=PATTERN[,PATTERN...]",
        help=(
            "[durations] Merge several regions into a single bar. Each "
            "value is 'NAME=PATTERN1,PATTERN2' where NAME is the combined "
            "bar's label and the comma-separated PATTERNs are regexes "
            "matched against region names (like --include). Repeat once per "
            "group. A region matched by more than one group is claimed by "
            "whichever group is given first. Example: --combine-regions "
            "'setup=^setup:.*'"
        ),
    )
    tuning.add_argument(
        "--log-scale",
        action="store_true",
        help=(
            "[durations, timeseries, histogram, imbalance, likwid] Use a "
            "logarithmic y-axis."
        ),
    )
    tuning.add_argument(
        "--histogram-bins",
        type=int,
        default=30,
        metavar="N",
        help="[histogram] Number of duration bins (default: 30).",
    )
    tuning.add_argument(
        "--imbalance-metric",
        choices=["avg", "min", "max", "total"],
        default="total",
        help=(
            "[imbalance] Per-call duration statistic plotted per rank "
            "(default: total, i.e. total time a rank spent in the region)."
        ),
    )
    tuning.add_argument(
        "--likwid-metric",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "[likwid] Name of the LIKWID derived metric or raw event to "
            "plot, e.g. 'CPI', 'MFlops/s'. Required when 'likwid' is in "
            "--plots. Run `scope-profiler inspect` on a LIKWID-enabled file "
            "to see the available names, or inspect "
            "ProfilingResults.get_likwid_regions()."
        ),
    )
    tuning.add_argument(
        "--speedup-x-field",
        type=str,
        default="num_ranks",
        metavar="FIELD",
        help=(
            "[speedup] What to plot on the x-axis (default: 'num_ranks'). "
            "One of 'num_ranks', 'omp_num_threads', 'total_cores' "
            "(num_ranks * omp_num_threads) — these are ordered numerically "
            "with an ideal-scaling line — or any other metadata field name, "
            "in which case files are kept in the order given on the command "
            "line and no ideal-scaling line is drawn."
        ),
    )

    exporting = parser.add_argument_group(
        "Exporting extra data",
        "Machine-readable outputs alongside the plot images. All require -o/--output.",
    )
    exporting.add_argument(
        "--export",
        nargs="*",
        type=str,
        choices=["data", "prof", "speedscope"],
        default=[],
        metavar="{data,prof,speedscope}",
        help=(
            "'data': the exact numbers behind each plot selected by "
            "--plots, as one <name>_data file per plot (see "
            "--export-data-format). 'prof': one profile_rank<N>.prof per "
            "exported rank, cProfile/pstats format (browse with `snakeviz "
            "profile_rank0.prof`). 'speedscope': one profile.speedscope.json "
            "covering every exported rank (browse at "
            "https://www.speedscope.app). 'prof'/'speedscope' reconstruct "
            "the call graph from region nesting and only cover the ranks "
            "selected by --ranks (default: rank 0). "
            "Example: --export data prof"
        ),
    )
    exporting.add_argument(
        "--export-data-format",
        choices=["csv", "json"],
        default="csv",
        help=(
            "File format for --export data (default: csv). 'json' also "
            "includes a 'colors' map matching the colors used in each plot, "
            "so the chart can be re-rendered (e.g. with Plotly) with "
            "consistent colors."
        ),
    )
    exporting.add_argument(
        "--skip-plot-images",
        action="store_true",
        help=(
            "Do not render/save the plot images themselves, only the "
            "--export outputs. Useful when charts are rendered entirely "
            "client-side (e.g. with Plotly) from exported data. Requires "
            "--export to select at least one output."
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
    args.combine_regions = parse_region_groups(args.combine_regions, parser)

    want_export_data = "data" in args.export
    want_export_prof = "prof" in args.export
    want_export_speedscope = "speedscope" in args.export

    if args.export and not args.output:
        parser.error("--export requires -o/--output.")

    if args.skip_plot_images and not args.export:
        parser.error(
            "--skip-plot-images requires --export to select at least one output."
        )

    selected_plots: set[str] = (
        set(args.plots) if args.plots is not None else set(_DEFAULT_PLOTS)
    )

    if "likwid" in selected_plots and not args.likwid_metric:
        parser.error("--plots likwid requires --likwid-metric.")

    if args.ranks:
        ranks = []
        for spec in args.ranks:
            ranks.extend(parse_ranks(spec))
        args.ranks = sorted(set(ranks))

    runs = [read_h5(file_path) for file_path in args.files]

    # Applied to the runs rather than passed down per plot: every output --
    # chart legends and panel titles, the JSON statistics, the exported
    # filenames -- names a run through ProfilingResults.display_label, so
    # overriding it here covers all of them at once. The files themselves
    # are not modified.
    if args.label is not None:
        if len(args.label) != len(runs):
            parser.error(
                f"--label given {len(args.label)} time(s) for "
                f"{len(runs)} file(s); pass one per file, in order."
            )
        for run, label in zip(runs, args.label):
            run.label = label

    # A file whose regions recorded no calls would produce empty charts.
    # Report what is there and stop, rather than failing deep inside the
    # plotting code.
    if not any(
        len(region[rank].durations)
        for run in runs
        for region in run.get_regions()
        for rank in region.regions
    ):
        print("No timing data found — these files recorded no calls.\n")
        for run in runs:
            print(f"{run.file_path}:")
            for region in run.get_regions():
                total = sum(r.num_calls for r in region.regions.values())
                print(f"  {region.name}: {total} calls")
        return

    gantt_path = None
    flame_path = None
    durations_path = None
    timeseries_path = None
    speedup_path = None
    histogram_path = None
    imbalance_path = None
    likwid_path = None
    statistics_path = None
    gantt_data_path = None
    flame_data_path = None
    durations_data_path = None
    timeseries_data_path = None
    speedup_data_path = None
    histogram_data_path = None
    imbalance_data_path = None
    likwid_data_path = None
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
            if len(runs) > 1 and "speedup" in selected_plots:
                speedup_path = os.path.join(args.output, f"speedup_plot.{ext}")
            if "histogram" in selected_plots:
                histogram_path = os.path.join(args.output, f"histogram_plot.{ext}")
            if "imbalance" in selected_plots:
                imbalance_path = os.path.join(args.output, f"imbalance_plot.{ext}")
            if "likwid" in selected_plots:
                likwid_path = os.path.join(args.output, f"likwid_plot.{ext}")
        statistics_path = os.path.join(args.output, "region_statistics.json")
        if want_export_data:
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
            if len(runs) > 1 and "speedup" in selected_plots:
                speedup_data_path = os.path.join(
                    args.output, f"speedup_data.{data_ext}"
                )
            if "histogram" in selected_plots:
                histogram_data_path = os.path.join(
                    args.output, f"histogram_data.{data_ext}"
                )
            if "imbalance" in selected_plots:
                imbalance_data_path = os.path.join(
                    args.output, f"imbalance_data.{data_ext}"
                )
            if "likwid" in selected_plots:
                likwid_data_path = os.path.join(args.output, f"likwid_data.{data_ext}")
        if want_export_prof:
            prof_path = os.path.join(args.output, "profile.prof")
        if want_export_speedscope:
            speedscope_path = os.path.join(args.output, "profile.speedscope.json")

    # --skip-plot-images still needs the plotting functions to produce the
    # --export data files, but an export-only run should not touch the
    # plotting backend at all.
    render_plots = not args.skip_plot_images or want_export_data

    if want_export_prof:
        prof_paths = export_prof(
            profiling_data=runs,
            filepath=prof_path,
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
            verbose=False,
        )

    if want_export_speedscope:
        speedscope_paths = export_speedscope(
            profiling_data=runs,
            filepath=speedscope_path,
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
            verbose=False,
        )

    if render_plots:
        if "gantt" in selected_plots:
            plot_gantt(
                profiling_data=runs,
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
                profiling_data=runs,
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
                profiling_data=runs,
                filepath=durations_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                metrics=args.duration_metrics,
                sort_by=args.sort_by,
                top_n=args.top_n,
                combine_regions=args.combine_regions,
                cmap=args.cmap,
                log_scale=args.log_scale,
                data_filepath=durations_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if "timeseries" in selected_plots:
            plot_duration_timeseries(
                profiling_data=runs,
                filepath=timeseries_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                cmap=args.cmap,
                log_scale=args.log_scale,
                data_filepath=timeseries_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if "histogram" in selected_plots:
            plot_duration_histogram(
                profiling_data=runs,
                filepath=histogram_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                bins=args.histogram_bins,
                cmap=args.cmap,
                log_scale=args.log_scale,
                data_filepath=histogram_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if "imbalance" in selected_plots:
            plot_imbalance(
                profiling_data=runs,
                metric=args.imbalance_metric,
                filepath=imbalance_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                cmap=args.cmap,
                log_scale=args.log_scale,
                data_filepath=imbalance_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if "likwid" in selected_plots:
            plot_likwid(
                profiling_data=runs,
                metric=args.likwid_metric,
                filepath=likwid_path,
                show=args.show,
                include=args.include,
                exclude=args.exclude,
                ranks=args.ranks,
                cmap=args.cmap,
                log_scale=args.log_scale,
                data_filepath=likwid_data_path,
                data_format=args.export_data_format,
                backend=args.backend,
            )

        if len(runs) > 1 and "speedup" in selected_plots:
            plot_speedup(
                profiling_data=runs,
                x_field=args.speedup_x_field,
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
            profiling_data=runs,
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
                histogram_path,
                imbalance_path,
                likwid_path,
                statistics_path,
                gantt_data_path,
                flame_data_path,
                durations_data_path,
                timeseries_data_path,
                speedup_data_path,
                histogram_data_path,
                imbalance_data_path,
                likwid_data_path,
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
