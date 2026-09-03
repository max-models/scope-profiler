"""CLI entry point for post-processing HDF5 profiling data."""

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from scope_profiler.chrome_trace_export import export_chrome_trace
from scope_profiler.plotting_scripts import (
    DEFAULT_CMAP,
    FLAME_CMAP,
    plot_callgraph,
    plot_duration_histogram,
    plot_duration_timeseries,
    plot_durations,
    plot_flame_chart,
    plot_flame_graph,
    plot_gantt,
    plot_imbalance,
    plot_likwid,
    plot_perf_events,
    plot_rank_heatmap,
    plot_scaling_efficiency,
    plot_speedup,
    plot_timeline_density,
    plot_weak_scaling,
    write_region_statistics_json,
)
from scope_profiler.prof_export import export_prof
from scope_profiler.profile_io import read_profile
from scope_profiler.speedscope_export import export_speedscope

# Single source of truth for --plots: name -> (one-line description, is a
# default plot). Everything else derives from this -- the argparse choices,
# the --plots help text, and the default set used when --plots is omitted --
# so the three can never drift out of sync.
_PLOT_CATALOG: dict[str, tuple[str, bool]] = {
    "gantt": ("per-rank timeline of every call", True),
    "density": ("binned timeline occupancy heatmap", False),
    "flame_chart": ("time-based nested call-stack flame chart", False),
    "flame_graph": ("aggregated call-stack flame graph", False),
    "callgraph": ("parent/child call graph from explicit call ids", False),
    "durations": ("bar chart of duration statistics per region", True),
    "timeseries": ("duration per call over wall-clock time", False),
    "speedup": ("scaling across multiple files (2+ files only)", False),
    "weak_scaling": ("weak scaling across multiple files (2+ files only)", False),
    "rank_heatmap": ("total duration by rank and region", False),
    "scaling_efficiency": ("measured versus ideal parallel efficiency", False),
    "histogram": ("call-duration distribution per region", False),
    "imbalance": ("per-rank duration comparison, to spot stragglers", False),
    "likwid": ("one LIKWID hardware-counter metric (needs --likwid-metric)", False),
    "perf_events": (
        "one Linux perf-event metric (needs --metric, e.g. ipc)",
        False,
    ),
}
_DEFAULT_PLOTS = frozenset(
    name for name, (_, is_default) in _PLOT_CATALOG.items() if is_default
)
_QUICK_PLOTS = frozenset(
    {"durations", "speedup", "weak_scaling", "rank_heatmap", "scaling_efficiency"}
)
_PLOTEXT_SIMPLE_PLOTS = frozenset(
    {
        "durations",
        "timeseries",
        "speedup",
        "weak_scaling",
        "scaling_efficiency",
        "histogram",
        "imbalance",
    }
)
_PYVIS_PLOTS = frozenset({"callgraph"})
_PLOT_ALIASES = {"flame": "flame_chart"}


@dataclass(frozen=True)
class OutputTargets:
    directory: str | None
    single_file: str | None
    statistics_path: str | None


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
                f"--combine-regions expects 'NAME=PATTERN1,PATTERN2' (got {spec!r})"
            )
        if name in groups:
            parser.error(f"--combine-regions group {name!r} given more than once.")
        groups[name] = [pattern for pattern in patterns.split(",") if pattern]
    return groups


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths or glob patterns for profiling_data.h5 files",
    )


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        metavar="LABEL",
        help=(
            "Name for a file in outputs. Repeat once per file, in the order "
            "the files are listed."
        ),
    )
    parser.add_argument(
        "--include",
        "-i",
        nargs="*",
        type=str,
        default=None,
        help="Region names to include (regex patterns).",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="*",
        type=str,
        default=None,
        help="Region names to exclude (regex patterns).",
    )
    parser.add_argument(
        "--ranks",
        "-r",
        nargs="*",
        type=str,
        default=None,
        help="Ranks to include. Supports comma-separated values and ranges.",
    )


def _add_callgraph_args(parser: argparse.ArgumentParser) -> None:
    """Register the callgraph-only flags.

    Shared by `plot` and `export plot-data`: `callgraph` is a --plots choice
    for both, and `_render_selected_plots` reads these unconditionally.
    """
    parser.add_argument(
        "--compact-callgraph",
        action="store_true",
        help="Collapse repeated callgraph invocations into one node per region.",
    )
    parser.add_argument(
        "--fluid-callgraph",
        action="store_true",
        help="Use an interactive force-directed layout for the compact callgraph.",
    )


def _add_plot_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help=(
            "Output directory. For a single plot kind, this may also be a "
            "target .png, .html, or .txt file."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively.",
    )
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "plotly", "pyvis", "plotext"],
        default="matplotlib",
        help=(
            "Renderer used for plots: matplotlib/plotly support chart plots; "
            "pyvis supports callgraph only; plotext supports simple plots only."
        ),
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
        help=f"Matplotlib colormap used for regions/files (default: {DEFAULT_CMAP!r}).",
    )
    _add_callgraph_args(parser)


def _add_timeline_args(parser: argparse.ArgumentParser) -> None:
    """Options shared by the raw Gantt and density timeline views."""
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeline start, relative to the first recorded event.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeline end, relative to the first recorded event.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Hide calls shorter than this duration.",
    )
    parser.add_argument(
        "--aggregate-calls",
        type=int,
        default=1,
        metavar="N",
        help="Represent each N calls of a region as one Gantt bar.",
    )
    parser.add_argument(
        "--collapse-depth",
        type=int,
        default=None,
        metavar="N",
        help="Show only calls at nesting depth N or shallower (0=root).",
    )


def _add_duration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metrics",
        nargs="*",
        type=str,
        choices=["avg", "min", "max", "total"],
        default=["total"],
        metavar="{avg,min,max,total}",
        help="Duration statistics to draw/export (default: total).",
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "avg", "min", "max", "total"],
        default=None,
        help="Order regions by this statistic, descending; name sorts alphabetically.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Keep only the top N regions.",
    )
    parser.add_argument(
        "--combine-regions",
        nargs="*",
        type=str,
        default=None,
        metavar="NAME=PATTERN[,PATTERN...]",
        help="Merge regions into named bars using regex patterns.",
    )
    parser.add_argument(
        "--stack-children",
        action="store_true",
        help=(
            "Split each bar into the region's own time plus one stacked "
            "segment per region called from it (total/avg only)."
        ),
    )


def _add_log_scale_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use a logarithmic y-axis where the plot supports it.",
    )


def _add_data_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Data export format (default: csv).",
    )


def _add_common_plot_args(parser: argparse.ArgumentParser) -> None:
    _add_input_args(parser)
    _add_selection_args(parser)
    _add_plot_output_args(parser)
    _add_timeline_args(parser)


def _add_common_export_args(parser: argparse.ArgumentParser) -> None:
    _add_input_args(parser)
    _add_selection_args(parser)
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output directory.",
    )


def _build_plot_kind_parser(
    subparsers: argparse._SubParsersAction, kind: str, description: str
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(kind, help=description, description=description)
    parser.set_defaults(plot_kind=kind)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scope-profiler plot",
        description="Render plots from HDF5 profiling files.",
    )
    subparsers = parser.add_subparsers(dest="plot_kind", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="List available plot kinds and presets.",
        description="List available plot kinds and presets.",
    )
    list_parser.set_defaults(plot_kind="list")

    for kind in ("default", "all", "quick"):
        preset = _build_plot_kind_parser(
            subparsers, kind, f"Render the {kind} plot preset."
        )
        _add_common_plot_args(preset)
        _add_duration_args(preset)
        _add_log_scale_arg(preset)
        preset.add_argument("--bins", type=int, default=30, metavar="N")
        preset.add_argument(
            "--metric",
            type=str,
            default=None,
            metavar="NAME",
            help="LIKWID metric name; includes the likwid plot in the all preset.",
        )
        preset.add_argument(
            "--x",
            type=str,
            default="num_ranks",
            metavar="FIELD",
            help="Speedup x-axis field.",
        )

    for kind, (description, _is_default) in _PLOT_CATALOG.items():
        plot_parser = _build_plot_kind_parser(subparsers, kind, description)
        _add_common_plot_args(plot_parser)
        if kind == "durations":
            _add_duration_args(plot_parser)
            _add_log_scale_arg(plot_parser)
        elif kind == "density":
            plot_parser.add_argument(
                "--bins",
                type=int,
                default=200,
                metavar="N",
                help="Number of time bins.",
            )
        elif kind == "timeseries":
            _add_log_scale_arg(plot_parser)
        elif kind in {"speedup", "weak_scaling", "scaling_efficiency"}:
            plot_parser.add_argument(
                "--x",
                type=str,
                default="num_ranks",
                metavar="FIELD",
                help="Scaling x-axis field.",
            )
        elif kind == "histogram":
            plot_parser.add_argument(
                "--bins",
                type=int,
                default=30,
                metavar="N",
                help="Number of duration bins.",
            )
            _add_log_scale_arg(plot_parser)
        elif kind == "imbalance":
            plot_parser.add_argument(
                "--metric",
                choices=["avg", "min", "max", "total"],
                default="total",
                help="Per-call duration statistic plotted per rank.",
            )
            _add_log_scale_arg(plot_parser)
        elif kind == "likwid":
            plot_parser.add_argument(
                "--metric",
                required=True,
                type=str,
                metavar="NAME",
                help="LIKWID derived metric or raw event name, e.g. CPI.",
            )
            plot_parser.add_argument(
                "--sort-by",
                choices=["name", "avg", "min", "max", "total"],
                default=None,
                help="Order regions by this statistic.",
            )
            plot_parser.add_argument("--top-n", type=int, default=None, metavar="N")
            _add_log_scale_arg(plot_parser)
        elif kind == "perf_events":
            plot_parser.add_argument(
                "--metric",
                required=True,
                metavar="NAME",
                help="Recorded event, ipc, or cache-misses-per-ki.",
            )
    return parser


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scope-profiler export",
        description="Export HDF5 profiling files to machine-readable formats.",
    )
    subparsers = parser.add_subparsers(dest="export_kind", required=True)

    for kind, description in {
        "prof": "Export cProfile/pstats files.",
        "speedscope": "Export speedscope JSON files.",
        "chrome-trace": "Export Chrome Trace Event JSON files for Perfetto.",
        "json": "Export the whole run as a JSON profile.",
    }.items():
        export_parser = subparsers.add_parser(kind, help=description)
        export_parser.set_defaults(export_kind=kind)
        _add_common_export_args(export_parser)
        if kind == "prof":
            export_parser.add_argument(
                "--no-call-paths",
                action="store_true",
                help=(
                    "Aggregate every call of a region into one entry, instead "
                    "of keeping 'parent > child' paths apart."
                ),
            )
        if kind == "json":
            export_parser.add_argument(
                "--gzip",
                action="store_true",
                help="Write profile.json.gz instead of profile.json",
            )
            export_parser.add_argument(
                "--indent",
                type=int,
                default=None,
                metavar="N",
                help="Indent the JSON by N spaces (default: one line, smallest)",
            )

    plot_data = subparsers.add_parser(
        "plot-data",
        help="Export the exact data behind plot kinds.",
        description="Export the exact data behind plot kinds.",
    )
    plot_data.set_defaults(export_kind="plot-data")
    _add_common_export_args(plot_data)
    _add_timeline_args(plot_data)
    _add_data_export_args(plot_data)
    _add_callgraph_args(plot_data)
    plot_data.add_argument(
        "--plots",
        "-p",
        nargs="*",
        choices=list(_PLOT_CATALOG),
        default=None,
        help=_plots_help(),
    )
    _add_duration_args(plot_data)
    plot_data.add_argument("--bins", type=int, default=30, metavar="N")
    plot_data.add_argument(
        "--imbalance-metric",
        choices=["avg", "min", "max", "total"],
        default="total",
        help="Per-call duration statistic exported for imbalance data.",
    )
    plot_data.add_argument(
        "--likwid-metric",
        type=str,
        default=None,
        metavar="NAME",
        help="LIKWID metric name; required if likwid is selected.",
    )
    plot_data.add_argument(
        "--x",
        type=str,
        default="num_ranks",
        metavar="FIELD",
        help="Speedup x-axis field.",
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


def _print_plot_list() -> None:
    print("presets:")
    print("  default    " + ", ".join(sorted(_DEFAULT_PLOTS)))
    print("  quick      " + ", ".join(sorted(_QUICK_PLOTS)))
    print("  all        every plot except counter plots (select those explicitly)")
    print("\nplots:")
    for name, (description, is_default) in _PLOT_CATALOG.items():
        marker = "default" if is_default else "optional"
        print(f"  {name:<10} {description} ({marker})")


def _normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.files = expand_file_patterns(args.files, parser)
    args.combine_regions = parse_region_groups(
        getattr(args, "combine_regions", None), parser
    )

    if args.ranks:
        ranks = []
        for spec in args.ranks:
            ranks.extend(parse_ranks(spec))
        args.ranks = sorted(set(ranks))


def _load_runs(args: argparse.Namespace, parser: argparse.ArgumentParser):
    runs = [read_profile(file_path) for file_path in args.files]
    if args.label is not None:
        if len(args.label) != len(runs):
            parser.error(
                f"--label given {len(args.label)} time(s) for "
                f"{len(runs)} file(s); pass one per file, in order."
            )
        for run, label in zip(runs, args.label):
            run.label = label
    return runs


def _has_timing_data(runs) -> bool:
    """Return whether any selected file recorded timed region calls."""
    return any(
        len(region[rank].durations)
        for run in runs
        for region in run.get_regions()
        for rank in region.regions
    )


def _report_no_timing_data(runs) -> None:
    print("No timing data found - these files recorded no calls.\n")
    for run in runs:
        print(f"{run.file_path}:")
        for region in run.get_regions():
            total = sum(r.num_calls for r in region.regions.values())
            print(f"  {region.name}: {total} calls")


def _selected_plots(args: argparse.Namespace) -> set[str]:
    kind = args.plot_kind
    if kind == "default":
        return set(_DEFAULT_PLOTS)
    if kind == "quick":
        return set(_QUICK_PLOTS)
    if kind == "all":
        plots = set(_PLOT_CATALOG)
        plots.difference_update({"likwid", "perf_events"})
        return plots
    return {kind}


def _plot_output_targets(
    args: argparse.Namespace, selected_plots: set[str]
) -> OutputTargets:
    if not args.output:
        return OutputTargets(directory=None, single_file=None, statistics_path=None)

    output_path = Path(args.output)
    is_single_plot_file = len(selected_plots) == 1 and output_path.suffix.lower() in {
        ".png",
        ".html",
        ".txt",
    }
    if is_single_plot_file:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return OutputTargets(
            directory=None,
            single_file=str(output_path),
            statistics_path=None,
        )

    os.makedirs(args.output, exist_ok=True)
    return OutputTargets(
        directory=args.output,
        single_file=None,
        statistics_path=os.path.join(args.output, "region_statistics.json"),
    )


def _plot_path(
    targets: OutputTargets,
    selected_plots: set[str],
    plot_name: str,
    default_filename: str,
    ext: str,
) -> str | None:
    if targets.single_file and selected_plots == {plot_name}:
        return targets.single_file
    if targets.directory:
        return os.path.join(targets.directory, f"{default_filename}.{ext}")
    return None


def _data_path(
    output_dir: str | None,
    selected_plots: set[str],
    plot_name: str,
    filename: str,
    fmt: str,
) -> str | None:
    if output_dir and plot_name in selected_plots:
        return os.path.join(output_dir, f"{filename}.{fmt}")
    return None


def _plot_options(args: argparse.Namespace, name: str):
    metric = getattr(args, "metric", None)
    imbalance_metric = getattr(args, "imbalance_metric", "total")
    return {
        "duration_metrics": getattr(args, "metrics", ["total"]),
        "sort_by": getattr(args, "sort_by", None),
        "top_n": getattr(args, "top_n", None),
        "combine_regions": getattr(args, "combine_regions", None),
        "stack_children": getattr(args, "stack_children", False),
        "log_scale": getattr(args, "log_scale", False),
        "histogram_bins": getattr(args, "bins", 30),
        "imbalance_metric": (
            metric if name == "imbalance" and metric else imbalance_metric
        ),
        "likwid_metric": (
            metric if name == "likwid" else getattr(args, "likwid_metric", None)
        ),
        "perf_event_metric": metric if name == "perf_events" else None,
        "speedup_x_field": getattr(args, "x", "num_ranks"),
        "start_time": getattr(args, "start_time", None),
        "end_time": getattr(args, "end_time", None),
        "min_duration": getattr(args, "min_duration", 0.0),
        "aggregate_calls": getattr(args, "aggregate_calls", 1),
        "collapse_depth": getattr(args, "collapse_depth", None),
        "timeline_bins": getattr(args, "bins", 200),
    }


def _render_selected_plots(
    args: argparse.Namespace,
    runs,
    selected_plots: set[str],
    parser: argparse.ArgumentParser,
    *,
    output_targets: OutputTargets,
    data_output_dir: str | None = None,
    data_format: str = "csv",
    render_images: bool = True,
) -> list[str]:
    if (
        "likwid" in selected_plots
        and not _plot_options(args, "likwid")["likwid_metric"]
    ):
        parser.error(
            "likwid requires --metric for plots or --likwid-metric for plot-data."
        )
    if (
        "perf_events" in selected_plots
        and not _plot_options(args, "perf_events")["perf_event_metric"]
    ):
        parser.error("perf_events requires --metric, e.g. --metric ipc.")

    ext = (
        "html"
        if getattr(args, "backend", "matplotlib") in {"plotly", "pyvis"}
        else "txt" if getattr(args, "backend", "matplotlib") == "plotext" else "png"
    )
    options = _plot_options(args, "")
    saved: list[str] = []

    def image_path(plot_name: str, filename: str) -> str | None:
        if not render_images:
            return None
        return _plot_path(output_targets, selected_plots, plot_name, filename, ext)

    if any(
        plot_name in selected_plots
        for plot_name in (
            "gantt",
            "flame_chart",
            "flame_graph",
            "durations",
            "timeseries",
            "histogram",
            "imbalance",
            "density",
        )
    ) and not _has_timing_data(runs):
        _report_no_timing_data(runs)
        return []

    gantt_data_path = _data_path(
        data_output_dir, selected_plots, "gantt", "gantt_data", data_format
    )
    density_data_path = _data_path(
        data_output_dir, selected_plots, "density", "timeline_density_data", data_format
    )
    flame_chart_data_path = _data_path(
        data_output_dir,
        selected_plots,
        "flame_chart",
        "flame_chart_data",
        data_format,
    )
    flame_graph_data_path = _data_path(
        data_output_dir,
        selected_plots,
        "flame_graph",
        "flame_graph_data",
        data_format,
    )
    callgraph_data_path = _data_path(
        data_output_dir, selected_plots, "callgraph", "callgraph_data", data_format
    )
    durations_data_path = _data_path(
        data_output_dir, selected_plots, "durations", "durations_data", data_format
    )
    timeseries_data_path = _data_path(
        data_output_dir,
        selected_plots,
        "timeseries",
        "duration_timeseries_data",
        data_format,
    )
    speedup_data_path = (
        _data_path(
            data_output_dir, selected_plots, "speedup", "speedup_data", data_format
        )
        if len(runs) > 1
        else None
    )
    weak_scaling_data_path = (
        _data_path(
            data_output_dir,
            selected_plots,
            "weak_scaling",
            "weak_scaling_data",
            data_format,
        )
        if len(runs) > 1
        else None
    )
    scaling_efficiency_data_path = _data_path(
        data_output_dir,
        selected_plots,
        "scaling_efficiency",
        "scaling_efficiency_data",
        data_format,
    )
    histogram_data_path = _data_path(
        data_output_dir, selected_plots, "histogram", "histogram_data", data_format
    )
    imbalance_data_path = _data_path(
        data_output_dir, selected_plots, "imbalance", "imbalance_data", data_format
    )
    rank_heatmap_data_path = _data_path(
        data_output_dir,
        selected_plots,
        "rank_heatmap",
        "rank_heatmap_data",
        data_format,
    )
    likwid_data_path = _data_path(
        data_output_dir, selected_plots, "likwid", "likwid_data", data_format
    )

    if "gantt" in selected_plots:
        path = image_path("gantt", "gantt_plot")
        plot_gantt(
            runs,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            cmap=args.cmap,
            data_filepath=gantt_data_path,
            data_format=data_format,
            backend=args.backend,
            min_duration=options["min_duration"],
            start_time=options["start_time"],
            end_time=options["end_time"],
            aggregate_calls=options["aggregate_calls"],
            collapse_depth=options["collapse_depth"],
        )
        saved.extend(path for path in (path, gantt_data_path) if path)

    if "density" in selected_plots:
        path = image_path("density", "timeline_density_plot")
        plot_timeline_density(
            runs,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            cmap=args.cmap,
            bins=options["timeline_bins"],
            min_duration=options["min_duration"],
            start_time=options["start_time"],
            end_time=options["end_time"],
            data_filepath=density_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, density_data_path) if path)

    if "flame_chart" in selected_plots:
        path = image_path("flame_chart", "flame_chart_plot")
        plot_flame_chart(
            runs,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            cmap=FLAME_CMAP if args.cmap == DEFAULT_CMAP else args.cmap,
            data_filepath=flame_chart_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, flame_chart_data_path) if path)

    if "flame_graph" in selected_plots:
        path = image_path("flame_graph", "flame_graph_plot")
        plot_flame_graph(
            runs,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            cmap=FLAME_CMAP if args.cmap == DEFAULT_CMAP else args.cmap,
            data_filepath=flame_graph_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, flame_graph_data_path) if path)

    if "callgraph" in selected_plots:
        path = image_path("callgraph", "callgraph_plot")
        plot_callgraph(
            runs[0],
            rank=(args.ranks[0] if args.ranks else 0),
            include=args.include,
            exclude=args.exclude,
            filepath=path,
            show=args.show,
            data_filepath=callgraph_data_path,
            data_format=data_format,
            backend=args.backend,
            compact=args.compact_callgraph,
            fluid=args.fluid_callgraph,
        )
        saved.extend(path for path in (path, callgraph_data_path) if path)

    if "durations" in selected_plots:
        path = image_path("durations", "durations_plot")
        durations_paths = []
        for metric in options["duration_metrics"]:
            durations_paths.extend(
                plot_durations(
                    runs,
                    metric=metric,
                    filepath=path,
                    show=args.show,
                    include=args.include,
                    exclude=args.exclude,
                    ranks=args.ranks,
                    sort_by=options["sort_by"],
                    top_n=options["top_n"],
                    combine_regions=options["combine_regions"],
                    stack_children=options["stack_children"],
                    cmap=args.cmap,
                    log_scale=options["log_scale"],
                    data_filepath=durations_data_path,
                    data_format=data_format,
                    backend=args.backend,
                )
            )
        saved.extend(str(path) for path in durations_paths if path)
        if durations_data_path:
            saved.append(durations_data_path)

    if "timeseries" in selected_plots:
        path = image_path("timeseries", "duration_timeseries_plot")
        plot_duration_timeseries(
            runs,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            cmap=args.cmap,
            log_scale=options["log_scale"],
            data_filepath=timeseries_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, timeseries_data_path) if path)

    if "histogram" in selected_plots:
        path = image_path("histogram", "histogram_plot")
        plot_duration_histogram(
            runs,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            bins=options["histogram_bins"],
            cmap=args.cmap,
            log_scale=options["log_scale"],
            data_filepath=histogram_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, histogram_data_path) if path)

    if "imbalance" in selected_plots:
        path = image_path("imbalance", "imbalance_plot")
        plot_imbalance(
            runs,
            metric=options["imbalance_metric"],
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            cmap=args.cmap,
            log_scale=options["log_scale"],
            data_filepath=imbalance_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, imbalance_data_path) if path)

    if "likwid" in selected_plots:
        path = image_path("likwid", "likwid_plot")
        plot_likwid(
            runs,
            metric=_plot_options(args, "likwid")["likwid_metric"],
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            cmap=args.cmap,
            log_scale=options["log_scale"],
            data_filepath=likwid_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, likwid_data_path) if path)

    if "perf_events" in selected_plots:
        path = image_path("perf_events", "perf_events_plot")
        plot_perf_events(
            runs,
            metric=_plot_options(args, "perf_events")["perf_event_metric"],
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            ranks=args.ranks,
            cmap=args.cmap,
            backend=args.backend,
        )
        if path:
            saved.append(path)

    if len(runs) > 1 and "speedup" in selected_plots:
        path = image_path("speedup", "speedup_plot")
        plot_speedup(
            runs,
            x_field=options["speedup_x_field"],
            ranks=args.ranks,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            cmap=args.cmap,
            data_filepath=speedup_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, speedup_data_path) if path)

    if len(runs) > 1 and "weak_scaling" in selected_plots:
        path = image_path("weak_scaling", "weak_scaling_plot")
        plot_weak_scaling(
            runs,
            x_field=options["speedup_x_field"],
            ranks=args.ranks,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            cmap=args.cmap,
            data_filepath=weak_scaling_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, weak_scaling_data_path) if path)

    if "rank_heatmap" in selected_plots:
        path = image_path("rank_heatmap", "rank_heatmap_plot")
        plot_rank_heatmap(
            runs,
            ranks=args.ranks,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            cmap=args.cmap,
            data_filepath=rank_heatmap_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, rank_heatmap_data_path) if path)

    if len(runs) > 1 and "scaling_efficiency" in selected_plots:
        path = image_path("scaling_efficiency", "scaling_efficiency_plot")
        plot_scaling_efficiency(
            runs,
            x_field=options["speedup_x_field"],
            ranks=args.ranks,
            filepath=path,
            show=args.show,
            include=args.include,
            exclude=args.exclude,
            cmap=args.cmap,
            data_filepath=scaling_efficiency_data_path,
            data_format=data_format,
            backend=args.backend,
        )
        saved.extend(path for path in (path, scaling_efficiency_data_path) if path)

    return saved


def main(argv: list[str] | None = None):
    """Render one plot kind or plot preset from HDF5 profiling data."""
    parser = build_parser()
    plot_argv = list(sys.argv[1:] if argv is None else argv)
    # Keep the concise form ``scope-profiler plot FILE [OPTIONS]`` as an
    # alias for the default preset. Explicit plot kinds continue to work as
    # subcommands, e.g. ``scope-profiler plot gantt FILE``.
    plot_kinds = {"list", "default", "all", "quick", *_PLOT_CATALOG, *_PLOT_ALIASES}
    if (
        plot_argv
        and not plot_argv[0].startswith("-")
        and plot_argv[0] not in plot_kinds
    ):
        plot_argv.insert(0, "default")
    if plot_argv and plot_argv[0] in _PLOT_ALIASES:
        plot_argv[0] = _PLOT_ALIASES[plot_argv[0]]
    args = parser.parse_args(plot_argv)

    if args.plot_kind == "list":
        _print_plot_list()
        return

    _normalize_args(args, parser)
    runs = _load_runs(args, parser)
    selected_plots = _selected_plots(args)
    if args.backend == "plotext":
        unsupported = sorted(selected_plots - _PLOTEXT_SIMPLE_PLOTS)
        if unsupported:
            parser.error(
                "--backend plotext supports simple plots only; unsupported plot(s): "
                + ", ".join(unsupported)
            )
    elif args.backend == "pyvis":
        unsupported = sorted(selected_plots - _PYVIS_PLOTS)
        if unsupported:
            parser.error(
                "--backend pyvis supports the interactive callgraph only; "
                "unsupported plot(s): "
                + ", ".join(unsupported)
                + ". Use --backend matplotlib or plotly for these plots."
            )
    output_targets = _plot_output_targets(args, selected_plots)

    saved = _render_selected_plots(
        args,
        runs,
        selected_plots,
        parser,
        output_targets=output_targets,
        render_images=True,
    )

    if output_targets.statistics_path:
        write_region_statistics_json(
            profiling_data=runs,
            filepath=output_targets.statistics_path,
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
        )
        saved.append(output_targets.statistics_path)

    if args.output and not args.show:
        print("Outputs saved to:\n  " + "\n  ".join(saved))


def export_main(argv: list[str] | None = None):
    """Export HDF5 profiling data without rendering plot images."""
    parser = build_export_parser()
    args = parser.parse_args(argv)
    _normalize_args(args, parser)
    runs = _load_runs(args, parser)
    os.makedirs(args.output, exist_ok=True)

    saved: list[str] = []
    if args.export_kind == "prof":
        prof_paths = export_prof(
            profiling_data=runs,
            filepath=os.path.join(args.output, "profile.prof"),
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
            call_paths=not args.no_call_paths,
            verbose=False,
        )
        saved.extend(str(path) for path in prof_paths)
    elif args.export_kind == "speedscope":
        speedscope_paths = export_speedscope(
            profiling_data=runs,
            filepath=os.path.join(args.output, "profile.speedscope.json"),
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
            verbose=False,
        )
        saved.extend(str(path) for path in speedscope_paths)
    elif args.export_kind == "chrome-trace":
        trace_paths = export_chrome_trace(
            profiling_data=runs,
            filepath=os.path.join(args.output, "profile.trace.json"),
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
            verbose=False,
        )
        saved.extend(str(path) for path in trace_paths)
    elif args.export_kind == "json":
        from scope_profiler.json_export import export_json

        json_paths = export_json(
            profiling_data=runs,
            filepath=os.path.join(args.output, f"profile.json{args.gzip * '.gz'}"),
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
            verbose=False,
            indent=args.indent,
        )
        saved.extend(str(path) for path in json_paths)
    elif args.export_kind == "plot-data":
        selected_plots = (
            set(args.plots) if args.plots is not None else set(_DEFAULT_PLOTS)
        )
        if "likwid" in selected_plots and not args.likwid_metric:
            parser.error("plot-data with likwid requires --likwid-metric.")
        if "perf_events" in selected_plots:
            parser.error(
                "plot-data does not yet support perf_events; use "
                "`scope-profiler plot perf_events --metric ...` instead."
            )
        args.show = False
        args.backend = "matplotlib"
        args.cmap = DEFAULT_CMAP
        saved.extend(
            _render_selected_plots(
                args,
                runs,
                selected_plots,
                parser,
                output_targets=OutputTargets(None, None, None),
                data_output_dir=args.output,
                data_format=args.format,
                render_images=False,
            )
        )
        statistics_path = os.path.join(args.output, "region_statistics.json")
        write_region_statistics_json(
            profiling_data=runs,
            filepath=statistics_path,
            ranks=args.ranks,
            include=args.include,
            exclude=args.exclude,
        )
        saved.append(statistics_path)

    print("Outputs saved to:\n  " + "\n  ".join(saved))
    if args.export_kind == "prof" and saved:
        print(f"\nView a .prof file with: snakeviz {saved[0]}")
    if args.export_kind == "json" and saved:
        print(f"\nRead one back with: scope-profiler inspect {saved[0]}")
    if args.export_kind == "speedscope" and saved:
        print(
            f"\nView {saved[0]} at https://www.speedscope.app "
            "(or: npx speedscope <file>)"
        )
    if args.export_kind == "chrome-trace" and saved:
        print(f"\nOpen {saved[0]} with https://ui.perfetto.dev or chrome://tracing")


if __name__ == "__main__":
    main()
