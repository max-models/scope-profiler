"""Command-line entry point: ``scope-profiler <command> ...``.

Also runnable as ``python -m scope_profiler <command> ...``.

Every subcommand that takes a profile reads an HDF5 file or a JSON one
(:mod:`scope_profiler.json_export`) interchangeably; the format is chosen by
the file name.

The subcommands:

- ``scope-profiler run script.py [args...]`` -- profiles a script's function
  calls without requiring any decorators or context managers in the script
  itself, similar to ``python -m cProfile``. By default only the script's
  own code is instrumented (the standard library and installed packages are
  skipped) to keep overhead low; pass ``--all`` to trace everything. The
  extension of ``-o`` picks the output format: HDF5 by default, a JSON
  profile for ``.json``/``.json.gz``, a rendered report for ``.html``.
- ``scope-profiler plot <kind> file.h5 [...]`` -- reads merged HDF5 profiling
  output and renders Gantt/flame/duration/speedup charts. See
  ``scope_profiler.post_processing`` for its full set of options.
- ``scope-profiler export <kind> file.h5 [...]`` -- writes plot data,
  cProfile/pstats files, speedscope or Chrome Trace JSON, or the whole run as
  a JSON profile,
  without rendering charts.
- ``scope-profiler inspect file.h5 [...]`` -- prints the run metadata and a
  per-region statistics table (including LIKWID hardware counters, when the
  run recorded any) for merged HDF5 profiling output, without producing any
  plots. See ``scope_profiler.inspection``.
- ``scope-profiler report file.h5 -o report.html`` -- writes a standalone HTML
  summary with metadata and per-region timing statistics.
- ``scope-profiler tui file.h5`` -- opens an interactive Textual browser for
  metadata, region statistics, per-rank calls, LIKWID counters and the raw
  HDF5 tree.
- ``scope-profiler line-profile file.h5 [...]`` -- prints persisted
  line-profiler timings from an HDF5 profile.
- ``scope-profiler diff a.h5 b.h5`` -- compares region statistics between two
  merged HDF5 profiling files, region by region, so a regression (or
  improvement) between two runs shows up in one table. See
  ``scope_profiler.diff``.
- ``scope-profiler check a.h5 b.h5`` -- applies a regression budget and
  returns a CI-friendly exit code.
- ``scope-profiler benchmark run config.toml`` -- runs a repeatable benchmark
  with a correctness gate and writes a JSON manifest.
- ``scope-profiler benchmark compare baseline.json candidate.json`` -- makes a
  median-based keep/reject decision for an AI agent or CI.
- ``scope-profiler import-native traces/ -o out.h5`` -- converts the trace
  files written by the Fortran region API
  (``scope_profiler/fortran/scope_profiler.f90``)
  into the usual HDF5 output, so a Fortran run post-processes exactly like a
  Python one. See ``scope_profiler.native_trace``.
"""

import argparse
import os
import sys

from scope_profiler import __version__
from scope_profiler.profile_manager import ProfileManager


def _parse_run_args(argv):
    parser = argparse.ArgumentParser(
        prog="scope-profiler run",
        description="Profile a script's function calls without modifying it.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default=None,
        help="Path of the output file (default: profiling_data.h5). The "
        "extension picks the format: .h5 for HDF5, .json / .json.gz for a "
        "JSON profile, .html for a rendered report",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="TOML file containing profiling settings ([profiling] table)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the per-region summary printed after the run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also instrument standard-library and installed-package calls "
        "(default: only the script's own code)",
    )
    parser.add_argument(
        "--line-profile",
        action="store_true",
        default=None,
        help="Also collect line-by-line timings via line_profiler "
        "(requires scope-profiler[line-profiler])",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        default=None,
        help="Record allocations with Memray (requires scope-profiler[extras]); "
        "writes an adjacent .memray.bin capture",
    )
    parser.add_argument(
        "--buffer-limit",
        type=int,
        default=None,
        help="Initial buffer capacity per region; grows as needed (default: 1024)",
    )
    parser.add_argument(
        "--aggregation-mode",
        action="store_true",
        default=None,
        help="Record aggregate timing statistics only; omit per-call timeline events.",
    )
    parser.add_argument("script", help="Script to run and profile")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the script",
    )
    return parser.parse_args(argv)


def _convert_run_output(profile_path, output_path, quiet=False):
    """Render the run's HDF5 output as JSON or HTML, then drop the HDF5.

    Under MPI only rank 0 holds the merged file, and only rank 0 wrote it, so
    only rank 0 converts and unlinks it.
    """
    from pathlib import Path

    from scope_profiler.profile_io import (
        FORMAT_JSON,
        profile_format,
        read_profile,
        write_profile,
    )

    config = ProfileManager.get_config()
    if config.comm is not None and config.comm.Get_rank() != 0:
        return
    if config.deactivate_profiling or config.deactivate_file_output:
        return
    try:
        results = read_profile(profile_path)
        # The temporary HDF5 is about to go, so a summary naming it would
        # print `scope-profiler inspect` hints for a file that no longer
        # exists. A JSON profile reads back exactly the same way, so the
        # summary points at that instead.
        if profile_format(output_path) == FORMAT_JSON:
            results._file_path = Path(output_path)
        written = write_profile(results, output_path)
        if not quiet:
            results.print_summary()
            print(f"\nwrote {written}")
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass


def _run(argv):
    """Handle ``scope-profiler run``: profile a script and write its output.

    The output format follows the ``-o`` extension, as viztracer's does: HDF5
    unless the name asks for a JSON profile or an HTML report. The run itself
    always writes HDF5 -- that is the format the parallel and rank-by-rank
    writers produce -- and the requested format is rendered from it once the
    script is done, so nothing about the measured run changes with ``-o``.
    """
    from scope_profiler.profile_io import FORMAT_HDF5, profile_format

    args = _parse_run_args(argv)

    if not os.path.isfile(args.script):
        print(
            f"scope-profiler run: can't open file {args.script!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    convert = args.outfile is not None and profile_format(args.outfile) != FORMAT_HDF5
    # Beside the requested file, so the writer's atomic publish stays a rename
    # within one directory, and so a read-only $TMPDIR cannot break the run.
    profile_path = args.outfile + ".scope-profiler.h5" if convert else args.outfile

    ProfileManager.setup(
        # ``run`` historically enables recursive profiling.  A TOML file may
        # override it, while the no-config path keeps that default.
        recursive_profile=True if args.config is None else None,
        use_likwid=None,
        use_line_profiler=args.line_profile,
        use_memray=args.memory_profile,
        buffer_limit=args.buffer_limit,
        aggregation_mode=args.aggregation_mode,
        file_path=profile_path,
        config_path=args.config,
    )

    try:
        ProfileManager.run_script(
            args.script,
            script_args=args.script_args,
            only_user_code=not args.all,
        )
    finally:
        # The summary names the file it came from, so with a conversion still
        # to come it is printed afterwards, against the file the user asked
        # for, rather than against a temporary that is about to be deleted.
        ProfileManager.finalize(verbose=not args.quiet and not convert)
        if convert:
            _convert_run_output(profile_path, args.outfile, quiet=args.quiet)


def _plot(argv):
    """Handle ``scope-profiler plot``: delegate to the post-processing CLI."""
    from scope_profiler.post_processing import main as plot_main

    return plot_main(argv)


def _export(argv):
    """Handle ``scope-profiler export``: delegate to the export CLI."""
    from scope_profiler.post_processing import export_main

    return export_main(argv)


def _inspect(argv):
    """Handle ``scope-profiler inspect``: delegate to the inspection CLI."""
    from scope_profiler.inspection import main as inspect_main

    return inspect_main(argv)


def _report(argv):
    """Handle ``scope-profiler report``: write a standalone HTML summary."""
    from scope_profiler.html_report import create_html_report
    from scope_profiler.post_processing import expand_file_patterns, parse_ranks
    from scope_profiler.summary import REGION_TABLE_COLUMNS, SORT_KEYS

    parser = argparse.ArgumentParser(
        prog="scope-profiler report",
        description="Write a standalone HTML report from profiling HDF5 files.",
    )
    parser.add_argument(
        "files", nargs="+", help="Profiling HDF5 files or glob patterns"
    )
    parser.add_argument(
        "-o", "--output", required=True, metavar="PATH", help="HTML file to write"
    )
    parser.add_argument(
        "--include", nargs="+", help="Only include matching region names"
    )
    parser.add_argument("--exclude", nargs="+", help="Exclude matching region names")
    parser.add_argument("--ranks", nargs="+", help="Ranks to include, e.g. 0 2 or 0-3")
    parser.add_argument(
        "--sort",
        choices=SORT_KEYS,
        default="start",
        help="Region ordering (default: start)",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        choices=REGION_TABLE_COLUMNS,
        help="Region table columns",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Omit embedded interactive charts",
    )
    parser.add_argument(
        "--charts-cdn",
        action="store_true",
        help=(
            "Load Plotly from https://cdn.plot.ly instead of embedding it: "
            "~4.7 MB smaller, but the charts then need a network connection"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated report in the default browser",
    )
    args = parser.parse_args(argv)
    ranks = None
    if args.ranks:
        ranks = sorted({rank for spec in args.ranks for rank in parse_ranks(spec)})
    output = create_html_report(
        expand_file_patterns(args.files, parser),
        args.output,
        include=args.include,
        exclude=args.exclude,
        ranks=ranks,
        sort=args.sort,
        columns=args.columns,
        include_charts=not args.no_charts,
        charts_cdn=args.charts_cdn,
    )
    print(f"Report written to: {output}")
    if args.show:
        import webbrowser

        webbrowser.open(output.resolve().as_uri())
    return 0


def _tui(argv):
    """Handle ``scope-profiler tui``: open the interactive HDF5 browser."""
    from scope_profiler.tui import main as tui_main

    return tui_main(argv)


def _line_profile(argv):
    """Handle ``scope-profiler line-profile``."""
    from scope_profiler.line_profile_cli import main as line_profile_main

    return line_profile_main(argv)


def _diff(argv):
    """Handle ``scope-profiler diff``: delegate to the diff CLI."""
    from scope_profiler.diff import main as diff_main

    return diff_main(argv)


def _check(argv):
    """Handle ``scope-profiler check``: enforce a regression budget."""
    from scope_profiler.diff import check_main

    return check_main(argv)


def _benchmark(argv):
    """Run or compare declarative, repeated benchmark manifests."""
    from scope_profiler.benchmark import (
        BenchmarkError,
        compare_benchmarks,
        load_config,
        run_benchmark,
    )

    parser = argparse.ArgumentParser(
        prog="scope-profiler benchmark",
        description="Run or compare repeatable AI/CI benchmark workflows.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="run a benchmark config")
    run_parser.add_argument("config", help="benchmark TOML configuration")
    run_parser.add_argument(
        "--label", default="candidate", help="baseline or candidate label"
    )
    run_parser.add_argument("--json", action="store_true", help="print only JSON")
    compare_parser = subparsers.add_parser(
        "compare", help="compare two benchmark manifests"
    )
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    compare_parser.add_argument("--json", action="store_true", help="print only JSON")
    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            result = run_benchmark(load_config(args.config), label=args.label)
            exit_code = 0 if result["correctness"]["passed"] else 1
        else:
            result = compare_benchmarks(args.baseline, args.candidate)
            exit_code = 0 if result["decision"] == "keep" else 1
    except BenchmarkError as exc:
        parser.error(str(exc))
    print(__import__("json").dumps(result, indent=None if args.json else 2))
    if exit_code:
        raise SystemExit(exit_code)
    return 0


def _import_fortran(argv):
    """Handle ``scope-profiler import-native``: Fortran traces -> HDF5."""
    from scope_profiler.native_trace import TRACE_SUFFIX, convert_traces

    parser = argparse.ArgumentParser(
        prog="scope-profiler import-native",
        description=(
            "Convert the trace files written by the Fortran region API into a "
            "standard scope-profiler HDF5 file."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=f"trace files (*{TRACE_SUFFIX}) and/or directories containing them",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="profiling_data.h5",
        help="HDF5 file to write (default: profiling_data.h5)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="name for the run in summaries, charts and exports",
    )
    parser.add_argument(
        "--merge",
        metavar="PROFILE.h5",
        default=None,
        help="an existing profile to combine the traces with, for a run whose "
        "Python side was profiled separately; region names must not clash",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="do not print the summary table",
    )
    args = parser.parse_args(argv)

    if args.merge is None:
        path = convert_traces(args.inputs, args.output, label=args.label)
    else:
        from scope_profiler.h5reader import read_h5
        from scope_profiler.native_trace import load_traces, write_results
        from scope_profiler.results import merge_results

        path = write_results(
            merge_results(
                read_h5(args.merge),
                load_traces(args.inputs),
                label=args.label,
                file_path=args.output,
            ),
            args.output,
        )

    if not args.quiet:
        from scope_profiler.h5reader import read_h5

        read_h5(path).print_summary()
        print(f"\nwrote {path}")
    return 0


_COMMANDS = {
    "run": _run,
    "plot": _plot,
    "export": _export,
    "inspect": _inspect,
    "report": _report,
    "tui": _tui,
    "line-profile": _line_profile,
    "diff": _diff,
    "check": _check,
    "benchmark": _benchmark,
    "import-native": _import_fortran,
}


def main(argv=None):
    """Dispatch to the requested subcommand."""
    argv = sys.argv[1:] if argv is None else list(argv)

    parser = argparse.ArgumentParser(
        prog="scope-profiler",
        description="Profile scripts and post-process the resulting HDF5 output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "run",
        add_help=False,
        help="Run and profile a script (see `scope-profiler run --help`)",
    )
    subparsers.add_parser(
        "plot",
        add_help=False,
        help="Post-process and plot HDF5 profiling data "
        "(see `scope-profiler plot --help`)",
    )
    subparsers.add_parser(
        "export",
        add_help=False,
        help="Export HDF5 profiling data without rendering charts "
        "(see `scope-profiler export --help`)",
    )
    subparsers.add_parser(
        "inspect",
        add_help=False,
        help="Print metadata and region statistics of HDF5 profiling data "
        "(see `scope-profiler inspect --help`)",
    )
    subparsers.add_parser(
        "report",
        add_help=False,
        help="Write a standalone HTML report (see `scope-profiler report --help`)",
    )
    subparsers.add_parser(
        "tui",
        add_help=False,
        help="Interactively browse HDF5 profiling data "
        "(see `scope-profiler tui --help`)",
    )
    subparsers.add_parser(
        "line-profile",
        add_help=False,
        help="Print persisted line-profiler timings from HDF5 data "
        "(see `scope-profiler line-profile --help`)",
    )
    subparsers.add_parser(
        "diff",
        add_help=False,
        help="Compare region statistics between two HDF5 profiling files "
        "(see `scope-profiler diff --help`)",
    )
    subparsers.add_parser(
        "check",
        add_help=False,
        help="Fail on profiling regressions (see `scope-profiler check --help`)",
    )
    subparsers.add_parser(
        "benchmark",
        add_help=False,
        help="Run or compare repeatable benchmarks (see `scope-profiler benchmark --help`)",
    )
    subparsers.add_parser(
        "import-native",
        add_help=False,
        help="Convert Fortran API trace files into HDF5 "
        "(see `scope-profiler import-native --help`)",
    )

    if not argv:
        parser.print_help()
        raise SystemExit(1)
    if argv[0] in ("-h", "--help", "--version"):
        parser.parse_args(argv)  # prints help/version and exits(0)
        return

    command, *rest = argv
    handler = _COMMANDS.get(command)
    if handler is None:
        parser.error(
            f"argument command: invalid choice: {command!r} "
            f"(choose from {', '.join(map(repr, _COMMANDS))})"
        )
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
