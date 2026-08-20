"""Command-line entry point: ``scope-profiler <command> ...``.

Also runnable as ``python -m scope_profiler <command> ...``.

Six subcommands:

- ``scope-profiler run script.py [args...]`` -- profiles a script's function
  calls without requiring any decorators or context managers in the script
  itself, similar to ``python -m cProfile``. By default only the script's
  own code is instrumented (the standard library and installed packages are
  skipped) to keep overhead low; pass ``--all`` to trace everything.
- ``scope-profiler plot <kind> file.h5 [...]`` -- reads merged HDF5 profiling
  output and renders Gantt/flame/duration/speedup charts. See
  ``scope_profiler.post_processing`` for its full set of options.
- ``scope-profiler export <kind> file.h5 [...]`` -- writes plot data,
  cProfile/pstats files, or speedscope JSON without rendering charts.
- ``scope-profiler inspect file.h5 [...]`` -- prints the run metadata and a
  per-region statistics table (including LIKWID hardware counters, when the
  run recorded any) for merged HDF5 profiling output, without producing any
  plots. See ``scope_profiler.inspection``.
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
        default="profiling_data.h5",
        help="Path to the merged HDF5 output file (default: profiling_data.h5)",
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
        help="Also collect line-by-line timings via line_profiler "
        "(requires scope-profiler[line-profiler])",
    )
    parser.add_argument(
        "--buffer-limit",
        type=int,
        default=1024,
        help="Initial buffer capacity per region; grows as needed (default: 1024)",
    )
    parser.add_argument("script", help="Script to run and profile")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the script",
    )
    return parser.parse_args(argv)


def _run(argv):
    """Handle ``scope-profiler run``: profile a script and write its HDF5 output."""
    args = _parse_run_args(argv)

    if not os.path.isfile(args.script):
        print(
            f"scope-profiler run: can't open file {args.script!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ProfileManager.setup(
        recursive_profile=True,
        use_likwid=False,
        use_line_profiler=args.line_profile,
        buffer_limit=args.buffer_limit,
        file_path=args.outfile,
    )

    try:
        ProfileManager.run_script(
            args.script,
            script_args=args.script_args,
            only_user_code=not args.all,
        )
    finally:
        ProfileManager.finalize(verbose=not args.quiet)


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
    "tui": _tui,
    "line-profile": _line_profile,
    "diff": _diff,
    "check": _check,
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
    main()
