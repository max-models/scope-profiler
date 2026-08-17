"""Command-line entry point: ``scope-profiler <command> ...``.

Also runnable as ``python -m scope_profiler <command> ...``.

Five subcommands:

- ``scope-profiler run script.py [args...]`` -- profiles a script's function
  calls without requiring any decorators or context managers in the script
  itself, similar to ``python -m cProfile``. By default only the script's
  own code is instrumented (the standard library and installed packages are
  skipped) to keep overhead low; pass ``--all`` to trace everything.
- ``scope-profiler plot file.h5 [...]`` -- reads merged HDF5 profiling
  output and renders Gantt/flame/duration/speedup charts, or exports the
  underlying data. See ``scope_profiler.post_processing`` for its full set
  of options. The old name ``pproc`` still works as a deprecated alias.
- ``scope-profiler inspect file.h5 [...]`` -- prints the run metadata and a
  per-region statistics table (including LIKWID hardware counters, when the
  run recorded any) for merged HDF5 profiling output, without producing any
  plots. See ``scope_profiler.inspection``.
- ``scope-profiler diff a.h5 b.h5`` -- compares region statistics between two
  merged HDF5 profiling files, region by region, so a regression (or
  improvement) between two runs shows up in one table. See
  ``scope_profiler.diff``.
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
        use_line_profiler=False,
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


def _inspect(argv):
    """Handle ``scope-profiler inspect``: delegate to the inspection CLI."""
    from scope_profiler.inspection import main as inspect_main

    return inspect_main(argv)


def _diff(argv):
    """Handle ``scope-profiler diff``: delegate to the diff CLI."""
    from scope_profiler.diff import main as diff_main

    return diff_main(argv)


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
    "inspect": _inspect,
    "diff": _diff,
    "import-native": _import_fortran,
}

# Old command names kept working for backwards compatibility. Deliberately
# not part of _COMMANDS or the add_subparsers() calls below, so they stay
# out of --help and out of the "invalid choice" error listing -- they still
# work if a user types them, with a deprecation warning.
_DEPRECATED_ALIASES = {
    "pproc": "plot",
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
        "inspect",
        add_help=False,
        help="Print metadata and region statistics of HDF5 profiling data "
        "(see `scope-profiler inspect --help`)",
    )
    subparsers.add_parser(
        "diff",
        add_help=False,
        help="Compare region statistics between two HDF5 profiling files "
        "(see `scope-profiler diff --help`)",
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
    if command in _DEPRECATED_ALIASES:
        target = _DEPRECATED_ALIASES[command]
        print(
            f"scope-profiler: {command!r} is deprecated and will be removed "
            f"in a future release; use {target!r} instead.",
            file=sys.stderr,
        )
        command = target
    handler = _COMMANDS.get(command)
    if handler is None:
        parser.error(
            f"argument command: invalid choice: {command!r} "
            f"(choose from {', '.join(map(repr, _COMMANDS))})"
        )
    return handler(rest)


if __name__ == "__main__":
    main()
