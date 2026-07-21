"""Command-line entry point: ``scope-profiler <command> ...``.

Also runnable as ``python -m scope_profiler <command> ...``.

Two subcommands:

- ``scope-profiler run script.py [args...]`` -- profiles a script's function
  calls without requiring any decorators or context managers in the script
  itself, similar to ``python -m cProfile``. By default only the script's
  own code is instrumented (the standard library and installed packages are
  skipped) to keep overhead low; pass ``--all`` to trace everything.
- ``scope-profiler pproc file.h5 [...]`` -- reads merged HDF5 profiling
  output and renders Gantt/flame/duration/speedup charts. See
  ``scope_profiler.post_processing`` for its full set of options.
"""

import argparse
import os
import sys

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
        default=100_000,
        help="Max buffered calls per region before flushing to disk (default: 100000)",
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
        time_trace=True,
        flush_to_disk=True,
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


def _pproc(argv):
    """Handle ``scope-profiler pproc``: delegate to the post-processing CLI."""
    from scope_profiler.post_processing import main as pproc_main

    return pproc_main(argv)


_COMMANDS = {
    "run": _run,
    "pproc": _pproc,
}


def main(argv=None):
    """Dispatch to the ``run`` or ``pproc`` subcommand."""
    argv = sys.argv[1:] if argv is None else list(argv)

    parser = argparse.ArgumentParser(
        prog="scope-profiler",
        description="Profile scripts and post-process the resulting HDF5 output.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "run",
        add_help=False,
        help="Run and profile a script (see `scope-profiler run --help`)",
    )
    subparsers.add_parser(
        "pproc",
        add_help=False,
        help="Post-process and plot HDF5 profiling data "
        "(see `scope-profiler pproc --help`)",
    )

    if not argv:
        parser.print_help()
        raise SystemExit(1)
    if argv[0] in ("-h", "--help"):
        parser.parse_args(argv)  # prints help and exits(0)
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
