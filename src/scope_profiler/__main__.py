"""Command-line entry point: ``scope-profiler script.py [args...]``.

Also runnable as ``python -m scope_profiler script.py [args...]``.

Profiles a script's function calls without requiring any decorators or
context managers in the script itself, similar to ``python -m cProfile``.
By default only the script's own code is instrumented (the standard library
and installed packages are skipped) to keep overhead low; pass ``--all`` to
trace everything.
"""

import argparse
import os
import sys

from scope_profiler.profile_manager import ProfileManager


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="scope-profiler",
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


def main(argv=None):
    """Parse CLI args, run the target script under profiling, and finalize."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not os.path.isfile(args.script):
        print(
            f"scope-profiler: can't open file {args.script!r}",
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


if __name__ == "__main__":
    main()
