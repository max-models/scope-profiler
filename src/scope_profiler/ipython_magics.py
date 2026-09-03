"""Jupyter/IPython magics for scope-profiler.

Load with::

    %load_ext scope_profiler.ipython_magics

This is a thin adapter over the same API the rest of scope-profiler uses --
``ProfileManager.session``/``profile_region`` for measuring, ``ProfilingResults
.print_summary`` for the table, and :mod:`scope_profiler.diff` for comparisons.
Nothing about timing, aggregation, or comparing runs is reimplemented here.

Importing this module requires IPython. It is not a dependency of the base
package (``pip install scope-profiler`` alone does not pull it in) -- install
it with the ``notebook`` extra: ``pip install "scope-profiler[notebook]"``.

Eleven magics are registered:

``%%scope [name] [-q] [-p] [--include PATTERN]``
    Cell magic. Runs the cell inside a profiling region named ``name``
    (default ``"cell"``) and prints its summary table. The result is kept
    in memory under ``name`` for ``%scope_last``/``%scope_compare``.
``%scope_timeit [-n N] [-q] statement``
    Line magic. Runs ``statement`` ``N`` times (default 7) inside a
    profiling region named ``"timeit"`` and prints per-call timing
    statistics -- like ``%timeit``, but backed by scope-profiler's region
    timer and table instead of the ``timeit`` module.
``%%scope_line [name] [-q] [-Q]``
    Cell magic. Runs the cell with ``use_line_profiler=True`` and prints
    per-line timing for whatever the cell itself profiles -- a
    ``@ProfileManager.profile``-decorated function or a
    ``with ProfileManager.profile_region(...):`` block written in the cell,
    exactly as in a script.
``%%scope_recursive [name] [-q] [--include PATTERN]``
    Cell magic. Records every Python call the cell makes as its own region,
    with no decorators or ``profile_region`` blocks -- ``recursive_profile
    =True``, as ``scope-profiler run`` applies it to a script.
``%%scope_agg [name] [-q] [--include PATTERN]``
    Cell magic. Runs the cell with ``aggregation_mode=True``: counts,
    totals and min/max per region, no per-call timeline, for cells that
    enter a region far too often to store every event.
``%scope_load filepath [-n name] [-q]``
    Load an HDF5 run produced elsewhere (an MPI job, ``scope-profiler
    run``) into the same registry, so the other magics can compare against
    it.
``%scope_df [-n name] [--events] [--per-rank] [--include PATTERN]``
    Return a recorded run as a pandas DataFrame, for analysis in the
    notebook rather than a printed table.
``%scope_last [name] [-p] [--include PATTERN]``
    Reprint the summary table for a previous ``%%scope``/``%scope_timeit``
    run. Defaults to the most recently recorded one.
``%scope_compare [name_a name_b] [--metric METRIC] [--sort KEY]``
    Compare two previous runs region-by-region with
    ``scope_profiler.diff.diff_rows``/``print_diff_table``. With no names,
    compares the two most recently recorded runs.
``%scope_export filepath [-n name] [--format prof|speedscope]``
    Export a previous run to a ``.prof`` (pstats/snakeviz) or speedscope
    JSON file via ``scope_profiler.prof_export``/``speedscope_export``.
    Format defaults to whatever ``filepath``'s extension implies.
    ``--no-call-paths`` makes the ``.prof`` one entry per region instead of
    one per ``parent > child`` call path.
``%scope_reset [name]``
    Drop one recorded run, or with no argument, every run recorded so far.

Every run uses ``deactivate_file_output=True``: nothing is written to disk,
results live only in the kernel's memory for the duration of the session.
"""

from __future__ import annotations

import sys
from pathlib import Path

from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from scope_profiler.diff import METRICS, SORT_KEYS, diff_rows, print_diff_table
from scope_profiler.profile_manager import ProfileManager
from scope_profiler.region_profiler import LineProfilerRegion
from scope_profiler.results import ProfilingResults

# The root region every ProfileManager.session() creates (see
# _ProfilingSession.ROOT_REGION_NAME in profile_manager.py). Its own
# auto-registered "function" is _ProfilingSession.__enter__ itself, not
# anything from the user's cell, so %%scope_line skips it when printing
# line-by-line tables.
_SESSION_ROOT_REGION = "scope_profiler.session"


@magics_class
class ScopeMagics(Magics):
    """IPython magics wrapping :class:`~scope_profiler.ProfileManager`."""

    def __init__(self, shell) -> None:
        super().__init__(shell)
        self._runs: dict[str, ProfilingResults] = {}
        self._order: list[str] = []

    def _store(self, name: str, results: ProfilingResults) -> None:
        self._runs[name] = results
        if name in self._order:
            self._order.remove(name)
        self._order.append(name)

    def _lookup(self, name: str | None) -> tuple[str, ProfilingResults]:
        """Resolve ``name`` (or the most recent run, if None) to a stored result."""
        if name is None:
            if not self._order:
                raise UsageError(
                    "no scope-profiler runs recorded yet; run %%scope or "
                    "%scope_timeit first",
                )
            name = self._order[-1]
        if name not in self._runs:
            available = ", ".join(self._order) or "(none)"
            raise UsageError(f"no recorded run named {name!r}; available: {available}")
        return name, self._runs[name]

    @magic_arguments()
    @argument(
        "name",
        nargs="?",
        default=None,
        help="name to store the run under (default: 'cell')",
    )
    @argument(
        "-q",
        "--quiet",
        action="store_true",
        help="don't print the summary table",
    )
    @argument(
        "-p",
        "--plot",
        action="store_true",
        help="show a duration bar chart after the summary",
    )
    @argument(
        "--include",
        default=None,
        help="regex selecting which regions to show/plot",
    )
    @cell_magic("scope")
    def scope_cell(self, line, cell):
        """Profile a cell as one region and print its summary table."""
        args = parse_argstring(self.scope_cell, line)
        name = args.name or "cell"
        with (
            ProfileManager.session(
                return_results=True,
                verbose=False,
                deactivate_file_output=True,
            ) as run,
            ProfileManager.profile_region(name),
        ):
            self.shell.run_cell(cell)
        results = run.results
        self._store(name, results)
        if not args.quiet:
            print(f"%%scope {name!r}")
            results.print_summary(include=args.include, suppress_notes=True)
        if args.plot:
            from scope_profiler import plot_durations

            plot_durations(results, include=args.include, show=True, verbose=False)

    @line_magic("scope_timeit")
    def scope_timeit(self, line):
        """Run a statement N times inside a profiling region and print timing stats."""
        opts, stmt = self.parse_options(line, "n:q", posix=False, strict=False)
        stmt = stmt.strip()
        if not stmt:
            raise UsageError("usage: %scope_timeit [-n N] statement")
        number = int(opts.get("n", 7))
        name = "timeit"
        with ProfileManager.session(
            return_results=True,
            verbose=False,
            deactivate_file_output=True,
        ) as run:
            region = ProfileManager.profile_region(name)
            for _ in range(number):
                with region:
                    self.shell.ex(stmt)
        results = run.results
        self._store(name, results)
        if "q" not in opts:
            print(f"%scope_timeit {stmt!r} ({number} runs)")
            results.print_summary(
                columns=["region", "calls", "total", "avg", "min", "max"],
                suppress_notes=True,
            )

    @magic_arguments()
    @argument(
        "name",
        nargs="?",
        default=None,
        help="name for the cell's outer region (default: 'cell')",
    )
    @argument(
        "-q",
        "--quiet",
        action="store_true",
        help="don't print the region summary table",
    )
    @argument(
        "-Q",
        "--quiet-lines",
        action="store_true",
        help="don't print the line-by-line tables",
    )
    @cell_magic("scope_line")
    def scope_line_cell(self, line, cell):
        """Profile a cell with line_profiler enabled and print per-line stats.

        Only what the cell itself hands to line_profiler gets a line-by-line
        breakdown -- exactly as in a script (see
        ``examples/ex_line_profiling.py``). A cell with neither still gets
        the usual region summary table.

        Prefer decorating the function you care about with
        ``@ProfileManager.profile``: it registers cleanly regardless of
        where it is called from. A bare ``with ProfileManager.profile_region
        (...):`` at the cell's top level also works, but line_profiler's own
        handling of a module-level code object containing a nested ``def``
        can cut the printed table off at that ``def`` -- the same limitation
        a plain script hits, not something specific to the magic.
        """
        args = parse_argstring(self.scope_line_cell, line)
        name = args.name or "cell"
        with ProfileManager.session(
            return_results=True,
            verbose=False,
            deactivate_file_output=True,
            use_line_profiler=True,
        ) as run:
            self.shell.run_cell(cell)
        results = run.results
        self._store(name, results)
        if not args.quiet:
            print(f"%%scope_line {name!r}")
            results.print_summary(suppress_notes=True)
        if not args.quiet_lines:
            for region_name, region in ProfileManager.get_all_regions().items():
                if region_name == _SESSION_ROOT_REGION:
                    continue
                if isinstance(region, LineProfilerRegion):
                    region.print_stats()

    @magic_arguments()
    @argument(
        "name",
        nargs="?",
        default=None,
        help="name to store the run under (default: 'cell')",
    )
    @argument(
        "-q",
        "--quiet",
        action="store_true",
        help="don't print the summary table",
    )
    @argument("--include", default=None, help="regex selecting which regions to show")
    @cell_magic("scope_recursive")
    def scope_recursive_cell(self, line, cell):
        """Profile every Python call the cell makes, with no instrumentation.

        The cell needs no decorators and no ``profile_region`` blocks: each
        function called while it runs becomes its own region, named
        ``<module>.<qualname>`` -- what ``recursive_profile=True`` does for a
        decorated function, and what ``scope-profiler run`` does for a
        script. Use it to find *where* a cell's time goes before deciding
        what to instrument properly.

        As in a script, tracing is not limited to your own functions:
        library internals called from the cell are recorded too, which is
        thorough but noisy and adds per-call overhead. Keep the cell small,
        and use ``--include`` to filter the table.

        The cell runs via ``exec`` rather than IPython's own execution, so a
        trailing expression is not echoed as ``Out[n]``; assignments land in
        the notebook namespace as usual.
        """
        args = parse_argstring(self.scope_recursive_cell, line)
        name = args.name or "cell"
        # Compile through IPython's caching compiler, as run_cell does, so the
        # cell's source is registered for tracebacks and inspection rather
        # than showing up as unavailable dynamically evaluated code.
        source = self.shell.transform_cell(cell)
        filename = self.shell.compile.cache(
            source,
            self.shell.execution_count,
            raw_code=cell,
        )
        code = compile(source, filename, "exec")
        exc_info = None
        with ProfileManager.session(
            return_results=True,
            verbose=False,
            deactivate_file_output=True,
        ) as run:
            # Install the tracer directly, as run_script() does for a script,
            # rather than calling a ``@ProfileManager.profile(recursive=True)``
            # helper: a decorated function stays registered for the life of
            # the kernel, so every later session would rebind it -- and a
            # %%scope_line session would hand it to line_profiler and print a
            # table for this module's own code.
            region = ProfileManager.profile_region(name)
            # Resolved before tracing starts: anything touched inside the
            # traced window is recorded as a region, and ``shell.user_ns`` is
            # a property, so looking it up there would put IPython's own
            # accessor in the user's table.
            namespace = self.shell.user_ns
            prev_profiler = sys.getprofile()
            tracer = ProfileManager._get_recursive_tracer(
                root_frame=sys._getframe(),
                prev_profiler=prev_profiler,
            )
            sys.setprofile(tracer)
            try:
                with region:
                    exec(code, namespace)  # noqa: S102 - the cell's own code
            except Exception:
                # Only captured here. Reporting it is deferred until tracing
                # has stopped and the session is closed: IPython's traceback
                # machinery is itself Python code, so formatting it inside the
                # traced window would bury the cell's profile under a million
                # regions from ultratb/stack_data/executing.
                etype, evalue, tb = sys.exc_info()
                # Drop this method's frame, so what the user sees starts at
                # their own cell rather than at the exec() call below.
                exc_info = (etype, evalue, tb.tb_next if tb is not None else tb)
            finally:
                sys.setprofile(prev_profiler)
        results = run.results
        if exc_info is not None:
            # What run_cell() does for the other cell magics: report the error
            # and carry on, so a cell that fails halfway still shows how far
            # it got. KeyboardInterrupt is not caught, and still propagates.
            #
            # tb_offset is pinned rather than left to the renderer. This frame
            # is already gone from `exc_info` above, and how many *more*
            # frames IPython drops on its own has varied between versions:
            # one of them left the cell's frame nowhere to be seen (an
            # exception with no source at all), another showed this method
            # alongside the cell. Zero means "drop nothing further", so the
            # slice above is the only one that happens, on every version.
            self.shell.showtraceback(exc_info, tb_offset=0)
        self._store(name, results)
        if not args.quiet:
            print(f"%%scope_recursive {name!r}")
            results.print_summary(
                include=args.include,
                sort="total",
                suppress_notes=True,
            )

    @magic_arguments()
    @argument(
        "name",
        nargs="?",
        default=None,
        help="name to store the run under (default: 'cell')",
    )
    @argument(
        "-q",
        "--quiet",
        action="store_true",
        help="don't print the summary table",
    )
    @argument("--include", default=None, help="regex selecting which regions to show")
    @cell_magic("scope_agg")
    def scope_agg_cell(self, line, cell):
        """Profile a cell in aggregation mode: counts and totals, no timeline.

        ``aggregation_mode=True`` keeps only each region's count, inclusive
        total, min/max and exclusive total instead of every call's
        timestamps, so a cell that enters a region millions of times stays
        cheap and bounded in memory. The per-call event data the Gantt
        chart, ``%scope_df --events`` and the timeline exports need is not
        recorded; the summary table and ``%scope_compare`` work as usual.
        """
        args = parse_argstring(self.scope_agg_cell, line)
        name = args.name or "cell"
        with (
            ProfileManager.session(
                return_results=True,
                verbose=False,
                deactivate_file_output=True,
                aggregation_mode=True,
            ) as run,
            ProfileManager.profile_region(name),
        ):
            self.shell.run_cell(cell)
        results = run.results
        self._store(name, results)
        if not args.quiet:
            print(f"%%scope_agg {name!r}")
            results.print_summary(include=args.include, suppress_notes=True)

    @magic_arguments()
    @argument(
        "name",
        nargs="?",
        default=None,
        help="run to reprint (default: the most recent one)",
    )
    @argument(
        "-p",
        "--plot",
        action="store_true",
        help="show a duration bar chart after the summary",
    )
    @argument(
        "--include",
        default=None,
        help="regex selecting which regions to show/plot",
    )
    @line_magic("scope_last")
    def scope_last(self, line):
        """Reprint the summary table for a previous %%scope/%scope_timeit run."""
        args = parse_argstring(self.scope_last, line)
        name, results = self._lookup(args.name)
        print(f"{name!r} (last)")
        results.print_summary(include=args.include, suppress_notes=True)
        if args.plot:
            from scope_profiler import plot_durations

            plot_durations(results, include=args.include, show=True, verbose=False)

    @magic_arguments()
    @argument(
        "name_a",
        nargs="?",
        default=None,
        help="baseline run (default: second-most-recent)",
    )
    @argument(
        "name_b",
        nargs="?",
        default=None,
        help="candidate run (default: most recent)",
    )
    @argument("--metric", default="total", choices=METRICS, help="statistic to compare")
    @argument("--sort", default="delta", choices=SORT_KEYS, help="row ordering")
    @line_magic("scope_compare")
    def scope_compare(self, line):
        """Compare two previous runs region-by-region."""
        args = parse_argstring(self.scope_compare, line)
        if args.name_a is None and args.name_b is None:
            if len(self._order) < 2:
                raise UsageError(
                    "need two recorded runs to compare; run %%scope/%scope_timeit "
                    "twice first, or pass two names",
                )
            name_a, name_b = self._order[-2], self._order[-1]
        elif args.name_a is not None and args.name_b is not None:
            name_a, name_b = args.name_a, args.name_b
        else:
            raise UsageError(
                "pass either two run names or none (compares the two most recent)",
            )
        name_a, results_a = self._lookup(name_a)
        name_b, results_b = self._lookup(name_b)
        rows = diff_rows(results_a, results_b, metric=args.metric, sort=args.sort)
        print_diff_table(rows, metric=args.metric, title=f"{name_a!r} -> {name_b!r}")

    @magic_arguments()
    @argument("filepath", help="output path, e.g. 'run.prof' or 'run.speedscope.json'")
    @argument(
        "-n",
        "--name",
        default=None,
        help="run to export (default: the most recent one)",
    )
    @argument(
        "--format",
        dest="fmt",
        choices=("prof", "speedscope"),
        default=None,
        help="output format (default: inferred from filepath)",
    )
    @argument("--include", default=None, help="regex selecting which regions to export")
    @argument("--exclude", default=None, help="regex selecting which regions to drop")
    @argument(
        "--no-call-paths",
        action="store_true",
        help="prof only: one entry per region instead of per 'parent > child' path",
    )
    @line_magic("scope_export")
    def scope_export(self, line):
        """Export a previous run to a .prof or speedscope JSON file."""
        args = parse_argstring(self.scope_export, line)
        _, results = self._lookup(args.name)
        fmt = args.fmt
        if fmt is None:
            fmt = (
                "speedscope"
                if "speedscope" in args.filepath or args.filepath.endswith(".json")
                else "prof"
            )
        if fmt == "prof":
            from scope_profiler.prof_export import export_prof

            paths = export_prof(
                results,
                args.filepath,
                include=args.include,
                exclude=args.exclude,
                call_paths=not args.no_call_paths,
                verbose=False,
            )
        else:
            from scope_profiler.speedscope_export import export_speedscope

            paths = export_speedscope(
                results,
                args.filepath,
                include=args.include,
                exclude=args.exclude,
                verbose=False,
            )
        for path in paths:
            print(f"wrote {path}")

    @magic_arguments()
    @argument(
        "name",
        nargs="?",
        default=None,
        help="run to drop (default: clear every recorded run)",
    )
    @line_magic("scope_reset")
    def scope_reset(self, line):
        """Drop one recorded run, or every recorded run with no argument."""
        args = parse_argstring(self.scope_reset, line)
        if args.name is None:
            self._runs.clear()
            self._order.clear()
            print("cleared all recorded scope-profiler runs")
            return
        name, _ = self._lookup(args.name)
        del self._runs[name]
        self._order.remove(name)
        print(f"cleared recorded run {name!r}")

    @magic_arguments()
    @argument("filepath", help="HDF5 profiling file to load, e.g. 'run_128ranks.h5'")
    @argument(
        "-n",
        "--name",
        default=None,
        help="name to store it under (default: the file's stem)",
    )
    @argument(
        "-q",
        "--quiet",
        action="store_true",
        help="don't print the summary table",
    )
    @line_magic("scope_load")
    def scope_load(self, line):
        """Load a profiling file into the notebook's recorded runs.

        Puts a run produced outside this notebook -- an MPI job, a
        ``scope-profiler run``, a colleague's file -- under the same name
        registry as ``%%scope``, so ``%scope_last``, ``%scope_compare``,
        ``%scope_df`` and ``%scope_export`` treat it like any cell's run.
        That makes "is my notebook version faster than the cluster run?" a
        one-liner.
        """
        from scope_profiler.h5reader import read_h5

        args = parse_argstring(self.scope_load, line)
        results = read_h5(args.filepath)
        name = args.name or Path(args.filepath).stem
        self._store(name, results)
        if not args.quiet:
            print(f"%scope_load {args.filepath} as {name!r}")
            results.print_summary(suppress_notes=True)

    @magic_arguments()
    @argument(
        "-n",
        "--name",
        default=None,
        help="run to convert (default: the most recent one)",
    )
    @argument(
        "--events",
        action="store_true",
        help="one row per recorded call instead of one per region",
    )
    @argument(
        "--per-rank",
        action="store_true",
        help="one row per (region, rank) instead of one aggregated row",
    )
    @argument("--include", default=None, help="regex selecting which regions to keep")
    @argument("--exclude", default=None, help="regex selecting which regions to drop")
    @line_magic("scope_df")
    def scope_df(self, line):
        """Return a recorded run as a pandas DataFrame.

        The frame is the magic's return value, so it renders as a table on
        its own and can be captured for further analysis::

            df = %scope_df
            df.nlargest(5, "total_duration")

        Needs pandas (the ``pproc`` extra). ``--events`` needs per-call event
        data, which an aggregation-mode run (``%%scope_agg``) does not record
        -- that comes back as an empty frame, with a note saying why.
        """
        args = parse_argstring(self.scope_df, line)
        _, results = self._lookup(args.name)
        if args.events:
            if args.per_rank:
                raise UsageError("--per-rank does not apply to --events")
            frame = results.to_events_dataframe(
                include=args.include,
                exclude=args.exclude,
            )
            if frame.empty:
                # An aggregation-mode run keeps per-region statistics but no
                # per-call timeline, and reports event data as "available",
                # so an empty frame here is otherwise unexplained.
                print(
                    "no events recorded: the run was either filtered down to "
                    "nothing or recorded with %%scope_agg (aggregation mode "
                    "keeps no per-call timeline)",
                )
            return frame
        return results.to_dataframe(
            include=args.include,
            exclude=args.exclude,
            per_rank=args.per_rank,
        )


def load_ipython_extension(ipython) -> None:
    """Register the magics: ``%load_ext scope_profiler.ipython_magics``."""
    ipython.register_magics(ScopeMagics)
