"""MCP server exposing scope-profiler to AI coding agents (e.g. Claude Code).

This module is a thin adapter: every tool below immediately delegates to a
plain function in :mod:`scope_profiler.mcp_server.tools`, which in turn
delegates to the existing ``scope_profiler`` Python API (the same functions
``inspect``/``plot``/``diff``/``run`` use). No profiling, HDF5-parsing,
summary or comparison logic lives here.

Run directly::

    scope-profiler-mcp

or::

    python -m scope_profiler.mcp_server

Requires the optional ``mcp`` extra: ``pip install "scope-profiler[mcp]"``.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from scope_profiler.mcp_server import tools


def create_server() -> FastMCP:
    """Build the MCP server and register every tool. Does not start it."""
    server = FastMCP(
        "scope-profiler",
        instructions=(
            "Tools for inspecting and comparing scope-profiler HDF5 profiling "
            "output, and for running repeatable benchmark workflows. Typical "
            "workflow: run_benchmark -> edit code -> run_benchmark -> "
            "compare_benchmarks; keep only faster candidates whose correctness "
            "gate passes."
        ),
    )

    @server.tool()
    def inspect_profile(
        file_path: str,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        ranks: list[int] | None = None,
        sort: str = "total",
        top_n: int | None = 20,
        full_metadata: bool = False,
    ) -> dict:
        """Summarize a scope-profiler HDF5 file: runtime, ranks, region stats, metadata.

        Use this to answer "what does this profile look like?" or "where did
        the time go?" for a single run. Returns structured JSON, not a
        formatted table.

        Args:
            file_path: Path to a merged ``profiling_data.h5`` file.
            include: Only report regions whose name matches one of these
                regex patterns.
            exclude: Skip regions whose name matches one of these regex
                patterns.
            ranks: Restrict region statistics to these MPI ranks (default:
                all ranks).
            sort: Region ordering: one of "total", "calls", "avg", "min",
                "max", "first", "last", "std" (all descending) or "name"
                (alphabetical).
            top_n: Maximum number of regions to return, ranked by `sort`
                (default: 20). Pass a larger number or 0/None for every
                region matching the filters -- the response also reports
                how many regions matched in total, so raise `top_n` only
                when the summary isn't enough.
            full_metadata: If False (default), raw environment variables and
                unrecognized metadata fields are collapsed to a count instead
                of being returned in full, since they are rarely useful and
                can be long (e.g. PATH). Set True to get everything.

        Returns:
            A dict with `file_path`, `label`, `num_ranks`, `num_regions`,
            `total_time_seconds`, `wall_clock_seconds`, `metadata` (grouped
            run/system/parallelism/slurm/modules info), `regions` (sorted,
            possibly truncated list of per-region statistics plus counts),
            and `likwid` (per-region hardware counters, or null if the run
            did not use LIKWID).
        """
        return tools.inspect_profile(
            file_path,
            include=include,
            exclude=exclude,
            ranks=ranks,
            sort=sort,
            top_n=top_n,
            full_metadata=full_metadata,
        )

    @server.tool()
    def compare_profiles(
        baseline_path: str,
        candidate_path: str,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        ranks: list[int] | None = None,
        metric: str = "total",
        threshold_pct: float = 5.0,
        top_n: int | None = 20,
    ) -> dict:
        """Compare two scope-profiler HDF5 files and report whether it got faster.

        Use this after changing code and re-profiling, to check the effect
        of the change: point `baseline_path` at the run before the change
        and `candidate_path` at the run after. All the arithmetic (absolute
        and relative change, per-region deltas, which regions regressed or
        improved) is computed here -- do not try to infer it from two
        separate `inspect_profile` calls.

        Args:
            baseline_path: The "before" profiling file.
            candidate_path: The "after" profiling file.
            include: Only compare regions whose name matches one of these
                regex patterns.
            exclude: Skip regions whose name matches one of these regex
                patterns.
            ranks: Restrict the comparison to these MPI ranks (default: all).
            metric: Which per-region statistic to compare: one of "total",
                "avg", "min", "max", "calls".
            threshold_pct: Percent change above which a region is listed in
                `regressions`, or below whose negation it is listed in
                `improvements` (default: 5.0).
            top_n: Maximum number of regions returned in `regions`,
                `regressions` and `improvements` (default: 20; 0/None for no
                limit). Each list also reports how many entries matched in
                total.

        Returns:
            A dict with `baseline`/`candidate` (each an `inspect_profile`-style
            headline: total_time_seconds, num_ranks, ...), `overall`
            (absolute_diff_seconds, relative_change_pct, speedup, and a
            `faster` boolean -- the direct answer to "did this get faster?"),
            `regions` (per-region deltas), and `regressions`/`improvements`
            (the subset exceeding `threshold_pct`, sorted by magnitude).
        """
        return tools.compare_profiles(
            baseline_path,
            candidate_path,
            include=include,
            exclude=exclude,
            ranks=ranks,
            metric=metric,
            threshold_pct=threshold_pct,
            top_n=top_n,
        )

    @server.tool()
    def run_profile(
        script_path: str,
        script_args: list[str] | None = None,
        only_user_code: bool = True,
        buffer_limit: int = 1024,
        output_path: str | None = None,
        timeout_seconds: float = 300.0,
        top_n: int | None = 20,
    ) -> dict:
        """Run a Python script under scope-profiler and summarize the result.

        Equivalent to ``scope-profiler run <script_path>`` followed by
        `inspect_profile` on the file it produces. The script runs in its
        own subprocess (not inside the MCP server), with a timeout, so it
        cannot hang or crash the server; failures (non-zero exit, timeout,
        missing script) are reported as tool errors rather than partial
        results.

        Args:
            script_path: Path to the Python script to run and profile.
            script_args: Command-line arguments to pass to the script
                (each a separate string, like `sys.argv[1:]` -- no shell
                involved, so shell metacharacters are not interpreted).
            only_user_code: If True (default), only the script's own code is
                instrumented (standard library and installed packages are
                skipped), which is faster and usually what you want. Set
                False to also trace library calls.
            buffer_limit: Initial per-region timestamp buffer size (default:
                1024); grows automatically, this is only a performance hint.
            output_path: Where to write the resulting HDF5 file. Defaults to
                a new temporary directory; pass a path (e.g. to compare
                against later with `compare_profiles`) to control where it
                lands.
            timeout_seconds: Kill the script if it runs longer than this
                (default: 300).
            top_n: Passed through to the `inspect_profile`-style summary of
                the resulting file (default: 20).

        Returns:
            The same structured summary `inspect_profile` returns for the
            file the run produced, plus `stdout_tail` (the last ~2000
            characters of the script's stdout, for debugging).
        """
        return tools.run_profile(
            script_path,
            script_args=script_args,
            only_user_code=only_user_code,
            buffer_limit=buffer_limit,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
            top_n=top_n,
        )

    @server.tool()
    def run_benchmark(config_path: str, label: str = "candidate") -> dict:
        """Run a TOML benchmark config with repeated profiles and correctness.

        The config is declarative and contains the benchmark script, number of
        repetitions, warmups, output directory, and an optional correctness
        command. Returns a JSON manifest with medians, variance, profile paths,
        and correctness status.
        """
        return tools.run_benchmark(config_path, label=label)

    @server.tool()
    def compare_benchmarks(baseline_path: str, candidate_path: str) -> dict:
        """Compare benchmark manifests and return an explicit keep/reject decision."""
        return tools.compare_benchmarks(baseline_path, candidate_path)

    @server.tool()
    def plot_profile(
        file_paths: list[str] | str,
        plot_type: str = "gantt",
        output_dir: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        ranks: list[int] | None = None,
        backend: str = "matplotlib",
    ) -> dict:
        """Render one figure from scope-profiler HDF5 file(s) and return its path.

        Most agent workflows should prefer `inspect_profile`/
        `compare_profiles`, which return numbers directly usable in
        reasoning. Use this only when a human will actually look at the
        resulting image (e.g. attaching it to a PR description or a chat
        message). Requires the `plot` extra
        (``pip install "scope-profiler[plot]"``).

        Args:
            file_paths: One path, or a list of paths, to profiling files.
                `plot_type="speedup"` requires at least two.
            plot_type: One of "gantt", "flame", "durations", "timeseries",
                "speedup".
            output_dir: Directory to write the figure into. Defaults to a
                new temporary directory.
            include: Only plot regions whose name matches one of these regex
                patterns.
            exclude: Skip regions whose name matches one of these regex
                patterns.
            ranks: Restrict the plot to these MPI ranks (default: all).
            backend: "matplotlib" (default, writes a `.png`) or "plotly"
                (writes an interactive `.html`).

        Returns:
            A dict with `plot_type`, `backend`, and `paths` (the file(s)
            written -- normally one, except `durations` with multiple
            requested metrics).
        """
        return tools.plot_profile(
            file_paths,
            plot_type=plot_type,
            output_dir=output_dir,
            include=include,
            exclude=exclude,
            ranks=ranks,
            backend=backend,
        )

    return server


def main() -> None:
    """Entry point for the ``scope-profiler-mcp`` console script."""
    create_server().run()


if __name__ == "__main__":
    main()
