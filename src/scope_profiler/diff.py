"""CLI entry point for ``scope-profiler diff``: compare two profiling files.

Aligns region statistics between two merged HDF5 profiling files by region
name and reports the change in one metric (default: total duration), so a
regression -- or an improvement -- between two runs (two commits, two
configs, two job sizes) shows up as a single table instead of two separate
``inspect`` runs a human has to cross-reference by eye.
"""

import argparse
import sys

from tabulate import tabulate

from scope_profiler.h5reader import read_h5, read_h5_summary
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import region_rows

METRICS = ("total", "avg", "min", "max", "p50", "p95", "p99", "imbalance", "calls")
_SUMMARY_METRICS = {"total", "avg", "min", "max", "imbalance", "calls"}
SORT_KEYS = ("delta", "pct", "name")

_METRIC_LABELS = {
    "total": "total [s]",
    "avg": "avg [s]",
    "min": "min [s]",
    "max": "max [s]",
    "p50": "p50 [s]",
    "p95": "p95 [s]",
    "p99": "p99 [s]",
    "imbalance": "imbalance [%]",
    "calls": "calls",
}


def diff_rows(
    results_a: ProfilingResults,
    results_b: ProfilingResults,
    include=None,
    exclude=None,
    ranks=None,
    metric: str = "total",
    sort: str = "delta",
    threshold: float | None = None,
) -> list:
    """Build one comparison row per region present in either result set.

    Parameters
    ----------
    results_a, results_b : ProfilingResults
        The "before" (``a``) and "after" (``b``) runs.
    include, exclude : list of str or str, optional
        Regex patterns selecting which regions to compare, matched as in
        :func:`~scope_profiler.summary.region_rows`.
    ranks : list of int, optional
        Restrict the statistics to these ranks (default: all).
    metric : str, optional
        Which per-region statistic to compare: one of :data:`METRICS`
        (default: ``total``).
    sort : str, optional
        One of :data:`SORT_KEYS`: ``delta`` and ``pct`` sort by descending
        magnitude of change (default: ``delta``), ``name`` alphabetically.
    threshold : float, optional
        Drop regions whose absolute percent change is below this many
        percent. Regions that only appear in one file always pass, since
        there is no baseline to compute a percentage against.

    Returns
    -------
    list of dict
        Each entry has ``name``, ``a``, ``b`` (the metric's value in each
        file, or None when the region is missing -- or timed no calls on the
        selected ranks -- there), ``delta`` (``b - a``, treating a missing
        side as 0) and ``pct`` (percent change, or None when ``a`` is None or
        0).
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if sort not in SORT_KEYS:
        raise ValueError(f"sort must be one of {SORT_KEYS}, got {sort!r}")

    rows_a = {
        row["name"]: row
        for row in region_rows(results_a, include=include, exclude=exclude, ranks=ranks)
    }
    rows_b = {
        row["name"]: row
        for row in region_rows(results_b, include=include, exclude=exclude, ranks=ranks)
    }

    rows = []
    for name in sorted(set(rows_a) | set(rows_b)):
        a = rows_a[name][metric] if name in rows_a else None
        b = rows_b[name][metric] if name in rows_b else None
        if a is None and b is None:
            continue

        delta = (b or 0) - (a or 0)
        pct = (delta / a * 100) if a else None

        if threshold is not None and pct is not None and abs(pct) < threshold:
            continue

        rows.append({"name": name, "a": a, "b": b, "delta": delta, "pct": pct})

    # Sort by name first so the stable sort below breaks ties alphabetically,
    # as region_rows() does.
    rows.sort(key=lambda row: row["name"])
    if sort != "name":
        rows.sort(
            key=lambda row: abs(row[sort]) if row[sort] is not None else -1,
            reverse=True,
        )
    return rows


def _format_value(value, metric) -> str:
    """Format one side's metric value, or a dash when the region is absent."""
    if value is None:
        return "-"
    return str(int(value)) if metric == "calls" else f"{value:.6g}"


def _format_delta(value, metric) -> str:
    """Format a signed delta, always showing the sign for readability."""
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{int(value)}" if metric == "calls" else f"{sign}{value:.6g}"


def _format_pct(value) -> str:
    """Format a signed percent change, or a dash when there is no baseline."""
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4g}%"


def print_diff_table(rows, metric: str = "total", title=None, stream=None) -> None:
    """Print the aligned region comparison table built by :func:`diff_rows`."""
    stream = sys.stdout if stream is None else stream
    label = _METRIC_LABELS[metric]

    if title:
        print(title, file=stream)
    if not rows:
        print("  (no regions to compare)", file=stream)
        return

    columns = (
        ("name", "region"),
        ("a", f"{label} (a)"),
        ("b", f"{label} (b)"),
        ("delta", "delta"),
        ("pct", "delta [%]"),
    )
    formatted = [
        {
            "name": row["name"],
            "a": _format_value(row["a"], metric),
            "b": _format_value(row["b"], metric),
            "delta": _format_delta(row["delta"], metric),
            "pct": _format_pct(row["pct"]),
        }
        for row in rows
    ]

    table_rows = [[row[key] for key, _ in columns] for row in formatted]
    for line in tabulate(
        table_rows,
        headers=[header for _, header in columns],
        tablefmt="rounded_outline",
        disable_numparse=True,
    ).splitlines():
        print(f"  {line}", file=stream)

    only_a = [row["name"] for row in rows if row["b"] is None]
    only_b = [row["name"] for row in rows if row["a"] is None]
    if only_a:
        print(f"\n  Only in a: {', '.join(only_a)}", file=stream)
    if only_b:
        print(f"  Only in b: {', '.join(only_b)}", file=stream)
    print(file=stream)


def diff_files(
    file_a,
    file_b,
    include=None,
    exclude=None,
    ranks=None,
    metric: str = "total",
    sort: str = "delta",
    threshold: float | None = None,
    stream=None,
) -> None:
    """Print a region-by-region comparison of two profiling HDF5 files.

    Parameters
    ----------
    file_a, file_b : str or Path
        The "before" and "after" merged profiling files.
    include, exclude, ranks, metric, sort, threshold
        See :func:`diff_rows`.
    stream : file-like, optional
        Where to write (default: stdout).
    """
    stream = sys.stdout if stream is None else stream
    reader = read_h5_summary if metric in _SUMMARY_METRICS else read_h5
    results_a = reader(file_a)
    results_b = reader(file_b)

    print("=" * 78, file=stream)
    print(f"a: {results_a.default_title()}", file=stream)
    print(f"b: {results_b.default_title()}", file=stream)
    print("=" * 78 + "\n", file=stream)

    rows = diff_rows(
        results_a,
        results_b,
        include=include,
        exclude=exclude,
        ranks=ranks,
        metric=metric,
        sort=sort,
        threshold=threshold,
    )
    print_diff_table(rows, metric=metric, title=f"Regions ({len(rows)})", stream=stream)


def check_rows(
    results_a: ProfilingResults,
    results_b: ProfilingResults,
    max_regression: float = 0.0,
    include=None,
    exclude=None,
    ranks=None,
    metric: str = "total",
    fail_on_new: bool = False,
) -> list:
    """Return rows that violate a performance regression budget.

    ``max_regression`` is a percentage. New regions have no meaningful
    percentage baseline and are only failures when ``fail_on_new`` is true.
    """
    if max_regression < 0:
        raise ValueError("max_regression must be non-negative")
    rows = diff_rows(
        results_a,
        results_b,
        include=include,
        exclude=exclude,
        ranks=ranks,
        metric=metric,
        sort="pct",
    )
    return [
        row
        for row in rows
        if (row["pct"] is not None and row["pct"] > max_regression)
        or (row["pct"] is None and row["a"] is None and fail_on_new)
    ]


def check_files(
    file_a,
    file_b,
    max_regression: float = 0.0,
    include=None,
    exclude=None,
    ranks=None,
    metric: str = "total",
    fail_on_new: bool = False,
    stream=None,
) -> int:
    """Print a CI regression report and return 0 for pass, 1 for failure."""
    stream = sys.stdout if stream is None else stream
    reader = read_h5_summary if metric in _SUMMARY_METRICS else read_h5
    results_a, results_b = reader(file_a), reader(file_b)
    failures = check_rows(
        results_a,
        results_b,
        max_regression=max_regression,
        include=include,
        exclude=exclude,
        ranks=ranks,
        metric=metric,
        fail_on_new=fail_on_new,
    )
    all_rows = diff_rows(
        results_a,
        results_b,
        include=include,
        exclude=exclude,
        ranks=ranks,
        metric=metric,
        sort="pct",
    )
    print(f"Baseline: {results_a.default_title()}", file=stream)
    print(f"Candidate: {results_b.default_title()}", file=stream)
    print(
        f"Regression budget: +{max_regression:g}% ({metric})",
        file=stream,
    )
    print_diff_table(
        all_rows,
        metric=metric,
        title=f"Regression check: {'FAIL' if failures else 'PASS'}",
        stream=stream,
    )
    if failures:
        print(
            "Budget violations: " + ", ".join(row["name"] for row in failures),
            file=stream,
        )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``scope-profiler diff``."""
    parser = argparse.ArgumentParser(
        prog="scope-profiler diff",
        description="Compare region statistics between two profiling files.",
    )
    parser.add_argument("file_a", help="Baseline profiling_data.h5 ('a')")
    parser.add_argument("file_b", help="Comparison profiling_data.h5 ('b')")
    parser.add_argument(
        "--include",
        nargs="+",
        help="Only compare regions whose name matches these regex patterns",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Skip regions whose name matches these regex patterns",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        help="Restrict region statistics to these ranks, e.g. 0 2 or 0-3",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="total",
        help="Statistic to compare (default: total)",
    )
    parser.add_argument(
        "--sort",
        choices=SORT_KEYS,
        default="delta",
        help="Order regions by descending |delta| (default), |delta %%| or name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        metavar="PCT",
        help="Only show regions whose absolute percent change is at least this",
    )
    return parser


def build_check_parser() -> argparse.ArgumentParser:
    """Build the parser for ``scope-profiler check``."""
    parser = argparse.ArgumentParser(
        prog="scope-profiler check",
        description="Fail when a profiling candidate exceeds a regression budget.",
    )
    parser.add_argument("file_a", help="Baseline profiling_data.h5")
    parser.add_argument("file_b", help="Candidate profiling_data.h5")
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Maximum allowed increase in percent (default: 0)",
    )
    parser.add_argument("--include", nargs="+", help="Only check matching regions")
    parser.add_argument("--exclude", nargs="+", help="Skip matching regions")
    parser.add_argument("--ranks", nargs="+", help="Restrict statistics to ranks")
    parser.add_argument("--metric", choices=METRICS, default="total")
    parser.add_argument(
        "--fail-on-new",
        action="store_true",
        help="Treat regions absent from the baseline as failures",
    )
    return parser


def main(argv: list | None = None):
    """Entry point for ``scope-profiler diff``."""
    from scope_profiler.post_processing import parse_ranks

    parser = build_parser()
    args = parser.parse_args(argv)

    ranks = None
    if args.ranks:
        ranks = sorted({rank for spec in args.ranks for rank in parse_ranks(spec)})

    diff_files(
        args.file_a,
        args.file_b,
        include=args.include,
        exclude=args.exclude,
        ranks=ranks,
        metric=args.metric,
        sort=args.sort,
        threshold=args.threshold,
    )
    return 0


def check_main(argv: list | None = None):
    """Entry point for ``scope-profiler check``."""
    from scope_profiler.post_processing import parse_ranks

    parser = build_check_parser()
    args = parser.parse_args(argv)
    ranks = None
    if args.ranks:
        ranks = sorted({rank for spec in args.ranks for rank in parse_ranks(spec)})
    return check_files(
        args.file_a,
        args.file_b,
        max_regression=args.max_regression,
        include=args.include,
        exclude=args.exclude,
        ranks=ranks,
        metric=args.metric,
        fail_on_new=args.fail_on_new,
    )
