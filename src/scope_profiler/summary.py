"""Shared per-region summary table.

One renderer, used by ``ProfileManager.finalize()``,
:meth:`ProfilingH5Reader.print_summary` and ``scope-profiler inspect``, so the
three agree on columns, units and formatting.

Everything here works off duck-typed reader/region objects, so this module has
no scope-profiler imports and cannot introduce an import cycle.
"""

import sys

import numpy as np

SORT_KEYS = ("total", "calls", "avg", "max", "name")

_COLUMNS = (
    ("name", "region"),
    ("ranks", "ranks"),
    ("calls", "calls"),
    ("total", "total [s]"),
    ("avg", "avg [s]"),
    ("min", "min [s]"),
    ("max", "max [s]"),
    ("std", "std [s]"),
)


def _region_durations(region, ranks=None) -> np.ndarray:
    """Pool every recorded call duration of a region, in seconds.

    Pooling the raw per-call durations rather than reusing ``MPIRegion``'s
    aggregates keeps a rank filter working, since those aggregate over every
    rank in the file.
    """
    selected = (
        region.regions
        if ranks is None
        else {rank: region.regions[rank] for rank in ranks if rank in region.regions}
    )
    values = [data.durations for data in selected.values() if data.durations.size]
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def region_row(region, ranks=None) -> dict:
    """Collect the summary statistics shown for one region.

    Duration entries are ``None`` when the region recorded no timestamps
    (``time_trace=False``), which the table renders as a dash.
    """
    if ranks is None:
        per_rank = region.regions
    else:
        per_rank = {
            rank: region.regions[rank] for rank in ranks if rank in region.regions
        }

    durations = _region_durations(region, ranks)
    return {
        "name": region.name,
        "num_ranks": len(per_rank),
        "calls": sum(data.num_calls for data in per_rank.values()),
        "total": float(np.sum(durations)) if durations.size else None,
        "avg": float(np.mean(durations)) if durations.size else None,
        "min": float(np.min(durations)) if durations.size else None,
        "max": float(np.max(durations)) if durations.size else None,
        "std": float(np.std(durations)) if durations.size else None,
    }


def region_rows(
    reader,
    include=None,
    exclude=None,
    ranks=None,
    sort: str = "total",
) -> list:
    """Build the summary rows for every region a reader exposes.

    Parameters
    ----------
    reader : ProfilingH5Reader
        Source of the regions.
    include, exclude : list of str or str, optional
        Regex patterns selecting which regions to summarize.
    ranks : list of int, optional
        Restrict the statistics to these ranks (default: all).
    sort : str, optional
        One of :data:`SORT_KEYS`: ``total``, ``calls``, ``avg`` and ``max``
        sort descending, ``name`` alphabetically.
    """
    rows = [
        region_row(region, ranks)
        for region in reader.get_regions(include=include, exclude=exclude)
    ]

    # Sort by name first so that the stable sort below breaks ties
    # alphabetically rather than by whatever order the file happened to use.
    rows.sort(key=lambda row: row["name"])
    if sort != "name":
        # None (no timing recorded) sorts last.
        rows.sort(key=lambda row: (row[sort] is not None, row[sort] or 0), reverse=True)
    return rows


def _format_duration(value) -> str:
    """Format a duration in seconds, or a dash when no timing was recorded."""
    return "-" if value is None else f"{value:.6g}"


def print_region_table(rows, title=None, stream=None) -> None:
    """Print the aligned per-region statistics table.

    Parameters
    ----------
    rows : list of dict
        Rows from :func:`region_rows`.
    title : str, optional
        Heading printed above the table.
    stream : file-like, optional
        Where to write (default: stdout).
    """
    stream = sys.stdout if stream is None else stream

    if title:
        print(title, file=stream)
    if not rows:
        print("  (no regions recorded)", file=stream)
        return

    formatted = [
        {
            "name": row["name"],
            "ranks": str(row["num_ranks"]),
            "calls": str(row["calls"]),
            "total": _format_duration(row["total"]),
            "avg": _format_duration(row["avg"]),
            "min": _format_duration(row["min"]),
            "max": _format_duration(row["max"]),
            "std": _format_duration(row["std"]),
        }
        for row in rows
    ]

    timed = [row["total"] for row in rows if row["total"] is not None]
    total_row = {
        "name": "TOTAL",
        "ranks": "",
        "calls": str(sum(row["calls"] for row in rows)),
        "total": _format_duration(sum(timed) if timed else None),
        "avg": "",
        "min": "",
        "max": "",
        "std": "",
    }

    widths = {
        key: max(len(header), max(len(row[key]) for row in [*formatted, total_row]))
        for key, header in _COLUMNS
    }

    def render(row):
        cells = [f"{row['name']:<{widths['name']}}"]
        cells += [f"{row[key]:>{widths[key]}}" for key, _ in _COLUMNS[1:]]
        return "  ".join(cells).rstrip()

    header_line = render({key: header for key, header in _COLUMNS})
    rule = "-" * len(header_line)

    print(f"  {header_line}", file=stream)
    print(f"  {rule}", file=stream)
    for row in formatted:
        print(f"  {render(row)}", file=stream)
    print(f"  {rule}", file=stream)
    print(f"  {render(total_row)}", file=stream)

    notes = []
    if len(rows) > 1:
        # Nested regions are counted in both the inner and the outer row, so
        # the summed total legitimately exceeds the run's wall-clock time.
        notes.append(
            "Regions may nest, so the summed total can exceed the wall-clock time."
        )
    if any(row["total"] is None for row in rows):
        notes.append(
            "Regions without timing were profiled with time_trace=False; "
            "only their call counts were recorded."
        )
    for note in notes:
        print(f"\n  {note}", file=stream)

    # Trailing blank line so whatever follows (line-profiler stats, a second
    # file's table) is not pressed against the last row.
    print(file=stream)
