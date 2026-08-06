"""Shared per-region summary table.

One renderer, used by ``ProfileManager.finalize()``,
:meth:`ProfilingH5Reader.print_summary` and ``scope-profiler inspect``, so the
three agree on columns, units and formatting.

Everything here works off duck-typed reader/region objects, so this module has
no scope-profiler imports and cannot introduce an import cycle.
"""

import re
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


# --------------------------------------------------------------------------
# LIKWID hardware counter table
# --------------------------------------------------------------------------


def _name_selected(name: str, include=None, exclude=None) -> bool:
    """Apply the same include/exclude regex rules the region table uses."""
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]
    if include is not None and not any(re.match(p, name) for p in include):
        return False
    if exclude is not None and any(re.match(p, name) for p in exclude):
        return False
    return True


def _set_cell(rows: dict, name: str, column: int, value) -> None:
    """Record one cell of the counter table, keyed by row name and column."""
    rows.setdefault(name, {})[column] = value


def _dense(rows: dict, width: int) -> list:
    """Turn the sparse ``{name: {column: value}}`` mapping into ordered rows.

    Insertion order is preserved, so counters keep the order LIKWID reports
    them in. Cells never filled (a region measured with a different event set)
    come back as ``None`` and render as a dash.
    """
    return [
        (name, [cells.get(column) for column in range(width)])
        for name, cells in rows.items()
    ]


def likwid_tables(reader, include=None, exclude=None, ranks=None) -> list:
    """Build one LIKWID counter table per (rank, event group).

    Regions become columns and counters become rows: a run typically has a
    handful of regions but a few dozen counters, so this orientation stays
    readable where the transpose would not. Splitting per event group keeps
    every column of a table comparable, since a group fixes which events and
    metrics exist.

    Parameters
    ----------
    reader : ProfilingH5Reader
        Source of the LIKWID results.
    include, exclude : list of str or str, optional
        Regex patterns selecting which regions to report, matched as for the
        region table.
    ranks : list of int, optional
        Restrict to these ranks (default: all).

    Returns
    -------
    list of dict
        One entry per table, each with ``rank``, ``group``, ``columns`` (the
        region labels) and ``sections`` (``(title, rows)`` pairs, where a row
        is ``(name, values)``). Empty when the file holds no LIKWID data.
    """
    tables = {}

    for rank, regions in sorted(reader.get_likwid_regions().items()):
        if ranks is not None and rank not in ranks:
            continue
        for tag, result in regions.items():
            if not _name_selected(tag, include, exclude):
                continue

            key = (rank, result.group_name)
            table = tables.setdefault(
                key,
                {
                    "rank": rank,
                    "group": result.group_name,
                    "columns": [],
                    "info": {},
                    "events": {},
                    "metrics": {},
                },
            )

            nthreads = len(result.times)
            for thread in range(nthreads):
                cpu = result.cpus[thread] if thread < len(result.cpus) else thread
                # One column per region, unless the region really did span
                # several hardware threads -- then name the CPU, rather than
                # inventing an aggregate LIKWID never reported.
                label = tag if nthreads == 1 else f"{tag}@cpu{cpu}"
                table["columns"].append(label)
                column = len(table["columns"]) - 1

                _set_cell(table["info"], "call count", column, result.call_counts[thread])
                _set_cell(table["info"], "runtime [s]", column, result.times[thread])
                # event_labels, not event_names: groups such as MEM_DP program
                # one event per memory channel, so the bare names repeat.
                for name, values in zip(result.event_labels, result.events):
                    _set_cell(table["events"], name, column, values[thread])
                for name, values in zip(result.metric_names, result.metrics):
                    _set_cell(table["metrics"], name, column, values[thread])

    built = []
    for (rank, group), table in sorted(tables.items()):
        width = len(table["columns"])
        built.append(
            {
                "rank": rank,
                "group": group,
                "columns": table["columns"],
                "sections": [
                    ("", _dense(table["info"], width)),
                    ("Events", _dense(table["events"], width)),
                    ("Metrics", _dense(table["metrics"], width)),
                ],
            }
        )
    return built


def _format_counter(value) -> str:
    """Format a counter or metric value for the table.

    Raw event counts are large integers and read far better as such than in
    the exponent notation ``%g`` would pick; derived metrics keep six
    significant digits.
    """
    if value is None:
        return "-"
    value = float(value)
    if value.is_integer() and abs(value) < 1e15:
        return f"{int(value):d}"
    return f"{value:.6g}"


def print_likwid_table(table, title=None, stream=None) -> None:
    """Print one LIKWID counter table from :func:`likwid_tables`."""
    stream = sys.stdout if stream is None else stream

    columns = table["columns"]
    if not columns:
        return

    if title is None:
        title = f"LIKWID counters (rank {table['rank']}"
        title += f", group {table['group']})" if table["group"] else ")"

    sections = [
        (heading, rows) for heading, rows in table["sections"] if rows
    ]
    all_rows = [row for _, rows in sections for row in rows]

    name_width = max(
        [len(name) for name, _ in all_rows]
        + [len(heading) for heading, _ in sections]
        + [len("counter")]
    )
    cell_widths = [
        max(
            len(label),
            max((len(_format_counter(values[i])) for _, values in all_rows), default=0),
        )
        for i, label in enumerate(columns)
    ]

    def render(name, cells):
        out = [f"{name:<{name_width}}"]
        out += [f"{cell:>{cell_widths[i]}}" for i, cell in enumerate(cells)]
        return "  ".join(out).rstrip()

    header_line = render("counter", list(columns))
    rule = "-" * len(header_line)

    print(title, file=stream)
    print(f"  {header_line}", file=stream)
    print(f"  {rule}", file=stream)
    for index, (heading, rows) in enumerate(sections):
        if index:
            print(f"  {rule}", file=stream)
        if heading:
            print(f"  {heading}", file=stream)
        for name, values in rows:
            print(f"  {render(name, [_format_counter(v) for v in values])}", file=stream)
    print(f"  {rule}", file=stream)
    print(file=stream)


def print_likwid_tables(reader, include=None, exclude=None, ranks=None, stream=None):
    """Print every LIKWID counter table a reader exposes.

    A no-op for files recorded without LIKWID, so callers can invoke it
    unconditionally.
    """
    for table in likwid_tables(reader, include=include, exclude=exclude, ranks=ranks):
        print_likwid_table(table, stream=stream)
