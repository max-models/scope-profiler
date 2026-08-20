"""Shared per-region summary table.

One renderer, used by ``ProfileManager.finalize()``,
:meth:`ProfilingResults.print_summary` and ``scope-profiler inspect``, so the
three agree on columns, units and formatting.

Everything here works off duck-typed results/region objects, so this module has
no scope-profiler imports and cannot introduce an import cycle.
"""

import re
import sys

import numpy as np

SORT_KEYS = (
    "total",
    "calls",
    "avg",
    "min",
    "max",
    "first",
    "last",
    "std",
    "p50",
    "p95",
    "p99",
    "imbalance",
    "name",
)

_COLUMNS = (
    ("name", "region"),
    ("ranks", "ranks"),
    ("calls", "calls"),
    ("total", "total [s]"),
    ("avg", "avg [s]"),
    ("min", "min [s]"),
    ("max", "max [s]"),
    ("first", "first [s]"),
    ("last", "last [s]"),
    ("std", "std [s]"),
    ("p50", "p50 [s]"),
    ("p95", "p95 [s]"),
    ("p99", "p99 [s]"),
    ("imbalance", "imbalance [%]"),
)

REGION_TABLE_COLUMN_NAMES = tuple(key for key, _ in _COLUMNS)
REGION_TABLE_COLUMNS = ("region", *REGION_TABLE_COLUMN_NAMES[1:])
DEFAULT_REGION_TABLE_COLUMNS = ("region", "ranks", "calls", "total", "avg")
_COLUMN_ALIASES = {"region": "name", "name": "name"}
_COLUMN_ALIASES.update({key: key for key, _ in _COLUMNS if key != "name"})


def normalize_region_table_columns(columns=None) -> tuple[tuple[str, str], ...]:
    """Return validated ``(row_key, header)`` pairs for a region table."""
    if columns is None:
        columns = DEFAULT_REGION_TABLE_COLUMNS
    if isinstance(columns, str):
        columns = [columns]

    by_key = dict(_COLUMNS)
    normalized = []
    unknown = []
    for column in columns:
        key = _COLUMN_ALIASES.get(column)
        if key is None:
            unknown.append(column)
            continue
        normalized.append((key, by_key[key]))

    if unknown:
        choices = ", ".join(REGION_TABLE_COLUMNS)
        raise ValueError(
            f"Unknown region summary column(s): {unknown}. Choices: {choices}"
        )
    if not normalized:
        raise ValueError("At least one region summary column must be selected.")
    return tuple(normalized)


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


def _first_last_durations(region, ranks=None):
    """Duration of the chronologically first and last call, across ranks.

    Pooling durations (as ``_region_durations`` does) loses call order across
    ranks, so first/last are found from each rank's own first/last call
    instead -- the earliest-starting rank supplies "first", the
    latest-ending rank supplies "last".
    """
    selected = (
        region.regions
        if ranks is None
        else {rank: region.regions[rank] for rank in ranks if rank in region.regions}
    )
    timed = [data for data in selected.values() if data.has_timing]
    if not timed:
        return None, None
    first = min(timed, key=lambda data: data.first_start_time).first_duration
    last = max(timed, key=lambda data: data.last_end_time).last_duration
    return first, last


def region_row(region, ranks=None) -> dict:
    """Collect the summary statistics shown for one region.

    Duration entries are ``None`` when the region has no recorded calls for
    the selected ranks, which the table renders as a dash.
    """
    if ranks is None:
        per_rank = region.regions
    else:
        per_rank = {
            rank: region.regions[rank] for rank in ranks if rank in region.regions
        }

    durations = _region_durations(region, ranks)
    first, last = _first_last_durations(region, ranks)
    return {
        "name": region.name,
        "num_ranks": len(per_rank),
        "calls": sum(data.num_calls for data in per_rank.values()),
        "total": float(np.sum(durations)) if durations.size else None,
        "avg": float(np.mean(durations)) if durations.size else None,
        "min": float(np.min(durations)) if durations.size else None,
        "max": float(np.max(durations)) if durations.size else None,
        "first": first,
        "last": last,
        "std": float(np.std(durations)) if durations.size else None,
        "p50": float(np.percentile(durations, 50)) if durations.size else None,
        "p95": float(np.percentile(durations, 95)) if durations.size else None,
        "p99": float(np.percentile(durations, 99)) if durations.size else None,
        "imbalance": (
            region.rank_imbalance_pct
            if ranks is None
            else _rank_imbalance_pct(region, ranks)
        ),
    }


def _rank_imbalance_pct(region, ranks=None) -> float:
    """Compute slowest-rank total excess over the selected-rank mean."""
    selected = (
        region.regions
        if ranks is None
        else {rank: region.regions[rank] for rank in ranks if rank in region.regions}
    )
    totals = [
        data.total_duration for data in selected.values() if data.total_duration > 0
    ]
    if len(totals) < 2:
        return 0.0
    return (max(totals) / float(np.mean(totals)) - 1.0) * 100.0


def region_rows(
    results,
    include=None,
    exclude=None,
    ranks=None,
    sort: str = "total",
) -> list:
    """Build the summary rows for every region a result set exposes.

    Parameters
    ----------
    results : ProfilingResults
        Source of the regions; loaded from a file or built in memory alike.
    include, exclude : list of str or str, optional
        Regex patterns selecting which regions to summarize.
    ranks : list of int, optional
        Restrict the statistics to these ranks (default: all).
    sort : str, optional
        One of :data:`SORT_KEYS`: ``total`` (default), ``calls``, ``avg``,
        ``min``, ``max`` and ``std`` sort descending, ``name`` alphabetically.
    """
    rows = [
        region_row(region, ranks)
        for region in results.get_regions(include=include, exclude=exclude)
    ]

    # Sort by name first so that the stable sort below breaks ties
    # alphabetically rather than by whatever order the file happened to use.
    rows.sort(key=lambda row: row["name"])
    if sort != "name":
        # None (nothing recorded for these ranks) sorts last.
        rows.sort(key=lambda row: (row[sort] is not None, row[sort] or 0), reverse=True)
    return rows


def _format_duration(value) -> str:
    """Format a duration in seconds, or a dash when no timing was recorded."""
    return "-" if value is None else f"{value:.6g}"


def print_region_table(
    rows,
    title=None,
    stream=None,
    suppress_notes: bool = False,
    total_time: float | None = None,
    columns=None,
) -> None:
    """Print the aligned per-region statistics table.

    Parameters
    ----------
    rows : list of dict
        Rows from :func:`region_rows`.
    title : str, optional
        Heading printed above the table.
    stream : file-like, optional
        Where to write (default: stdout).
    suppress_notes : bool, optional
        Don't print the explanatory notes below the table (default: False).
    total_time : float, optional
        Wall-clock seconds from ``setup()`` to ``finalize()`` (see
        :attr:`~scope_profiler.results.ProfilingResults.total_time`), printed
        below the TOTAL row when given. Unlike that row -- which sums region
        durations and so can exceed the run's real duration when regions
        nest -- this is the run's own actual wall-clock time.
    columns : list of str or str, optional
        Region summary columns to print. Defaults to ``region``, ``ranks``,
        ``calls``, ``total`` and ``avg``. The public name for the first column
        is ``region``; ``name`` is accepted as an alias for Python callers.
    """
    stream = sys.stdout if stream is None else stream
    selected_columns = normalize_region_table_columns(columns)

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
            "first": _format_duration(row["first"]),
            "last": _format_duration(row["last"]),
            "std": _format_duration(row["std"]),
            "p50": _format_duration(row["p50"]),
            "p95": _format_duration(row["p95"]),
            "p99": _format_duration(row["p99"]),
            "imbalance": _format_duration(row["imbalance"]),
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
        "first": "",
        "last": "",
        "std": "",
        "p50": "",
        "p95": "",
        "p99": "",
        "imbalance": "",
    }

    widths = {
        key: max(len(header), max(len(row[key]) for row in [*formatted, total_row]))
        for key, header in selected_columns
    }

    def render(row):
        cells = []
        for index, (key, _) in enumerate(selected_columns):
            if index == 0 and key == "name":
                cells.append(f"{row[key]:<{widths[key]}}")
            else:
                cells.append(f"{row[key]:>{widths[key]}}")
        return "  ".join(cells).rstrip()

    header_line = render({key: header for key, header in selected_columns})
    rule = "-" * len(header_line)

    print(f"  {header_line}", file=stream)
    print(f"  {rule}", file=stream)
    for row in formatted:
        print(f"  {render(row)}", file=stream)
    if not suppress_notes:
        notes = [
            "Durations are in seconds.",
            "TOTAL row sums over all ranks.",
        ]
        for note in notes:
            print(f"\n  {note}", file=stream)
    print(f"  {rule}", file=stream)
    print(f"  {render(total_row)}", file=stream)
    if total_time is not None:
        print(f"\n  Total time (setup to finalize): {total_time:.6g} s", file=stream)

    notes = []
    if len(rows) > 1:
        # Nested regions are counted in both the inner and the outer row, so
        # the summed total legitimately exceeds the run's wall-clock time.
        notes.append(
            "Regions may nest, so the summed total can exceed the wall-clock time."
        )
    if any(row["total"] is None for row in rows):
        notes.append(
            "Regions shown without timing recorded no calls on the selected ranks."
        )
    if not suppress_notes and notes:
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


def likwid_tables(results, include=None, exclude=None, ranks=None) -> list:
    """Build one LIKWID counter table per (rank, event group).

    Regions become columns and counters become rows: a run typically has a
    handful of regions but a few dozen counters, so this orientation stays
    readable where the transpose would not. Splitting per event group keeps
    every column of a table comparable, since a group fixes which events and
    metrics exist.

    Columns are ordered by descending LIKWID runtime (ties alphabetically), so
    the costliest regions lead, as in the region table.

    Parameters
    ----------
    results : ProfilingResults
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
    # Collect first, so the columns of each table can be ordered before any
    # cell is placed. HDF5 hands regions back alphabetically; showing the
    # costliest first instead matches how the region table is sorted, so the
    # two tables read together.
    grouped = {}
    for rank, regions in sorted(results.get_likwid_regions().items()):
        if ranks is not None and rank not in ranks:
            continue
        for tag, result in regions.items():
            if not _name_selected(tag, include, exclude):
                continue
            grouped.setdefault((rank, result.group_name), []).append((tag, result))

    for entries in grouped.values():
        entries.sort(
            key=lambda item: (
                -float(np.max(item[1].times)) if len(item[1].times) else 0.0,
                item[0],
            )
        )

    tables = {}
    for key, entries in grouped.items():
        rank, group_name = key
        for tag, result in entries:
            table = tables.setdefault(
                key,
                {
                    "rank": rank,
                    "group": group_name,
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

                _set_cell(
                    table["info"], "call count", column, result.call_counts[thread]
                )
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

    sections = [(heading, rows) for heading, rows in table["sections"] if rows]
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
            print(
                f"  {render(name, [_format_counter(v) for v in values])}", file=stream
            )
    print(f"  {rule}", file=stream)
    print(file=stream)


def print_likwid_tables(results, include=None, exclude=None, ranks=None, stream=None):
    """Print every LIKWID counter table a result set exposes.

    A no-op for files recorded without LIKWID, so callers can invoke it
    unconditionally.
    """
    for table in likwid_tables(results, include=include, exclude=exclude, ranks=ranks):
        print_likwid_table(table, stream=stream)
