"""Shared per-region summary table.

One renderer, used by ``ProfileManager.finalize()``,
:meth:`ProfilingResults.print_summary` and ``scope-profiler inspect``, so the
three agree on columns, units and formatting.

Everything here works off duck-typed results/region objects, so this module has
no scope-profiler imports and cannot introduce an import cycle.
"""

import os
import re
import shlex
import sys

import numpy as np
from tabulate import tabulate


def _print_table(rows, headers, stream, title=None) -> None:
    """Print a rounded, header-separated table."""
    lines = tabulate(
        rows,
        headers=headers,
        tablefmt="rounded_outline",
        disable_numparse=True,
    ).splitlines()
    if title:
        width = max(len(line) for line in lines)
        centered_title = title.center(width - 4).lstrip()
        if getattr(stream, "isatty", lambda: False)():
            centered_title = f"\033[1;36m{centered_title}\033[0m"
        # Keep the heading flush-left so labels are immediately visible and
        # can be consumed reliably by callers parsing summary output.
        print(centered_title, file=stream)
    for line in lines:
        print(f"  {line}", file=stream)


def _print_heading(text, stream) -> None:
    if getattr(stream, "isatty", lambda: False)():
        text = f"\033[1;36m{text}\033[0m"
    print(text, file=stream)


SORT_KEYS = (
    "start",
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
    ("calls", "n"),
    ("total", "total [s]"),
    ("percent", "% session"),
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
DEFAULT_REGION_TABLE_COLUMNS = (
    "region",
    "calls",
    "percent",
    "total",
    "avg",
)
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
            f"Unknown region summary column(s): {unknown}. Choices: {choices}",
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


def _stored_distribution_statistics(per_rank):
    """Combine fixed-size per-rank moments without loading event arrays."""
    summaries = [
        data.stored_summary
        for data in per_rank.values()
        if data.num_calls and data.stored_summary is not None
    ]
    if not summaries:
        return None, None, None

    first_candidates = [
        item for item in summaries if "first" in item and "start_minimum" in item
    ]
    last_candidates = [
        item for item in summaries if "last" in item and "end_maximum" in item
    ]
    first = (
        min(first_candidates, key=lambda item: item["start_minimum"])["first"] / 1e9
        if first_candidates
        else None
    )
    last = (
        max(last_candidates, key=lambda item: item["end_maximum"])["last"] / 1e9
        if last_candidates
        else None
    )

    if any("mean" not in item or "m2" not in item for item in summaries):
        return first, last, None
    count = sum(int(item["count"]) for item in summaries)
    if not count:
        return first, last, None
    mean = sum(int(item["count"]) * float(item["mean"]) for item in summaries) / count
    m2 = sum(
        float(item["m2"]) + int(item["count"]) * (float(item["mean"]) - mean) ** 2
        for item in summaries
    )
    return first, last, float(np.sqrt(max(m2, 0.0) / count)) / 1e9


def _covered_duration(per_rank) -> float | None:
    """Return wall-clock coverage, counting overlapping calls only once."""
    total = 0.0
    found = False
    for data in per_rank.values():
        if not data.num_calls:
            continue
        if not data.has_timing:
            total += data.total_duration
            found = True
            continue
        intervals = sorted(zip(data.start_times, data.end_times))
        if not intervals:
            total += data.total_duration
            found = True
            continue
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        total += end - start
        found = True
    return total if found else None


def region_row(region, ranks=None, *, include_exclusive: bool = False) -> dict:
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
    calls = sum(data.num_calls for data in per_rank.values())
    coverage = _covered_duration(per_rank)
    exclusive = None
    if include_exclusive:
        from scope_profiler.call_stack import NestingError
        from scope_profiler.region import EventDataUnavailableError

        try:
            exclusive = float(
                sum(data.exclusive_duration for data in per_rank.values()),
            )
        except (NestingError, EventDataUnavailableError):
            # Legacy profiles may contain overlapping events, for which an
            # exclusive call tree cannot be reconstructed.
            exclusive = float(sum(data.total_duration for data in per_rank.values()))
    if not durations.size and calls:
        # Aggregate-only regions intentionally have no per-call duration
        # array. Their scalar statistics are still sufficient for the summary
        # table (except distribution statistics such as percentiles).
        totals = [data.total_duration for data in per_rank.values()]
        minimums = [data.min_duration for data in per_rank.values() if data.num_calls]
        maximums = [data.max_duration for data in per_rank.values() if data.num_calls]
        total = float(sum(totals))
        first, last, std = _stored_distribution_statistics(per_rank)
        return {
            "name": region.name,
            "num_ranks": len(per_rank),
            "calls": calls,
            "total": total,
            "coverage": coverage,
            "exclusive": exclusive,
            "avg": total / calls,
            "min": min(minimums),
            "max": max(maximums),
            "first": first,
            "last": last,
            "std": std,
            "p50": None,
            "p95": None,
            "p99": None,
            "imbalance": (
                region.rank_imbalance_pct
                if ranks is None
                else _rank_imbalance_pct(region, ranks)
            ),
        }
    return {
        "name": region.name,
        "num_ranks": len(per_rank),
        "calls": calls,
        "total": float(np.sum(durations)) if durations.size else None,
        "coverage": coverage,
        "exclusive": exclusive,
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


def region_rows(
    results,
    include=None,
    exclude=None,
    ranks=None,
    sort: str = "start",
    percentage_mode: str = "coverage",
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
        One of :data:`SORT_KEYS`: ``start`` (default), ``total``, ``calls``, ``avg``,
        ``min``, ``max`` and ``std`` sort descending, ``name`` alphabetically.
    percentage_mode : {"coverage", "exclusive"}, optional
        Quantity used for the ``% session`` column. Wall-clock coverage is the
        default; exclusive time can be selected for attribution-focused tables.
    """
    if percentage_mode not in {"coverage", "exclusive"}:
        raise ValueError("percentage_mode must be 'coverage' or 'exclusive'")
    rows = [
        region_row(region, ranks, include_exclusive=percentage_mode == "exclusive")
        for region in results.get_regions(include=include, exclude=exclude)
    ]
    rows_by_name = {row["name"]: row for row in rows}

    # Build display rows from the call tree rather than from the flat region
    # registry. A region can occur below multiple parents, in which case each
    # distinct path gets its own row. Consecutive copies of the same name are
    # one recursive call chain, not distinct aggregate rows: the timing values
    # already include every invocation of that region.
    display_rows = []
    seen_paths = set()
    display_by_path = {}
    selected_ranks = range(results.num_ranks) if ranks is None else ranks
    from scope_profiler.call_stack import NestingError
    from scope_profiler.region import EventDataUnavailableError

    for rank in selected_ranks:
        try:
            calls = results.call_stack(rank=rank, include=include, exclude=exclude)
        except (NestingError, EventDataUnavailableError):
            # Keep summary output available for legacy profiles containing
            # overlapping intervals that cannot form a call tree.
            continue
        by_id = {call["call_id"]: call for call in calls}
        for call in calls:
            name = call["name"]
            if name not in rows_by_name:
                continue
            path = []
            parent = call
            while parent is not None:
                path.append(parent["name"])
                parent_id = parent.get("parent")
                parent = by_id.get(parent_id) if parent_id is not None else None
            path = tuple(reversed(path))
            collapsed_path = tuple(
                name
                for index, name in enumerate(path)
                if index == 0 or name != path[index - 1]
            )
            if collapsed_path in seen_paths:
                if len(collapsed_path) != len(path):
                    display_by_path[collapsed_path]["recursive"] = True
                continue
            seen_paths.add(collapsed_path)
            row = dict(rows_by_name[name])
            row["depth"] = len(collapsed_path) - 1
            row["recursive"] = len(collapsed_path) != len(path)
            row["start"] = float(call["start"])
            display_rows.append(row)
            display_by_path[collapsed_path] = row

    # Keep regions with no reconstructable call tree in the summary.
    for row in rows:
        if row["name"] not in seen_paths and not any(
            display["name"] == row["name"] for display in display_rows
        ):
            fallback = dict(row)
            fallback["depth"] = 0
            fallback["recursive"] = False
            region = results.get_region(row["name"])
            fallback["start"] = (
                region.first_start_time if region.has_timing else float("inf")
            )
            display_rows.append(fallback)

    rows = display_rows

    if sort == "start":
        rows.sort(key=lambda row: row["start"])
    else:
        # Sort by name first so that the stable sort below breaks ties
        # alphabetically rather than by whatever order the file happened to use.
        rows.sort(key=lambda row: row["name"])
        # None (nothing recorded for these ranks) sorts last.
        if sort != "name":
            rows.sort(
                key=lambda row: (row[sort] is not None, row[sort] or 0),
                reverse=True,
            )
    return rows


def _format_duration(value) -> str:
    """Format a duration in seconds, or a dash when no timing was recorded."""
    return "-" if value is None else f"{value:.1e}"


def _format_count(value) -> str:
    """Format a call count compactly while keeping small counts exact."""
    if value is None:
        return "-"
    value = int(value)
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if value >= threshold:
            return f"{value / threshold:.3g}{suffix}"
    return str(value)


def _format_percentage(value, denominator) -> str:
    """Format a duration as a readable percentage of the session duration.

    Fixed-point notation is easier to scan in terminal tables. Scientific
    notation is retained only below 0.01%, where two decimal places would
    otherwise turn a non-zero value into ``0.00%``.
    """
    if value is None or denominator is None or denominator <= 0:
        return "-"
    percentage = 100.0 * value / denominator
    if percentage and abs(percentage) < 0.01:
        return f"{percentage:.1e}%"
    return f"{percentage:.2f}%"


def _display_region_name(row) -> str:
    """Render a hierarchical name, marking a collapsed recursive chain."""
    depth = row.get("depth", 0)
    prefix = f"{'│ ' * (depth - 1)}└─ " if depth else ""
    recursive = " ↻" if row.get("recursive") else ""
    return f"{prefix}{row['name']}{recursive}"


def print_region_table(
    rows,
    title=None,
    stream=None,
    suppress_notes: bool = False,
    total_time: float | None = None,
    columns=None,
    percentage_mode: str = "coverage",
    file_path=None,
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
        Region summary columns to print. Defaults to ``region``, ``calls``,
        ``percent`` and ``avg``. The optional ``total`` column remains
        available for callers that need aggregate duration. The percentage is
        relative to ``scope_profiler.session``. The public name for the first
        column is ``region``; ``name`` is accepted as an alias for Python
        callers.
    percentage_mode : {"coverage", "exclusive"}, optional
        Quantity used for the ``% session`` column. Wall-clock coverage is the
        default; exclusive time can be selected for attribution-focused tables.
    file_path : str or Path, optional
        File represented by the table, used for the help hints in the info box.
    """
    stream = sys.stdout if stream is None else stream
    if percentage_mode not in {"coverage", "exclusive"}:
        raise ValueError("percentage_mode must be 'coverage' or 'exclusive'")
    if not rows:
        if title:
            print(title, file=stream)
        print("  (no regions recorded)", file=stream)
        return

    session_total = next(
        (root["total"] for root in rows if root["name"] == "scope_profiler.session"),
        None,
    )
    if columns is None and session_total is None:
        # Percentages are defined relative to the session root. When a
        # filtered table does not contain that root, omit the unusable column
        # from the default layout rather than filling it with dashes.
        columns = ("region", "calls", "total", "avg")
    selected_columns = normalize_region_table_columns(columns)

    formatted = [
        {
            "name": _display_region_name(row),
            "ranks": str(row["num_ranks"]),
            "calls": _format_count(row["calls"]),
            "total": _format_duration(row["total"]),
            # The session root represents the complete run, so keep it at
            # 100% even when exclusive attribution is selected.
            "percent": _format_percentage(
                (
                    row["total"]
                    if percentage_mode == "exclusive"
                    and row["name"] == "scope_profiler.session"
                    else row.get(percentage_mode)
                ),
                session_total,
            ),
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

    # ``total_time`` is supplied by finalize()/print_summary(), but not by
    # the inspect renderer. Keep the latter's historical region-only output.
    if total_time is not None:
        timed = [row["total"] for row in rows if row["total"] is not None]
        total_row = {
            "name": "TOTAL",
            "ranks": "",
            "calls": _format_count(sum(row["calls"] for row in rows)),
            "total": _format_duration(sum(timed) if timed else None),
            # TOTAL is the run represented by the session root, rather than
            # the sum of the root and its nested contribution rows.
            "percent": _format_percentage(session_total, session_total),
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
        formatted.append(total_row)

    headers = [header for _, header in selected_columns]
    table_rows = [[row[key] for key, _ in selected_columns] for row in formatted]
    _print_table(table_rows, headers, stream)
    notes = []
    if title:
        notes.append(f"Summary: {title}")
    if file_path is not None:
        command_path = shlex.quote(os.path.relpath(str(file_path)))
        notes.extend(
            (
                "",
                "Explore:",
                f"  Inspect: scope-profiler inspect {command_path}",
                f"  TUI:     scope-profiler tui {command_path}",
                "",
                "Visualize and export:",
                f"  Plot:    scope-profiler plot default {command_path} -o plots --show",
                f"  Report:  scope-profiler report {command_path} -o report.html",
                f"  Export:  scope-profiler export plot-data {command_path} -o data",
                f"  Lines:   scope-profiler line-profile {command_path}",
                "",
                "Compare runs:",
                "  Diff:    scope-profiler diff BASE.h5 CANDIDATE.h5",
                "  Check:   scope-profiler check BASE.h5 CANDIDATE.h5",
            ),
        )
        notes.append("")
    notes.append("Durations are in seconds.")
    if len(rows) > 1:
        # Nested regions are counted in both the inner and the outer row, so
        # the summed total legitimately exceeds the run's wall-clock time.
        notes.append(
            "Regions may nest, so the summed total can exceed the wall-clock time.",
        )
    if any(row.get("recursive") for row in rows):
        notes.append("↻ Recursive rows aggregate all invocations of that region.")
    if session_total is not None:
        notes.append(
            (
                "% session uses wall-clock coverage; overlapping recursive calls "
                "count once."
                if percentage_mode == "coverage"
                else "% session uses exclusive time for each region."
            ),
        )
    if any(row["total"] is None for row in rows):
        notes.append(
            "Regions shown without timing recorded no calls on the selected ranks.",
        )
    if not suppress_notes and notes:
        width = max(len(note) for note in notes)
        print(file=stream)
        print(f"  ╭─ Info {'─' * max(1, width - 5)}╮", file=stream)
        for note in notes:
            print(f"  │ {note:<{width}} │", file=stream)
        print(f"  ╰{'─' * (width + 2)}╯", file=stream)

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
    return not (exclude is not None and any(re.match(p, name) for p in exclude))


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
            ),
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
                    table["info"],
                    "call count",
                    column,
                    result.call_counts[thread],
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
            },
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

    _print_heading(title, stream)
    for index, (heading, rows) in enumerate(sections):
        if heading:
            _print_heading(f"  {heading}", stream)
        table_rows = [
            [name, *[_format_counter(value) for value in values]]
            for name, values in rows
        ]
        _print_table(table_rows, ("counter", *columns), stream)
    print(file=stream)


def print_likwid_tables(results, include=None, exclude=None, ranks=None, stream=None):
    """Print every LIKWID counter table a result set exposes.

    A no-op for files recorded without LIKWID, so callers can invoke it
    unconditionally.
    """
    for table in likwid_tables(results, include=include, exclude=exclude, ranks=ranks):
        print_likwid_table(table, stream=stream)


def perf_event_tables(results, include=None, exclude=None, ranks=None) -> list:
    """Build one built-in perf-event counter table per rank.

    These counters are already summed across calls by the collector, so each
    region occupies one row and calls/events occupy columns. Missing events
    use ``-`` rather than being treated as zero.
    """
    tables = []
    for rank, regions in sorted(results.get_perf_events().items()):
        if ranks is not None and rank not in ranks:
            continue
        selected = [
            (name, totals)
            for name, totals in regions.items()
            if _name_selected(name, include, exclude)
        ]
        if not selected:
            continue
        selected.sort(key=lambda item: item[0])
        event_names = sorted(
            {event for _, totals in selected for event in totals.values},
        )
        tables.append(
            {
                "rank": rank,
                "events": event_names,
                "rows": [
                    (
                        name,
                        totals.calls,
                        *[totals.values.get(event) for event in event_names],
                    )
                    for name, totals in selected
                ],
            },
        )
    return tables


def print_perf_event_tables(
    results,
    include=None,
    exclude=None,
    ranks=None,
    stream=None,
):
    """Print every built-in Linux perf-event counter table in ``results``."""
    stream = sys.stdout if stream is None else stream
    for table in perf_event_tables(
        results,
        include=include,
        exclude=exclude,
        ranks=ranks,
    ):
        _print_heading(f"Perf events (rank {table['rank']})", stream)
        _print_table(
            [
                [
                    name,
                    _format_counter(calls),
                    *[_format_counter(value) for value in values],
                ]
                for name, calls, *values in table["rows"]
            ],
            ("region", "calls", *table["events"]),
            stream,
        )
        print(file=stream)
