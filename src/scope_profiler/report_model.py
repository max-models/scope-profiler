"""What a profiling report says, as plain data.

The HTML report and the Astro catalogue pages show the same things: headline
numbers, findings, hotspots, tables, charts. The decisions about what to say
(which findings apply, how a change is classified, which rows lead) live here,
as JSON-serializable models, so the two renderers cannot drift apart: each only
decides how to draw them.

Text that mentions a region is a list of *parts*: a plain string, or a mapping
such as ``{"region": "solve"}`` (a region to link to its row in the table) or
``{"anchor": "balance", "text": "load balance"}`` (a link to a section). The
renderer escapes the strings and decides where the links point.
"""

from __future__ import annotations

import linecache
import re
from pathlib import Path

import numpy as np

from scope_profiler.report_data import (
    _BALANCE_ROWS,
    _CHANGE_COUNT,
    _COMPARE_METRICS,
    _HOT_CALL_AVG_SECONDS,
    _HOT_CALL_THRESHOLD,
    _HOTSPOT_COUNT,
    _IMBALANCE_FLAG_PCT,
    _LP_FUNCTIONS,
    _MATRIX_RANKS,
    _MATRIX_REGIONS,
    _SAME_PCT,
    _SESSION,
    _SPEEDUP_REGIONS,
    _UNCOVERED_FLAG_PCT,
    _change_items,
    _comparison_entries,
    _dedent_width,
    _duration_text,
    _format_timestamp,
    _highlight_python,
    _hotspot_base,
    _hotspot_entries,
    _line_profile_functions,
    _metric_text,
    _metric_value,
    _per_rank_row,
    _plural,
    _rank_balance,
    _seconds,
    _signed_pct,
    _source_path,
    _threads,
)
from scope_profiler.summary import (
    _format_counter,
    _region_durations,
    _session_total,
    format_region_table,
    gpu_timing_warnings,
    likwid_tables,
    perf_event_tables,
    region_rows,
)

# The tree glyphs and column indentation that format_region_table() puts in
# front of a cell's value, kept apart from it so that re-sorting a table by a
# column, which breaks the hierarchy, can hide them.
_CELL_INDENT = re.compile(r"^[ │└─]*")


def _kpi(label, value, detail="", tone=""):
    """One headline number; ``value`` and ``detail`` are lists of parts or strings."""
    return {"label": label, "value": value, "detail": detail, "tone": tone}


def summary_model(results, rows, entries, base, balance) -> dict:
    """Headline numbers and a few sentences on what stands out in one run.

    Returns ``{"empty": True}`` without timed regions; otherwise ``{"kpis": [...],
    "findings": [...]}``, where each finding is ``{"tone": "info" | "warn",
    "parts": [...]}``.
    """
    timed = [row for row in rows if row["total"] is not None]
    if not timed:
        return {"empty": True, "kpis": [], "findings": []}

    kpis = []
    if results.time_span:
        kpis.append(
            _kpi("Wall time", [_duration_text(results.time_span)], "setup to finalize")
        )
    kpis.append(_kpi("Ranks", [str(results.num_ranks)]))
    kpis.append(_kpi("Regions", [str(len({row["name"] for row in rows}))]))
    findings: list[dict] = []

    def finding(tone, *parts):
        findings.append({"tone": tone, "parts": list(parts)})

    regions_only = [entry for entry in entries if entry["kind"] != "outside"]
    outside = next((entry for entry in entries if entry["kind"] == "outside"), None)
    # A lone session root has no regions for the outside time to be outside of.
    if outside is not None and base and regions_only:
        uncovered = 100.0 * outside["time"] / base
        kpis.append(
            _kpi(
                "In regions",
                [f"{100.0 - uncovered:.0f}%"],
                "of the session",
                "warn" if uncovered >= _UNCOVERED_FLAG_PCT else "",
            )
        )
        if uncovered >= _UNCOVERED_FLAG_PCT:
            finding(
                "warn",
                f"{uncovered:.0f}% of the session ({_duration_text(outside['time'])}) "
                "is outside every region; adding regions there would show "
                "where that time goes.",
            )

    if regions_only:
        top = regions_only[0]
        share = f"{100.0 * top['time'] / base:.1f}%" if base else ""
        kpis.append(
            _kpi(
                "Top hotspot",
                [{"region": top["name"]}],
                f"{share} of the session" if share else "",
            )
        )
        where = f" (in {' › '.join(top['context'])})" if top["context"] else ""
        subject = (
            [
                "Time spent directly in ",
                {"region": top["name"]},
                where,
                ", outside its nested regions,",
            ]
            if top["kind"] == "own"
            else [{"region": top["name"]}, where]
        )
        findings.insert(
            0,
            {
                "tone": "info",
                "parts": [
                    *subject,
                    f" is the largest hotspot: {_duration_text(top['time'])} over "
                    f"{_plural(top['calls'], 'call')}"
                    + (f", {share} of the session." if share else "."),
                ],
            },
        )
        if len(regions_only) >= 3 and base:
            top_three = sum(entry["time"] for entry in regions_only[:3])
            findings.insert(
                1,
                {
                    "tone": "info",
                    "parts": [
                        (
                            "The three largest hotspots account for "
                            f"{100.0 * top_three / base:.0f}% of the session."
                        )
                    ],
                },
            )

    if balance is not None:
        worst = balance["regions"][0]
        tone = "warn" if worst["imbalance"] >= _IMBALANCE_FLAG_PCT else ""
        kpis.append(
            _kpi(
                "Load imbalance",
                [f"+{worst['imbalance']:.0f}%"],
                f"{worst['name']}, slowest on rank {worst['slowest']}",
                tone,
            )
        )
        if tone:
            finding(
                "warn",
                {"region": worst["name"]},
                f" is unevenly distributed across ranks: rank {worst['slowest']} "
                f"spends {worst['imbalance']:.0f}% more time in it than the "
                f"per-rank average ({_duration_text(worst['excess'])} more). See ",
                {"anchor": "balance", "text": "load balance"},
                ".",
            )
        counts = balance["slowest_counts"]
        if counts:
            rank, slowest_in = counts.most_common(1)[0]
            if slowest_in >= 2 and slowest_in * 2 >= len(balance["regions"]):
                finding(
                    "info",
                    f"Rank {rank} is the slowest rank in {slowest_in} of "
                    f"{len(balance['regions'])} regions.",
                )

    chatty = [
        row
        for row in timed
        if row["calls"] >= _HOT_CALL_THRESHOLD
        and row["avg"] is not None
        and row["avg"] < _HOT_CALL_AVG_SECONDS
    ]
    if chatty:
        worst_row = max(chatty, key=lambda row: row["calls"])
        finding(
            "warn",
            {"region": worst_row["name"]},
            f" was called {worst_row['calls']} times at ~{worst_row['avg'] * 1e6:.1f} "
            "µs on average; frequent short calls like this can make timer overhead "
            "itself measurable.",
        )

    untimed = len(rows) - len(timed)
    if untimed:
        finding("info", f"{untimed} region(s) recorded no calls on the selected ranks.")
    return {"empty": False, "kpis": kpis, "findings": findings}


def deviation(value: float, mean: float) -> dict | None:
    """How a value shades against its mean: red above, blue below, none near it.

    Returns ``{"sign": 1 | -1, "alpha": float}``, or None for a value within
    the noise of the mean.
    """
    if mean <= 0:
        return None
    relative = (value - mean) / mean
    if abs(relative) < _SAME_PCT / 100.0:
        return None
    strength = min(abs(relative) / 0.5, 1.0)
    return {
        "sign": 1 if relative > 0 else -1,
        "alpha": round(0.08 + 0.47 * strength, 2),
    }


def matrix_model(balance: dict) -> dict:
    """Every rank side by side, region by region, against the region's mean.

    ``rows`` and ``totals`` hold one cell per rank: its ``value`` in seconds, its
    ``label`` (the percent against the mean), its ``title`` and its ``shade``.
    """
    ranks = balance["ranks"][:_MATRIX_RANKS]
    columns = len(ranks)
    by_size = sorted(balance["regions"], key=lambda entry: -entry["mean"])
    shown = by_size[:_MATRIX_REGIONS]
    totals = balance["totals"][:columns]
    totals_mean = float(np.mean(totals))
    # Ranks that wait for each other end up with the same time in regions;
    # the largest of near-equal totals is noise, not a slowest rank.
    slowest_rank = (
        ranks[int(np.argmax(totals))]
        if totals_mean
        and (float(np.max(totals)) - totals_mean) / totals_mean >= _SAME_PCT / 100.0
        else None
    )

    def cells(values, mean) -> list[dict]:
        rendered = []
        for rank, value in zip(ranks, values[:columns]):
            change = 100.0 * (value - mean) / mean if mean else 0.0
            text = _signed_pct(change) if abs(change) >= 0.05 else "0%"
            rendered.append(
                {
                    "rank": rank,
                    "value": float(value),
                    "label": text,
                    "title": f"rank {rank}: {_duration_text(value)} ({text} vs mean)",
                    "shade": deviation(value, mean),
                }
            )
        return rendered

    notes = []
    if len(by_size) > len(shown):
        notes.append(f"The {len(shown)} regions with the most time per rank are shown.")
    if len(balance["ranks"]) > columns:
        notes.append(
            f"The first {columns} of {len(balance['ranks'])} ranks are shown; the "
            "rank heatmap chart shows all of them."
        )
    overall = (
        f"Rank {slowest_rank}, in red in the header, spends the most time in "
        "regions overall."
        if slowest_rank is not None
        else "Every rank spends about the same time in regions overall, so the "
        "differences above are ranks waiting for each other."
    )
    return {
        "ranks": ranks,
        "slowest_rank": slowest_rank,
        "rows": [
            {
                "name": entry["name"],
                "mean": entry["mean"],
                "mean_text": _duration_text(entry["mean"]),
                "cells": cells(entry["values"], entry["mean"]),
            }
            for entry in shown
        ],
        "totals": {
            "mean": totals_mean,
            "mean_text": _duration_text(totals_mean),
            "cells": cells(totals, totals_mean),
        },
        "notes": notes,
        "overall": overall,
    }


def change_model(base, value, metric: str) -> dict:
    """How ``value`` compares with ``base``, for a comparison table cell.

    ``kind`` is ``"none"`` (neither has it), ``"new"``, ``"gone"`` or ``"delta"``;
    a delta has a ``tone`` (``faster``, ``slower``, ``same``, or, for call counts,
    ``neutral``), a ``pct`` change and a ``title`` giving its size. ``size`` is
    the magnitude of the change, for sorting by it.
    """
    from scope_profiler.report_data import _signed_pct as signed

    if base is None and value is None:
        return {"kind": "none", "size": None}
    if base is None:
        return {"kind": "new", "size": abs(value)}
    if value is None:
        return {"kind": "gone", "size": abs(base)}
    delta = value - base
    if not base:
        return {"kind": "new", "size": abs(delta)}
    pct = 100.0 * delta / base
    if metric == "calls":
        tone = "same" if delta == 0 else "neutral down" if delta < 0 else "neutral"
        title = f"{delta:+,} calls"
    else:
        tone = "same" if abs(pct) < _SAME_PCT else "faster" if delta < 0 else "slower"
        speed = (
            ""
            if not value
            else (
                f" · {base / value:.2f}× faster"
                if delta < 0
                else f" · {value / base:.2f}× slower"
            )
        )
        title = f"{'+' if delta >= 0 else '−'}{_duration_text(abs(delta))}{speed}"
    return {
        "kind": "delta",
        "tone": tone,
        "pct": pct,
        "label": signed(pct),
        "title": title,
        "size": abs(delta),
    }


def line_profile_model(entry: dict) -> dict:
    """One profiled function's source, each line with the time spent on it.

    ``rows`` has a row per line: ``number``, ``source`` (HTML with syntax
    classes) and, for a line that was timed, ``hits``, ``seconds``, ``per_hit``
    and ``percent``; the hottest line has ``hot`` set. ``has_source`` is false
    when no line's source could be found, and then only timed lines are listed.
    """
    recorded = entry["lines"]
    total = entry["total"]
    first = min(entry["first_lineno"], min(recorded))
    last = max(recorded)
    stored = entry.get("stored") or {}
    path = _source_path(entry["filename"])
    sources = {
        number: (
            stored[number]
            if number in stored
            else linecache.getline(path, number).rstrip("\n")
        ).expandtabs()
        for number in range(first, last + 1)
    }
    # With the source at hand, show the function as it reads, unrecorded
    # lines (comments, blank lines, the def) included; without it, only
    # the lines that have timings.
    has_source = any(sources.values())
    numbers = list(range(first, last + 1)) if has_source else sorted(recorded)
    strip = _dedent_width(sources[number] for number in numbers)
    highlighted = dict(
        zip(numbers, _highlight_python([sources[number][strip:] for number in numbers]))
    )
    hottest = max(recorded, key=lambda number: recorded[number][1])
    rows = []
    for number in numbers:
        row = {"number": number, "source": highlighted[number], "recorded": False}
        if number in recorded:
            hits, seconds = recorded[number]
            row.update(
                recorded=True,
                hits=hits,
                seconds=seconds,
                seconds_text=_duration_text(seconds),
                per_hit=seconds / hits if hits else 0.0,
                per_hit_text=_duration_text(seconds / hits if hits else 0.0),
                percent=100.0 * seconds / total if total else 0.0,
                hot=bool(number == hottest and total),
            )
        rows.append(row)
    return {"rows": rows, "has_source": has_source, "filename": entry["filename"]}


def _cell(text: str) -> dict:
    indent = _CELL_INDENT.match(text).group()  # type: ignore[union-attr]
    return {"indent": indent, "text": text[len(indent) :]}


def _sort_value(row: dict, key: str, session_total) -> str:
    if key == "ranks":
        value = row["num_ranks"]
    elif key == "percent":
        value = (
            100.0 * row["coverage"] / session_total
            if row.get("coverage") is not None and session_total
            else None
        )
    elif key == "parent_percent":
        value = (
            100.0 * row["coverage"] / row["parent_coverage"]
            if row.get("coverage") is not None and row.get("parent_coverage")
            else None
        )
    else:
        value = row[key]
    return "" if value is None else str(value)


def line_profile_location(results, entry: dict) -> dict:
    """Where a profiled function is, and on how many ranks it ran."""
    location = f"{Path(entry['filename']).name}:{entry['first_lineno']}"
    ranks_note = (
        f" · {_plural(len(entry['ranks']), 'rank')}"
        if len(entry["ranks"]) > 1
        else f" · rank {entry['ranks'][0]}" if results.num_ranks > 1 else ""
    )
    return {"text": location + ranks_note, "title": entry["filename"]}


def _rank_breakdown(region, ranks) -> list[dict]:
    selected = sorted(
        region.ranks if ranks is None else [r for r in ranks if r in region.regions],
    )
    rows = []
    for rank in selected:
        data = region.regions[rank]
        called = bool(data.num_calls)
        rows.append(
            {
                "rank": rank,
                "calls": data.num_calls,
                "total": data.total_duration,
                "avg": data.average_duration if called else None,
                "min": data.min_duration if called else None,
                "max": data.max_duration if called else None,
            }
        )
    return rows


def region_detail_model(results, region, ranks, functions=()) -> dict:
    """What a region's expandable detail shows: call site, tags, ranks, lines."""
    source = None
    if region.has_source:
        # A run with line profiles shows the code, with its timings, there;
        # the captured snippet would only repeat it.
        snippet = (
            region.source_text.rstrip()
            if region.source_text and not any(results.line_profile.values())
            else None
        )
        source = {
            "file": str(region.source_file),
            "line": region.source_lineno,
            "snippet": snippet,
        }
    return {
        "tags": [str(tag) for tag in region.tags],
        "source": source,
        "gpu": (
            {"total": region.gpu_total_duration, "average": region.gpu_average_duration}
            if region.has_gpu_timing
            else None
        ),
        "ranks": _rank_breakdown(region, ranks),
        "line_profiles": [
            {
                "function": entry["function"],
                "location": line_profile_location(results, entry),
                "total": entry["total"],
                "total_text": _duration_text(entry["total"]),
                "table": line_profile_model(entry),
            }
            for entry in functions
        ],
    }


def region_table_model(results, rows, ranks, columns, line_profiles=None) -> dict:
    """Region statistics laid out like the terminal summary table.

    Cell text comes from :func:`~scope_profiler.summary.format_region_table`, so
    every renderer shows the same tree, call counts and ``(own)`` rows. Each
    region is one entry of ``groups``, holding its row, its ``own`` row when it
    has children, its sort keys, its call durations for a trend line and its
    ``detail``.
    """
    line_profiles = {} if line_profiles is None else line_profiles
    selected_columns, display_rows = format_region_table(rows, columns)
    keys = [key for key, _ in selected_columns]
    max_total = max((row["total"] or 0.0 for row in rows), default=0.0)
    session_total = _session_total(rows)
    # (own) rows follow exactly the regions that have children in the table.
    parents = {row.get("call_path") for row, _, is_own in display_rows if is_own}

    def bar(value):
        return 100.0 * (value or 0.0) / max_total if max_total else None

    groups: list[dict] = []
    for row, display, is_own in display_rows:
        cells = {key: _cell(display[key]) for key in keys}
        if is_own:
            groups[-1]["own"] = {"cells": cells, "bar": bar(row["exclusive"])}
            continue
        region = results.get_region(row["name"])
        durations = np.asarray(_region_durations(region, ranks), dtype=float)
        # A trend line cannot show more points than it has pixels.
        if durations.size > 200:
            durations = durations[np.linspace(0, durations.size - 1, 200).astype(int)]
        groups.append(
            {
                "name": row["name"],
                "run": results.display_label,
                "path": row.get("call_path"),
                "depth": row["depth"] if "call_path" in row else None,
                "sort": {key: _sort_value(row, key, session_total) for key in keys},
                "has_children": row.get("call_path") in parents,
                "cells": cells,
                "bar": bar(row["total"]),
                "own": None,
                "durations": [float(value) for value in durations],
                "detail": region_detail_model(
                    results, region, ranks, line_profiles.get(row["name"], ())
                ),
            }
        )
    notes = ["Durations are in seconds.", *gpu_timing_warnings(rows)]
    if parents:
        notes.append("(own) rows show a region's time excluding its children.")
    if session_total is not None and "percent" in keys:
        notes.append(
            "% session uses wall-clock coverage; overlapping recursive calls count once."
        )
    if rows:
        notes.append(
            "Click a row for its call site, line profile and per-rank breakdown."
            if line_profiles
            else "Click a row for its call site and per-rank breakdown."
        )
    return {
        "columns": [{"key": key, "header": header} for key, header in selected_columns],
        "groups": groups,
        "notes": notes,
        "tree": bool(parents),
        "empty": not rows,
    }


def hotspots_model(entries, base) -> dict:
    """The largest leaves of the call tree, ranked, with their share of ``base``.

    ``items`` is empty when there is nothing to rank against (fewer than two
    entries). Each item has the entry's ``name``, ``context`` and ``kind``
    (``leaf``, ``own`` or ``outside``), its ``share`` in percent, ``details``
    (time, calls, time per call) and, for a slow leaf, a ``flag``.
    """
    if len(entries) < 2:
        return {"items": []}
    items = []
    for position, entry in enumerate(entries[:_HOTSPOT_COUNT], start=1):
        share = 100.0 * entry["time"] / base if base else 0.0
        details = [_duration_text(entry["time"])]
        if entry["kind"] != "outside":
            details.append(_plural(entry["calls"], "call"))
            if entry["calls"]:
                details.append(f"{_duration_text(entry['time'] / entry['calls'])}/call")
        flag = None
        if entry["imbalance"] is not None and entry["imbalance"] >= _IMBALANCE_FLAG_PCT:
            flag = f"slowest rank +{entry['imbalance']:.0f}%"
        items.append(
            {
                "rank": position,
                "name": entry["name"],
                "context": list(entry["context"]),
                "kind": entry["kind"],
                "time": entry["time"],
                "calls": entry["calls"],
                "share": share,
                "details": details,
                "flag": flag,
                "title": " › ".join([*entry["context"], entry["name"]]),
            }
        )
    return {"items": items}


def run_meta_model(results, region_count: int, metadata_keys=None) -> dict:
    """One line on where, when and for how long a run was profiled.

    ``metadata_keys`` limits which of the host-describing metadata fields
    (``hostname``, ``chip_information``) are included; all of them by default.
    """
    metadata = results.metadata or {}
    keys = ("hostname", "chip_information")
    keys = (
        keys if metadata_keys is None else tuple(k for k in keys if k in metadata_keys)
    )
    file = None
    if results.file_path and str(results.file_path) != ".":
        path = Path(results.file_path)
        file = {"name": path.name, "path": str(path)}
    return {
        "file": file,
        "timestamp": (
            _format_timestamp(metadata["timestamp"])
            if metadata.get("timestamp")
            else None
        ),
        "host": [str(metadata[key]) for key in keys if metadata.get(key)],
        "ranks": _plural(results.num_ranks, "rank"),
        "regions": _plural(region_count, "region"),
        "profiled": (
            {
                "text": f"{_seconds(results.time_span)} profiled",
                "title": f"setup to finalize: {_seconds(results.total_time)}",
            }
            if results.time_span is not None
            else None
        ),
    }


def balance_model(results, balance: dict) -> dict:
    """How evenly each region's time is spread over the ranks.

    ``rows`` are ordered by excess (the time balancing could save); each has the
    mean, min and max per-rank time, the slowest rank, the imbalance in percent,
    and ``low``, ``high`` and ``mean_pos`` as percentages of the widest bar for
    the min-mean-max spread. ``limit`` rows are shown before "show all".
    """
    scale = max(entry["max"] for entry in balance["regions"]) or 1.0
    rows = []
    for entry in balance["regions"]:
        rows.append(
            {
                "name": entry["name"],
                "mean": entry["mean"],
                "min": entry["min"],
                "max": entry["max"],
                "slowest": entry["slowest"],
                "imbalance": entry["imbalance"],
                "excess": entry["excess"],
                "mean_text": _duration_text(entry["mean"]),
                "min_text": _duration_text(entry["min"]),
                "max_text": _duration_text(entry["max"]),
                "excess_text": _duration_text(entry["excess"]),
                "flag": entry["imbalance"] >= _IMBALANCE_FLAG_PCT,
                "low": 100.0 * entry["min"] / scale,
                "high": 100.0 * entry["max"] / scale,
                "mean_pos": 100.0 * entry["mean"] / scale,
                "values": [float(value) for value in entry["values"]],
            }
        )
    return {
        "metric": "own time" if balance["own"] else "total time",
        "own": balance["own"],
        "ranks": list(balance["ranks"]),
        "rows": rows,
        "limit": _BALANCE_ROWS,
    }


def hardware_model(runs, include=None, exclude=None, ranks=None) -> list[dict]:
    """Counter tables, only for the runs and ranks that recorded hardware metrics.

    Each panel has a ``kind`` (``likwid`` or ``perf``), a ``title``, and tables:
    a ``columns`` list and ``rows`` of already formatted cell texts.
    """

    def formatted(rows):
        return [
            [
                str(value) if index == 0 else _format_counter(value)
                for index, value in enumerate(row)
            ]
            for row in rows
        ]

    panels = []
    for run in runs:
        for table in likwid_tables(run, include=include, exclude=exclude, ranks=ranks):
            sections = [
                {
                    "heading": heading,
                    "columns": ["counter", *map(str, table["columns"])],
                    "rows": formatted((name, *values) for name, values in rows),
                }
                for heading, rows in table["sections"]
                if rows
            ]
            panels.append(
                {
                    "kind": "likwid",
                    "title": f"LIKWID: {run.display_label}, rank {table['rank']}, "
                    f"group {table['group']}",
                    "sections": sections,
                }
            )
        for table in perf_event_tables(
            run, include=include, exclude=exclude, ranks=ranks
        ):
            panels.append(
                {
                    "kind": "perf",
                    "title": f"Linux perf events: {run.display_label}, rank {table['rank']}",
                    "sections": [
                        {
                            "heading": "",
                            "columns": ["region", "calls", *map(str, table["events"])],
                            "rows": formatted(table["rows"]),
                        }
                    ],
                }
            )
    return panels


def metadata_model(metadata: dict, keys=None) -> list[list[str]]:
    """The run's recorded metadata as sorted ``[key, value]`` pairs.

    ``keys`` limits which fields are included; all of them by default.
    """
    from scope_profiler.inspection import _json_safe

    rows = []
    for key, value in sorted((metadata or {}).items()):
        if keys is not None and key not in keys:
            continue
        value = _json_safe(value)
        if isinstance(value, (list, dict)):
            import json

            value = json.dumps(value, ensure_ascii=False)
        rows.append([str(key), str(value)])
    return rows


def runs_table_model(runs) -> dict:
    """One line per compared run: its recording, size, wall time and change.

    ``change`` is a :func:`change_model` of the run's wall time against the
    baseline's, or ``{"kind": "base"}`` for the baseline itself.
    """
    base_span = runs[0].time_span
    threads = [_threads(run) for run in runs]
    # Threads are worth a column once they differ, or are more than one.
    show_threads = len(set(threads)) > 1 or any((count or 1) > 1 for count in threads)
    rows = []
    for index, run in enumerate(runs):
        metadata = run.metadata or {}
        file = None
        if run.file_path and str(run.file_path) != ".":
            path = Path(run.file_path)
            file = {"name": path.name, "path": str(path)}
        if index == 0:
            change = {"kind": "base"}
        elif not base_span or not run.time_span:
            change = {"kind": "none", "size": None}
        else:
            change = change_model(base_span, run.time_span, "total")
        rows.append(
            {
                "label": run.display_label,
                "file": file,
                "recorded": (
                    _format_timestamp(metadata["timestamp"])
                    if metadata.get("timestamp")
                    else None
                ),
                "host": metadata.get("hostname"),
                "ranks": run.num_ranks,
                "threads": threads[index],
                "wall": run.time_span,
                "wall_text": _duration_text(run.time_span),
                "change": change,
            }
        )
    return {"show_threads": show_threads, "rows": rows}


def wall_model(base_run, run) -> dict:
    """Wall time of ``run`` against ``base_run``: both, the change and its tone."""
    before, after = base_run.time_span, run.time_span
    if not before or not after:
        return {"after_text": _duration_text(after)}
    pct = 100.0 * (after - before) / before
    return {
        "before_text": _duration_text(before),
        "after_text": _duration_text(after),
        "pct": pct,
        "pct_text": _signed_pct(pct),
        "tone": "" if abs(pct) < _SAME_PCT else "good" if pct < 0 else "bad",
        "speed": (
            f"{before / after:.2f}× faster"
            if after < before
            else f"{after / before:.2f}× slower"
        ),
    }


def _change_item_model(change, scale) -> dict:
    return {
        "name": change["name"],
        "context": list(change["context"]),
        "outside": change["name"] == _SESSION,
        "delta": change["delta"],
        "delta_text": _duration_text(abs(change["delta"])),
        "sign": "+" if change["delta"] > 0 else "−",
        "pct_text": (
            f" ({_signed_pct(100.0 * change['delta'] / change['base'])})"
            if change["base"]
            else " (new)"
        ),
        "width": 100.0 * abs(change["delta"]) / scale if scale else 0.0,
    }


def candidate_model(runs, entries, run_index: int) -> dict:
    """One run against the baseline: headline numbers and what moved.

    ``improvements`` and ``regressions`` are the largest own-time changes, each
    with its size, share of the widest change (``width``) and a label.
    """
    base_run, run = runs[0], runs[run_index]
    improvements = _change_items(entries, run_index, -1)
    regressions = _change_items(entries, run_index, 1)
    scale = max(
        (abs(change["delta"]) for change in improvements + regressions), default=0.0
    )
    extremes = {}
    for title, changes in (
        ("Largest improvement", improvements),
        ("Largest regression", regressions),
    ):
        extremes[title] = (
            {
                "name": changes[0]["name"],
                "sign": "+" if changes[0]["delta"] > 0 else "−",
                "delta_text": _duration_text(abs(changes[0]["delta"])),
            }
            if changes
            else None
        )
    return {
        "label": run.display_label,
        "base_label": base_run.display_label,
        "wall": wall_model(base_run, run),
        "extremes": extremes,
        "improvements": [_change_item_model(change, scale) for change in improvements],
        "regressions": [_change_item_model(change, scale) for change in regressions],
        "limit": _CHANGE_COUNT,
    }


def compare_table_model(runs, entries, per_rank: bool = False) -> dict:
    """Every region's duration in every run, against the baseline.

    ``rows`` follow the baseline's call tree. Each row has, per run, the text of
    every metric in ``metrics`` and, for the runs after the baseline, a
    :func:`change_model` against the baseline; ``sizes`` holds the size of the
    last run's change per metric, to sort by.
    """
    rows = []
    for order, entry in enumerate(entries):
        runs_cells = []
        sizes = {}
        for run_index, row in enumerate(entry["values"]):
            values = {}
            changes = {}
            for metric, _ in _COMPARE_METRICS:
                value = _metric_value(row, metric)
                values[metric] = _metric_text(value, metric)
                if run_index:
                    base = _metric_value(entry["values"][0], metric)
                    changes[metric] = change_model(base, value, metric)
                    # Sorting by change follows the last run against the baseline.
                    sizes[metric] = changes[metric]["size"]
            runs_cells.append({"values": values, "changes": changes or None})
        rows.append(
            {
                "name": entry["name"],
                "depth": entry["depth"],
                "context": list(entry["context"]),
                "title": " › ".join([*entry["context"], entry["name"]]),
                "order": order,
                "sizes": sizes,
                "cells": runs_cells,
            }
        )
    return {
        "labels": [run.display_label for run in runs],
        "metrics": [{"key": key, "label": label} for key, label in _COMPARE_METRICS],
        "rows": rows,
        "per_rank": per_rank,
    }


def chart_description_text(title: str, payload: dict, comparison: bool = False) -> str:
    """Explain how to read one chart; ``""`` for a chart without an explanation.

    A scaling chart's text names its x-axis as ``{axis}``, which the renderer
    replaces by the current axis label, since the reader can switch axes.
    """
    if title.startswith("Timeline:"):
        return (
            "Each bar is one recorded region call on rank 0. Its position and "
            "width show when the call started and how long it ran; colors "
            "identify regions."
        )
    if title == "Region durations":
        what = (
            "mean call duration"
            if payload.get("metrics") == ["avg"]
            else "total recorded duration"
        )
        if payload.get("options", {}).get("stack_children"):
            return (
                f"Each bar shows a region's {what}, summed over the selected "
                "ranks. The stacked segments divide that time between the "
                "region itself (self) and the regions it calls directly."
            )
        if not comparison:
            return (
                f"Each bar shows a region's {what}, summed over the selected "
                "ranks, longest first."
            )
        return f"Grouped bars compare each region's {what} across the profiled runs."
    if title.startswith("Change:"):
        return (
            "Percent change in each region's total duration, candidate over "
            "baseline. Bars above zero got slower; a region measured in only "
            "one run leaves a gap rather than reading as a 100% change."
        )
    if title == "Speedup":
        return (
            "Each line is one region's speedup: its mean call duration on the run "
            "with the fewest {axis} divided by its mean call duration on "
            "each run. The dashed line is ideal scaling; a region below it gains "
            "less than its extra {axis} would allow. The "
            f"{_SPEEDUP_REGIONS} regions with the most time on that first run are "
            "shown. Valid for a strong-scaling study: the same total problem size "
            "on every run."
        )
    if title == "Weak scaling":
        return (
            "Each line is one region's weak-scaling efficiency: its mean call "
            "duration on the run with the fewest {axis} divided by its "
            "mean call duration on each run. The dashed line at 1 is ideal: the "
            "same time per call however large the run; a region below it loses "
            "time to costs that grow with the run, such as communication or load "
            f"imbalance. The {_SPEEDUP_REGIONS} regions with the most time on that "
            "first run are shown. Valid for a weak-scaling study: the problem "
            "grows with the run, so every run has the same problem size per "
            "rank or core."
        )
    if title == "Rank heatmap":
        return (
            "This heatmap uses exclusive timings. Exclusive duration is the time "
            "spent in a region itself, excluding time spent in nested child "
            "regions; this prevents the enclosing session region from dominating "
            "the heatmap."
        )
    if title == "Rank imbalance":
        return (
            "Each line compares a region's total duration by rank; its dashed line "
            "marks the mean across ranks. Points far from the mean identify stragglers."
        )
    if title.startswith("LIKWID:"):
        return (
            "Bars compare the selected LIKWID metric across regions and ranks. "
            "The hardware-counter tables above retain every recorded event and metric."
        )
    return ""


def chart_panels(charts, comparison: bool = False, scaling: str = "both") -> list[dict]:
    """The chart panels of a report, from ``collect_chart_payloads``' charts.

    Each panel has the chart's ``title``, ``payload`` and view ``options``, its
    ``description`` (with ``{axis}`` placeholders, see
    :func:`chart_description_text`), its ``axis`` label, whether it starts
    ``open``, and, for a scaling chart, its x-axis ``variants`` (each with a
    ``label`` and its own ``payload``) with the index of the ``default_variant``.
    """
    # Open only the views that orient a reader; the rest wait, collapsed, for
    # a reader looking for them, rather than all competing for attention.
    # With both scaling charts, the speedup leads and the weak-scaling chart
    # waits; asked for on its own, the weak-scaling chart leads instead.
    lead = "Weak scaling" if scaling == "weak" else "Speedup"
    kinds = (
        (lead, "Change:", "Region durations")
        if comparison
        else ("Region durations", "Timeline:")
    )
    opened = {
        next(index for index, chart in enumerate(charts) if chart[0].startswith(kind))
        for kind in kinds
        if any(chart[0].startswith(kind) for chart in charts)
    }
    if not opened and charts:
        # An aggregated profile has no timeline; never leave every panel shut.
        opened = {0}
    panels = []
    for index, (title, payload, chart_options) in enumerate(charts):
        options = dict(chart_options)
        variants = options.pop("variants", None)
        alternate = options.pop("alternate", None)
        stacked = options.pop("stacked", None)
        is_duration_chart = payload.get("plot") == "durations"
        panels.append(
            {
                "id": f"scope-profiler-chart-{index}",
                "title": title,
                "payload": payload,
                "options": (
                    {**options, "layout": {"height": 680}}
                    if is_duration_chart
                    else options
                ),
                "variants": (
                    [
                        {
                            "field": v.get("field"),
                            "label": v["label"],
                            "payload": v["payload"],
                        }
                        for v in variants
                    ]
                    if variants
                    else None
                ),
                "default_variant": (
                    next(
                        (i for i, v in enumerate(variants) if v["payload"] is payload),
                        0,
                    )
                    if variants
                    else None
                ),
                # The durations chart drawn the other way, to switch to (the page's own
                # "stack children" checkbox); the HTML report does not use it.
                "stack": (
                    {"alternate": alternate, "stacked": stacked}
                    if alternate is not None
                    else None
                ),
                "description": chart_description_text(title, payload, comparison),
                "axis": payload.get("options", {}).get("x_label", "ranks"),
                "open": index in opened,
                "duration": is_duration_chart,
            }
        )
    return panels


def report_rows(results, include, exclude, ranks, sort) -> list[dict]:
    """The region rows a report is built from."""
    return region_rows(
        results,
        include=include,
        exclude=exclude,
        ranks=ranks,
        sort=sort,
        # Populates each row's "exclusive" time. The table's own % column
        # is computed from the inclusive total either way; the hotspots
        # need exclusive time to name the leaves rather than their parents.
        percentage_mode="exclusive",
    )


def partition_line_profiles(results, rows, ranks):
    """Line profiles split into those of tabled regions and the rest.

    Returns ``(by_region, unmatched)``: ``by_region`` maps a region in the table
    to its profiled functions, which its detail shows; ``unmatched`` lists the
    functions of regions the table does not show, largest first.
    """
    tabled = {row["name"] for row in rows}
    by_region: dict[str, list[dict]] = {}
    unmatched = []
    for entry in _line_profile_functions(results, ranks):
        if entry["region"] in tabled:
            by_region.setdefault(entry["region"], []).append(entry)
        else:
            unmatched.append(entry)
    return by_region, unmatched


def line_profile_list_model(results, functions) -> dict:
    """Line profiles of regions the table does not show: one entry per function."""
    grand_total = sum(entry["total"] for entry in functions)
    return {
        "functions": [
            {
                "function": entry["function"],
                "region": entry["region"],
                "location": line_profile_location(results, entry),
                "total": entry["total"],
                "total_text": _duration_text(entry["total"]),
                "share": 100.0 * entry["total"] / grand_total if grand_total else 0.0,
                "table": line_profile_model(entry),
            }
            for entry in functions
        ],
        "limit": _LP_FUNCTIONS,
    }


def run_page_model(
    results,
    include=None,
    exclude=None,
    ranks=None,
    sort="start",
    columns=None,
    host_keys=None,
    metadata_keys=None,
) -> dict:
    """Everything the report shows about one run, except its charts.

    The sections are ``meta``, ``summary``, ``hotspots``, ``regions`` (the table),
    ``balance`` and ``matrix`` (None for a single rank), ``line_profiles`` (None
    without any), ``metadata`` and ``hardware``. ``region_ids`` maps a region to
    the ``id`` of its first row in ``regions["groups"]``, for links to it.

    ``host_keys`` and ``metadata_keys`` limit which host-describing fields the
    header shows and which metadata fields are listed (all by default), for a
    report that is published.
    """
    rows = report_rows(results, include, exclude, ranks, sort)
    entries = _hotspot_entries(rows)
    base = _hotspot_base(rows, entries)
    balance = _rank_balance(results, rows, ranks)
    # A region on several call paths has a row per path.
    region_count = len({row["name"] for row in rows})
    by_region, unmatched = partition_line_profiles(results, rows, ranks)
    regions = region_table_model(results, rows, ranks, columns, by_region)
    region_ids: dict[str, str] = {}
    for index, group in enumerate(regions["groups"]):
        group["id"] = None
        if group["name"] not in region_ids:
            group["id"] = region_ids[group["name"]] = f"region-{index}"
    return {
        "kind": "run",
        "label": results.display_label,
        "region_count": region_count,
        "meta": run_meta_model(results, region_count, host_keys),
        "summary": summary_model(results, rows, entries, base, balance),
        "hotspots": hotspots_model(entries, base),
        "regions": regions,
        "region_ids": region_ids,
        "balance": balance_model(results, balance) if balance is not None else None,
        "matrix": matrix_model(balance) if balance is not None else None,
        "line_profiles": (
            line_profile_list_model(results, unmatched) if unmatched else None
        ),
        "metadata": metadata_model(results.metadata, metadata_keys),
        "hardware": hardware_model([results], include, exclude, ranks),
    }


def compare_dataset(runs, per_run_rows, per_rank: bool) -> dict:
    """Everything needed to compare any two of the runs, compactly, for a browser.

    A comparison of 100 runs has 9,900 ordered pairs, too many to compute in
    advance, so a page that lets the reader pick the two runs computes the pair in
    the browser from this: ``regions`` (``{key: {name, depth, context}}``, a key
    being the region's call path) and, per run, its wall time, the ``order`` of
    its rows (the call tree) and ``values``, ``{key: [total, own, avg, calls]}``
    with ``None`` where the run did not time the region. ``per_rank`` says that
    the runs differ in rank counts, so totals and own times are per rank.
    """
    regions: dict[str, dict] = {}
    run_data = []
    for run, rows in zip(runs, per_run_rows):
        order = []
        values = {}
        for row in rows:
            key = (
                f"path:{row['call_path']}"
                if "call_path" in row
                else f"name:{row['name']}"
            )
            order.append(key)
            regions.setdefault(
                key,
                {
                    "name": row["name"],
                    "depth": row.get("depth", 0),
                    "context": (
                        [
                            part
                            for part in row["call_path"].split(" > ")[:-1]
                            if part != _SESSION
                        ]
                        if "call_path" in row
                        else []
                    ),
                },
            )
            values[key] = [_metric_value(row, metric) for metric, _ in _COMPARE_METRICS]
        run_data.append(
            {
                "label": run.display_label,
                "wall": run.time_span,
                "order": order,
                "values": values,
            }
        )
    return {
        "per_rank": per_rank,
        "metrics": [{"key": key, "label": label} for key, label in _COMPARE_METRICS],
        "regions": regions,
        "runs": run_data,
    }


def comparison_page_model(
    runs, include=None, exclude=None, ranks=None, sort="start", reference=None
) -> dict:
    """Everything the report shows about a comparison of runs, except its charts.

    ``dataset`` (see :func:`compare_dataset`) holds what a page needs to compare any
    two runs. ``candidates`` and ``table`` are the report's comparison of the runs
    against the first; with ``reference`` they cover only that many runs after the
    first, as the reference a page's own computation is checked against.
    """
    per_run_rows = [report_rows(run, include, exclude, ranks, sort) for run in runs]
    per_rank = len({run.num_ranks for run in runs}) > 1
    if per_rank:
        per_run_rows = [[_per_rank_row(row) for row in rows] for rows in per_run_rows]
    count = len(runs) if reference is None else min(len(runs), reference + 1)
    entries = _comparison_entries(per_run_rows[:count])
    labels = [run.display_label for run in runs]
    return {
        "kind": "comparison",
        "labels": labels,
        "base_label": labels[0],
        "per_rank": per_rank,
        "runs": runs_table_model(runs),
        "candidates": [
            candidate_model(runs[:count], entries, index) for index in range(1, count)
        ],
        "table": compare_table_model(runs[:count], entries, per_rank),
        "dataset": compare_dataset(runs, per_run_rows, per_rank),
    }
