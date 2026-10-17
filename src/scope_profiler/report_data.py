"""Data behind the report charts and its scaling axes, without any HTML.

The HTML report and the Astro site generator both need the same chart payloads
(the JSON the ``plot_*`` functions export) and the same rules for what a
comparison scales over; they live here so neither depends on the other.
"""

from __future__ import annotations

import builtins
import functools
import html
import io
import json
import keyword
import os
import re
import sys
import tempfile
import tokenize
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scope_profiler.summary import _session_total

#: Regions drawn in a scaling chart: the ones with the most time on the
#: smallest run, which is where scaling matters.
_SPEEDUP_REGIONS = 8


def _threads(run) -> int | None:
    """A run's OpenMP thread count, when its metadata recorded one."""
    try:
        return int(run.metadata["omp_num_threads"])
    except (KeyError, TypeError, ValueError):
        return None


def _has_call_timestamps(run) -> bool:
    """Whether a run recorded its calls, rather than only summary statistics.

    Aggregated and summary-only profiles keep per-region totals but no call
    timestamps, so the nesting a stacked durations bar needs cannot be
    reconstructed from them.
    """
    return all(
        rank_region.stored_summary is None
        for region in run.get_regions()
        for rank_region in region.regions.values()
    )


def _scaling_field(runs) -> str | None:
    """The parallelism compared runs differ in, for a speedup chart.

    MPI ranks whenever they change, even when threads change too (the cores
    axis is then a button away); threads when only they change; None when the
    runs share both, as two versions of the same code do.
    """
    if len({run.num_ranks for run in runs}) > 1:
        return "num_ranks"
    threads = [_threads(run) for run in runs]
    threads_vary = None not in threads and len(set(threads)) > 1
    return "omp_num_threads" if threads_vary else None


#: ``speedup_x`` choices for a comparison report's speedup chart, each naming
#: the run property it puts on the x-axis. ``"auto"`` picks for itself (see
#: :func:`_scaling_field`).
SPEEDUP_X_FIELDS = {
    "ranks": "num_ranks",
    "nodes": "num_nodes",
    "threads": "omp_num_threads",
    "cores": "total_cores",
}
SPEEDUP_X_CHOICES = ("auto", *SPEEDUP_X_FIELDS)


def _scaling_value(run, field: str) -> int | None:
    """A run's value of a scaling field, or None when the run did not record it."""
    if field == "num_ranks":
        return run.num_ranks
    if field == "num_nodes":
        return run.num_nodes
    threads = _threads(run)
    if threads is None:
        return None
    return threads if field == "omp_num_threads" else run.num_ranks * threads


class SpeedupAxisError(ValueError):
    """The requested speedup x-axis is unknown, or some run did not record it."""


def _check_speedup_x(runs, speedup_x: str) -> str | None:
    """The scaling field ``speedup_x`` names, checked against every run.

    ``"auto"`` resolves to :func:`_scaling_field`'s choice. A named field
    every run has to have recorded: a speedup chart silently missing some
    runs, or silently drawn over another axis, would misread the study.
    """
    if speedup_x not in SPEEDUP_X_CHOICES:
        raise SpeedupAxisError(
            f"speedup_x must be one of {', '.join(SPEEDUP_X_CHOICES)}, "
            f"got {speedup_x!r}",
        )
    if speedup_x == "auto":
        return _scaling_field(runs)
    field = SPEEDUP_X_FIELDS[speedup_x]
    missing = [run.display_label for run in runs if _scaling_value(run, field) is None]
    if missing:
        raise SpeedupAxisError(
            f"the {speedup_x} axis needs {field!r} in every run's metadata, "
            f"but {', '.join(missing)} did not record it"
            + (
                " (files written by older scope-profiler versions lack the node "
                "count of a multi-rank run)"
                if field == "num_nodes"
                else ""
            )
            + "; choose another axis, or 'auto'",
        )
    return field


def _speedup_axes(runs, field: str) -> list[str]:
    """The scaling fields a speedup chart over ``field`` can switch to.

    Every field all runs recorded and differ in, in :data:`SPEEDUP_X_FIELDS`
    order; ``field`` itself is always one. A field whose values match another
    one's run for run - cores when every run has one thread - would draw the
    same chart again and is left out.
    """
    order = list(SPEEDUP_X_FIELDS.values())
    kept: dict[str, tuple] = {}
    for candidate in [field, *(other for other in order if other != field)]:
        values = tuple(_scaling_value(run, candidate) for run in runs)
        if candidate != field and (
            None in values or len(set(values)) < 2 or values in kept.values()
        ):
            continue
        kept[candidate] = values
    return [candidate for candidate in order if candidate in kept]


def collect_chart_payloads(
    runs,
    include,
    exclude,
    ranks,
    comparison: bool = False,
    scaling_field: str | None = None,
    scaling: str = "both",
) -> tuple[list[tuple[str, dict, dict]], list[str]]:
    """Build the chart payloads of one run or of a comparison of runs.

    Returns ``(charts, failures)``: ``charts`` holds ``(title, payload,
    options)`` in display order, where ``payload`` is the JSON document a
    ``plot_*`` function exports and ``options`` its extra view settings (the
    ``variants`` of a scaling chart, a layout override); ``failures`` says why
    a chart is missing. Needs ``scope-profiler[pproc]`` and raises
    :class:`ImportError` without it.

    A comparison keeps only the charts that compare runs; each run's timeline
    and rank charts belong to its own page. ``scaling`` picks the chart(s) for
    runs of different sizes: "strong" the speedup, "weak" the weak-scaling
    efficiency, "both" the two of them.
    """
    from scope_profiler.plotting_scripts import (
        available_likwid_metrics,
        collect_region_statistics,
        plot_durations,
        plot_gantt,
        plot_imbalance,
        plot_likwid,
        plot_rank_heatmap,
        plot_speedup,
        plot_weak_scaling_efficiency,
    )

    charts: list[tuple[str, dict, dict]] = []
    failures: list[str] = []

    def payload_of(title, plotter, path: Path, *args, **kwargs) -> dict | None:
        try:
            plotter(
                *args,
                data_filepath=path,
                data_format="json",
                show=False,
                verbose=False,
                # Plot functions prepare their export payload before handing
                # the Canvas to a renderer. This private sentinel reaches the
                # no-output path and avoids materializing an unused Python
                # figure; the browser package owns rendering for reports.
                backend="data-only",
                **kwargs,
            )
        except (ImportError, ValueError) as exc:
            failures.append(f"{title}: {exc}")
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def collect(
        title,
        plotter,
        path: Path,
        *args,
        chart_options: dict | None = None,
        **kwargs,
    ) -> None:
        payload = payload_of(title, plotter, path, *args, **kwargs)
        if payload is not None:
            charts.append((title, payload, chart_options or {}))

    selected_ranks = [
        [
            rank
            for rank in (range(run.num_ranks) if ranks is None else ranks)
            if 0 <= rank < run.num_ranks
        ]
        for run in runs
    ]
    # The x-axis of a comparison's speedup chart (see _check_speedup_x).
    scaling_field = scaling_field if comparison else None

    with tempfile.TemporaryDirectory(prefix="scope-profiler-report-") as directory:
        payload_dir = Path(directory)
        for index, run in enumerate([] if comparison else runs):
            title = (
                # Naming the rank only says something with more than one.
                f"Timeline: {run.display_label}"
                + (" (rank 0)" if run.num_ranks > 1 else "")
            )
            payload = payload_of(
                title,
                plot_gantt,
                payload_dir / f"gantt-{index}.json",
                run,
                include=include,
                exclude=exclude,
                ranks=[0],
            )
            if payload is None:
                continue
            # Every row is already labelled with its region.
            charts.append((title, payload, {"layout": {"showlegend": False}}))

        scaling_values = (
            [_scaling_value(run, scaling_field) for run in runs]
            if scaling_field is not None
            else []
        )
        if scaling_field is not None and len(set(scaling_values)) < 2:
            # Only a named axis gets here; "auto" picks one the runs differ in.
            failures.append(
                f"Speedup: every run has the same {scaling_field} "
                f"({scaling_values[0]}), so there is no scaling to show",
            )
        elif scaling_field is not None:
            baseline_run = min(runs, key=lambda run: _scaling_value(run, scaling_field))
            # A line per region gets unreadable fast: the regions with the
            # most time on the smallest run, which is where scaling matters.
            # The same regions on every axis, so switching axes keeps the lines.
            leading = sorted(
                baseline_run.get_regions(include=include, exclude=exclude),
                key=lambda region: -region.total_duration,
            )[:_SPEEDUP_REGIONS]
            # One payload per axis the runs differ in, for each scaling chart;
            # the report's buttons switch between them without recomputing
            # anything in the browser.
            scaling_charts = (
                ("Speedup", plot_speedup, "speedup", ("strong", "both")),
                (
                    "Weak scaling",
                    plot_weak_scaling_efficiency,
                    "weak-scaling",
                    ("weak", "both"),
                ),
            )
            axes = _speedup_axes(runs, scaling_field)
            for title, plotter, file_stem, modes in scaling_charts:
                if scaling not in modes:
                    continue
                variants = []
                for field in axes:
                    payload = payload_of(
                        title if field == scaling_field else f"{title} over {field}",
                        plotter,
                        payload_dir / f"{file_stem}-{field}.json",
                        runs,
                        x_field=field,
                        include=[f"{re.escape(region.name)}$" for region in leading],
                        exclude=exclude,
                        ranks=ranks,
                    )
                    if payload is not None:
                        variants.append(
                            {
                                "field": field,
                                "label": payload["options"]["x_label"],
                                "payload": payload,
                            },
                        )
                default = next(
                    (item for item in variants if item["field"] == scaling_field),
                    None,
                )
                if default is not None:
                    charts.append(
                        (
                            title,
                            default["payload"],
                            {"variants": variants} if len(variants) > 1 else {},
                        ),
                    )
        if len(runs) == 2:
            # A baseline/candidate pair wants "what changed?", which reads
            # better as one signed bar per region than as two bars a viewer
            # has to subtract by eye.
            statistics = collect_region_statistics(
                runs,
                ranks=ranks,
                include=include,
                exclude=exclude,
            )
            if statistics["files"]:
                baseline, candidate = (file["label"] for file in statistics["files"])
                charts.append(
                    (
                        f"Change: {candidate} vs {baseline}",
                        {"plot": "region_statistics", **statistics},
                        {"comparison": "percent"},
                    ),
                )
        collect(
            "Region durations",
            plot_durations,
            payload_dir / "durations.json",
            runs,
            include=include,
            exclude=exclude,
            ranks=ranks,
            # Totals summed over ranks grow with the rank count; across
            # run sizes, the mean call reads like the speedup chart.
            metric="total" if len({run.num_ranks for run in runs}) == 1 else "avg",
            sort_by="total",
            # One run's bars split into the region's own time and its direct
            # children, which the ranked table cannot show at a glance. Runs
            # compare as grouped bars instead: stacking both ways on one axis
            # would merge equal segment names across runs. The split needs
            # call timestamps, which aggregated profiles do not store.
            stack_children=len(runs) == 1 and _has_call_timestamps(runs[0]),
        )
        if not comparison and charts and charts[-1][0] == "Region durations":
            # Where the time went is the first question about one run, so its
            # durations lead, above the timeline.
            charts.insert(0, charts.pop())

        # Both rank views are empty or trivial with a single rank.
        if not comparison and any(len(selected) > 1 for selected in selected_ranks):
            collect(
                "Rank imbalance",
                plot_imbalance,
                payload_dir / "rank-imbalance.json",
                runs,
                metric="total",
                include=include,
                exclude=exclude,
                ranks=ranks,
            )
            collect(
                "Rank heatmap",
                plot_rank_heatmap,
                payload_dir / "rank-heatmap.json",
                runs,
                include=include,
                exclude=exclude,
                ranks=ranks,
                exclusive=True,
            )

        likwid_metrics = available_likwid_metrics(runs)
        if likwid_metrics:
            metric = likwid_metrics[0]
            collect(
                f"LIKWID: {metric}",
                plot_likwid,
                payload_dir / "likwid.json",
                runs,
                metric=metric,
                include=include,
                exclude=exclude,
                ranks=ranks,
            )

    return charts, failures


# --- Moved from html_report.py: the data and formatting behind the report. ---

_IMBALANCE_FLAG_PCT = 15.0
_HOT_CALL_THRESHOLD = 1000
_HOT_CALL_AVG_SECONDS = 1e-5
_UNCOVERED_FLAG_PCT = 25.0
_HOTSPOT_COUNT = 8
_SESSION = "scope_profiler.session"
# Past this many regions, the Regions slider opens on only the ones with the
# most time: more lanes and bars than this cannot be read anyway, and the
# browser spends seconds drawing them.
_TIMELINE_REGIONS = 500
# Which scaling chart(s) a comparison of run sizes shows. Nothing in a profile
# records the problem size, so the report cannot tell a strong-scaling study
# (fixed total problem) from a weak-scaling one (fixed problem per rank); the
# caller says which, and by default both are drawn.
SCALING_MODES = ("strong", "weak", "both")
_BALANCE_ROWS = 12
_MATRIX_REGIONS = 20
_MATRIX_RANKS = 64
_MATRIX_TEXT_RANKS = 24
_CHANGE_COUNT = 5
# Line profiles: functions listed before "Show all", the share of a function's
# time that makes a line worth showing, and the fewest lines worth folding.
_LP_FUNCTIONS = 10
# Changes smaller than this read as noise rather than as faster or slower.
_SAME_PCT = 2.0
# Own-time changes below this share of the baseline are not listed as changes.
_CHANGE_FLOOR_PCT = 0.5
# Line profiles of scope-profiler's own frames - the session's __enter__,
# for one - say nothing about the code being profiled.
_PACKAGE_DIR = Path(__file__).resolve().parent
_BUILTIN_NAMES = frozenset(dir(builtins))
# f-strings (3.12+) and t-strings (3.14+) arrive as several tokens.
_STRING_TOKENS = frozenset(
    {tokenize.STRING}
    | {
        getattr(tokenize, name)
        for name in (
            "FSTRING_START",
            "FSTRING_MIDDLE",
            "FSTRING_END",
            "TSTRING_START",
            "TSTRING_MIDDLE",
            "TSTRING_END",
        )
        if hasattr(tokenize, name)
    }
)
_LAYOUT_TOKENS = frozenset(
    {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}
)
_COMPARE_METRICS = (
    ("total", "total"),
    ("own", "own"),
    ("avg", "avg/call"),
    ("calls", "calls"),
)


def _seconds(value) -> str:
    return "-" if value is None else f"{value:.6g} s"


def _duration_text(seconds) -> str:
    """A duration in the unit that keeps it short: 51.2 ms, not 0.0512 s."""
    if seconds is None:
        return "-"
    magnitude = abs(seconds)
    if magnitude == 0:
        return "0 s"
    if magnitude < 1e-6:
        return f"{seconds * 1e9:.3g} ns"
    if magnitude < 1e-3:
        return f"{seconds * 1e6:.3g} µs"
    if magnitude < 1:
        return f"{seconds * 1e3:.3g} ms"
    if magnitude < 1000:
        return f"{seconds:.4g} s"
    return f"{seconds:,.0f} s"


def _signed_pct(value: float) -> str:
    """A percent change with a real minus sign, so columns of them line up."""
    return f"{value:+.1f}%".replace("-", "−")


def _plural(count, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _hotspot_entries(rows) -> list[dict]:
    """Where the time goes: the leaves of the call tree, largest first.

    A region's own time - its time outside every nested region - is a leaf
    of the tree the region table draws: a leaf region's own time is all of
    its time, and a parent's is its ``(own)`` row. Ranking leaves rather
    than inclusive totals names the code that costs the time instead of
    whatever encloses it, and the entries add up to the session.
    """
    timed = [row for row in rows if row["total"] is not None]
    paths = {row["call_path"] for row in timed if "call_path" in row}
    parents = {path.rsplit(" > ", 1)[0] for path in paths if " > " in path}
    entries = []
    for row in timed:
        if "call_path" in row:
            path = row["call_path"]
            if row["name"] == _SESSION:
                # The session root's own time is the time outside every region.
                kind = "outside"
            else:
                kind = "own" if path in parents else "leaf"
            context = [part for part in path.split(" > ")[:-1] if part != _SESSION]
        elif row["name"] == _SESSION:
            # Without a call tree, the session's time outside every region
            # is unknown: its fallback "exclusive" time is its whole total.
            continue
        else:
            kind, context = "leaf", []
        # Legacy profiles whose call tree cannot be rebuilt have no exclusive
        # figure; inclusive is the only thing left to rank them by.
        own = row["total"] if row["exclusive"] is None else row["exclusive"]
        if own <= 0:
            continue
        entries.append(
            {
                "name": row["name"],
                "context": context,
                "kind": kind,
                "time": own,
                "calls": row["calls"],
                # A parent's imbalance is that of its total, not its own time.
                "imbalance": row["imbalance"] if kind == "leaf" else None,
            }
        )
    entries.sort(key=lambda entry: -entry["time"])
    return entries


def _hotspot_base(rows, entries) -> float:
    """What a hotspot's share is a share of: the session, when recorded."""
    session = _session_total(rows)
    if session and any("call_path" in row for row in rows):
        return session
    return sum(entry["time"] for entry in entries)


def _format_timestamp(value) -> str:
    """An ISO timestamp as minutes in UTC, or as recorded when unparseable."""
    text = str(value)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return text
    if moment.tzinfo is None:
        return moment.strftime("%Y-%m-%d %H:%M")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _is_own_frame(filename) -> bool:
    try:
        return Path(filename).resolve().is_relative_to(_PACKAGE_DIR)
    except (OSError, ValueError):
        return False


def _line_profile_functions(results, ranks) -> list[dict]:
    """Line timings per profiled function, summed over the selected ranks."""
    available = results.line_profile
    selected_ranks = sorted(
        available if ranks is None else [rank for rank in ranks if rank in available],
    )
    functions: dict[tuple, dict] = {}
    for rank in selected_ranks:
        for record in available.get(rank, []):
            if _is_own_frame(record["filename"]):
                continue
            key = (
                record["region"],
                record["filename"],
                record["function"],
                int(record["first_lineno"]),
            )
            entry = functions.setdefault(
                key, {"lines": {}, "ranks": set(), "stored": {}}
            )
            entry["ranks"].add(rank)
            if record.get("source") and not entry["stored"]:
                first = int(record["source_first_lineno"])
                entry["stored"] = dict(
                    enumerate(str(record["source"]).split("\n"), start=first)
                )
            unit = float(record["unit"])
            for line, hits, elapsed in zip(
                record["line_numbers"], record["hits"], record["times"]
            ):
                stats = entry["lines"].setdefault(int(line), [0, 0.0])
                stats[0] += int(hits)
                stats[1] += float(elapsed) * unit
    result = []
    for (region, filename, function, first_lineno), entry in functions.items():
        result.append(
            {
                "region": region,
                "filename": filename,
                "function": function,
                "first_lineno": first_lineno,
                "lines": entry["lines"],
                # The source stored with the timings; files recorded before
                # it was stored are read from the source file instead.
                "stored": entry["stored"],
                "ranks": sorted(entry["ranks"]),
                "total": sum(seconds for _, seconds in entry["lines"].values()),
            }
        )
    result.sort(key=lambda entry: -entry["total"])
    return result


def _dedent_width(lines) -> int:
    """The indentation every non-blank line shares, to strip from all of them."""
    widths = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    return min(widths, default=0)


def _highlight_python(lines: list[str]) -> list[str]:
    """Python source lines as HTML, with each token in a classed span.

    The standard library's tokenizer rather than a highlighting package: the
    report takes no dependency for it, and line profiles are always Python.
    The lines are a slice of a file, so they may end inside a statement or a
    string; whatever tokenizes before that is highlighted, the rest is plain.
    """
    marks: list[list[tuple[int, int, str]]] = [[] for _ in lines]
    previous = None
    decorator = False
    try:
        tokens = tokenize.generate_tokens(io.StringIO("\n".join(lines) + "\n").readline)
        for token in tokens:
            kind = None
            if token.type == tokenize.COMMENT:
                kind = "tk-com"
            elif token.type in _STRING_TOKENS:
                kind = "tk-str"
            elif token.type == tokenize.NUMBER:
                kind = "tk-num"
            elif token.type == tokenize.OP:
                if token.string == "@" and (
                    previous is None or previous.type == tokenize.NEWLINE
                ):
                    decorator = True
                elif token.string != ".":
                    decorator = False
                if decorator:
                    kind = "tk-dec"
            elif token.type == tokenize.NAME:
                after_dot = previous is not None and previous.string == "."
                if decorator:
                    kind = "tk-dec"
                elif previous is not None and previous.string in ("def", "class"):
                    kind = "tk-fn"
                elif token.string in ("True", "False", "None"):
                    kind = "tk-const"
                elif keyword.iskeyword(token.string):
                    kind = "tk-kw"
                elif token.string in ("self", "cls"):
                    kind = "tk-self"
                elif token.string in _BUILTIN_NAMES and not after_dot:
                    kind = "tk-bi"
            if token.type not in _LAYOUT_TOKENS or token.type == tokenize.NEWLINE:
                previous = token
            if kind is None:
                continue
            (first_row, first_col), (last_row, last_col) = token.start, token.end
            # A triple-quoted string spans lines: mark its part of each.
            for row in range(first_row, min(last_row, len(lines)) + 1):
                start = first_col if row == first_row else 0
                end = last_col if row == last_row else len(lines[row - 1])
                if end > start:
                    marks[row - 1].append((start, end, kind))
    except (tokenize.TokenError, SyntaxError):
        pass
    rendered = []
    for line, line_marks in zip(lines, marks):
        parts = []
        cursor = 0
        for start, end, kind in sorted(line_marks):
            if start < cursor:
                continue
            parts.append(html.escape(line[cursor:start]))
            parts.append(f'<span class="{kind}">{html.escape(line[start:end])}</span>')
            cursor = end
        parts.append(html.escape(line[cursor:]))
        rendered.append("".join(parts))
    return rendered


@functools.lru_cache(maxsize=256)
def _source_path(filename: str) -> str:
    """Where to read a recorded source file from, here.

    Profiles store paths relative to the working directory or to the
    ``sys.path`` entry the file was imported from, so a relative path is
    looked up against both, in that order.
    """
    if os.path.isabs(filename) or os.path.isfile(filename):
        return filename
    for root in sys.path:
        candidate = os.path.join(root or os.curdir, filename)
        if os.path.isfile(candidate):
            return candidate
    return filename


def _region_index(runs, include, exclude, ranks) -> dict[str, list]:
    """``{region: [seconds, depth]}`` for the Regions and Depth sliders.

    ``seconds`` is the region's largest total over the runs, summed over the
    selected ranks, so a region costly in any run ranks high. ``depth`` is the
    shallowest level its calls reach on the first rank of any run - 0 for a
    top-level call - or None when that rank's calls do not nest.
    """
    from scope_profiler.call_stack import NestingError, build_call_arrays

    index: dict[str, list] = {}
    for run in runs:
        for region in run.get_regions(include=include, exclude=exclude):
            seconds = sum(
                rank_region.total_duration
                for rank, rank_region in region.regions.items()
                if ranks is None or rank in ranks
            )
            entry = index.setdefault(region.name, [0.0, None])
            entry[0] = max(entry[0], seconds)
        rank = 0 if ranks is None else min(ranks, default=0)
        try:
            arrays = build_call_arrays(run.get_regions(), rank)
        except NestingError:
            continue
        shallowest: dict[str, int] = {}
        for row, depth in zip(arrays.region_index.tolist(), arrays.depth.tolist()):
            name = arrays.names[row]
            shallowest[name] = min(depth, shallowest.get(name, depth))
        for name, depth in shallowest.items():
            if name in index:
                known = index[name][1]
                index[name][1] = depth if known is None else min(known, depth)
    return index


def _rank_balance(results, rows, ranks) -> dict | None:
    """Per-rank time in each region, for the load-balance views.

    Own time rather than inclusive: a parent's inclusive total repeats every
    child's imbalance, so ranking it would mostly name the session. Own times
    are disjoint, so they also add up to each rank's time inside regions.
    Without a call tree there is no own time, and totals are used instead.
    """
    selected = sorted(
        rank
        for rank in (range(results.num_ranks) if ranks is None else ranks)
        if 0 <= rank < results.num_ranks
    )
    if len(selected) < 2:
        return None
    timed = [
        row for row in rows if row["total"] is not None and row["name"] != _SESSION
    ]
    own = bool(timed) and all("call_path" in row for row in timed)
    regions = []
    for name in dict.fromkeys(row["name"] for row in timed):
        region = results.get_region(name)
        values = np.asarray(
            [
                (
                    (
                        region.regions[rank].total_exclusive_duration
                        if own
                        else region.regions[rank].total_duration
                    )
                    if rank in region.regions
                    else 0.0
                )
                for rank in selected
            ],
            dtype=float,
        )
        mean = float(values.mean())
        if mean <= 0:
            continue
        peak = int(values.argmax())
        regions.append(
            {
                "name": name,
                "values": values,
                "mean": mean,
                "min": float(values.min()),
                "max": float(values.max()),
                "slowest": selected[peak],
                "imbalance": (float(values.max()) / mean - 1.0) * 100.0,
                "excess": float(values.max()) - mean,
            }
        )
    if not regions:
        return None
    # Largest excess first: the time balancing a region could save.
    regions.sort(key=lambda entry: -entry["excess"])
    return {
        "ranks": selected,
        "regions": regions,
        "own": own,
        "totals": np.sum([entry["values"] for entry in regions], axis=0),
        "slowest_counts": Counter(
            entry["slowest"] for entry in regions if entry["excess"] > 0
        ),
    }


def _comparison_entries(per_run_rows) -> list[dict]:
    """Align every run's region rows by call path, in call-tree order.

    The baseline's tree sets the order. A path only a later run has is placed
    after the last row below its parent, so the merged rows still read as one
    tree; rows without a call path match by name.
    """
    entries: list[dict] = []
    index: dict[str, int] = {}

    def key_of(row) -> str:
        return (
            f"path:{row['call_path']}" if "call_path" in row else f"name:{row['name']}"
        )

    def insert_at(row) -> int:
        if "call_path" not in row or " > " not in row["call_path"]:
            return len(entries)
        parent = "path:" + row["call_path"].rsplit(" > ", 1)[0]
        if parent not in index:
            return len(entries)
        position = index[parent] + 1
        prefix = parent + " > "
        while position < len(entries) and entries[position]["key"].startswith(prefix):
            position += 1
        return position

    for run_index, rows in enumerate(per_run_rows):
        for row in rows:
            key = key_of(row)
            if key not in index:
                position = insert_at(row)
                entries.insert(
                    position,
                    {
                        "key": key,
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
                        "values": [None] * len(per_run_rows),
                    },
                )
                index = {entry["key"]: i for i, entry in enumerate(entries)}
            entries[index[key]]["values"][run_index] = row
    return entries


def _metric_value(row, metric):
    if row is None or row["total"] is None:
        return None
    if metric == "own":
        return row["total"] if row["exclusive"] is None else row["exclusive"]
    if metric == "calls":
        return row["calls"]
    return row[metric]


def _metric_text(value, metric) -> str:
    if value is None:
        return "-"
    return f"{value:,}" if metric == "calls" else _duration_text(value)


def _per_rank_row(row) -> dict:
    """A row's times per rank: its totals over the ranks that ran it.

    Summed over ranks, a region's time is what it costs; compared across
    runs of different sizes, that grows with the rank count even as the run
    gets faster. Per rank, it follows what a reader waits for.
    """
    count = row.get("num_ranks") or 1
    return {
        **row,
        "total": None if row["total"] is None else row["total"] / count,
        "exclusive": None if row["exclusive"] is None else row["exclusive"] / count,
    }


def _change_items(entries, run_index, direction) -> list[dict]:
    """The largest own-time changes of one run against the baseline.

    Only changes that matter on both scales: at least 2% of the region's own
    baseline time, so timer jitter on a large region does not count, and at
    least 0.5% of the baseline's time in regions, so a tiny region doubling
    does not headline the comparison.
    """
    owns = [
        (
            _metric_value(entry["values"][0], "own") or 0.0,
            _metric_value(entry["values"][run_index], "own") or 0.0,
        )
        for entry in entries
    ]
    floor = _CHANGE_FLOOR_PCT / 100.0 * sum(base for base, _ in owns)
    changes = []
    for entry, (base, value) in zip(entries, owns):
        delta = value - base
        if (
            delta * direction > 0
            and abs(delta) >= floor
            and (not base or 100.0 * abs(delta) / base >= _SAME_PCT)
        ):
            changes.append({**entry, "base": base, "value": value, "delta": delta})
    changes.sort(key=lambda change: -abs(change["delta"]))
    return changes[:_CHANGE_COUNT]
