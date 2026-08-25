"""Interactive Textual browser for scope-profiler HDF5 files."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import linecache
import re
import socket
import subprocess
import sys
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import h5py
import numpy as np

from scope_profiler.h5reader import read_h5
from scope_profiler.inspection import _metadata_sections, _time_span
from scope_profiler.plotting_scripts import (
    available_likwid_metrics,
    plot_duration_histogram,
    plot_duration_timeseries,
    plot_durations,
    plot_flame,
    plot_gantt,
    plot_imbalance,
    plot_likwid,
)
from scope_profiler.post_processing import parse_ranks
from scope_profiler.prof_export import export_prof
from scope_profiler.summary import region_row, region_rows

_PLOT_CATALOG = {
    "gantt": "Per-rank timeline of recorded calls",
    "durations": "Duration statistics by region",
    "flame": "Reconstructed nested call-stack flame graph",
    "timeseries": "Duration of each call over time",
    "histogram": "Call-duration distribution by region",
    "imbalance": "Per-rank duration comparison",
}
_PLOTEXT_TUI_PLOTS = frozenset({"durations"})


@dataclass
class BrowserNode:
    """One selectable item in the TUI navigation tree."""

    label: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    children: list["BrowserNode"] = field(default_factory=list)


@dataclass
class BrowserModel:
    """Loaded data and navigation tree for one HDF5 profile."""

    file_path: Path
    results: Any
    root: BrowserNode


def _format_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _format_scalar(value.item())
        return ", ".join(_format_scalar(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_scalar(item) for item in value)
    return str(value)


def _dataset_preview(dataset) -> str:
    if dataset.shape == ():
        return _format_scalar(dataset[()])
    if dataset.size == 0:
        return "empty"
    slices = tuple(slice(0, min(size, 8)) for size in dataset.shape)
    preview = np.asarray(dataset[slices])
    suffix = "" if dataset.size <= preview.size else " ..."
    return np.array2string(preview, threshold=16, edgeitems=4) + suffix


def _attrs_payload(obj) -> dict[str, str]:
    return {key: _format_scalar(value) for key, value in obj.attrs.items()}


def _build_raw_h5_node(name: str, obj) -> BrowserNode:
    if isinstance(obj, h5py.Dataset):
        return BrowserNode(
            label=name,
            kind="h5_dataset",
            payload={
                "name": name,
                "path": obj.name,
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "size": int(obj.size),
                "chunks": obj.chunks,
                "compression": obj.compression,
                "attrs": _attrs_payload(obj),
                "preview": _dataset_preview(obj),
            },
        )

    children = [_build_raw_h5_node(key, obj[key]) for key in sorted(obj.keys())]
    return BrowserNode(
        label=name,
        kind="h5_group",
        payload={"name": name, "path": obj.name, "attrs": _attrs_payload(obj)},
        children=children,
    )


def _line_profile_total_seconds(record: dict) -> float:
    return float(np.sum(record["times"])) * float(record["unit"])


def _line_profile_label(record: dict) -> str:
    return str(record["function"])


def _line_profile_by_region(results) -> dict:
    by_region = defaultdict(lambda: defaultdict(list))
    for rank, records in sorted(results.line_profile.items()):
        for record in records:
            by_region[record["region"]][rank].append(record)
    return by_region


def _build_region_line_profile_node(region_name: str, by_rank: dict) -> BrowserNode:
    rank_children = []
    all_records = []
    for rank, records in sorted(by_rank.items()):
        all_records.extend({**record, "rank": rank} for record in records)
        if len(records) == 1:
            rank_children.append(
                BrowserNode(
                    f"Rank {rank}",
                    "line_profile_record",
                    {"rank": rank, "record": records[0]},
                )
            )
            continue
        rank_children.append(
            BrowserNode(
                f"Rank {rank} ({len(records)} record(s))",
                "line_profile_rank",
                {"rank": rank, "records": records},
                [
                    BrowserNode(
                        _line_profile_label(record),
                        "line_profile_record",
                        {"rank": rank, "record": record},
                    )
                    for record in records
                ],
            )
        )

    return BrowserNode(
        "Line Profile",
        "line_profile_region",
        {"region": region_name, "records": all_records},
        rank_children,
    )


def build_browser_model(file_path: str | Path) -> BrowserModel:
    """Load one HDF5 profile and build the selectable TUI navigation tree."""
    results = read_h5(file_path)
    path = Path(results.file_path)

    metadata_children = []
    sections, modules = _metadata_sections(results.metadata)
    for title, entries in sections:
        if entries:
            metadata_children.append(
                BrowserNode(title, "metadata_section", {"entries": entries})
            )
    if modules is not None:
        metadata_children.append(
            BrowserNode("Modules", "modules", {"modules": list(modules)})
        )

    region_children = []
    sorted_rows = {row["name"]: row for row in region_rows(results, sort="total")}
    line_profile_by_region = _line_profile_by_region(results)
    for region in results.get_regions():
        calls_children = [
            BrowserNode(
                f"Rank {rank}",
                "rank_calls",
                {"region": region, "rank": rank},
            )
            for rank in region.ranks
        ]
        extra_children = [
            BrowserNode("Summary", "region_summary", {"region": region}),
            BrowserNode("Calls", "region_calls", {"region": region}, calls_children),
        ]
        if region.has_source:
            extra_children.append(BrowserNode("Source", "source", {"region": region}))
        line_profile = line_profile_by_region.pop(region.name, None)
        if line_profile:
            extra_children.append(
                _build_region_line_profile_node(region.name, line_profile)
            )
        likwid_children = []
        for rank, by_region in sorted(results.get_likwid_regions().items()):
            counters = by_region.get(region.name)
            if counters is not None:
                likwid_children.append(
                    BrowserNode(
                        f"LIKWID rank {rank}",
                        "likwid_rank",
                        {"rank": rank, "region_name": region.name, "likwid": counters},
                    )
                )
        extra_children.extend(likwid_children)
        region_children.append(
            BrowserNode(
                region.name,
                "region",
                {"region": region, "row": sorted_rows.get(region.name)},
                extra_children,
            )
        )
    for region_name, by_rank in sorted(line_profile_by_region.items()):
        line_profile_node = _build_region_line_profile_node(str(region_name), by_rank)
        region_children.append(
            BrowserNode(
                str(region_name),
                "line_profile_region",
                line_profile_node.payload,
                line_profile_node.children,
            )
        )

    with h5py.File(path, "r") as h5file:
        raw_node = _build_raw_h5_node("/", h5file)

    plot_children = [
        BrowserNode(
            name.title(),
            "plot",
            {"results": results, "plot_name": name, "description": description},
        )
        for name, description in _PLOT_CATALOG.items()
    ]
    for metric in available_likwid_metrics(results):
        plot_children.append(
            BrowserNode(
                f"LIKWID: {metric}",
                "plot_likwid",
                {
                    "results": results,
                    "plot_name": "likwid",
                    "metric": metric,
                    "description": f"LIKWID metric/event {metric}",
                },
            )
        )

    root = BrowserNode(
        path.name,
        "root",
        {"file_path": path},
        [
            BrowserNode("Overview", "overview", {"results": results}),
            BrowserNode("Plots", "plots", {"results": results}, plot_children),
            BrowserNode(
                "Metadata",
                "metadata",
                {"metadata": results.metadata},
                metadata_children,
            ),
            BrowserNode("Regions", "regions", {"results": results}, region_children),
            BrowserNode("Raw HDF5", "h5_group", raw_node.payload, raw_node.children),
        ],
    )
    return BrowserModel(file_path=path, results=results, root=root)


def _duration(value) -> str:
    return "-" if value is None else f"{value:.6g} s"


def _line_table(
    headers, rows, *, compact: bool = False, maxcolwidths: tuple[int, ...] | None = None
) -> str:
    rows = list(rows)
    from tabulate import tabulate

    return tabulate(
        rows,
        headers=headers,
        tablefmt="plain" if compact else "rounded_outline",
        disable_numparse=True,
        maxcolwidths=maxcolwidths,
    )


def node_detail_text(node: BrowserNode) -> str:
    """Return a plain-text detail view for a selected navigation node."""
    kind = node.kind
    payload = node.payload

    if kind == "overview":
        results = payload["results"]
        path = Path(results.file_path)
        size_mb = path.stat().st_size / 1024**2
        span = _time_span(results)
        rows = region_rows(results, sort="total")
        total_region_time = sum(
            row["total"] for row in rows if row["total"] is not None
        )
        total_calls = sum(row["calls"] for row in rows)
        max_imbalance = max(
            (row["imbalance"] for row in rows if row["imbalance"] is not None),
            default=None,
        )
        top_rows = []
        for row in rows[:5]:
            duration = row["total"] or 0.0
            bar_width = (
                round(24 * duration / total_region_time)
                if total_region_time
                else 0
            )
            top_rows.append(
                (
                    row["name"],
                    _duration(row["total"]),
                    f"{duration / total_region_time:.1%}"
                    if total_region_time
                    else "-",
                    "█" * bar_width,
                )
            )
        metrics = _line_table(
            ("Metric", "Value"),
            (
                ("File", path),
                ("Label", results.label or "-"),
                ("Ranks", results.num_ranks),
                ("Regions", len(results.get_regions())),
                ("Calls", total_calls),
                ("Size", f"{size_mb:.2f} MiB"),
                *(
                    [("Profiled wall clock", f"{span:.6g} s")]
                    if span is not None
                    else []
                ),
                *(
                    [("Setup to finalize", f"{results.total_time:.6g} s")]
                    if results.total_time is not None
                    else []
                ),
                *(
                    [("Max rank imbalance", f"{max_imbalance:.6g}%")]
                    if max_imbalance is not None
                    else []
                ),
            ),
        )
        if not top_rows:
            return metrics + "\n\nNo regions recorded."
        return (
            metrics
            + "\n\nTop regions by total time\n\n"
            + _line_table(("Region", "Total", "Share", ""), top_rows, compact=True)
            + "\n\nSelect Regions for the complete breakdown."
        )

    if kind == "plots":
        lines = [
            "Plots",
            "",
            "Select a plot below, then press:",
            "  g  Show it in Matplotlib",
            "  t  Show simple plots with Plotext",
            "  s  Save it as a PNG",
            "Edit the region filters to change the selected regions.",
            "",
            "Available plots:",
        ]
        lines.extend(
            f"  {name:<12} {description}" for name, description in _PLOT_CATALOG.items()
        )
        if any(child.kind == "plot_likwid" for child in node.children):
            lines.append("  LIKWID       One chart per selected hardware metric")
        return "\n".join(lines)

    if kind in {"plot", "plot_likwid"}:
        lines = [
            f"Plot: {payload.get('plot_name', 'likwid')}",
            payload["description"],
            "",
            "Press g for Matplotlib, t for Plotext (simple plots), p for Plotly in a browser, or s to save PNG.",
        ]
        if payload.get("plot_name") == "flame":
            lines.append("Press v to open the reconstructed profile in Snakeviz.")
        if payload.get("metric"):
            lines.append(f"Metric: {payload['metric']}")
        return "\n".join(lines)

    if kind == "metadata":
        metadata = payload["metadata"]
        if not metadata:
            return "No metadata recorded."
        sections, modules = _metadata_sections(metadata)
        lines = [
            "Metadata",
            "",
            f"{len(metadata)} metadata entries recorded.",
            "Select a section below to inspect its values:",
            "",
        ]
        lines.extend(f"• {title} ({len(entries)} entries)" for title, entries in sections)
        if modules is not None:
            lines.append(f"• Modules ({len(modules)} entries)")
        return "\n".join(lines)

    if kind == "metadata_section":
        return _line_table(
            ("Key", "Value"),
            payload["entries"],
            maxcolwidths=(28, 72),
        )

    if kind == "modules":
        modules = payload["modules"]
        return "\n".join(modules) if modules else "No modules recorded."

    if kind == "regions":
        rows = region_rows(payload["results"], sort="total")
        if not rows:
            return "No regions recorded."
        return _line_table(
            ("Region", "Ranks", "Calls", "Total", "Avg", "Imbalance"),
            (
                (
                    row["name"],
                    row["num_ranks"],
                    row["calls"],
                    _duration(row["total"]),
                    _duration(row["avg"]),
                    "-" if row["imbalance"] is None else f"{row['imbalance']:.6g}%",
                )
                for row in rows
            ),
        )

    if kind == "line_profile":
        line_profile = payload["line_profile"]
        if not line_profile:
            return "No line-profiler records stored in this file."
        rows = []
        for rank, records in sorted(line_profile.items()):
            total = sum(_line_profile_total_seconds(record) for record in records)
            rows.append((rank, len(records), f"{total:.6g} s"))
        return _line_table(("Rank", "Records", "Total"), rows, compact=True)

    if kind == "line_profile_rank":
        records = payload["records"]
        if not records:
            return "No line-profiler records stored for this rank."
        by_region = defaultdict(list)
        for record in records:
            by_region[record["region"]].append(record)
        rows = []
        for region, region_records in sorted(by_region.items()):
            total = sum(
                _line_profile_total_seconds(record) for record in region_records
            )
            rows.append((region, len(region_records), f"{total:.6g} s"))
        return f"Line profile rank {payload['rank']}\n\n" + _line_table(
            ("Region", "Functions", "Total"), rows, compact=True
        )

    if kind == "line_profile_region":
        records = payload["records"]
        rows = [
            (
                record.get("rank", payload.get("rank", "-")),
                record["function"],
                f"{record['filename']}:{record['first_lineno']}",
                len(record["line_numbers"]),
                f"{_line_profile_total_seconds(record):.6g} s",
            )
            for record in records
        ]
        title = f"Line profile | {payload['region']}"
        if "rank" in payload:
            title = f"Line profile rank {payload['rank']} | {payload['region']}"
        return (
            title
            + "\n\n"
            + _line_table(
                ("Rank", "Function", "Location", "Lines", "Total"),
                rows,
                compact=True,
            )
        )

    if kind == "line_profile_record":
        record = payload["record"]
        unit = float(record["unit"])
        total_raw = float(np.sum(record["times"]))
        rows = []
        for line, hits, elapsed in zip(
            record["line_numbers"], record["hits"], record["times"]
        ):
            seconds = float(elapsed) * unit
            hits = int(hits)
            per_hit = seconds / hits if hits else 0.0
            percent = float(elapsed) / total_raw * 100 if total_raw else 0.0
            source = linecache.getline(record["filename"], int(line)).strip()
            rows.append(
                (
                    int(line),
                    hits,
                    f"{seconds:.6g}",
                    f"{per_hit:.6g}",
                    f"{percent:.2f}",
                    source,
                )
            )
        header = (
            f"Rank {payload['rank']} | {record['region']} | {record['function']}\n"
            f"{record['filename']}:{record['first_lineno']}\n"
            f"Total: {_line_profile_total_seconds(record):.6g} s"
        )
        if not rows:
            return header + "\n\nNo per-line timings recorded for this function."
        return (
            header
            + "\n\n"
            + _line_table(
                ("Line", "Hits", "Time [s]", "Per hit [s]", "%", "Source"),
                rows,
                compact=True,
            )
        )

    if kind in {"region", "region_summary"}:
        region = payload["region"]
        row = payload.get("row") or region_row(region)
        summary_rows = [
            ("Region", region.name),
            ("Ranks", row["num_ranks"]),
            ("Calls", row["calls"]),
            ("Total", _duration(row["total"])),
            ("Average", _duration(row["avg"])),
            ("Min / max", f"{_duration(row['min'])} / {_duration(row['max'])}"),
            (
                "P50 / P95 / P99",
                f"{_duration(row['p50'])} / {_duration(row['p95'])} / "
                f"{_duration(row['p99'])}",
            ),
            (
                "Rank imbalance",
                "-" if row["imbalance"] is None else f"{row['imbalance']:.6g}%",
            ),
        ]
        if region.tags:
            summary_rows.append(("Tags", ", ".join(region.tags)))
        if region.has_source:
            summary_rows.append(
                ("Source", f"{region.source_file}:{region.source_lineno}")
            )
        return _line_table(("Metric", "Value"), summary_rows)

    if kind == "rank_region":
        region = payload["region"]
        rank = payload["rank"]
        data = region[rank]
        rows = [
            ("Calls", data.num_calls),
            ("Total", _duration(data.total_duration)),
            ("Exclusive", _duration(data.exclusive_duration)),
            ("Average", _duration(data.average_duration)),
            ("Min", _duration(data.min_duration)),
            ("Max", _duration(data.max_duration)),
            ("First", _duration(data.first_duration)),
            ("Last", _duration(data.last_duration)),
            ("Std", _duration(data.std_duration)),
        ]
        return f"{region.name} on rank {rank}\n\n" + _line_table(
            ("Metric", "Value"), rows
        )

    if kind == "region_calls":
        region = payload["region"]
        rows = []
        for rank in region.ranks:
            events = region.events(ranks=rank, origin=0.0)
            rows.append((rank, len(events)))
        return f"Calls | {region.name}\n\n" + _line_table(("Rank", "Calls"), rows)

    if kind == "rank_calls":
        region = payload["region"]
        rank = payload["rank"]
        events = region.events(ranks=rank, origin=0.0)
        if not events:
            return "No calls recorded."
        return _line_table(
            ("#", "Start", "End", "Duration"),
            (
                (
                    event["call_index"],
                    f"{event['start']:.9g}",
                    f"{event['end']:.9g}",
                    f"{event['duration']:.9g}",
                )
                for event in events[:200]
            ),
        )

    if kind == "source":
        region = payload["region"]
        if not region.has_source:
            return "Source not captured."
        return f"{region.source_file}:{region.source_lineno}\n\n{region.source_text}"

    if kind == "likwid_rank":
        counters = payload["likwid"]
        lines = [
            f"Region: {counters.tag}",
            f"Rank: {payload['rank']}",
            f"Group: {counters.group_name or counters.group_id}",
            f"Source: {counters.source}",
            f"CPUs: {', '.join(map(str, counters.cpus)) or '-'}",
            "",
        ]
        if counters.event_labels:
            lines.append(
                _line_table(
                    ("Event", *[f"CPU {cpu}" for cpu in counters.cpus]),
                    zip(counters.event_labels, *counters.events.tolist()),
                )
            )
        if counters.metric_names:
            lines.append("")
            lines.append(
                _line_table(
                    ("Metric", *[f"CPU {cpu}" for cpu in counters.cpus]),
                    zip(counters.metric_names, *counters.metrics.tolist()),
                )
            )
        return "\n".join(lines)

    if kind == "h5_group":
        attrs = payload.get("attrs", {})
        lines = [f"HDF5 group: {payload['path']}"]
        lines.append(f"Children: {len(node.children)}")
        if attrs:
            lines.append("\nAttributes")
            lines.append(
                _line_table(
                    ("Key", "Value"),
                    sorted(attrs.items()),
                    maxcolwidths=(28, 72),
                )
            )
        return "\n".join(lines)

    if kind == "h5_dataset":
        attrs = payload.get("attrs", {})
        lines = [
            f"HDF5 dataset: {payload['path']}",
            f"Shape: {payload['shape']}",
            f"Dtype: {payload['dtype']}",
            f"Size: {payload['size']}",
            f"Chunks: {payload['chunks'] or '-'}",
            f"Compression: {payload['compression'] or '-'}",
            "",
            "Preview",
            payload["preview"],
        ]
        if attrs:
            lines.append("\nAttributes")
            lines.append(
                _line_table(
                    ("Key", "Value"),
                    sorted(attrs.items()),
                    maxcolwidths=(28, 72),
                )
            )
        return "\n".join(lines)

    return node.label


def render_plot(
    node: BrowserNode,
    *,
    filepath: str | Path | None = None,
    show: bool = False,
    settings: dict[str, Any] | None = None,
) -> str | None:
    """Render one TUI plot node through the existing plotting functions."""
    if node.kind not in {"plot", "plot_likwid"}:
        raise ValueError("Select an individual plot before rendering.")

    results = node.payload["results"]
    name = node.payload["plot_name"]
    settings = settings or {}

    def patterns(key: str) -> list[str] | None:
        value = settings.get(key, "")
        values = [item.strip() for item in str(value).split(",") if item.strip()]
        return values or None

    ranks = None
    if settings.get("ranks", "").strip():
        ranks = []
        for spec in str(settings["ranks"]).split(","):
            ranks.extend(parse_ranks(spec.strip()))
        ranks = sorted(set(ranks))

    cmap = settings.get("cmap", "tab20") or "tab20"
    common = {
        "filepath": str(filepath) if filepath else None,
        "show": show,
        "verbose": False,
        "backend": settings.get("backend", "matplotlib") or "matplotlib",
        "include": patterns("include"),
        "exclude": patterns("exclude"),
        "ranks": ranks,
        "cmap": cmap,
    }
    functions = {
        "gantt": plot_gantt,
        "flame": plot_flame,
        "durations": plot_durations,
        "timeseries": plot_duration_timeseries,
        "histogram": plot_duration_histogram,
        "imbalance": plot_imbalance,
    }
    if name == "likwid":
        plot_likwid(
            results,
            metric=node.payload["metric"],
            log_scale=bool(settings.get("log_scale", False)),
            **common,
        )
    elif name == "durations":
        metric = str(settings.get("metric", "total")).strip() or "total"
        top_n = settings.get("top_n", "").strip()
        plot_durations(
            results,
            metric=metric,
            sort_by=settings.get("sort_by") or None,
            top_n=int(top_n) if top_n else None,
            log_scale=bool(settings.get("log_scale", False)),
            **common,
        )
    elif name == "histogram":
        bins = int(settings.get("bins", 30) or 30)
        plot_duration_histogram(
            results,
            bins=bins,
            log_scale=bool(settings.get("log_scale", False)),
            **common,
        )
    elif name == "imbalance":
        plot_imbalance(
            results,
            metric=settings.get("imbalance_metric", "total") or "total",
            log_scale=bool(settings.get("log_scale", False)),
            **common,
        )
    elif name == "timeseries":
        plot_duration_timeseries(
            results, log_scale=bool(settings.get("log_scale", False)), **common
        )
    else:
        functions[name](results, **common)
    return str(filepath) if filepath else None


def render_plotext_text(
    node: BrowserNode,
    *,
    settings: dict[str, Any] | None = None,
    width: int = 100,
    height: int = 35,
) -> str:
    """Render a simple Plotext chart as text for display inside Textual."""
    if node.kind not in {"plot", "plot_likwid"}:
        raise ValueError("Select an individual plot before rendering.")
    if node.payload.get("plot_name") not in _PLOTEXT_TUI_PLOTS:
        raise ValueError("Plotext is available in the TUI for simple plots only.")

    # maxplotlib creates a fresh Plotext figure for every render. Plotext's
    # default terminal size is 140 columns, which is wider than a typical
    # Textual detail pane, so constrain the defaults while that figure is built.
    try:
        import plotext._figure as plotext_figure
    except ImportError:
        plotext_figure = None
        default_figure_class = None
        canvas_class = None
        legend_method = None
    else:
        from maxplotlib import Canvas

        canvas_class = Canvas
        legend_method = canvas_class.set_legend
        figure_globals = plotext_figure._figure_class.__init__.__globals__
        default_figure_class = figure_globals["default_figure_class"]
        utility = figure_globals["ut"]
        terminal_size = utility.terminal_size

        def constrained_defaults():
            defaults = default_figure_class()
            defaults.width_term = max(40, int(width))
            defaults.height_term = max(12, int(height))
            defaults.size_term = [defaults.width_term, defaults.height_term]
            return defaults

        figure_globals["default_figure_class"] = constrained_defaults
        utility.terminal_size = lambda: [max(40, int(width)), max(12, int(height))]
        # Region legends are useful in image output, but their vertical list
        # can consume nearly the whole Plotext canvas in a narrow TUI pane.
        canvas_class.set_legend = lambda self, *args, **kwargs: None
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            try:
                render_plot(
                    node,
                    show=True,
                    settings={**(settings or {}), "backend": "plotext"},
                )
            except re.error as exc:
                raise ValueError(f"Invalid region filter: {exc}") from exc
    finally:
        if plotext_figure is not None:
            figure_globals["default_figure_class"] = default_figure_class
            utility.terminal_size = terminal_size
            canvas_class.set_legend = legend_method
    return output.getvalue().rstrip()


def _matplotlib_child_script() -> str:
    """Return the isolated runner used for genuine Matplotlib figures."""
    return (
        "import sys\n"
        "import json\n"
        "from scope_profiler.tui import build_browser_model, render_plot\n"
        "file_path, plot_name, metric, settings_json = sys.argv[1:]\n"
        "model = build_browser_model(file_path)\n"
        "nodes = list(model.root.children)\n"
        "while nodes:\n"
        "    node = nodes.pop(0)\n"
        "    if node.kind in {'plot', 'plot_likwid'} and node.payload.get('plot_name') == plot_name and node.payload.get('metric') == (metric or None):\n"
        "        render_plot(node, show=True, settings=json.loads(settings_json))\n"
        "        break\n"
        "    nodes.extend(node.children)\n"
    )

def _build_textual_app_class():
    try:
        from rich.text import Text
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.suggester import Suggester
        from textual.widgets import (
            Button,
            Checkbox,
            Footer,
            Header,
            Input,
            Select,
            Static,
            Tree,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The interactive TUI requires Textual. Install it with "
            "`pip install scope-profiler[tui]` or `pip install textual`."
        ) from exc

    class H5BrowserApp(App):
        """Textual application for browsing a profile file."""

        class Detail(Static):
            """Scrollable detail view that can receive keyboard focus."""

            can_focus = True

        class RegionSuggester(Suggester):
            """Complete the current comma-separated region filter token."""

            def __init__(self, region_names: list[str]):
                super().__init__(case_sensitive=False)
                self.region_names = region_names

            async def get_suggestion(self, value: str) -> str | None:
                prefix, _, token = value.rpartition(",")
                token = token.strip().casefold()
                for name in self.region_names:
                    if name.casefold().startswith(token):
                        separator = f"{prefix}, " if prefix else ""
                        return separator + name
                return None

        CSS = """
        Screen {
            layout: vertical;
        }

        Header {
            height: 1;
        }

        Footer {
            height: 2;
        }

        #body {
            height: 1fr;
        }

        #nav {
            width: 28%;
            min-width: 24;
            border: solid $accent;
            padding: 0 1;
            overflow: hidden;
        }

        #detail {
            height: 1fr;
            min-height: 20;
            border: solid $accent;
            padding: 1 2;
            overflow-x: auto;
            overflow-y: auto;
            text-wrap: nowrap;
        }

        #right {
            width: 1fr;
        }

        #settings {
            height: auto;
            max-height: 35%;
            border: solid $accent;
            padding: 0 1;
            display: none;
            overflow-y: auto;
        }

        #selected-regions {
            height: 5;
            border: solid $secondary;
            padding: 0 1;
            overflow-y: auto;
        }

        #settings-title {
            height: 1;
        }

        .setting {
            width: 1fr;
        }
        """
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("escape", "focus_navigation", "Focus navigation"),
            ("g", "show_matplotlib", "Show Matplotlib"),
            ("p", "show_plotly", "Open Plotly"),
            ("t", "show_plotext", "Show Plotext"),
            ("s", "save_plot", "Save plot"),
            ("v", "show_snakeviz", "Open in Snakeviz"),
        ]

        def __init__(self, model: BrowserModel, output_dir: str | Path | None = None):
            super().__init__()
            self.model = model
            self.output_dir = Path(output_dir) if output_dir else None
            self.selected_browser_node: BrowserNode | None = None
            self.plot_settings = {
                "include": "",
                "exclude": "",
                "ranks": "",
                "cmap": "tab20",
                "log_scale": False,
                "metric": "total",
                "sort_by": "",
                "top_n": "",
                "bins": "30",
                "imbalance_metric": "total",
            }
            self._plotext_refresh_timer = None

        def _detail(self, node: BrowserNode):
            return Text(
                node_detail_text(node),
                no_wrap=node.kind not in {"metadata", "metadata_section", "modules"},
            )

        def compose(self) -> ComposeResult:
            region_names = [region.name for region in self.model.results.get_regions()]
            region_suggester = self.RegionSuggester(region_names)
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                yield Tree(self.model.root.label, id="nav")
                with Vertical(id="right"):
                    yield self.Detail(
                        self._detail(self.model.root.children[0]), id="detail"
                    )
                    with Vertical(id="settings"):
                        yield Static("Plot settings", id="settings-title")
                        yield Input(
                            placeholder="Include regions (comma-separated)",
                            id="include",
                            classes="setting",
                            suggester=region_suggester,
                        )
                        yield Input(
                            placeholder="Exclude regions (comma-separated)",
                            id="exclude",
                            classes="setting",
                            suggester=region_suggester,
                        )
                        yield Static("Selected regions", classes="setting")
                        yield Static(
                            "All regions", id="selected-regions", classes="setting"
                        )
                        yield Input(
                            placeholder="Ranks (e.g. 0,2-4)",
                            id="ranks",
                            classes="setting",
                        )
                        yield Input(
                            value="tab20",
                            placeholder="Colormap",
                            id="cmap",
                            classes="setting",
                        )
                        yield Select(
                            [(label.title(), label) for label in ("total", "avg", "min", "max")],
                            prompt="Duration metric",
                            value="total",
                            id="metric",
                            classes="setting",
                        )
                        yield Select(
                            [("Default", ""), *[(label.title(), label) for label in ("name", "avg", "min", "max", "total")]],
                            prompt="Sort by",
                            value="",
                            id="sort_by",
                            classes="setting",
                        )
                        yield Input(
                            placeholder="Top N regions", id="top_n", classes="setting"
                        )
                        yield Input(
                            value="30",
                            placeholder="Histogram bins",
                            id="bins",
                            classes="setting",
                        )
                        yield Select(
                            [(label.title(), label) for label in ("total", "avg", "min", "max")],
                            prompt="Imbalance metric",
                            value="total",
                            id="imbalance_metric",
                            classes="setting",
                        )
                        yield Checkbox("Log scale", id="log_scale")
                        yield Button(
                            "Apply settings", id="apply-settings", variant="primary"
                        )
            yield Footer()

        def on_mount(self) -> None:
            tree = self.query_one("#nav", Tree)
            tree.root.data = self.model.root
            self._populate_tree(tree.root, self.model.root.children)
            tree.root.expand()
            if tree.root.children:
                first = tree.root.children[0]
                if hasattr(tree, "select_node"):
                    tree.select_node(first)
                else:
                    tree.move_cursor(first)
                self.selected_browser_node = first.data
                self.query_one("#detail", Static).update(self._detail(first.data))
                self._update_selected_regions()

        def _populate_tree(self, tree_node, browser_nodes) -> None:
            for browser_node in browser_nodes:
                child = tree_node.add(browser_node.label, data=browser_node)
                if browser_node.children:
                    self._populate_tree(child, browser_node.children)

        def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
            node = event.node.data
            if isinstance(node, BrowserNode):
                self.selected_browser_node = node
                self._update_plot_settings_visibility(node)
                if node.kind in {"plot", "plot_likwid"} and node.payload.get(
                    "plot_name"
                ) in _PLOTEXT_TUI_PLOTS:
                    # Let Textual finish laying out the detail/settings panes
                    # before measuring them for the Plotext canvas.
                    self.call_after_refresh(self._show_plotext_in_detail, node)
                else:
                    self.query_one("#detail", Static).update(self._detail(node))

        def _update_plot_settings_visibility(self, node: BrowserNode) -> None:
            panel = self.query_one("#settings", Vertical)
            panel.styles.display = (
                "block" if node.kind in {"plot", "plot_likwid"} else "none"
            )

        def _read_plot_settings(self) -> None:
            for key in (
                "include",
                "exclude",
                "ranks",
                "cmap",
                "top_n",
                "bins",
            ):
                self.plot_settings[key] = self.query_one(f"#{key}", Input).value
            for key in ("metric", "sort_by", "imbalance_metric"):
                value = self.query_one(f"#{key}", Select).value
                self.plot_settings[key] = "" if value is Select.NULL else str(value)
            self.plot_settings["log_scale"] = self.query_one(
                "#log_scale", Checkbox
            ).value

        def _update_selected_regions(self) -> None:
            include = self.query_one("#include", Input).value
            exclude = self.query_one("#exclude", Input).value
            include_patterns = [
                item.strip() for item in include.split(",") if item.strip()
            ]
            exclude_patterns = [
                item.strip() for item in exclude.split(",") if item.strip()
            ]
            regions = [region.name for region in self.model.results.get_regions()]
            try:
                if include_patterns:
                    regions = [
                        name
                        for name in regions
                        if any(re.search(pattern, name) for pattern in include_patterns)
                    ]
                if exclude_patterns:
                    regions = [
                        name
                        for name in regions
                        if not any(
                            re.search(pattern, name) for pattern in exclude_patterns
                        )
                    ]
                text = ", ".join(regions) if regions else "No matching regions"
            except re.error as exc:
                text = f"Invalid filter: {exc}"
            self.query_one("#selected-regions", Static).update(text)

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id in {"include", "exclude"}:
                self._update_selected_regions()
            if event.input.id in {
                "include",
                "exclude",
                "ranks",
                "cmap",
                "metric",
                "sort_by",
                "top_n",
                "bins",
                "imbalance_metric",
            }:
                self._schedule_plotext_refresh()

        def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
            if event.checkbox.id == "log_scale":
                self._schedule_plotext_refresh()

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id in {"metric", "sort_by", "imbalance_metric"}:
                self._schedule_plotext_refresh()

        def _schedule_plotext_refresh(self) -> None:
            node = self.selected_browser_node
            if node is not None and node.payload.get("plot_name") in _PLOTEXT_TUI_PLOTS:
                if self._plotext_refresh_timer is not None:
                    self._plotext_refresh_timer.stop()
                self._plotext_refresh_timer = self.set_timer(
                    0.2, self._show_plotext_in_detail, node
                )

        def action_focus_navigation(self) -> None:
            """Return focus to the tree so its global shortcuts are active."""
            self.query_one("#nav", Tree).focus()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "apply-settings":
                self._read_plot_settings()
                self.notify("Plot settings applied.", timeout=3)

        def _run_selected_plot(self, show: bool, backend: str = "matplotlib") -> None:
            self._read_plot_settings()
            settings = {**self.plot_settings, "backend": backend}
            node = self.selected_browser_node
            if node is None or node.kind not in {"plot", "plot_likwid"}:
                self.notify("Select an individual plot first.", severity="warning")
                return
            if (
                backend == "plotext"
                and node.payload.get("plot_name") not in _PLOTEXT_TUI_PLOTS
            ):
                self.notify(
                    "Plotext is available in the TUI for simple plots only.",
                    severity="warning",
                    timeout=5,
                )
                return

            filepath = None
            if not show:
                directory = self.output_dir or (
                    self.model.file_path.parent / f"{self.model.file_path.stem}_plots"
                )
                directory.mkdir(parents=True, exist_ok=True)
                filename = node.payload["plot_name"]
                if node.payload.get("metric"):
                    filename += "_" + "_".join(
                        part
                        for part in node.payload["metric"].split()
                        if part.isalnum()
                    )
                suffix = ".txt" if backend == "plotext" else ".png"
                filepath = directory / f"{filename}{suffix}"
            try:
                if show:
                    saved = self._show_plot_in_child_process(node, settings)
                else:
                    saved = render_plot(
                        node, filepath=filepath, show=False, settings=settings
                    )
            except (ImportError, RuntimeError, ValueError) as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
            if saved:
                self.notify(f"Saved {saved}", timeout=8)
            else:
                self.notify("Plot opened in Matplotlib.", timeout=5)

        def _show_plotext_in_detail(self, node: BrowserNode | None = None) -> None:
            """Render Plotext into the selected detail pane, not the terminal."""
            node = node or self.selected_browser_node
            if node is None:
                return
            self._read_plot_settings()
            detail = self.query_one("#detail", Static)
            width = max(40, detail.content_size.width - 4)
            # Leave room for the pane border, padding, and Textual's line
            # accounting. Plotext uses nearly all of the requested height.
            height = max(8, detail.content_size.height - 5)
            try:
                plot_text = render_plotext_text(
                    node,
                    settings=self.plot_settings,
                    width=width,
                    height=height,
                )
            except (ImportError, RuntimeError, ValueError) as exc:
                if isinstance(exc, ValueError) and str(exc).startswith(
                    "Invalid region filter:"
                ):
                    # Filters are edited one character at a time. Intermediate
                    # regexes can be invalid, so keep the last chart visible
                    # and let the selected-regions field show the validation.
                    return
                detail.update(self._detail(node))
                self.notify(str(exc), severity="error", timeout=8)
                return
            from rich.text import Text

            detail.update(Text.from_ansi(plot_text))

        def action_show_snakeviz(self) -> None:
            node = self.selected_browser_node
            if (
                node is None
                or node.kind != "plot"
                or node.payload.get("plot_name") != "flame"
            ):
                self.notify("Select the Flame plot first.", severity="warning")
                return

            def split_patterns(key: str) -> list[str] | None:
                values = [
                    item.strip()
                    for item in str(self.plot_settings.get(key, "")).split(",")
                    if item.strip()
                ]
                return values or None

            ranks = None
            rank_text = self.plot_settings.get("ranks", "").strip()
            if rank_text:
                ranks = []
                for spec in rank_text.split(","):
                    ranks.extend(parse_ranks(spec.strip()))
                ranks = sorted(set(ranks))

            directory = self.output_dir or (
                self.model.file_path.parent / f"{self.model.file_path.stem}_plots"
            )
            directory.mkdir(parents=True, exist_ok=True)
            try:
                prof_paths = export_prof(
                    self.model.results,
                    directory / "profile.prof",
                    ranks=ranks,
                    include=split_patterns("include"),
                    exclude=split_patterns("exclude"),
                    verbose=False,
                )
                if not prof_paths:
                    raise ValueError("No profile data was exported.")
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    port = probe.getsockname()[1]
                subprocess.Popen(
                    [
                        "snakeviz",
                        "--server",
                        "--hostname",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        str(prof_paths[0]),
                    ],
                    start_new_session=True,
                )
                profile_url = quote(str(prof_paths[0]), safe="")
                url = f"http://127.0.0.1:{port}/snakeviz/{profile_url}"
                self.set_timer(1.0, lambda: webbrowser.open(url))
            except FileNotFoundError:
                self.notify(
                    "Snakeviz is not installed. Install it with `pip install snakeviz`.",
                    severity="error",
                    timeout=8,
                )
                return
            except (RuntimeError, ValueError, OSError) as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
            self.notify(f"Opened {url} for {prof_paths[0]} in Snakeviz.", timeout=8)

        def _show_plot_in_child_process(
            self, node: BrowserNode, settings: dict[str, Any]
        ) -> None:
            """Render a real Matplotlib figure outside Textual's event loop."""
            viewer = _matplotlib_child_script()
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    viewer,
                    str(self.model.file_path),
                    node.payload["plot_name"],
                    node.payload.get("metric") or "",
                    json.dumps(settings),
                ],
                start_new_session=True,
            )

        def action_show_matplotlib(self) -> None:
            self._run_selected_plot(show=True, backend="matplotlib")

        def action_show_plotext(self) -> None:
            node = self.selected_browser_node
            if node is None or node.payload.get("plot_name") not in _PLOTEXT_TUI_PLOTS:
                self.notify(
                    "Plotext is available in the TUI for simple plots only.",
                    severity="warning",
                    timeout=5,
                )
                return
            self._show_plotext_in_detail(node)

        def action_show_plotly(self) -> None:
            self._open_plotly_browser()

        def _open_plotly_browser(self) -> None:
            node = self.selected_browser_node
            if node is None or node.kind not in {"plot", "plot_likwid"}:
                self.notify("Select an individual plot first.", severity="warning")
                return
            self._read_plot_settings()
            settings = {**self.plot_settings, "backend": "plotly"}
            directory = self.output_dir or (
                self.model.file_path.parent / f"{self.model.file_path.stem}_plots"
            )
            directory.mkdir(parents=True, exist_ok=True)
            filename = node.payload["plot_name"]
            if node.payload.get("metric"):
                filename += "_" + "_".join(
                    part for part in node.payload["metric"].split() if part.isalnum()
                )
            filepath = directory / f"{filename}.html"
            try:
                render_plot(node, filepath=filepath, show=False, settings=settings)
                webbrowser.open(filepath.resolve().as_uri())
            except (ImportError, RuntimeError, ValueError, OSError) as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
            self.notify(f"Opened {filepath} in the browser.", timeout=8)

        def action_save_plot(self) -> None:
            self._run_selected_plot(show=False)

    return H5BrowserApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scope-profiler tui",
        description="Interactively browse a scope-profiler HDF5 output file.",
    )
    parser.add_argument("file", help="Path to a profiling_data.h5 file")
    parser.add_argument(
        "--plot-output",
        help="Directory for plots saved from the TUI "
        "(default: <file stem>_plots next to the input file)",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        app_cls = _build_textual_app_class()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    model = build_browser_model(args.file)
    app_cls(model, output_dir=args.plot_output).run()
    return 0
