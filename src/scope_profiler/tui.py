"""Interactive Textual browser for scope-profiler HDF5 files."""

from __future__ import annotations

import argparse
import linecache
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from scope_profiler.h5reader import read_h5
from scope_profiler.inspection import _metadata_sections, _time_span
from scope_profiler.summary import region_row, region_rows


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

    root = BrowserNode(
        path.name,
        "root",
        {"file_path": path},
        [
            BrowserNode("Overview", "overview", {"results": results}),
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


def _line_table(headers, rows) -> str:
    rows = list(rows)
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    if tabulate is not None:
        return tabulate(
            rows,
            headers=headers,
            tablefmt="plain",
            disable_numparse=True,
        )

    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max([len(header), *(len(row[index]) for row in text_rows)])
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in text_rows
    )
    return "\n".join(lines)


def node_detail_text(node: BrowserNode) -> str:
    """Return a plain-text detail view for a selected navigation node."""
    kind = node.kind
    payload = node.payload

    if kind == "overview":
        results = payload["results"]
        path = Path(results.file_path)
        size_mb = path.stat().st_size / 1024**2
        span = _time_span(results)
        lines = [
            f"File: {path}",
            f"Label: {results.label or '-'}",
            f"Ranks: {results.num_ranks}",
            f"Regions: {len(results.get_regions())}",
            f"Size: {size_mb:.2f} MiB",
        ]
        if span is not None:
            lines.append(f"Profiled wall clock: {span:.6g} s")
        if results.total_time is not None:
            lines.append(f"Setup to finalize: {results.total_time:.6g} s")
        return "\n".join(lines)

    if kind == "metadata":
        metadata = payload["metadata"]
        if not metadata:
            return "No metadata recorded."
        return _line_table(("Key", "Value"), sorted(metadata.items()))

    if kind == "metadata_section":
        return _line_table(("Key", "Value"), payload["entries"])

    if kind == "modules":
        modules = payload["modules"]
        return "\n".join(modules) if modules else "No modules recorded."

    if kind == "regions":
        rows = region_rows(payload["results"], sort="total")
        if not rows:
            return "No regions recorded."
        return _line_table(
            ("Region", "Ranks", "Calls", "Total", "Avg", "P95", "Imbalance"),
            (
                (
                    row["name"],
                    row["num_ranks"],
                    row["calls"],
                    _duration(row["total"]),
                    _duration(row["avg"]),
                    _duration(row["p95"]),
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
        return _line_table(("Rank", "Records", "Total"), rows)

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
            ("Region", "Functions", "Total"), rows
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
            + _line_table(("Rank", "Function", "Location", "Lines", "Total"), rows)
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
                ("Line", "Hits", "Time [s]", "Per hit [s]", "%", "Source"), rows
            )
        )

    if kind in {"region", "region_summary"}:
        region = payload["region"]
        row = payload.get("row") or region_row(region)
        lines = [
            f"Region: {region.name}",
            f"Ranks: {row['num_ranks']}",
            f"Calls: {row['calls']}",
            f"Total: {_duration(row['total'])}",
            f"Average: {_duration(row['avg'])}",
            f"Min / max: {_duration(row['min'])} / {_duration(row['max'])}",
            "P50 / P95 / P99: "
            f"{_duration(row['p50'])} / {_duration(row['p95'])} / "
            f"{_duration(row['p99'])}",
            f"Rank imbalance: {row['imbalance']:.6g}%",
        ]
        if region.tags:
            lines.append(f"Tags: {', '.join(region.tags)}")
        if region.has_source:
            lines.append(f"Source: {region.source_file}:{region.source_lineno}")
        return "\n".join(lines)

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
            lines.append(_line_table(("Key", "Value"), sorted(attrs.items())))
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
            lines.append(_line_table(("Key", "Value"), sorted(attrs.items())))
        return "\n".join(lines)

    return node.label


def _build_textual_app_class():
    try:
        from rich.text import Text
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal
        from textual.widgets import Footer, Header, Static, Tree
    except ImportError as exc:
        raise RuntimeError(
            "The interactive TUI requires Textual. Install it with "
            "`pip install scope-profiler[tui]` or `pip install textual`."
        ) from exc

    class H5BrowserApp(App):
        """Textual application for browsing a profile file."""

        CSS = """
        Screen {
            layout: vertical;
        }

        #body {
            height: 1fr;
        }

        #nav {
            width: 36%;
            min-width: 28;
            border: solid $accent;
            padding: 0 1;
            overflow: hidden;
        }

        #detail {
            width: 1fr;
            border: solid $accent;
            padding: 1 2;
            overflow-x: auto;
            overflow-y: auto;
            text-wrap: nowrap;
        }
        """
        BINDINGS = [("q", "quit", "Quit")]

        def __init__(self, model: BrowserModel):
            super().__init__()
            self.model = model

        def _detail(self, node: BrowserNode):
            return Text(node_detail_text(node), no_wrap=True)

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                yield Tree(self.model.root.label, id="nav")
                yield Static(self._detail(self.model.root.children[0]), id="detail")
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
                self.query_one("#detail", Static).update(self._detail(first.data))

        def _populate_tree(self, tree_node, browser_nodes) -> None:
            for browser_node in browser_nodes:
                child = tree_node.add(browser_node.label, data=browser_node)
                if browser_node.children:
                    self._populate_tree(child, browser_node.children)

        def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
            node = event.node.data
            if isinstance(node, BrowserNode):
                self.query_one("#detail", Static).update(self._detail(node))

    return H5BrowserApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scope-profiler tui",
        description="Interactively browse a scope-profiler HDF5 output file.",
    )
    parser.add_argument("file", help="Path to a profiling_data.h5 file")
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
    app_cls(model).run()
    return 0
