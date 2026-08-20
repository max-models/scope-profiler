"""Interactive Textual browser for scope-profiler HDF5 files."""

from __future__ import annotations

import argparse
import sys
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
    for region in results.get_regions():
        rank_children = [
            BrowserNode(
                f"Rank {rank}",
                "rank_region",
                {"region": region, "rank": rank},
                [BrowserNode("Calls", "rank_calls", {"region": region, "rank": rank})],
            )
            for rank in region.ranks
        ]
        extra_children = [
            BrowserNode("Summary", "region_summary", {"region": region}),
            *rank_children,
        ]
        if region.has_source:
            extra_children.append(BrowserNode("Source", "source", {"region": region}))
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
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        )
    ]
    lines.append("  ".join("-" * width for width in widths))
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
        }

        #detail {
            width: 1fr;
            border: solid $accent;
            padding: 1 2;
            overflow: auto;
        }
        """
        BINDINGS = [("q", "quit", "Quit")]

        def __init__(self, model: BrowserModel):
            super().__init__()
            self.model = model

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                yield Tree(self.model.root.label, id="nav")
                yield Static(node_detail_text(self.model.root.children[0]), id="detail")
            yield Footer()

        def on_mount(self) -> None:
            tree = self.query_one("#nav", Tree)
            tree.root.data = self.model.root
            self._populate_tree(tree.root, self.model.root.children)
            tree.root.expand()
            if tree.root.children:
                tree.root.children[0].select()

        def _populate_tree(self, tree_node, browser_nodes) -> None:
            for browser_node in browser_nodes:
                child = tree_node.add(browser_node.label, data=browser_node)
                if browser_node.children:
                    self._populate_tree(child, browser_node.children)

        def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
            node = event.node.data
            if isinstance(node, BrowserNode):
                self.query_one("#detail", Static).update(node_detail_text(node))

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
