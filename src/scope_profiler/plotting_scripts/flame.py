"""Flame chart rendering: nested call-stack frames sized by duration."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.call_stack import build_call_stack
from scope_profiler.plotting_scripts._utils import (
    _as_runs,
    _normalize_ranks,
    _panel_gridspec,
    _region_color_map,
    _to_hex,
    _unique_labels,
    _write_csv,
    _write_json,
)
from scope_profiler.results import ProfilingResults

FLAME_CMAP = "inferno"


def _print_interactive_backend_hint(backend: str, verbose: bool) -> None:
    if verbose and backend == "matplotlib":
        print("For interactive flame-chart hover details, use " "--backend plotly.")


def plot_flame_chart(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = FLAME_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
) -> object | None:
    """
    Plot a flame chart reconstructing the call stack from region timings.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    return_fig : bool
        Return the rendered figure instead of the default ``None``. Matplotlib
        returns ``(fig, axes)``; Plotly returns its figure object.
    """
    Canvas = _ps._get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return

    normalized_ranks = _normalize_ranks(ranks) if ranks is not None else [0]

    reader_regions = []
    all_region_names: set[str] = set()
    for run in runs:
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        all_region_names.update(region.name for region in regions)
        reader_regions.append((run, regions))

    color_map = _region_color_map(all_region_names, cmap=cmap)
    for _, regions in reader_regions:
        for region in regions:
            region.color = color_map[region.name]

    prepared = []
    for run, regions in reader_regions:
        for rank in normalized_ranks:
            if rank < 0 or rank >= run.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = build_call_stack(regions, rank)
            if calls:
                prepared.append((run, rank, calls))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    if data_filepath:
        labels = _unique_labels([run.display_label for run, _, _ in prepared])
        if data_format == "json":
            call_records = []
            colors = {}
            for label, (_, rank, calls) in zip(labels, prepared):
                for call in calls:
                    colors[call["name"]] = _to_hex(call["color"])
                    call_records.append(
                        {
                            "file": label,
                            "rank": rank,
                            "call_id": call["call_id"],
                            "parent_call_id": call["parent"],
                            "region": call["name"],
                            "call_path": call["call_path"],
                            "source_file": call["source_file"],
                            "source_lineno": call["source_lineno"],
                            "depth": call["depth"],
                            "start_seconds": call["start"],
                            "end_seconds": call["end"],
                            "inclusive_duration_seconds": call["inclusive_duration"],
                            "exclusive_duration_seconds": call["exclusive_duration"],
                        }
                    )
            _write_json(
                data_filepath,
                {"calls": call_records, "colors": colors},
                plot="flame",
            )
        else:
            rows = []
            for label, (_, rank, calls) in zip(labels, prepared):
                for call in calls:
                    rows.append(
                        [
                            label,
                            rank,
                            call["call_id"],
                            call["parent"],
                            call["name"],
                            call["call_path"],
                            call["source_file"],
                            call["source_lineno"],
                            call["depth"],
                            call["start"],
                            call["end"],
                            call["inclusive_duration"],
                            call["exclusive_duration"],
                        ]
                    )
            _write_csv(
                data_filepath,
                [
                    "file",
                    "rank",
                    "call_id",
                    "parent_call_id",
                    "region",
                    "call_path",
                    "source_file",
                    "source_lineno",
                    "depth",
                    "start_seconds",
                    "end_seconds",
                    "inclusive_duration_seconds",
                    "exclusive_duration_seconds",
                ],
                rows,
            )

    if verbose:
        print(
            "Plotting flame chart for: "
            + ", ".join(
                f"{run.display_label} (rank {rank})" for run, rank, _ in prepared
            )
        )
    _print_interactive_backend_hint(backend, verbose)

    single_panel = len(prepared) == 1
    panel_heights = [
        max(2.0, 0.6 * (max(call["depth"] for call in calls) + 1))
        for _, _, calls in prepared
    ]
    fig_width, fig_height = 12.0, 1.0 + sum(panel_heights)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        # Depth numbers plus the "Call depth" axis label.
        gridspec_kw=_panel_gridspec(fig_width, fig_height, 8, not single_panel),
    )

    def add_region_legend(fig, axes) -> None:
        """Make Matplotlib flame colors identify regions, not stack depth."""
        if backend != "matplotlib":
            return
        from matplotlib.patches import Patch

        axes_list = np.asarray(axes).reshape(-1)
        for axis, (_, _, calls) in zip(axes_list, prepared):
            for patch, call in zip(axis.patches, calls):
                patch.set_facecolor(_to_hex(color_map[call["name"]]))

        handles = [
            Patch(
                facecolor=_to_hex(color_map[name]),
                edgecolor="black",
                label=name,
            )
            for name in sorted(all_region_names)
        ]
        if handles:
            fig.legend(
                handles=handles,
                title="Regions",
                loc="center left",
                bbox_to_anchor=(0.82, 0.5),
            )
            fig.subplots_adjust(right=0.8)

    hover_enabled = backend == "plotly"
    for idx, (run, rank, calls) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        first_start = min(call["start"] for call in calls)
        total_span = max(call["end"] for call in calls) - first_start
        max_depth = max(call["depth"] for call in calls)

        hover_texts = None
        if hover_enabled:
            hover_texts = []
            for call in calls:
                extra = [
                    ("call", call["call_path"]),
                    ("this call", f"{call['inclusive_duration']:.6g} s"),
                    ("this call, self", f"{call['exclusive_duration']:.6g} s"),
                ]
                if call["source_file"]:
                    location = call["source_file"]
                    if call["source_lineno"] is not None:
                        location += f":{call['source_lineno']}"
                    extra.append(("source", location))
                hover_texts.append(
                    _ps._hover_summary(
                        run.get_region(call["name"])[rank],
                        title=f"{call['name']} (rank {rank})",
                        extra=extra,
                    )
                )

        canvas.flame_chart(
            [call["call_path"] for call in calls],
            [call["parent"] for call in calls],
            [call["end"] - call["start"] for call in calls],
            start_times=[call["start"] - first_start for call in calls],
            row=row,
            col=col,
            edgecolor="black",
            hover=hover_texts,
            colors=[_to_hex(color_map[call["name"]]) for call in calls],
            # Canvas.flame_chart colors frames by depth from a colormap and
            # ignores per-frame colors. Only the matplotlib backend takes a
            # matplotlib colormap name; the Plotly one needs a Plotly
            # colorscale, so it keeps its own default.
            **({} if backend == "plotly" else {"colormap": cmap}),
        )

        canvas.set_yticks(list(range(max_depth + 1)), row=row, col=col)
        # The frames are drawn as rectangles, which don't drive autorange, so
        # frame the panel from the data.
        canvas.set_xlim(0, total_span, row=row, col=col)
        canvas.set_ylim(-0.6, max_depth + 1.0, row=row, col=col)
        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_ylabel("Call depth", row=row, col=col)
        canvas.set_title(f"{run.display_label} (rank {rank})", row=row, col=col)
        canvas.set_grid(True, row=row, col=col)

    if not single_panel:
        canvas.suptitle("Flame Graphs")

    def improve_plotly_flame(fig) -> None:
        """Keep frame labels available and make depth rows touch cleanly."""
        for trace in fig.data:
            if getattr(trace, "type", None) != "bar" or trace.customdata is None:
                continue
            trace.update(
                width=1.0,
                cliponaxis=False,
            )
        # maxplotlib adds labels for wide frames as layout annotations. The
        # legend and hover text are less intrusive and remain available for
        # every frame, so remove those background labels from Plotly output.
        fig.layout.annotations = tuple(
            annotation
            for annotation in (fig.layout.annotations or ())
            if annotation.text not in all_region_names
        )
        from plotly.graph_objects import Scatter

        for name in sorted(all_region_names):
            fig.add_trace(
                Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={"color": _to_hex(color_map[name]), "size": 9},
                    name=name,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        fig.update_layout(
            bargap=0,
            legend={"title": "Regions"},
        )

    rendered = _ps._render(
        canvas,
        filepath,
        show,
        backend,
        return_fig=return_fig,
        matplotlib_postprocess=add_region_legend if backend == "matplotlib" else None,
        plotly_postprocess=improve_plotly_flame if backend == "plotly" else None,
    )
    return rendered if return_fig else None


def _aggregate_flame_calls(calls: list[dict]) -> list[dict]:
    """Collapse repeated call paths into the nodes of a flame graph."""
    nodes: dict[tuple[str, ...], dict] = {}
    order: list[tuple[str, ...]] = []
    for call in calls:
        path = tuple(str(call["call_path"]).split(" > "))
        node = nodes.get(path)
        if node is None:
            node = {
                "name": path[-1],
                "path": path,
                "inclusive_duration": 0.0,
                "exclusive_duration": 0.0,
                "source_file": call["source_file"],
                "source_lineno": call["source_lineno"],
                "color": call["color"],
            }
            nodes[path] = node
            order.append(path)
        node["inclusive_duration"] += call["inclusive_duration"]
        node["exclusive_duration"] += call["exclusive_duration"]

    index = {path: i for i, path in enumerate(order)}
    children: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for path in order:
        parent = path[:-1]
        children.setdefault(parent, []).append(path)

    starts: dict[tuple[str, ...], float] = {}

    def layout(path: tuple[str, ...], start: float) -> None:
        starts[path] = start
        cursor = start
        for child in children.get(path, []):
            layout(child, cursor)
            cursor += nodes[child]["inclusive_duration"]

    roots = children.get((), [])
    cursor = 0.0
    for root in roots:
        layout(root, cursor)
        cursor += nodes[root]["inclusive_duration"]

    aggregated = []
    for call_id, path in enumerate(order):
        node = nodes[path]
        parent_path = path[:-1]
        start = starts[path]
        aggregated.append(
            {
                "call_id": call_id,
                "parent": None if not parent_path else index[parent_path],
                "name": node["name"],
                "call_path": " > ".join(path),
                "start": start,
                "end": start + node["inclusive_duration"],
                "inclusive_duration": node["inclusive_duration"],
                "exclusive_duration": node["exclusive_duration"],
                "depth": len(path) - 1,
                "source_file": node["source_file"],
                "source_lineno": node["source_lineno"],
                "color": node["color"],
            }
        )
    return aggregated


def plot_flame_graph(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = FLAME_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
) -> object | None:
    """Plot an aggregated flame graph whose x-axis represents total time."""
    Canvas = _ps._get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        return
    normalized_ranks = _normalize_ranks(ranks) if ranks is not None else [0]

    reader_regions = []
    all_region_names: set[str] = set()
    for run in runs:
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        all_region_names.update(region.name for region in regions)
        reader_regions.append((run, regions))
    color_map = _region_color_map(all_region_names, cmap=cmap)
    for _, regions in reader_regions:
        for region in regions:
            region.color = color_map[region.name]

    prepared = []
    for run, regions in reader_regions:
        for rank in normalized_ranks:
            if rank < 0 or rank >= run.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = _aggregate_flame_calls(build_call_stack(regions, rank))
            if calls:
                prepared.append((run, rank, calls))
    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")
    if data_filepath:
        labels = _unique_labels([run.display_label for run, _, _ in prepared])
        rows = [
            {
                "file": label,
                "rank": rank,
                "call_id": call["call_id"],
                "parent_call_id": call["parent"],
                "region": call["name"],
                "call_path": call["call_path"],
                "depth": call["depth"],
                "start_seconds": call["start"],
                "end_seconds": call["end"],
                "inclusive_duration_seconds": call["inclusive_duration"],
                "exclusive_duration_seconds": call["exclusive_duration"],
            }
            for label, (_, rank, calls) in zip(labels, prepared)
            for call in calls
        ]
        if data_format == "json":
            _write_json(data_filepath, {"calls": rows}, plot="flame_graph")
        else:
            headers = [
                "file",
                "rank",
                "call_id",
                "parent_call_id",
                "region",
                "call_path",
                "depth",
                "start_seconds",
                "end_seconds",
                "inclusive_duration_seconds",
                "exclusive_duration_seconds",
            ]
            _write_csv(
                data_filepath, headers, [[row[key] for key in headers] for row in rows]
            )
    if verbose:
        print(
            "Plotting flame graph for: "
            + ", ".join(
                f"{run.display_label} (rank {rank})" for run, rank, _ in prepared
            )
        )
    _print_interactive_backend_hint(backend, verbose)

    single_panel = len(prepared) == 1
    panel_heights = [
        max(2.0, 0.6 * (max(call["depth"] for call in calls) + 1))
        for _, _, calls in prepared
    ]
    fig_width, fig_height = 12.0, 1.0 + sum(panel_heights)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw=_panel_gridspec(fig_width, fig_height, 8, not single_panel),
    )
    for idx, (run, rank, calls) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0
        total_span = sum(
            call["inclusive_duration"] for call in calls if call["parent"] is None
        )
        max_depth = max(call["depth"] for call in calls)
        hover = None
        if backend == "plotly":
            hover = [
                _ps._hover_summary(
                    run.get_region(call["name"])[rank],
                    title=f"{call['name']} (rank {rank})",
                    extra=[
                        ("aggregated path", call["call_path"]),
                        ("total", f"{call['inclusive_duration']:.6g} s"),
                        ("self", f"{call['exclusive_duration']:.6g} s"),
                    ],
                )
                for call in calls
            ]
        canvas.flame_chart(
            [call["name"] for call in calls],
            [call["parent"] for call in calls],
            [call["inclusive_duration"] for call in calls],
            start_times=[call["start"] for call in calls],
            row=row,
            col=col,
            colors=[_to_hex(color_map[call["name"]]) for call in calls],
            edgecolor="black",
            hover=hover,
            **({} if backend == "plotly" else {"colormap": cmap}),
        )
        canvas.set_xlim(0, total_span, row=row, col=col)
        canvas.set_ylim(-0.6, max_depth + 1.0, row=row, col=col)
        canvas.set_xlabel("Accumulated time (seconds)", row=row, col=col)
        canvas.set_ylabel("Call depth", row=row, col=col)
        canvas.set_title(f"{run.display_label} (rank {rank})", row=row, col=col)
        canvas.set_grid(True, row=row, col=col)
    if not single_panel:
        canvas.suptitle("Flame Graphs")
    rendered = _ps._render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None


# Backward-compatible name for the former time-based plot.
plot_flame = plot_flame_chart
