"""Flame chart rendering: nested call-stack frames sized by duration."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.call_stack import build_call_stack
from scope_profiler.plotting_scripts._utils import (
    DEFAULT_CMAP,
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


def plot_flame(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
) -> object | None:
    """
    Plot a flame graph reconstructing the call stack from region timings using maxplotlib.

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
            _write_json(data_filepath, {"calls": call_records, "colors": colors})
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
            "Plotting flame graph for: "
            + ", ".join(
                f"{run.display_label} (rank {rank})" for run, rank, _ in prepared
            )
        )

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

    rendered = _ps._render(
        canvas,
        filepath,
        show,
        backend,
        return_fig=return_fig,
        matplotlib_postprocess=add_region_legend if backend == "matplotlib" else None,
    )
    return rendered if return_fig else None
