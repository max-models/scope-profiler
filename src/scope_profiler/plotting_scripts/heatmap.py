"""Rank x region duration heatmap."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.call_stack import NestingError
from scope_profiler.plotting_scripts._utils import (
    _as_runs,
    _normalize_ranks,
    _write_csv,
    _write_json,
)
from scope_profiler.results import ProfilingResults


def plot_rank_heatmap(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = "viridis",
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
    exclusive: bool = False,
) -> object | None:
    """Plot total or exclusive region time as a rank-by-region heatmap."""
    Canvas = _ps._get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        return

    normalized_ranks = _normalize_ranks(ranks)
    prepared = []
    all_records = []
    for run in runs:
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        selected_ranks = (
            normalized_ranks
            if normalized_ranks is not None
            else list(range(run.num_ranks))
        )
        invalid = [rank for rank in selected_ranks if rank < 0 or rank >= run.num_ranks]
        if invalid:
            raise ValueError(f"Invalid ranks requested: {invalid}")
        region_names = [region.name for region in regions]
        matrix = np.zeros((len(selected_ranks), len(region_names)), dtype=float)
        for col, region in enumerate(regions):
            for row, rank in enumerate(selected_ranks):
                if rank in region and region[rank].durations.size:
                    if exclusive:
                        try:
                            value = region[rank].exclusive_duration
                        except NestingError:
                            value = region[rank].total_duration
                    else:
                        value = float(np.sum(region[rank].durations))
                    matrix[row, col] = float(value)
                all_records.append(
                    [run.display_label, rank, region.name, matrix[row, col]]
                )
        prepared.append((run, selected_ranks, region_names, matrix))

    if verbose:
        print("Plotting rank × region duration heatmap")

    if data_filepath:
        duration_name = (
            "exclusive_duration_seconds" if exclusive else "total_duration_seconds"
        )
        header = ["file", "rank", "region", duration_name]
        if data_format == "json":
            _write_json(
                data_filepath,
                {"points": [dict(zip(header, record)) for record in all_records]},
            )
        else:
            _write_csv(data_filepath, header, all_records)

    fig_width = max(10.0, 0.8 * max(len(item[2]) for item in prepared) + 3.0)
    fig_height = max(3.5, 1.0 + 0.35 * sum(len(item[1]) for item in prepared))
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"right": 0.86},
    )
    single_panel = len(prepared) == 1
    hover_enabled = backend == "plotly"
    for index, (run, selected_ranks, region_names, matrix) in enumerate(prepared):
        row = None if single_panel else index
        col = None if single_panel else 0
        # One hover text per cell, in the matrix's own (rank, region) shape:
        # a cell is one region on one rank, so that rank's Region describes
        # it.
        cell_hover = None
        if hover_enabled:
            cell_hover = [
                [
                    (
                        _ps._hover_summary(
                            run.get_region(region_name)[rank],
                            title=f"{region_name} (rank {rank})",
                        )
                        if rank in run.get_region(region_name)
                        else f"<b>{region_name} (rank {rank})</b><br>not entered"
                    )
                    for region_name in region_names
                ]
                for rank in selected_ranks
            ]
        canvas.imshow(
            matrix, cmap=cmap, aspect="auto", row=row, col=col, hover=cell_hover
        )
        canvas.set_xticks(
            list(range(len(region_names))), labels=region_names, row=row, col=col
        )
        canvas.set_yticks(
            list(range(len(selected_ranks))),
            labels=[str(rank) for rank in selected_ranks],
            row=row,
            col=col,
        )
        canvas.set_xlabel("Region", row=row, col=col)
        canvas.set_ylabel("MPI rank", row=row, col=col)
        canvas.set_title(run.display_label, row=row, col=col)
        canvas.colorbar(
            "Exclusive duration (seconds)" if exclusive else "Total duration (seconds)",
            row=row,
            col=col,
        )

    if not single_panel:
        canvas.suptitle(
            "Rank × region exclusive duration"
            if exclusive
            else "Rank × region duration"
        )
    rendered = _ps._render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None
