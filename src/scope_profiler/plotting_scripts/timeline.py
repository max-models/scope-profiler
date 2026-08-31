"""Timeline density heatmap: call-count intensity over time, per region."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.plotting_scripts._utils import (
    _as_runs,
    _normalize_ranks,
    _set_xticks,
    _write_csv,
    _write_json,
)
from scope_profiler.results import ProfilingResults


def plot_timeline_density(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = "magma",
    bins: int = 200,
    start_time: float | None = None,
    end_time: float | None = None,
    min_duration: float = 0.0,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
) -> object | None:
    """Plot binned region occupancy instead of individual timeline bars.

    Each cell contains the seconds of a region overlapping that time bin.
    This is intentionally an aggregate visualization: it remains readable
    when a run contains millions of short calls.
    """
    Canvas = _ps._get_canvas()
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if min_duration < 0:
        raise ValueError("min_duration must be non-negative")
    runs = _as_runs(profiling_data)
    if not runs:
        return
    normalized_ranks = _normalize_ranks(ranks)
    prepared = []
    records = []
    for run in runs:
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        selected_ranks = normalized_ranks or list(range(run.num_ranks))
        all_events = []
        for region in regions:
            for rank in selected_ranks:
                if rank in region:
                    data = region[rank]
                    all_events.extend(
                        (
                            region.name,
                            start - run.minimum_start_time,
                            end - run.minimum_start_time,
                        )
                        for start, end in zip(data.start_times, data.end_times)
                        if end - start >= min_duration
                    )
        if not all_events:
            raise ValueError("No calls recorded for the requested filters.")
        lower = min(
            start_time if start_time is not None else 0.0,
            min(event[1] for event in all_events),
        )
        upper = max(event[2] for event in all_events) if end_time is None else end_time
        if start_time is not None:
            lower = start_time
        if upper <= lower:
            raise ValueError("end_time must be greater than start_time")
        names = [region.name for region in regions]
        edges = np.linspace(lower, upper, bins + 1)
        matrix = np.zeros((len(names), bins), dtype=float)
        name_index = {name: index for index, name in enumerate(names)}
        for name, start, end in all_events:
            start, end = max(start, lower), min(end, upper)
            if end <= start:
                continue
            left = max(0, int(np.searchsorted(edges, start, side="right") - 1))
            right = min(bins - 1, int(np.searchsorted(edges, end, side="left")))
            for index in range(left, right + 1):
                matrix[name_index[name], index] += max(
                    0.0, min(end, edges[index + 1]) - max(start, edges[index])
                )
        prepared.append((run, names, edges, matrix))
        for row, name in enumerate(names):
            for col in range(bins):
                records.append(
                    [
                        run.display_label,
                        name,
                        float(edges[col]),
                        float(edges[col + 1]),
                        float(matrix[row, col]),
                    ]
                )

    if verbose:
        print("Plotting timeline density")
    if data_filepath:
        header = [
            "file",
            "region",
            "bin_start_seconds",
            "bin_end_seconds",
            "occupied_seconds",
        ]
        if data_format == "json":
            _write_json(
                data_filepath, {"points": [dict(zip(header, row)) for row in records]}
            )
        else:
            _write_csv(data_filepath, header, records)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(12.0, max(3.5, 1.0 + 0.35 * sum(len(x[1]) for x in prepared))),
    )
    single_panel = len(prepared) == 1
    for index, (run, names, edges, matrix) in enumerate(prepared):
        row = None if single_panel else index
        col = None if single_panel else 0
        canvas.imshow(matrix, cmap=cmap, aspect="auto", row=row, col=col)
        tick_count = min(10, len(edges))
        ticks = np.linspace(0, len(edges) - 1, tick_count, dtype=int)
        _set_xticks(
            canvas, ticks, labels=[f"{edges[t]:.3g}" for t in ticks], row=row, col=col
        )
        canvas.set_yticks(list(range(len(names))), labels=names, row=row, col=col)
        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_ylabel("Region", row=row, col=col)
        canvas.set_title(run.display_label, row=row, col=col)
        canvas.colorbar("Occupied seconds per bin", row=row, col=col)
    if not single_panel:
        canvas.suptitle("Timeline density")
    rendered = _ps._render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None
