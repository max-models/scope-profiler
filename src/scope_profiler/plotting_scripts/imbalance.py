"""Rank-imbalance plots: per-region duration spread across ranks."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
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
from scope_profiler.plotting_scripts.durations import _DURATION_METRICS
from scope_profiler.plotting_scripts.statistics import _stats_from_values
from scope_profiler.results import ProfilingResults


def plot_imbalance(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    metric: str = "total",
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    log_scale: bool = False,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
) -> None:
    """Plot each region's duration statistic per rank, to surface load imbalance.

    One point per (region, rank), connected by a line in rank order, with a
    dashed horizontal line at the mean over ranks -- so a straggler rank shows
    up as a point sitting well off its region's line. This is a per-rank view
    of the same statistics :func:`plot_durations` aggregates across ranks.

    Parameters
    ----------
    metric : str
        Which per-call duration statistic to plot per rank; one of
        ``"avg"``, ``"min"``, ``"max"``, ``"total"`` (default: ``"total"``,
        the total time a rank spent in the region).
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    """
    Canvas = _ps._get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return

    if metric not in _DURATION_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Valid options are: {list(_DURATION_METRICS)}",
        )
    stat_key, metric_label = _DURATION_METRICS[metric]

    normalized_ranks = _normalize_ranks(ranks)

    reader_regions = []
    all_region_names: set[str] = set()
    for run in runs:
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        all_region_names.update(region.name for region in regions)
        reader_regions.append((run, regions))

    color_map = _region_color_map(all_region_names, cmap=cmap)

    prepared = []
    for run, regions in reader_regions:
        available_ranks = (
            normalized_ranks
            if normalized_ranks is not None
            else list(range(run.num_ranks))
        )
        series = []
        for region in regions:
            rank_list: list[int] = []
            value_list: list[float] = []
            for rank in sorted(available_ranks):
                if rank not in region:
                    continue
                durations = region[rank].durations
                if not durations.size:
                    continue
                value = _stats_from_values(durations)[stat_key]
                if value is None:
                    continue
                rank_list.append(rank)
                value_list.append(value)
            if rank_list:
                series.append(
                    (
                        region.name,
                        np.asarray(rank_list, dtype=int),
                        np.asarray(value_list, dtype=float),
                    ),
                )
        if series:
            prepared.append((run, series))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    labels = _unique_labels([run.display_label for run, _ in prepared])

    if data_filepath:
        records = []
        for label, (_, series) in zip(labels, prepared):
            for region_name, region_ranks, values in series:
                mean_value = float(np.mean(values))
                for rank, value in zip(region_ranks, values):
                    records.append(
                        [label, region_name, int(rank), float(value), mean_value],
                    )
        header = ["file", "region", "rank", "value_seconds", "mean_over_ranks_seconds"]
        if data_format == "json":
            points = [dict(zip(header, record)) for record in records]
            colors_map = {
                name: _to_hex(color) for name, color in sorted(color_map.items())
            }
            _write_json(
                data_filepath,
                {"metric": metric, "points": points, "colors": colors_map},
                plot="imbalance",
            )
        else:
            _write_csv(data_filepath, header, records)

    if verbose:
        print(f"Plotting per-rank imbalance ({metric}) for files: " + ", ".join(labels))

    single_panel = len(prepared) == 1
    fig_width, fig_height = 12.0, 1.0 + 4.0 * len(prepared)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw=_panel_gridspec(fig_width, fig_height, 10, not single_panel),
    )

    hover_enabled = backend == "plotly"
    for idx, (run, series) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        for region_name, region_ranks, values in series:
            color = _to_hex(color_map[region_name])
            # A point is one region on one rank, described by that rank's
            # own Region; the line and its markers share the text.
            texts = (
                [
                    # No value line: the plotted statistic is one of the
                    # ones that rank's own summary already lists.
                    _ps._hover_summary(
                        run.get_region(region_name)[int(rank)],
                        title=f"{region_name} (rank {int(rank)})",
                    )
                    for rank in region_ranks
                ]
                if hover_enabled
                else None
            )
            canvas.add_line(
                region_ranks,
                values,
                row=row,
                col=col,
                linewidth=1.4,
                color=color,
                label=region_name,
                hover=texts,
            )
            canvas.scatter(
                region_ranks,
                values,
                row=row,
                col=col,
                color=color,
                hover=texts,
            )
            canvas.axhline(
                float(np.mean(values)),
                row=row,
                col=col,
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.5,
            )

        canvas.set_xlabel("Rank", row=row, col=col)
        canvas.set_ylabel(metric_label, row=row, col=col)
        canvas.set_title(
            "Per-rank load imbalance" if single_panel else run.display_label,
            row=row,
            col=col,
        )
        canvas.set_grid(True, row=row, col=col)
        canvas.set_legend(row=row, col=col)
        if log_scale:
            canvas.set_yscale("log", row=row, col=col)

    if not single_panel:
        canvas.suptitle("Per-rank load imbalance")

    _ps._render(canvas, filepath, show, backend)
