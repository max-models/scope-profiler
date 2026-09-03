"""Duration-vs-time scatter: per-region call durations across the run."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.plotting_scripts._utils import (
    DEFAULT_CMAP,
    _as_runs,
    _hover_region,
    _normalize_ranks,
    _panel_gridspec,
    _region_color_map,
    _to_hex,
    _unique_labels,
    _write_csv,
    _write_json,
)
from scope_profiler.results import ProfilingResults


def _duration_timeseries(
    region,
    ranks: list[int] | None,
    first_start_time: float,
) -> dict[str, np.ndarray] | None:
    """Aggregate one region's calls over ranks into a duration-versus-time series.

    Calls are matched across ranks by call index, so point ``i`` describes the
    i-th call of the region on every rank that got that far. Ranks that stopped
    calling the region earlier simply drop out of the later points, which keeps
    ragged call counts (e.g. a rank-dependent number of iterations) usable.

    Returns ``None`` if no selected rank recorded any call.
    """
    if ranks is None:
        selected_ranks = sorted(region.regions)
    else:
        selected_ranks = [rank for rank in ranks if rank in region.regions]

    per_rank = [
        (rank, region.regions[rank])
        for rank in selected_ranks
        if region.regions[rank].durations.size
    ]
    if not per_rank:
        return None

    max_calls = max(len(region_data.durations) for _, region_data in per_rank)
    times, means, minima, maxima, counts = [], [], [], [], []
    for index in range(max_calls):
        starts = [
            float(region_data.start_times[index]) - first_start_time
            for _, region_data in per_rank
            if len(region_data.durations) > index
        ]
        durations = [
            float(region_data.durations[index])
            for _, region_data in per_rank
            if len(region_data.durations) > index
        ]
        times.append(float(np.mean(starts)))
        means.append(float(np.mean(durations)))
        minima.append(float(np.min(durations)))
        maxima.append(float(np.max(durations)))
        counts.append(len(durations))

    return {
        "time": np.asarray(times, dtype=float),
        "mean": np.asarray(means, dtype=float),
        "min": np.asarray(minima, dtype=float),
        "max": np.asarray(maxima, dtype=float),
        "num_ranks": np.asarray(counts, dtype=int),
        "ranks": np.asarray([rank for rank, _ in per_rank], dtype=int),
    }


def plot_duration_timeseries(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
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
    return_fig: bool = False,
) -> None:
    """Plot each region's call duration over wall-clock time, with a min-max band.

    One line per region tracks the mean duration over the ranks that recorded
    each call, shaded between the minimum and maximum duration seen across
    those ranks, so rank imbalance shows up as a widening band.

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
        # As in _prepare_gantt_data: charts are framed on the first entry.
        first_start_time = run.minimum_start_time
        series = [
            (
                region.name,
                _duration_timeseries(region, normalized_ranks, first_start_time),
            )
            for region in regions
        ]
        series = [(name, values) for name, values in series if values is not None]
        if series:
            prepared.append((run, series))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    labels = _unique_labels([run.display_label for run, _ in prepared])

    if data_filepath:
        records = []
        for label, (_, series) in zip(labels, prepared):
            for region_name, values in series:
                for index in range(values["time"].size):
                    records.append(
                        [
                            label,
                            region_name,
                            index,
                            float(values["time"][index]),
                            float(values["mean"][index]),
                            float(values["min"][index]),
                            float(values["max"][index]),
                            int(values["num_ranks"][index]),
                        ],
                    )
        header = [
            "file",
            "region",
            "call_index",
            "time_seconds",
            "mean_duration_seconds",
            "min_duration_seconds",
            "max_duration_seconds",
            "num_ranks",
        ]
        if data_format == "json":
            points = [dict(zip(header, record)) for record in records]
            colors_map = {
                name: _to_hex(color) for name, color in sorted(color_map.items())
            }
            _write_json(
                data_filepath,
                {"points": points, "colors": colors_map},
                plot="timeseries",
            )
        else:
            _write_csv(data_filepath, header, records)

    if verbose:
        print("Plotting duration over time for files: " + ", ".join(labels))

    single_panel = len(prepared) == 1
    fig_width, fig_height = 12.0, 1.0 + 4.0 * len(prepared)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        # Duration tick labels plus the "Duration (seconds)" axis label.
        gridspec_kw=_panel_gridspec(fig_width, fig_height, 12, not single_panel),
    )

    hover_enabled = backend == "plotly"
    for idx, (run, series) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        for region_name, values in series:
            color = _to_hex(color_map[region_name])
            # The band is the line's min-max envelope; hover belongs on the
            # line, one entry per call.
            canvas.fill_between(
                values["time"],
                values["min"],
                values["max"],
                row=row,
                col=col,
                color=color,
                alpha=0.25,
            )
            line_hover = None
            if hover_enabled:
                region, title = _hover_region(
                    run.get_region(region_name),
                    normalized_ranks,
                )
                line_hover = [
                    _ps._hover_summary(
                        region,
                        title=title,
                        extra=[
                            ("call", index),
                            ("at", f"{values['time'][index]:.6g} s"),
                            ("mean", f"{values['mean'][index]:.6g} s"),
                            (
                                "min-max",
                                (
                                    f"{values['min'][index]:.6g} - "
                                    f"{values['max'][index]:.6g} s"
                                ),
                            ),
                            ("over ranks", int(values["num_ranks"][index])),
                        ],
                    )
                    for index in range(values["time"].size)
                ]
            canvas.add_line(
                values["time"],
                values["mean"],
                row=row,
                col=col,
                linewidth=1.8,
                color=color,
                label=region_name,
                hover=line_hover,
            )

        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_ylabel("Duration per call (seconds)", row=row, col=col)
        canvas.set_title(
            "Region duration over time" if single_panel else run.display_label,
            row=row,
            col=col,
        )
        canvas.set_grid(True, row=row, col=col)
        canvas.set_legend(row=row, col=col)
        if log_scale:
            canvas.set_yscale("log", row=row, col=col)

    if not single_panel:
        canvas.suptitle("Region duration over time")

    rendered = _ps._render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None
