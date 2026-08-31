"""Gantt chart rendering: per-rank timeline bars for one or more profiling runs."""

from collections import defaultdict
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


def _add_gantt_bars(
    canvas,
    row: int | None,
    lanes: Sequence[str],
    bars: Sequence[tuple[int, float, float, str]],
    alpha: float = 0.7,
    hover_enabled: bool = False,
    lane_hover: Sequence[str] | None = None,
) -> None:
    """Draw ``Canvas.gantt`` bars, every call of a lane on that lane's row.

    ``Canvas.gantt`` puts bar *i* of a call on row *i* (``y =
    arange(len(tasks))``, on both backends) and takes a single color for the
    whole call, since maxplotlib's Plotly backend reads ``color`` as one
    RGB(A) value. Each drawing call therefore passes one entry per lane --- so
    the entry index is the lane's row --- and only fills in the bars sharing
    one color; the rest are NaN, which draws nothing on either backend.

    Lanes hold several calls, so bars are grouped by their position within the
    lane as well: the *k*-th call of every lane goes into the same group,
    leaving at most one bar per lane per drawing call.

    ``lane_hover`` gives each lane's region summary; the hovered call's own
    start and duration are added to it per drawing call, since that is what
    distinguishes one bar of a lane from the next. ``hover=`` is a
    backend-neutral ``Canvas`` kwarg: Plotly renders it, other backends
    ignore it.
    """
    col = None if row is None else 0
    n_lanes = len(lanes)

    # (color, index within the lane) -> the bars drawn together.
    groups: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    calls_per_lane: dict[int, int] = defaultdict(int)
    for lane, start, duration, color in bars:
        slot = calls_per_lane[lane]
        calls_per_lane[lane] += 1
        groups[(color, slot)].append((lane, start, duration))

    for (color, _), group in groups.items():
        starts = np.full(n_lanes, np.nan)
        durations = np.full(n_lanes, np.nan)
        for lane, start, duration in group:
            starts[lane] = start
            durations[lane] = duration
        hover_kwargs = {}
        if hover_enabled:
            texts = [""] * n_lanes
            if lane_hover:
                for lane, start, duration in group:
                    texts[lane] = (
                        f"{lane_hover[lane]}<br>this call: {start:.6g} s "
                        f"+ {duration:.6g} s"
                    )
            hover_kwargs["hover"] = texts
        canvas.gantt(
            list(lanes),
            starts,
            durations,
            row=row,
            col=col,
            color=color,
            edgecolor="black",
            alpha=alpha,
            **hover_kwargs,
        )



def _prepare_gantt_data(
    profiling_data: ProfilingResults,
    ranks: list[int] | int | None,
    include: list[str] | str | None,
    exclude: list[str] | str | None,
) -> tuple[list, list[int], float]:
    regions = profiling_data.get_regions(include=include, exclude=exclude)
    if not regions:
        raise ValueError("No regions matched the selected filters.")

    normalized_ranks = _normalize_ranks(ranks)
    if normalized_ranks is None:
        normalized_ranks = list(range(profiling_data.num_ranks))
    else:
        invalid_ranks = [
            rank
            for rank in normalized_ranks
            if rank < 0 or rank >= profiling_data.num_ranks
        ]
        if invalid_ranks:
            raise ValueError(f"Invalid ranks requested: {invalid_ranks}")

    # Charts frame the recorded window: x = 0 is the first region entry, not
    # the run's registered start (run.time_origin), so that a long gap
    # between setup() and the first region does not push the bars off to one
    # side. run.events(origin=run.minimum_start_time) matches this.
    return regions, normalized_ranks, profiling_data.minimum_start_time


def _aggregate_gantt_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    min_duration: float = 0.0,
    start_time: float | None = None,
    end_time: float | None = None,
    block_size: int = 1,
) -> list[tuple[float, float, int]]:
    """Filter and optionally coalesce timeline intervals.

    ``start_time``/``end_time`` are relative to the plot origin. Intervals
    crossing a window boundary are clipped. Coalescing preserves event order
    and reports the number of calls represented by each returned bar.
    """
    if min_duration < 0:
        raise ValueError("min_duration must be non-negative")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    selected: list[tuple[float, float]] = []
    for start, end in intervals:
        if end - start < min_duration:
            continue
        if start_time is not None and end <= start_time:
            continue
        if end_time is not None and start >= end_time:
            continue
        start = max(start, start_time) if start_time is not None else start
        end = min(end, end_time) if end_time is not None else end
        if end > start:
            selected.append((start, end))
    if block_size == 1:
        return [(start, end, 1) for start, end in selected]
    return [
        (
            selected[index][0],
            selected[min(index + block_size - 1, len(selected) - 1)][1],
            min(block_size, len(selected) - index),
        )
        for index in range(0, len(selected), block_size)
    ]


def plot_gantt(
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
    min_duration: float = 0.0,
    start_time: float | None = None,
    end_time: float | None = None,
    aggregate_calls: int = 1,
    collapse_depth: int | None = None,
) -> object | None:
    """
    Plot a Gantt chart of all (or selected) regions with per-rank lanes using maxplotlib.

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

    if collapse_depth is not None and collapse_depth < 0:
        raise ValueError("collapse_depth must be non-negative")
    prepared = []
    for run in runs:
        regions, selected_ranks, first_start_time = _prepare_gantt_data(
            run,
            ranks,
            include,
            exclude,
        )
        prepared.append((run, regions, selected_ranks, first_start_time))

    color_map = _region_color_map(
        (region.name for _, regions, _, _ in prepared for region in regions),
        cmap=cmap,
    )
    for _, regions, _, _ in prepared:
        for region in regions:
            region.color = color_map[region.name]

    labels = _unique_labels([run.display_label for run, _, _, _ in prepared])

    if data_filepath:
        if data_format == "json":
            intervals = []
            colors = {}
            for label, (_, regions, selected_ranks, first_start_time) in zip(
                labels, prepared
            ):
                for region in regions:
                    colors[region.name] = _to_hex(region.color)
                    for rank in selected_ranks:
                        region_data = region[rank]
                        for start, end in zip(
                            region_data.start_times, region_data.end_times
                        ):
                            intervals.append(
                                {
                                    "file": label,
                                    "rank": rank,
                                    "region": region.name,
                                    "start_seconds": start - first_start_time,
                                    "end_seconds": end - first_start_time,
                                }
                            )
            _write_json(data_filepath, {"intervals": intervals, "colors": colors})
        else:
            rows = []
            for label, (_, regions, selected_ranks, first_start_time) in zip(
                labels, prepared
            ):
                for region in regions:
                    for rank in selected_ranks:
                        region_data = region[rank]
                        for start, end in zip(
                            region_data.start_times, region_data.end_times
                        ):
                            rows.append(
                                [
                                    label,
                                    rank,
                                    region.name,
                                    start - first_start_time,
                                    end - first_start_time,
                                ]
                            )
            _write_csv(
                data_filepath,
                ["file", "rank", "region", "start_seconds", "end_seconds"],
                rows,
            )

    if verbose:
        if len(prepared) == 1:
            print(f"Plotting Gantt chart for ranks: {prepared[0][2]}")
        else:
            print(f"Plotting combined Gantt chart for files: {', '.join(labels)}")

    # One lane per (region, rank): every call of a region lands on the same
    # row, so a panel is as tall as the number of lanes it draws.
    single_panel = len(prepared) == 1
    hover_enabled = backend == "plotly"
    panel_lanes: list[list[str]] = []
    panel_bars: list[list[tuple[int, float, float, str]]] = []
    panel_lane_hover: list[list[str]] = []
    for _, regions, selected_ranks, first_start_time in prepared:
        lanes: list[str] = []
        bars: list[tuple[int, float, float, str]] = []
        lane_hover: list[str] = []
        for region in regions:
            for rank in selected_ranks:
                # A region need not have been entered on every rank.
                if rank not in region:
                    continue
                region_data = region[rank]
                if not len(region_data.start_times):
                    continue
                lane = len(lanes)
                lanes.append(f"{region.name} (rank {rank})")
                # A lane is one region on one rank, so it is that rank's
                # Region that describes it.
                if hover_enabled:
                    lane_hover.append(
                        _ps._hover_summary(
                            region_data, title=f"{region.name} (rank {rank})"
                        )
                    )
                color = _to_hex(region.color)
                visible_calls = None
                if collapse_depth is not None:
                    calls = build_call_stack(regions, rank, origin=first_start_time)
                    visible_calls = {
                        call["call_index"]
                        for call in calls
                        if call["name"] == region.name
                        and call["depth"] <= collapse_depth
                    }
                raw_intervals = [
                    (float(start - first_start_time), float(end - first_start_time))
                    for call_index, (start, end) in enumerate(
                        zip(region_data.start_times, region_data.end_times)
                    )
                    if visible_calls is None or call_index in visible_calls
                ]
                for start, end, count in _aggregate_gantt_intervals(
                    raw_intervals,
                    min_duration=min_duration,
                    start_time=start_time,
                    end_time=end_time,
                    block_size=aggregate_calls,
                ):
                    bars.append(
                        (
                            lane,
                            start,
                            end - start,
                            color,
                        )
                    )
                if aggregate_calls > 1:
                    lanes[-1] += f" (blocks of {aggregate_calls})"
        panel_lanes.append(lanes)
        panel_bars.append(bars)
        panel_lane_hover.append(lane_hover)

    if not any(panel_bars):
        raise ValueError("No calls recorded for the requested ranks.")

    panel_heights = [max(2.5, 0.35 * len(lanes)) for lanes in panel_lanes]
    fig_width, fig_height = 12.0, 1.0 + sum(panel_heights)
    lane_label_chars = max(len(lane) for lanes in panel_lanes for lane in lanes)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw=_panel_gridspec(
            fig_width, fig_height, lane_label_chars, not single_panel
        ),
    )

    for idx, (label, lanes, bars, lane_hover) in enumerate(
        zip(labels, panel_lanes, panel_bars, panel_lane_hover)
    ):
        row = None if single_panel else idx
        col = None if single_panel else 0

        _add_gantt_bars(
            canvas, row, lanes, bars, hover_enabled=hover_enabled, lane_hover=lane_hover
        )

        canvas.set_yticks(list(range(len(lanes))), labels=lanes, row=row, col=col)
        canvas.set_xlim(
            0 if start_time is None else start_time,
            (
                end_time
                if end_time is not None
                else max(start + duration for _, start, duration, _ in bars)
            ),
            row=row,
            col=col,
        )
        canvas.set_ylim(-0.6, len(lanes) - 0.4, row=row, col=col)
        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_title(
            "Profiling Gantt Chart" if single_panel else label, row=row, col=col
        )
        canvas.set_grid(True, row=row, col=col)

    if not single_panel:
        canvas.suptitle("Combined Profiling Gantt Chart")

    # One Plotly bar trace per region color; without "overlay" they would
    # share each row's height instead of each drawing on the full row.
    rendered = _ps._render(
        canvas,
        filepath,
        show,
        backend,
        plotly_layout={"barmode": "overlay"},
        return_fig=return_fig,
    )
    return rendered if return_fig else None

