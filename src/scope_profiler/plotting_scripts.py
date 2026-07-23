"""Plotting utilities for visualizing profiling data.

Figures are rendered with `maxplotlib <https://github.com/max-models/maxplotlib>`_
using its Plotly backend.
"""

import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.h5reader import ProfilingH5Reader


def _get_canvas_cls():
    try:
        from maxplotlib import Canvas
    except ImportError as exc:
        raise ImportError(
            "maxplotlib is required for plotting. Install scope-profiler[pproc] "
            "(see https://github.com/max-models/maxplotlib)."
        ) from exc
    return Canvas


# Qualitative 20-colour palette (matplotlib's "tab20" values), inlined so
# plotting only depends on maxplotlib.
_PALETTE = (
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
)


def _palette_colors(n: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(max(n, 1))]


def _add_bar(
    subplot,
    hovertexts: list[str],
    x0: float,
    x1: float,
    y_bottom: float,
    y_top: float,
    color: str,
    alpha: float,
    hovertext: str,
) -> None:
    """Draw one filled rectangle as a maxplotlib ``fill_between`` region.

    maxplotlib renders each of these as a single ``fill='toself'`` Plotly
    trace, in insertion order; ``_style_bars`` walks the finished figure in
    that same order to attach the outline and hover text, which
    ``fill_between`` itself does not forward to the backend.
    """
    subplot.fill_between([x0, x1], y_top, y_bottom, color=color, alpha=alpha)
    hovertexts.append(hovertext)


def _style_bars(fig, hovertexts: Sequence[str]) -> None:
    """Attach bar outlines and hover text to the traces made by ``_add_bar``."""
    bar_traces = [
        trace for trace in fig.data if getattr(trace, "fill", None) == "toself"
    ]
    for trace, hovertext in zip(bar_traces, hovertexts):
        # Plotly defaults short scatter traces to "lines+markers", which would
        # dot every bar corner.
        trace.mode = "lines"
        trace.line.color = "black"
        trace.line.width = 0.5
        trace.hoveron = "fills"
        trace.hoverinfo = "text"
        trace.hovertext = hovertext


def _export_plotly_figure(fig, filepath: str | None, show: bool) -> None:
    if filepath:
        extension = Path(filepath).suffix.lower()
        if extension in {".html", ".htm"}:
            fig.write_html(filepath)
        else:
            try:
                fig.write_image(filepath)
            except Exception as exc:
                raise RuntimeError(
                    "Plotly image export failed. For PNG/PDF/SVG export, install "
                    "kaleido (e.g., `pip install -U kaleido`), or export to an "
                    ".html filepath instead."
                ) from exc
    if show:
        fig.show()


def _as_readers(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
) -> list[ProfilingH5Reader]:
    if isinstance(profiling_data, ProfilingH5Reader):
        return [profiling_data]
    return list(profiling_data)


def _unique_labels(labels: Sequence[str]) -> list[str]:
    label_counts: dict[str, int] = {}
    unique_labels: list[str] = []
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts[label] > 1:
            unique_labels.append(f"{label} ({label_counts[label]})")
        else:
            unique_labels.append(label)
    return unique_labels


def _normalize_ranks(
    ranks: list[int] | int | None,
) -> list[int] | None:
    if ranks is None:
        return None
    if isinstance(ranks, int):
        return [ranks]
    return list(ranks)


def _region_average_duration(
    region,
    ranks: list[int] | None = None,
) -> float:
    if ranks is None:
        selected_ranks = list(region.regions.keys())
    else:
        selected_ranks = [rank for rank in ranks if rank in region.regions]

    durations = [
        region.regions[rank].durations
        for rank in selected_ranks
        if region.regions[rank].durations.size
    ]
    if not durations:
        return float("nan")

    values = np.concatenate(durations)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _region_duration_values(
    region,
    ranks: list[int] | None = None,
) -> np.ndarray:
    if ranks is None:
        selected_ranks = list(region.regions.keys())
    else:
        selected_ranks = [rank for rank in ranks if rank in region.regions]

    durations = [
        region.regions[rank].durations
        for rank in selected_ranks
        if region.regions[rank].durations.size
    ]
    if not durations:
        return np.array([], dtype=float)
    return np.concatenate(durations)


def _stats_from_values(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "count": 0,
            "average_duration_seconds": None,
            "min_duration_seconds": None,
            "max_duration_seconds": None,
            "std_duration_seconds": None,
            "total_duration_seconds": None,
        }

    return {
        "count": int(values.size),
        "average_duration_seconds": float(np.mean(values)),
        "min_duration_seconds": float(np.min(values)),
        "max_duration_seconds": float(np.max(values)),
        "std_duration_seconds": float(np.std(values)),
        "total_duration_seconds": float(np.sum(values)),
    }


def _common_region_names(
    readers: Sequence[ProfilingH5Reader],
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
) -> list[str]:
    filtered_regions = [
        reader.get_regions(include=include, exclude=exclude) for reader in readers
    ]
    if not filtered_regions or not filtered_regions[0]:
        return []

    region_name_sets = [
        {candidate.name for candidate in regions} for regions in filtered_regions[1:]
    ]
    return [
        region.name
        for region in filtered_regions[0]
        if all(region.name in names for names in region_name_sets)
    ]


def collect_region_statistics(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
) -> dict:
    """Collect aggregate region-duration statistics for one or more profiling files."""
    readers = _as_readers(profiling_data)
    selected_ranks = _normalize_ranks(ranks)

    if labels is None:
        labels = _unique_labels([reader.file_path.stem for reader in readers])
    else:
        labels = list(labels)

    if len(labels) != len(readers):
        raise ValueError("labels must match the number of profiling files.")

    files_payload = []
    for label, reader in zip(labels, readers):
        regions = reader.get_regions(include=include, exclude=exclude)
        region_payload = {}
        for region in regions:
            values = _region_duration_values(region, selected_ranks)
            per_rank_stats = {}
            for rank in sorted(region.regions.keys()):
                if selected_ranks is not None and rank not in selected_ranks:
                    continue
                rank_values = region.regions[rank].durations
                per_rank_stats[str(rank)] = _stats_from_values(rank_values)
            region_payload[region.name] = {
                **_stats_from_values(values),
                "per_rank": per_rank_stats,
            }

        files_payload.append(
            {
                "label": label,
                "file_path": str(Path(reader.file_path).resolve()),
                "num_ranks": reader.num_ranks,
                "region_statistics": region_payload,
            }
        )

    return {
        "units": {"durations": "seconds"},
        "filters": {
            "include": include,
            "exclude": exclude,
            "ranks": selected_ranks,
        },
        "common_regions": (
            _common_region_names(readers, include=include, exclude=exclude)
            if len(readers) > 1
            else list(files_payload[0]["region_statistics"].keys())
        ),
        "files": files_payload,
    }


def write_region_statistics_json(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    filepath: str | Path,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
) -> dict:
    """Write aggregate region-duration statistics to a JSON file."""
    payload = collect_region_statistics(
        profiling_data=profiling_data,
        ranks=ranks,
        include=include,
        exclude=exclude,
        labels=labels,
    )
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _prepare_gantt_data(
    profiling_data: ProfilingH5Reader,
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

    return regions, normalized_ranks, profiling_data.minimum_start_time


def _draw_gantt_subplot(
    subplot,
    regions: list,
    ranks: list[int],
    first_start_time: float,
    hovertexts: list[str],
) -> None:
    num_ranks = len(ranks)
    colors = _palette_colors(len(regions))

    yticks = []
    yticklabels = []
    max_end = 0.0
    for i, region in enumerate(regions):
        for irank, rank in enumerate(ranks):
            starts = region[rank].start_times - first_start_time
            ends = region[rank].end_times - first_start_time
            y = i * num_ranks + irank
            for start, end in zip(starts, ends):
                _add_bar(
                    subplot,
                    hovertexts,
                    x0=start,
                    x1=end,
                    y_bottom=y - 0.5,
                    y_top=y + 0.5,
                    color=colors[i],
                    alpha=0.7,
                    hovertext=(
                        f"{region.name}<br>"
                        f"rank: {rank}<br>"
                        f"start: {start:.6f} s<br>"
                        f"end: {end:.6f} s<br>"
                        f"duration: {end - start:.6f} s"
                    ),
                )
                max_end = max(max_end, float(end))
            yticks.append(y)
            yticklabels.append(f"{region.name} (rank {rank})")

    subplot.set_yticks(yticks, yticklabels)
    # Pin the axes to the data so the bars fill the panel instead of being
    # framed by Plotly's default autorange padding.
    subplot.set_xlim(0, max_end)
    subplot.set_ylim(-0.5, len(yticks) - 0.5)


def _gantt_panel_height(num_rows: int) -> int:
    return max(220, 150 + 40 * num_rows)


def plot_gantt(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
) -> None:
    """
    Plot a Gantt chart of all (or selected) regions with per-rank lanes.

    Parameters
    ----------
    ranks : list[int] | None
        List of ranks to include. If None, include all ranks.
    regions : list[str] | str | None
        List of region names to plot, or a single region name as a string.
        If None, plot all regions.
    filepath : str | None
        Path to save the figure. If None, figure is not saved.
    show : bool
        Whether to display the plot. Default is False.
    """
    Canvas = _get_canvas_cls()
    readers = _as_readers(profiling_data)
    if not readers:
        raise ValueError("No profiling data provided.")

    prepared = []
    for reader in readers:
        regions, selected_ranks, first_start_time = _prepare_gantt_data(
            reader,
            ranks,
            include,
            exclude,
        )
        prepared.append((reader, regions, selected_ranks, first_start_time))

    labels = _unique_labels([reader.file_path.stem for reader, _, _, _ in prepared])
    if verbose:
        if len(prepared) == 1:
            print(f"Plotting Gantt chart for ranks: {prepared[0][2]}")
        else:
            print(f"Plotting combined Gantt chart for files: {', '.join(labels)}")

    canvas = Canvas(nrows=len(prepared), ncols=1)
    panel_row_counts = [
        len(regions) * len(selected_ranks) for _, regions, selected_ranks, _ in prepared
    ]

    hovertexts: list[str] = []
    for row, (label, (_, regions, selected_ranks, first_start_time)) in enumerate(
        zip(labels, prepared)
    ):
        title = "Profiling Gantt Chart" if len(prepared) == 1 else label
        subplot = canvas.add_subplot(
            row=row,
            col=0,
            title=title,
            xlabel="Time (seconds)",
            grid=True,
        )
        _draw_gantt_subplot(
            subplot, regions, selected_ranks, first_start_time, hovertexts
        )

    if len(prepared) > 1:
        canvas.suptitle("Combined Profiling Gantt Chart")

    fig = canvas.plot(backend="plotly")
    _style_bars(fig, hovertexts)
    fig.update_layout(
        width=1100,
        height=sum(_gantt_panel_height(n) for n in panel_row_counts),
    )

    _export_plotly_figure(fig, filepath, show)


def _build_call_stack_intervals(regions: list, rank: int) -> list[dict]:
    """Reconstruct per-call nesting depth for one rank from region intervals.

    Regions only store flat (start, end) pairs per call, with no explicit
    parent/child link. Since profiling scopes are always properly nested or
    sequential in a single rank's execution (recursive self-nesting included,
    see the recursion fix in region_profiler.py), the call tree can be
    rebuilt by treating each interval's parent as the innermost still-open
    interval that encloses it - the same rule used to match parentheses.
    """
    calls = []
    for region in regions:
        if rank not in region.regions:
            continue
        region_data = region.regions[rank]
        for start, end in zip(region_data.start_times, region_data.end_times):
            calls.append(
                {"name": region.name, "start": float(start), "end": float(end)}
            )

    # Longer-running calls that start at the same instant must be sorted
    # first so they end up enclosing (not enclosed by) shorter siblings.
    calls.sort(key=lambda call: (call["start"], -call["end"]))

    open_stack: list[dict] = []
    for call in calls:
        while open_stack and open_stack[-1]["end"] <= call["start"]:
            open_stack.pop()
        call["depth"] = len(open_stack)
        open_stack.append(call)

    return calls


def _draw_flame_subplot(subplot, calls: list[dict], hovertexts: list[str]) -> None:
    first_start = min(call["start"] for call in calls)
    total_span = max(call["end"] for call in calls) - first_start
    max_depth = max(call["depth"] for call in calls)

    region_names = sorted({call["name"] for call in calls})
    colors = _palette_colors(len(region_names))
    color_map = dict(zip(region_names, colors))

    for call in calls:
        start = call["start"] - first_start
        width = call["end"] - call["start"]
        _add_bar(
            subplot,
            hovertexts,
            x0=start,
            x1=start + width,
            y_bottom=call["depth"] - 0.5,
            y_top=call["depth"] + 0.5,
            color=color_map[call["name"]],
            alpha=0.85,
            hovertext=(
                f"{call['name']}<br>"
                f"depth: {call['depth']}<br>"
                f"start: {start:.6f} s<br>"
                f"end: {start + width:.6f} s<br>"
                f"duration: {width:.6f} s"
            ),
        )
        if total_span > 0 and width / total_span > 0.02:
            subplot.text(
                start + width / 2,
                call["depth"],
                call["name"],
                ha="center",
                va="center",
                fontsize=9,
            )

    subplot.set_yticks(list(range(max_depth + 1)))
    # Pin the axes to the data so the bars fill the panel instead of being
    # framed by Plotly's default autorange padding.
    subplot.set_xlim(0, total_span)
    subplot.set_ylim(-0.5, max_depth + 0.5)


def _flame_panel_height(max_depth: int) -> int:
    return max(220, 150 + 45 * (max_depth + 1))


def plot_flame(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
) -> None:
    """
    Plot a flame graph reconstructing the call stack from region timings.

    Unlike ``plot_gantt``, which lays out one row per region name, this
    reconstructs call nesting from each call's (start, end) interval, so
    recursive calls into the same region are shown at their correct depth,
    producing a classic flame-graph shape.

    Parameters
    ----------
    ranks : list[int] | int | None
        Ranks to plot, one flame graph per rank. Defaults to rank 0 only,
        since a flame graph represents a single execution's call stack.
    filepath : str | None
        Path to save the figure. If None, figure is not saved.
    show : bool
        Whether to display the plot. Default is False.
    """
    Canvas = _get_canvas_cls()
    readers = _as_readers(profiling_data)
    if not readers:
        raise ValueError("No profiling data provided.")

    normalized_ranks = _normalize_ranks(ranks) if ranks is not None else [0]

    prepared = []
    for reader in readers:
        regions = reader.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        for rank in normalized_ranks:
            if rank < 0 or rank >= reader.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = _build_call_stack_intervals(regions, rank)
            if calls:
                prepared.append((reader, rank, calls))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    if verbose:
        print(
            "Plotting flame graph for: "
            + ", ".join(
                f"{reader.file_path.stem} (rank {rank})" for reader, rank, _ in prepared
            )
        )

    canvas = Canvas(nrows=len(prepared), ncols=1)
    panel_max_depths = [
        max(call["depth"] for call in calls) for _, _, calls in prepared
    ]

    hovertexts: list[str] = []
    for row, (reader, rank, calls) in enumerate(prepared):
        subplot = canvas.add_subplot(
            row=row,
            col=0,
            title=f"{reader.file_path.stem} (rank {rank})",
            xlabel="Time (seconds)",
            ylabel="Call depth",
            grid=True,
        )
        _draw_flame_subplot(subplot, calls, hovertexts)

    if len(prepared) > 1:
        canvas.suptitle("Flame Graphs")

    fig = canvas.plot(backend="plotly")
    _style_bars(fig, hovertexts)
    fig.update_layout(
        width=1100,
        height=sum(_flame_panel_height(depth) for depth in panel_max_depths),
    )

    _export_plotly_figure(fig, filepath, show)


_DURATION_METRICS: dict[str, tuple[str, str]] = {
    "avg": ("average_duration_seconds", "Average duration per call (seconds)"),
    "min": ("min_duration_seconds", "Minimum duration per call (seconds)"),
    "max": ("max_duration_seconds", "Maximum duration per call (seconds)"),
    "total": ("total_duration_seconds", "Total duration (seconds)"),
}


def _region_metric_value(
    region,
    metric_key: str,
    ranks: list[int] | None = None,
) -> float:
    values = _region_duration_values(region, ranks=ranks)
    stats = _stats_from_values(values)
    stat_value = stats[metric_key]
    return float("nan") if stat_value is None else stat_value


def _metric_filepath(filepath: str, metric_key: str, single_metric: bool) -> str:
    if single_metric:
        return filepath
    base, ext = os.path.splitext(filepath)
    return f"{base}_{metric_key}{ext}"


def plot_durations(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
    metrics: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Plot duration bar charts for one or more profiling files.

    Each requested metric is rendered as its own separate figure.

    Args:
        metrics: Which statistics to plot, any of "avg", "min", "max", "total".
            Defaults to all four.
        filepath: Base output path. When multiple metrics are plotted, each
            figure is saved with the metric name inserted before the
            extension (e.g. "durations_plot.png" -> "durations_plot_avg.png").

    Returns:
        List of filepaths that were written (empty if filepath is None).
    """
    import plotly.graph_objects as go

    Canvas = _get_canvas_cls()
    readers = _as_readers(profiling_data)
    ranks = _normalize_ranks(ranks)

    if metrics is None:
        metric_keys = list(_DURATION_METRICS)
    elif isinstance(metrics, str):
        metric_keys = [metrics]
    else:
        metric_keys = list(metrics)

    unknown_metrics = [key for key in metric_keys if key not in _DURATION_METRICS]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metric(s) {unknown_metrics}. "
            f"Valid options are: {list(_DURATION_METRICS)}"
        )

    if labels is None:
        labels = _unique_labels([reader.file_path.stem for reader in readers])
    else:
        labels = list(labels)

    if len(labels) != len(readers):
        raise ValueError("labels must match the number of profiling files.")

    region_names = _common_region_names(readers, include=include, exclude=exclude)
    if not region_names:
        raise ValueError("No regions matched the selected filters.")

    if verbose:
        print(
            f"Plotting duration comparison ({', '.join(metric_keys)}) "
            f"for files: {', '.join(labels)}"
        )

    x = np.arange(len(region_names))
    num_readers = len(readers)
    width = min(0.8 / max(num_readers, 1), 0.35)
    colors = _palette_colors(num_readers)
    offset_start = -0.5 * width * (num_readers - 1)
    fig_width = max(700, 70 * len(region_names) + 300)
    fig_height = max(450, 300 + 40 * num_readers)

    saved_paths: list[str] = []

    for metric_key in metric_keys:
        stat_key, ylabel = _DURATION_METRICS[metric_key]

        values = [
            [
                _region_metric_value(
                    reader.get_region(region_name), stat_key, ranks=ranks
                )
                for region_name in region_names
            ]
            for reader in readers
        ]

        canvas = Canvas(nrows=1, ncols=1)
        subplot = canvas.add_subplot(
            title=f"Region duration comparison ({metric_key})",
            ylabel=ylabel,
            grid=True,
            legend=num_readers > 1,
        )
        for idx, (label, file_values) in enumerate(zip(labels, values)):
            offsets = x + offset_start + idx * width
            subplot.bar(
                offsets,
                file_values,
                color=colors[idx],
                label=label if num_readers > 1 else None,
            )
        subplot.set_xticks(x.tolist(), region_names)

        fig = canvas.plot(backend="plotly")
        for trace in fig.data:
            if isinstance(trace, go.Bar):
                trace.width = width
        fig.update_layout(width=fig_width, height=fig_height)

        metric_filepath = None
        if filepath:
            metric_filepath = _metric_filepath(
                filepath, metric_key, single_metric=len(metric_keys) == 1
            )
            saved_paths.append(metric_filepath)

        _export_plotly_figure(fig, metric_filepath, show)

    return saved_paths


def plot_speedup(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
) -> None:
    """Plot scope speedup versus MPI rank count across one or more files.

    The speedup is computed from matching scopes' average per-call durations
    and normalized against the smallest MPI rank count present in the inputs.
    """
    Canvas = _get_canvas_cls()
    readers = _as_readers(profiling_data)
    if len(readers) < 2:
        raise ValueError("Speedup plot requires at least two profiling files.")

    region_names = _common_region_names(readers, include=include, exclude=exclude)
    if not region_names:
        raise ValueError("No regions matched the selected filters.")

    rank_counts = sorted({reader.num_ranks for reader in readers})
    if verbose:
        print(
            "Plotting speedup comparison for files with ranks: "
            + ", ".join(map(str, rank_counts))
        )

    duration_samples: dict[str, dict[int, list[float]]] = {
        region_name: defaultdict(list) for region_name in region_names
    }
    for reader in readers:
        for region_name in region_names:
            duration = _region_average_duration(
                reader.get_region(region_name),
                ranks=ranks,
            )
            if np.isfinite(duration) and duration > 0:
                duration_samples[region_name][reader.num_ranks].append(duration)

    baseline_ranks = rank_counts[0]
    colors = _palette_colors(len(region_names))

    canvas = Canvas(nrows=1, ncols=1)
    subplot = canvas.add_subplot(
        title=f"Region speedup scaling (baseline: {baseline_ranks} ranks)",
        xlabel="MPI ranks",
        ylabel="Speedup",
        grid=True,
        legend=True,
    )

    plotted = 0
    for idx, region_name in enumerate(region_names):
        region_counts = duration_samples[region_name]
        baseline_samples = region_counts.get(baseline_ranks, [])
        if not baseline_samples:
            continue

        baseline_duration = float(np.mean(baseline_samples))
        if not np.isfinite(baseline_duration) or baseline_duration <= 0:
            continue

        x_values = []
        speedups = []
        for rank_count in rank_counts:
            samples = region_counts.get(rank_count, [])
            if not samples:
                continue
            mean_duration = float(np.mean(samples))
            if not np.isfinite(mean_duration) or mean_duration <= 0:
                continue
            x_values.append(rank_count)
            speedups.append(baseline_duration / mean_duration)

        if not x_values:
            continue

        plotted += 1
        subplot.plot(
            x_values,
            speedups,
            marker="o",
            color=colors[idx],
            label=region_name,
        )

    if plotted == 0:
        raise ValueError("No valid speedup data could be computed.")

    x_line = np.array(rank_counts, dtype=float)
    subplot.plot(
        x_line,
        x_line / baseline_ranks,
        linestyle="dashed",
        color="black",
        label="Optimal speedup",
    )
    subplot.set_xticks(rank_counts)

    fig_width = max(700, 90 * len(rank_counts) + 300)
    fig_height = max(450, 300 + 35 * len(region_names))

    fig = canvas.plot(backend="plotly")
    fig.update_layout(width=fig_width, height=fig_height)

    _export_plotly_figure(fig, filepath, show)
