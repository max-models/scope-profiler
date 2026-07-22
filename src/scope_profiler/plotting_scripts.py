"""Plotting utilities for visualizing profiling data."""

import csv
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.h5reader import ProfilingH5Reader


def _write_csv(
    filepath: str | Path, header: Sequence[str], rows: Sequence[Sequence]
) -> None:
    """Write rows of plotting data to a plain-text CSV file.

    Used by the ``data_filepath`` argument of the ``plot_*`` functions so the
    exact data behind a chart can be re-parsed and re-plotted later without
    needing the original HDF5 file.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(filepath: str | Path, payload: dict) -> None:
    """Write the exact data behind a plot to a JSON file.

    Used by the ``data_filepath``/``data_format="json"`` arguments of the
    ``plot_*`` functions so charts can be reconstructed (e.g. with Plotly)
    later without needing the original HDF5 file or matplotlib.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _to_hex(color) -> str:
    """Convert a matplotlib color (e.g. an RGBA tuple) to a ``#rrggbb`` string."""
    from matplotlib.colors import to_hex

    return to_hex(color)


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install scope-profiler[dev] or matplotlib."
        ) from exc
    return plt


DEFAULT_CMAP = "tab20"


def _get_cmap(plt, cmap: str):
    """Resolve a colormap name to a callable, with a friendly error on typos."""
    try:
        return plt.get_cmap(cmap)
    except ValueError as exc:
        raise ValueError(
            f"Unknown colormap {cmap!r}. See "
            "https://matplotlib.org/stable/users/explain/colors/colormaps.html "
            "for valid names."
        ) from exc


def _region_color_map(plt, region_names, cmap: str = DEFAULT_CMAP) -> dict:
    """Assign each region name a stable color from a canonical sorted order.

    Sorting by name (rather than data/insertion order) means gantt and flame
    charts built from the same region names get identical colors, even
    though they derive their region lists independently and gantt charts
    order regions by first-appearance while flame charts don't.
    """
    names = sorted(set(region_names))
    colors = _get_cmap(plt, cmap)(np.linspace(0, 1, len(names)))
    return dict(zip(names, colors))


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


def _draw_gantt_axes(
    plt,
    ax,
    regions: list,
    ranks: list[int],
    first_start_time: float,
) -> None:
    num_ranks = len(ranks)

    for i, region in enumerate(regions):
        for irank, rank in enumerate(ranks):
            starts = region[rank].start_times - first_start_time
            ends = region[rank].end_times - first_start_time
            y = i * num_ranks + irank
            for start, end in zip(starts, ends):
                ax.barh(
                    y=y,
                    width=end - start,
                    left=start,
                    height=1.0,
                    color=region.color,
                    edgecolor="black",
                    alpha=0.7,
                )

    yticks = []
    yticklabels = []
    for i, region in enumerate(regions):
        region_name = region.name
        for irank, rank in enumerate(ranks):
            yticks.append(i * num_ranks + irank)
            yticklabels.append(f"{region_name} (rank {rank})")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time (seconds)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)


def plot_gantt(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
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
    cmap : str
        Name of the matplotlib colormap used to color regions (default: "tab20").
    data_filepath : str | Path | None
        If given, write the underlying (file, rank, region, start, end)
        intervals plotted here to this path, so the chart can be
        reconstructed later without the original HDF5 file.
    data_format : str
        Format for ``data_filepath``: "csv" (default) or "json". The JSON
        payload additionally includes a "colors" map of region name to
        ``#rrggbb`` string, matching the colors used in this plot.
    """
    plt = _get_pyplot()
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

    color_map = _region_color_map(
        plt,
        (region.name for _, regions, _, _ in prepared for region in regions),
        cmap=cmap,
    )
    for _, regions, _, _ in prepared:
        for region in regions:
            region.color = color_map[region.name]

    labels = _unique_labels([reader.file_path.stem for reader, _, _, _ in prepared])

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

    if len(prepared) == 1:
        _, regions, selected_ranks, first_start_time = prepared[0]
        fig, ax = plt.subplots(
            figsize=(12, max(3.0, 1 * len(regions) * len(selected_ranks)))
        )
        _draw_gantt_axes(plt, ax, regions, selected_ranks, first_start_time)
        ax.set_title("Profiling Gantt Chart")
    else:
        subplot_heights = [
            max(2.5, 1 * len(regions) * len(selected_ranks))
            for _, regions, selected_ranks, _ in prepared
        ]
        fig, axes = plt.subplots(
            nrows=len(prepared),
            ncols=1,
            figsize=(12, max(4.0, sum(subplot_heights))),
        )
        axes = np.atleast_1d(axes).ravel()
        for ax, label, (_, regions, selected_ranks, first_start_time) in zip(
            axes,
            labels,
            prepared,
        ):
            _draw_gantt_axes(plt, ax, regions, selected_ranks, first_start_time)
            ax.set_title(label)
        fig.suptitle("Combined Profiling Gantt Chart")
        fig.tight_layout(rect=(0, 0, 1, 0.98))

    if filepath:
        plt.savefig(filepath, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


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
                {
                    "name": region.name,
                    "start": float(start),
                    "end": float(end),
                    "color": region.color,
                }
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


def _draw_flame_axis(plt, ax, calls: list[dict]) -> None:
    first_start = min(call["start"] for call in calls)
    total_span = max(call["end"] for call in calls) - first_start
    max_depth = max(call["depth"] for call in calls)

    for call in calls:
        start = call["start"] - first_start
        width = call["end"] - call["start"]
        ax.barh(
            y=call["depth"],
            width=width,
            left=start,
            height=1.0,
            color=call["color"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        if total_span > 0 and width / total_span > 0.02:
            ax.text(
                start + width / 2,
                call["depth"],
                call["name"],
                ha="center",
                va="center",
                fontsize=7,
                clip_on=True,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Call depth")
    ax.set_yticks(range(max_depth + 1))
    ax.set_ylim(-0.5, max_depth + 0.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)


def plot_flame(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
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
    cmap : str
        Name of the matplotlib colormap used to color regions (default: "tab20").
        Pass the same value used for ``plot_gantt`` to keep colors matching.
    data_filepath : str | Path | None
        If given, write the reconstructed (file, rank, region, depth, start,
        end) call intervals to this path, so the flame graph can be
        reconstructed later without the original HDF5 file.
    data_format : str
        Format for ``data_filepath``: "csv" (default) or "json". The JSON
        payload additionally includes a "colors" map of region name to
        ``#rrggbb`` string, matching the colors used in this plot.
    """
    plt = _get_pyplot()
    readers = _as_readers(profiling_data)
    if not readers:
        raise ValueError("No profiling data provided.")

    normalized_ranks = _normalize_ranks(ranks) if ranks is not None else [0]

    reader_regions = []
    all_region_names: set[str] = set()
    for reader in readers:
        regions = reader.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        all_region_names.update(region.name for region in regions)
        reader_regions.append((reader, regions))

    color_map = _region_color_map(plt, all_region_names, cmap=cmap)
    for _, regions in reader_regions:
        for region in regions:
            region.color = color_map[region.name]

    prepared = []
    for reader, regions in reader_regions:
        for rank in normalized_ranks:
            if rank < 0 or rank >= reader.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = _build_call_stack_intervals(regions, rank)
            if calls:
                prepared.append((reader, rank, calls))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    if data_filepath:
        labels = _unique_labels([reader.file_path.stem for reader, _, _ in prepared])
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
                            "region": call["name"],
                            "depth": call["depth"],
                            "start_seconds": call["start"],
                            "end_seconds": call["end"],
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
                            call["name"],
                            call["depth"],
                            call["start"],
                            call["end"],
                        ]
                    )
            _write_csv(
                data_filepath,
                ["file", "rank", "region", "depth", "start_seconds", "end_seconds"],
                rows,
            )

    if verbose:
        print(
            "Plotting flame graph for: "
            + ", ".join(
                f"{reader.file_path.stem} (rank {rank})" for reader, rank, _ in prepared
            )
        )

    subplot_heights = [
        max(2.0, 0.6 * (max(call["depth"] for call in calls) + 1))
        for _, _, calls in prepared
    ]
    fig, axes = plt.subplots(
        nrows=len(prepared),
        ncols=1,
        figsize=(12, max(3.0, sum(subplot_heights))),
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (reader, rank, calls) in zip(axes, prepared):
        _draw_flame_axis(plt, ax, calls)
        ax.set_title(f"{reader.file_path.stem} (rank {rank})")

    if len(prepared) == 1:
        fig.tight_layout()
    else:
        fig.suptitle("Flame Graphs")
        fig.tight_layout(rect=(0, 0, 1, 0.98))

    if filepath:
        plt.savefig(filepath, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


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
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
) -> list[str]:
    """Plot duration bar charts for one or more profiling files.

    Each requested metric is rendered as its own separate figure.

    Args:
        metrics: Which statistics to plot, any of "avg", "min", "max", "total".
            Defaults to all four.
        filepath: Base output path. When multiple metrics are plotted, each
            figure is saved with the metric name inserted before the
            extension (e.g. "durations_plot.png" -> "durations_plot_avg.png").
        cmap: Name of the matplotlib colormap used to color files (default:
            "tab20"). Bars here are colored per-file, not per-region, so this
            is independent of the region colors used by ``plot_gantt``/
            ``plot_flame``.
        data_filepath: If given, write the (file, region, metric, value) bars
            plotted across all requested metrics to this single path.
        data_format: Format for ``data_filepath``: "csv" (default) or "json".
            The JSON payload additionally includes a "colors" map of file
            label to ``#rrggbb`` string, matching the colors used in this
            plot, and the list of "metrics" plotted.

    Returns:
        List of filepaths that were written (empty if filepath is None).
    """
    plt = _get_pyplot()
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
    colors = _get_cmap(plt, cmap)(np.linspace(0, 1, max(num_readers, 1)))
    fig_width = max(10, 0.85 * len(region_names) + 2)
    fig_height = max(4.5, 2.5 + 0.35 * num_readers)
    offset_start = -0.5 * width * (num_readers - 1)

    saved_paths: list[str] = []
    figs = []
    data_rows = []

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

        if data_filepath:
            for label, file_values in zip(labels, values):
                for region_name, value in zip(region_names, file_values):
                    data_rows.append([label, region_name, metric_key, value])

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        for idx, (label, file_values) in enumerate(zip(labels, values)):
            offsets = x + offset_start + idx * width
            ax.bar(
                offsets,
                file_values,
                width=width,
                label=label if num_readers > 1 else None,
                color=colors[idx],
                edgecolor="black",
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(region_names, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Region duration comparison ({metric_key})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        if num_readers > 1:
            ax.legend(frameon=False)
        fig.tight_layout()

        if filepath:
            metric_filepath = _metric_filepath(
                filepath, metric_key, single_metric=len(metric_keys) == 1
            )
            fig.savefig(metric_filepath, dpi=300)
            saved_paths.append(metric_filepath)

        figs.append(fig)

    if data_filepath:
        if data_format == "json":
            bars = [
                {
                    "file": file,
                    "region": region,
                    "metric": metric,
                    "value_seconds": value,
                }
                for file, region, metric, value in data_rows
            ]
            colors_map = {label: _to_hex(color) for label, color in zip(labels, colors)}
            _write_json(
                data_filepath,
                {"bars": bars, "colors": colors_map, "metrics": metric_keys},
            )
        else:
            _write_csv(
                data_filepath, ["file", "region", "metric", "value_seconds"], data_rows
            )

    if show:
        plt.show()
    for fig in figs:
        plt.close(fig)

    return saved_paths


def plot_speedup(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
) -> None:
    """Plot scope speedup versus MPI rank count across one or more files.

    The speedup is computed from matching scopes' average per-call durations
    and normalized against the smallest MPI rank count present in the inputs.

    Parameters
    ----------
    cmap : str
        Name of the matplotlib colormap used to color regions (default: "tab20").
    data_filepath : str | Path | None
        If given, write the (region, rank_count, speedup) points plotted
        here to this path.
    data_format : str
        Format for ``data_filepath``: "csv" (default) or "json". The JSON
        payload additionally includes a "colors" map of region name to
        ``#rrggbb`` string, matching the colors used in this plot.
    """
    plt = _get_pyplot()
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
    colors = _get_cmap(plt, cmap)(np.linspace(0, 1, len(region_names)))
    fig_width = max(10, 1.2 * len(rank_counts) + 3)
    fig_height = max(4.5, 2.8 + 0.35 * len(region_names))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plotted = 0
    data_rows = []
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
        ax.plot(
            x_values,
            speedups,
            marker="o",
            linewidth=1.8,
            color=colors[idx],
            label=region_name,
        )
        if data_filepath:
            for rank_count, speedup in zip(x_values, speedups):
                data_rows.append([region_name, rank_count, speedup])

    if plotted == 0:
        raise ValueError("No valid speedup data could be computed.")

    if data_filepath:
        if data_format == "json":
            points = [
                {"region": region, "rank_count": rank_count, "speedup": speedup}
                for region, rank_count, speedup in data_rows
            ]
            colors_map = {
                name: _to_hex(color) for name, color in zip(region_names, colors)
            }
            _write_json(data_filepath, {"points": points, "colors": colors_map})
        else:
            _write_csv(data_filepath, ["region", "rank_count", "speedup"], data_rows)

    x_line = np.array(rank_counts, dtype=float)
    ax.plot(
        x_line,
        x_line / baseline_ranks,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Optimal speedup",
    )
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Region speedup scaling (baseline: {baseline_ranks} ranks)")
    ax.set_xticks(rank_counts)
    ax.grid(True, axis="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
