"""Plotting utilities for visualizing profiling data."""

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.h5reader import ProfilingH5Reader


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install scope-profiler[dev] or matplotlib."
        ) from exc
    return plt


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
    colors = plt.cm.tab20(np.linspace(0, 1, len(regions)))

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
                    color=colors[i],
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

    labels = _unique_labels([reader.file_path.stem for reader, _, _, _ in prepared])
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


def plot_durations(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
) -> None:
    """Plot average-duration bar charts for one or more profiling files."""
    plt = _get_pyplot()
    readers = _as_readers(profiling_data)
    ranks = _normalize_ranks(ranks)

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
        print(f"Plotting duration comparison for files: {', '.join(labels)}")

    values = [
        [
            _region_average_duration(reader.get_region(region_name), ranks=ranks)
            for region_name in region_names
        ]
        for reader in readers
    ]

    x = np.arange(len(region_names))
    num_readers = len(readers)
    width = min(0.8 / max(num_readers, 1), 0.35)
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_readers, 1)))
    fig_width = max(10, 0.85 * len(region_names) + 2)
    fig_height = max(4.5, 2.5 + 0.35 * num_readers)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    offset_start = -0.5 * width * (num_readers - 1)
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
    ax.set_ylabel("Average duration per call (seconds)")
    ax.set_title("Region duration comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if num_readers > 1:
        ax.legend(frameon=False)
    fig.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


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
    colors = plt.cm.tab20(np.linspace(0, 1, len(region_names)))
    fig_width = max(10, 1.2 * len(rank_counts) + 3)
    fig_height = max(4.5, 2.8 + 0.35 * len(region_names))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

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
        ax.plot(
            x_values,
            speedups,
            marker="o",
            linewidth=1.8,
            color=colors[idx],
            label=region_name,
        )

    if plotted == 0:
        raise ValueError("No valid speedup data could be computed.")

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
