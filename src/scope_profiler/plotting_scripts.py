"""Plotting utilities for visualizing profiling data."""

from collections.abc import Sequence

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


def plot_gantt(
    profiling_data: ProfilingH5Reader,
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
    first_start_time = profiling_data.minimum_start_time
    regions = profiling_data.get_regions(include=include, exclude=exclude)
    if not regions:
        raise ValueError("No regions matched the selected filters.")

    ranks = _normalize_ranks(ranks)
    if ranks is None:
        ranks = list(range(profiling_data.num_ranks))
    else:
        invalid_ranks = [
            rank for rank in ranks if rank < 0 or rank >= profiling_data.num_ranks
        ]
        if invalid_ranks:
            raise ValueError(f"Invalid ranks requested: {invalid_ranks}")

    if verbose:
        print(f"Plotting Gantt chart for ranks: {ranks}")
    num_ranks = len(ranks)
    fig, ax = plt.subplots(figsize=(12, 1 * len(regions) * num_ranks))
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
    ax.set_title("Profiling Gantt Chart")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

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
        labels = [reader.file_path.stem for reader in readers]
        label_counts = {}
        unique_labels = []
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            if label_counts[label] > 1:
                unique_labels.append(f"{label} ({label_counts[label]})")
            else:
                unique_labels.append(label)
        labels = unique_labels
    else:
        labels = list(labels)

    if len(labels) != len(readers):
        raise ValueError("labels must match the number of profiling files.")

    filtered_regions = [
        reader.get_regions(include=include, exclude=exclude) for reader in readers
    ]
    if not filtered_regions[0]:
        raise ValueError("No regions matched the selected filters.")

    region_name_sets = [
        {candidate.name for candidate in regions} for regions in filtered_regions[1:]
    ]
    region_names = [
        region.name
        for region in filtered_regions[0]
        if all(region.name in names for names in region_name_sets)
    ]
    if not region_names:
        raise ValueError("No common regions matched the selected filters.")

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
