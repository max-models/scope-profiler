import matplotlib.pyplot as plt
import numpy as np

from scope_profiler.h5reader import ProfilingH5Reader


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
    first_start_time = profiling_data.minimum_start_time
    regions = profiling_data.get_regions(include=include, exclude=exclude)

    if ranks is None:
        ranks = list(range(profiling_data.num_ranks))
    elif isinstance(ranks, int):
        ranks = [ranks]
    else:
        num_ranks = profiling_data.num_ranks
        assert all(0 <= r < num_ranks for r in ranks), "Invalid rank in ranks list."
    if verbose:
        print(f"Plotting Gantt chart for ranks: {ranks}")
    num_ranks = len(ranks)
    # Compute figure height: 0.5 per rank per region
    fig, ax = plt.subplots(figsize=(12, 1 * len(regions) * num_ranks))
    colors = plt.cm.tab20(np.linspace(0, 1, len(regions)))

    # Draw bars
    for i, region in enumerate(
        profiling_data.get_regions(
            include=include,
            exclude=exclude,
        )
    ):
        for irank, rank in enumerate(ranks):
            starts = region[rank].start_times - first_start_time
            ends = region[rank].end_times - first_start_time
            y = i * num_ranks + irank  # stack ranks vertically within the region
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

    # Configure y-axis labels
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
