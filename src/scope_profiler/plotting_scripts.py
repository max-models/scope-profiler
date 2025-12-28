import matplotlib.pyplot as plt
from scope_profiler.h5reader import ProfilingH5Reader
import numpy as np


def plot_gantt(
    profiling_data: ProfilingH5Reader,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
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
    regions = profiling_data.get_regions(include=include, exclude=exclude)
    num_ranks = profiling_data.num_ranks

    if ranks is None:
        ranks = list(range(num_ranks))
    elif isinstance(ranks, int):
        ranks = [ranks]
    else:
        assert all(0 <= r < num_ranks for r in ranks), "Invalid rank in ranks list."

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
        for r in ranks:
            starts = region[r].start_times
            ends = region[r].end_times
            y = i * num_ranks + r  # stack ranks vertically within the region
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
        for r in range(num_ranks):
            yticks.append(i * num_ranks + r)
            yticklabels.append(f"{region_name} (rank {r})")

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


def plot_durations(
    profiling_data: ProfilingH5Reader,
    ranks: list[int] | None = None,
    regions: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    bins: int = 30,
) -> None:
    """
    Plot duration histograms for each region with per-rank lanes.

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
    bins : int
        Number of histogram bins. Default is 30.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if regions is None:
        regions = list(profiling_data._region_dict.keys())
    elif isinstance(regions, str):
        regions = [regions]

    # Determine number of ranks from first region
    first_region = profiling_data._region_dict[regions[0]]
    num_ranks = len(first_region.keys())
    if ranks is None:
        ranks = list(range(num_ranks))

    # Compute figure height: 1 unit per rank per region
    fig, axes = plt.subplots(
        nrows=len(regions), ncols=1, figsize=(10, 1 * len(regions) * num_ranks)
    )
    if len(regions) == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, num_ranks))

    for ax, region_name in zip(axes, regions):
        region = profiling_data._region_dict[region_name]

        # Determine max y for proper stacking
        y_positions = {r: r for r in ranks}  # rank -> vertical offset within region
        max_y = max(y_positions.values()) + 1

        for r in ranks:
            subregion = region[r]
            starts = subregion.start_times
            ends = subregion.end_times
            y = y_positions[r]
            if len(starts) == 0:
                continue
            ax.hist(
                starts,  # use start times as representative events for histogram
                bins=bins,
                alpha=0.6,
                color=colors[r],
                label=f"Rank {r}",
            )

        ax.set_title(f"Region: {region_name}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.4)

    fig.suptitle("Region Duration Distributions per Rank", fontsize=14)
    fig.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300)
    if show:
        plt.show()
