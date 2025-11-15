from pathlib import Path
from typing import Any, Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np


class Region:
    def __init__(self, start_times: np.ndarray, end_times: np.ndarray) -> None:
        self._start_times = start_times
        self._end_times = end_times
        self._durations = end_times - start_times

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the region’s statistics as a dictionary."""
        return {
            "num_calls": self.num_calls,
            "total_duration": self.total_duration,
            "average_duration": self.average_duration,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "std_duration": self.std_duration,
        }

    @property
    def start_times(self) -> np.ndarray:
        return self._start_times / 1e9

    @property
    def end_times(self) -> np.ndarray:
        return self._end_times / 1e9

    @property
    def durations(self) -> np.ndarray:
        return self._durations / 1e9

    @property
    def num_calls(self) -> int:
        """Number of recorded calls."""
        return len(self._durations)

    @property
    def total_duration(self) -> float:
        """Total time spent in this region (sum of all durations)."""
        return float(np.sum(self._durations)) if self.num_calls else 0.0

    @property
    def average_duration(self) -> float:
        """Average duration per call."""
        return float(np.mean(self._durations)) if self.num_calls else 0.0

    @property
    def min_duration(self) -> float:
        """Minimum duration among all calls."""
        return float(np.min(self._durations)) if self.num_calls else 0.0

    @property
    def max_duration(self) -> float:
        """Maximum duration among all calls."""
        return float(np.max(self._durations)) if self.num_calls else 0.0

    @property
    def std_duration(self) -> float:
        """Standard deviation of durations."""
        return float(np.std(self._durations)) if self.num_calls else 0.0

    def __repr__(self) -> str:
        """Print summaries for all regions in the file."""
        # print(f"\nProfiling data summary for: {self.file_path}")
        _out = "-" * 60 + "\n"
        stats = self.get_summary()
        for key, value in stats.items():
            _out += f"  {key:>18}: {value}\n"
        _out += "-" * 60 + "\n\n"
        return _out


class ProfilingH5Reader:
    """
    Reads profiling data stored by ProfileRegion in an HDF5 file.
    """

    def __init__(self, file_path: str | Path):
        self._file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        # Read the file
        self._region_dict = {}

        with h5py.File(self.file_path, "r") as f:
            # Iterate over all rank groups
            for rank_group_name, rank_group in f.items():
                # print(f"{rank_group_name = }")
                # print(rank_group_name, rank_group)
                rank = int(rank_group_name.replace("rank", ""))
                if "regions" not in rank_group:
                    continue
                regions_group = rank_group["regions"]

                for region_name, region_grp in regions_group.items():
                    starts = region_grp["start_times"][()]
                    ends = region_grp["end_times"][()]
                    # print(f"{region_name = }")
                    # Merge if region already exists (from another rank)
                    if region_name in self._region_dict:
                        self._region_dict[region_name][rank] = Region(starts, ends)
                    else:
                        self._region_dict[region_name] = {rank: Region(starts, ends)}
        # print(f"{self._region_dict["main"].keys() = }")

    def get_region(self, region_name: str) -> Region:
        return self._region_dict[region_name]

    def plot_gantt(
        self,
        ranks: list[int] | None = None,
        regions: list[str] | str | None = None,
        filepath: str | None = None,
        show: bool = False,
    ) -> None:
        """
        Plot a Gantt chart of all (or selected) regions with per-rank lanes.

        Parameters
        ----------
        regions : list[str] | None
            List of region names to plot. If None, plot all.
        ranks : list[int] | None
            List of ranks to include. If None, include all ranks.
        """
        if regions is None:
            regions = list(self._region_dict.keys())
        elif isinstance(regions, str):
            regions = [regions]

        # Determine number of ranks from the first region

        # print(f"{self._region_dict = }")
        first_region = self._region_dict[regions[0]]
        n_ranks = len(first_region.keys())

        if ranks is None:
            ranks = list(range(n_ranks))

        # Compute figure height: 0.5 per rank per region
        fig, ax = plt.subplots(figsize=(12, 1 * len(regions) * n_ranks))
        colors = plt.cm.tab20(np.linspace(0, 1, len(regions)))

        # Draw bars
        for i, region_name in enumerate(regions):
            region = self._region_dict[region_name]
            for r in ranks:
                starts = region[r].start_times
                ends = region[r].end_times
                y = i * n_ranks + r  # stack ranks vertically within the region
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
        for i, region_name in enumerate(regions):
            for r in range(n_ranks):
                yticks.append(i * n_ranks + r)
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
        self,
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
        regions : list[str] | None
            List of region names to plot. If None, plot all.
        ranks : list[int] | None
            List of ranks to include. If None, include all ranks.
        bins : int
            Number of histogram bins.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if regions is None:
            regions = list(self._region_dict.keys())
        elif isinstance(regions, str):
            regions = [regions]

        # Determine number of ranks from first region
        first_region = self._region_dict[regions[0]]
        n_ranks = len(first_region.keys())
        if ranks is None:
            ranks = list(range(n_ranks))

        # Compute figure height: 1 unit per rank per region
        fig, axes = plt.subplots(
            nrows=len(regions), ncols=1, figsize=(10, 1 * len(regions) * n_ranks)
        )
        if len(regions) == 1:
            axes = [axes]

        colors = plt.cm.tab20(np.linspace(0, 1, n_ranks))

        for ax, region_name in zip(axes, regions):
            region = self._region_dict[region_name]

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

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def regions(self) -> List[Region]:
        return self._regions

    def __repr__(self) -> str:
        _out = ""
        for region_name, region in self._region_dict.items():
            _out += f"Region: {region_name}\n"
            _out += str(region[0])
        return _out
