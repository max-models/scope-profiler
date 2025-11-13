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

    def __repr__(self) -> str:
        """Print summaries for all regions in the file."""
        # print(f"\nProfiling data summary for: {self.file_path}")
        _out = "-" * 60 + "\n"
        stats = self.get_summary()
        for key, value in stats.items():
            _out += f"  {key:>18}: {value}\n"
        _out += "-" * 60 + "\n\n"
        return _out

    @property
    def start_times(self) -> np.ndarray:
        return self._start_times

    @property
    def end_times(self) -> np.ndarray:
        return self._end_times

    @property
    def durations(self) -> np.ndarray:
        return self._durations

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


class ProfilingH5Reader:
    """
    Reads profiling data stored by ProfileRegion in an HDF5 file.
    """

    def __init__(self, file_path: str | Path):
        self._file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        # Verify it's an HDF5 file
        try:
            with h5py.File(self.file_path, "r") as f:
                if "regions" not in f:
                    raise ValueError("Invalid profiling file: missing 'regions' group.")
        except OSError as e:
            raise ValueError(f"Cannot open {self.file_path} as an HDF5 file") from e

        # Read the file
        self._region_dict = {}
        with h5py.File(self.file_path, "r") as f:
            # region_names = list(f["regions"].keys())

            for region_name, region in f["regions"].items():
                # grp = f[f"regions/{region_name}"]
                self._region_dict[region_name] = Region(
                    region["start_times"][()],
                    region["end_times"][()],
                )

    def get_region(self, region_name: str) -> Region:
        return self._region_dict[region_name]

    def plot_gantt(
        self,
        regions: List[str] | str | None = None,
        filepath: str | None = None,
        show: bool = False,
    ) -> None:
        """
        Plot a Gantt chart of all (or selected) regions.

        Parameters
        ----------
        regions : list[str] | None
            List of region names to plot. If None, plot all.
        """
        if regions is None:
            regions = list(self._region_dict.keys())
        elif isinstance(regions, str):
            regions = [regions]

        fig, ax = plt.subplots(figsize=(10, 0.7 * len(regions)))
        colors = plt.cm.tab20(np.linspace(0, 1, len(regions)))

        for i, region_name in enumerate(regions):
            region = self._region_dict[region_name]
            for start, end in zip(region.start_times, region.end_times):
                ax.barh(
                    y=i,
                    width=end - start,
                    left=start,
                    height=0.4,
                    color=colors[i],
                    edgecolor="black",
                    alpha=0.7,
                )

        ax.set_xlabel("Time (seconds)")
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions)
        ax.set_title("Profiling Gantt Chart")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=300)
        if show:
            plt.show()

    def plot_durations(
        self,
        regions: List[str] | str | None = None,
        filepath: str | None = None,
        show: bool = False,
        bins: int = 30,
    ) -> None:
        """
        Plot duration histograms for each region.

        Parameters
        ----------
        regions : list[str] | None
            List of region names to plot. If None, plot all.
        bins : int
            Number of histogram bins.
        """
        if regions is None:
            regions = list(self._region_dict.keys())
        elif isinstance(regions, str):
            regions = [regions]

        n = len(regions)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3 * n))
        if n == 1:
            axes = [axes]

        for ax, region_name in zip(axes, regions):
            region = self._region_dict[region_name]
            durations = region.durations
            if len(durations) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            ax.hist(
                durations, bins=bins, color="steelblue", alpha=0.7, edgecolor="black"
            )
            ax.set_title(f"Region: {region_name}")
            ax.set_xlabel("Duration (s)")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.4)

        fig.suptitle("Region Duration Distributions", fontsize=14)
        fig.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=300)
        if show:
            plt.show()

    def __repr__(self) -> str:
        _out = ""
        for region_name, region in self._region_dict.items():
            _out += f"Region: {region_name}\n"
            _out += str(region)
        return _out

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def regions(self) -> List[Region]:
        return self._regions
