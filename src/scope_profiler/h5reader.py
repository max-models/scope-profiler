from pathlib import Path
from typing import Any, Dict, List

import h5py
from matplotlib.pylab import f
import numpy as np
import re


class Region:
    def __init__(
        self,
        start_times: np.ndarray,
        end_times: np.ndarray,
    ) -> None:
        """
        Initialize a Region with timing information for multiple calls.

        Parameters
        ----------
        start_times : np.ndarray
            Start times of all calls in nanoseconds.
        end_times : np.ndarray
            End times of all calls in nanoseconds.
        """
        self._start_times = start_times
        self._end_times = end_times
        self._durations = end_times - start_times

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the region's statistics as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing statistics: num_calls, total_duration,
            average_duration, min_duration, max_duration, and std_duration.
        """
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
        """Start times of all calls in seconds."""
        return self._start_times / 1e9

    @property
    def first_start_time(self) -> float:
        """First start time in seconds."""
        return float(np.min(self._start_times)) / 1e9 if self.num_calls else 0.0

    @property
    def end_times(self) -> np.ndarray:
        """End times of all calls in seconds."""
        return self._end_times / 1e9

    @property
    def durations(self) -> np.ndarray:
        """Duration of all calls in seconds."""
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
        """
        Return a string representation of the region's statistics.

        Returns
        -------
        str
            Formatted string with region statistics.
        """
        # print(f"\nProfiling data summary for: {self.file_path}")
        _out = "-" * 60 + "\n"
        stats = self.get_summary()
        for key, value in stats.items():
            _out += f"  {key:>18}: {value}\n"
        _out += "-" * 60 + "\n\n"
        return _out


class MPIRegion:
    def __init__(self, name: str, regions: Dict[int, Region]) -> None:
        """
        Initialize an MPIRegion containing Region data for multiple ranks.

        Parameters
        ----------
        regions : Dict[int, Region]
            Dictionary mapping rank IDs to their corresponding Region objects.
        """
        self._name = name
        self._regions = regions

    @property
    def name(self) -> str:
        """Name of the region."""
        return self._name

    @property
    def regions(self) -> Dict[int, Region]:
        """Dictionary of rank IDs to their corresponding Region objects."""
        return self._regions

    def __getitem__(self, rank: int) -> Region:
        """
        Get the Region object for a specific rank.

        Parameters
        ----------
        rank : int
            Rank ID.

        Returns
        -------
        Region
            Region object for the specified rank.
        """
        return self._regions[rank]

    def average_durations(self) -> Dict[int, float]:
        """
        Get the average duration for each rank.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping rank IDs to their average durations.
        """
        return {rank: region.average_duration for rank, region in self._regions.items()}

    def min_durations(self) -> Dict[int, float]:
        """
        Get the minimum duration for each rank.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping rank IDs to their minimum durations.
        """
        return {rank: region.min_duration for rank, region in self._regions.items()}

    def max_durations(self) -> Dict[int, float]:
        """
        Get the maximum duration for each rank.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping rank IDs to their maximum durations.
        """
        return {rank: region.max_duration for rank, region in self._regions.items()}

    @property
    def min_duration(self) -> float:
        """
        Get the minimum duration across all ranks.

        Returns
        -------
        float
            The minimum duration among all ranks.
        """
        return min(region.min_duration for region in self._regions.values())

    @property
    def max_duration(self) -> float:
        """
        Get the maximum duration across all ranks.

        Returns
        -------
        float
            The maximum duration among all ranks.
        """
        return max(region.max_duration for region in self._regions.values())

    @property
    def first_start_time(self) -> float:
        """
        Get the earliest start time across all ranks.

        Returns
        -------
        float
            The earliest start time among all ranks.
        """
        return min(region.first_start_time for region in self._regions.values())


class ProfilingH5Reader:
    """
    Reads profiling data stored by ProfileRegion in an HDF5 file.
    """

    def __init__(
        self,
        file_path: str | Path,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the HDF5 reader by loading profiling data from the specified file.

        Parameters
        ----------
        file_path : str | Path
            Path to the HDF5 file containing profiling data.

        Raises
        ------
        FileNotFoundError
            If the specified HDF5 file does not exist.
        """
        self._file_path = Path(file_path)
        self._num_ranks = 0
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        # Read the file
        _region_dict = {}
        region_names = []
        with h5py.File(self.file_path, "r") as f:
            # Iterate over all rank groups
            for rank_group_name, rank_group in f.items():
                self._num_ranks += 1
                if verbose:
                    print(f"{rank_group_name = }")
                    print(rank_group_name, rank_group)
                rank = int(rank_group_name.replace("rank", ""))
                if "regions" not in rank_group:
                    continue
                regions_group = rank_group["regions"]

                for region_name, region_grp in regions_group.items():
                    region_names.append(region_name)
                    starts = region_grp["start_times"][()]
                    ends = region_grp["end_times"][()]
                    # print(f"{region_name = }")
                    # Merge if region already exists (from another rank)
                    if region_name in _region_dict:
                        _region_dict[region_name][rank] = Region(starts, ends)
                    else:
                        _region_dict[region_name] = {rank: Region(starts, ends)}

        self._region_dict = {}

        for region_name in region_names:
            self._region_dict[region_name] = MPIRegion(
                name=region_name, regions=_region_dict[region_name]
            )

    def get_region(self, region_name: str) -> MPIRegion:
        """
        Retrieve profiling data for a specific region.

        Parameters
        ----------
        region_name : str
            Name of the region to retrieve.

        Returns
        -------
        Region
            Region object containing profiling data for all ranks.

        Raises
        ------
        KeyError
            If the specified region name does not exist.
        """
        return self._region_dict[region_name]

    @property
    def file_path(self) -> Path:
        """
        Get the path to the HDF5 file.

        Returns
        -------
        Path
            The file path as a pathlib.Path object.
        """
        return self._file_path

    @property
    def num_ranks(self) -> int:
        """
        Get the number of ranks recorded in the profiling data.

        Returns
        -------
        int
            Number of ranks.
        """
        return self._num_ranks

    def get_regions(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
    ) -> List[MPIRegion]:
        """Get a list of all regions in order of appearance.

        Returns
        -------
        List[Region]
            List of Region objects.
        """

        if isinstance(include, str):
            include = [include]
        if isinstance(exclude, str):
            exclude = [exclude]

        regions = []

        # Collect regions based on include/exclude filters
        for region_name, region in self._region_dict.items():
            # print(f"{region_name = } {region = }")
            # Match with regex patterns if provided
            if include is not None:
                if not any([re.match(pattern, region_name) for pattern in include]):
                    continue
            if exclude is not None:
                if any([re.match(pattern, region_name) for pattern in exclude]):
                    continue

            regions.append(region)

        # Sort regions based on first start time across all ranks
        regions.sort(
            key=lambda r: min(region.first_start_time for region in r.regions.values())
        )

        return regions

    def __repr__(self) -> str:
        """
        Return a string representation of all regions and their profiling statistics.

        Returns
        -------
        str
            Formatted string containing profiling data for all regions.
        """
        _out = ""
        for region_name, region in self._region_dict.items():
            _out += f"Region: {region_name}\n"
            _out += str(region[0])
        return _out
