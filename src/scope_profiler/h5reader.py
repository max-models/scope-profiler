import re
from pathlib import Path
from typing import List

import h5py

from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import Region


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

    @property
    def minimum_start_time(self) -> float:
        """
        Get the minimum start time across all regions and ranks.

        Returns
        -------
        float
            Minimum start time in seconds.
        """
        min_start = float("inf")
        for region in self.get_regions():
            region_min = min(r.first_start_time for r in region.regions.values())
            if region_min < min_start:
                min_start = region_min
        return min_start

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
