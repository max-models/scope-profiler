from typing import Dict

from scope_profiler.region import Region


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
