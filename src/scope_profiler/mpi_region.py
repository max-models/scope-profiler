"""Container for per-rank Region data within an MPI-parallel profiling run."""

from collections.abc import Iterator
from typing import Any

import numpy as np

from scope_profiler.region import NS_PER_SECOND, Region


class MPIRegion:
    """One named region across all ranks that recorded it.

    Indexing gives the per-rank :class:`~scope_profiler.region.Region`
    (``region[0]``), while the properties on this class aggregate over every
    rank. All durations are in seconds.
    """

    def __init__(self, name: str, regions: dict[int, Region]) -> None:
        """
        Initialize an MPIRegion containing Region data for multiple ranks.

        Parameters
        ----------
        regions : Dict[int, Region]
            Dictionary mapping rank IDs to their corresponding Region objects.
        """
        self._name = name
        self._regions = regions
        # Set externally by plotting code so different charts drawn from the
        # same regions (e.g. gantt and flame) agree on a color per name.
        self.color = None

    @property
    def name(self) -> str:
        """Name of the region."""
        return self._name

    @property
    def regions(self) -> dict[int, Region]:
        """Dictionary of rank IDs to their corresponding Region objects."""
        return self._regions

    @property
    def ranks(self) -> list[int]:
        """Sorted list of ranks that recorded this region."""
        return sorted(self._regions)

    @property
    def has_timing(self) -> bool:
        """Whether any rank recorded timestamps for this region."""
        return any(region.has_timing for region in self._regions.values())

    @property
    def has_event_data(self) -> bool:
        """Whether every represented rank has per-call timestamps loaded."""
        return all(region.has_event_data for region in self._regions.values())

    def _first_captured(self, attr: str):
        """First non-None value of ``attr`` across ranks, in rank order.

        Every rank that created this region from the same code path captured
        the same source, so any one of them stands in for the region as a
        whole.
        """
        for rank in self.ranks:
            value = getattr(self._regions[rank], attr)
            if value is not None:
                return value
        return None

    @property
    def has_source(self) -> bool:
        """Whether any rank captured this region's call-site source."""
        return self.source_file is not None

    @property
    def has_gpu_timing(self) -> bool:
        """Whether any rank recorded CUDA-event timings for this region."""
        return any(region.has_gpu_timing for region in self._regions.values())

    @property
    def source_file(self) -> str | None:
        """Path of the file this region is defined in, or None if not captured."""
        return self._first_captured("source_file")

    @property
    def source_lineno(self) -> int | None:
        """Line the region's call site starts at, or None if not captured."""
        return self._first_captured("source_lineno")

    @property
    def source_text(self) -> str | None:
        """Source text of the region's ``with`` block or decorated function."""
        return self._first_captured("source_text")

    @property
    def tags(self) -> tuple[str, ...]:
        """User-defined tags attached to this region."""
        return self._first_captured("tags") or ()

    def get_summary(self) -> dict[str, Any]:
        """
        Return statistics aggregated over every rank.

        Returns
        -------
        Dict[str, Any]
            Dictionary with name, num_ranks, num_calls (summed over ranks) and
            duration statistics in seconds, pooled over all calls on all ranks.
        """
        summary = {
            "name": self.name,
            "num_ranks": len(self._regions),
            "num_calls": self.num_calls,
            "total_duration": self.total_duration,
            "inclusive_duration": self.inclusive_duration,
            "exclusive_duration": self.exclusive_duration,
            "average_duration": self.average_duration,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "first_duration": self.first_duration,
            "last_duration": self.last_duration,
            "std_duration": self.std_duration,
        }
        if self.has_gpu_timing:
            summary["gpu_total_duration"] = self.gpu_total_duration
            summary["gpu_average_duration"] = self.gpu_average_duration
        return summary

    def events(
        self, ranks: list[int] | int | None = None, origin: float = 0.0
    ) -> list[dict[str, Any]]:
        """
        Return one dict per recorded call, on every rank.

        This is the long-form view custom plots and dataframes want: each
        entry is a single call rather than a per-region aggregate.

        Parameters
        ----------
        ranks : list of int or int, optional
            Restrict to these ranks (default: all ranks that recorded the
            region). Ranks without data for this region are skipped.
        origin : float, optional
            Seconds subtracted from every timestamp, so passing
            ``results.minimum_start_time`` yields a timeline starting at zero
            (default: 0.0, i.e. raw timestamps).

        Returns
        -------
        list of dict
            Entries with keys ``name``, ``rank``, ``call_index``, ``start``,
            ``end`` and ``duration``, in seconds, ordered by rank and then by
            call order.
        """
        if ranks is None:
            selected = self.ranks
        elif isinstance(ranks, int):
            selected = [ranks]
        else:
            selected = list(ranks)

        events = []
        for rank in selected:
            region = self._regions.get(rank)
            if region is None:
                continue
            for event in region.events(origin=origin):
                events.append({"name": self.name, "rank": rank, **event})
        return events

    @property
    def durations(self) -> np.ndarray:
        """
        Get every recorded call duration on every rank, in seconds.

        Returns
        -------
        np.ndarray
            Pooled durations. Empty if no rank recorded timing.
        """
        values = [
            region.durations for region in self._regions.values() if region.has_timing
        ]
        if not values:
            return np.array([], dtype=float)
        return np.concatenate(values)

    @property
    def gpu_durations(self) -> np.ndarray | None:
        """CUDA-event elapsed device times pooled across ranks, or None if absent."""
        values = [
            region.gpu_durations
            for region in self._regions.values()
            if region.gpu_durations is not None
        ]
        if not values:
            return None
        return np.concatenate(values)

    @property
    def inclusive_durations(self) -> np.ndarray:
        """Inclusive durations pooled across ranks, in seconds."""
        return self.durations

    @property
    def exclusive_durations(self) -> np.ndarray:
        """Exclusive durations pooled across ranks, in seconds."""
        values = [
            region.exclusive_durations
            for region in self._regions.values()
            if region.has_timing
        ]
        if not values:
            return np.array([], dtype=float)
        return np.concatenate(values)

    @property
    def num_calls(self) -> int:
        """
        Total number of calls summed over all ranks.

        Returns
        -------
        int
            Summed call count.
        """
        return sum(region.num_calls for region in self._regions.values())

    def num_calls_per_rank(self) -> dict[int, int]:
        """
        Get the call count for each rank.

        Returns
        -------
        Dict[int, int]
            Dictionary mapping rank IDs to their call counts.
        """
        return {rank: region.num_calls for rank, region in self._regions.items()}

    def average_durations(self) -> dict[int, float]:
        """
        Get the average duration for each rank.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping rank IDs to their average durations.
        """
        return {rank: region.average_duration for rank, region in self._regions.items()}

    def min_durations(self) -> dict[int, float]:
        """
        Get the minimum duration for each rank.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping rank IDs to their minimum durations.
        """
        return {rank: region.min_duration for rank, region in self._regions.items()}

    def max_durations(self) -> dict[int, float]:
        """
        Get the maximum duration for each rank.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping rank IDs to their maximum durations.
        """
        return {rank: region.max_duration for rank, region in self._regions.items()}

    def total_durations(self) -> dict[int, float]:
        """
        Get the total duration for each rank.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping rank IDs to their total durations.
        """
        return {rank: region.total_duration for rank, region in self._regions.items()}

    @property
    def total_duration(self) -> float:
        """
        Get the total duration summed over all ranks and calls.

        Returns
        -------
        float
            Total duration in seconds.
        """
        return sum(region.total_duration for region in self._regions.values())

    @property
    def gpu_total_duration(self) -> float | None:
        """Total CUDA-event elapsed device time across ranks, or None if absent."""
        values = [
            region.gpu_total_duration
            for region in self._regions.values()
            if region.gpu_total_duration is not None
        ]
        if not values:
            return None
        return float(sum(values))

    @property
    def inclusive_duration(self) -> float:
        """Total inclusive time across ranks, in seconds."""
        return self.total_duration

    @property
    def total_exclusive_duration(self) -> float:
        """Total time excluding nested regions, across ranks, in seconds."""
        return sum(region.total_exclusive_duration for region in self._regions.values())

    @property
    def exclusive_duration(self) -> float:
        """Alias for :attr:`total_exclusive_duration`."""
        return self.total_exclusive_duration

    @property
    def average_duration(self) -> float:
        """
        Get the mean duration over every call on every rank.

        Note this pools all calls, so ranks with more calls weigh more heavily;
        use ``average_durations()`` for the per-rank breakdown.

        Returns
        -------
        float
            Average duration in seconds, or 0.0 if no timing was recorded.
        """
        values = self.durations
        if values.size:
            return float(np.mean(values))
        return self.total_duration / self.num_calls if self.num_calls else 0.0

    @property
    def gpu_average_duration(self) -> float | None:
        """Average CUDA-event elapsed device time across ranks, or None if absent."""
        values = self.gpu_durations
        if values is not None and values.size:
            return float(np.mean(values))
        count = sum(
            int(region.stored_summary.get("gpu_count", 0))
            for region in self._regions.values()
            if region.stored_summary is not None
        )
        total = self.gpu_total_duration
        return total / count if total is not None and count else None

    @property
    def std_duration(self) -> float:
        """
        Get the standard deviation over every call on every rank.

        Returns
        -------
        float
            Standard deviation in seconds, or 0.0 if no timing was recorded.
        """
        values = self.durations
        if values.size:
            return float(np.std(values))
        summaries = [
            region.stored_summary
            for region in self._regions.values()
            if region.num_calls and region.stored_summary is not None
        ]
        if not summaries or any(
            "mean" not in summary or "m2" not in summary for summary in summaries
        ):
            return 0.0
        count = sum(int(summary["count"]) for summary in summaries)
        mean = (
            sum(int(summary["count"]) * float(summary["mean"]) for summary in summaries)
            / count
        )
        m2 = sum(
            float(summary["m2"])
            + int(summary["count"]) * (float(summary["mean"]) - mean) ** 2
            for summary in summaries
        )
        return float(np.sqrt(max(m2, 0.0) / count)) / NS_PER_SECOND

    def percentile_duration(self, percentile: float) -> float | None:
        """Return a pooled duration percentile across all ranks."""
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        values = self.durations
        if values.size:
            return float(np.percentile(values, percentile))
        return None if self.num_calls else 0.0

    @property
    def p50_duration(self) -> float | None:
        """Median duration across all ranks, in seconds."""
        return self.percentile_duration(50)

    @property
    def p95_duration(self) -> float | None:
        """95th-percentile duration across all ranks, in seconds."""
        return self.percentile_duration(95)

    @property
    def p99_duration(self) -> float | None:
        """99th-percentile duration across all ranks, in seconds."""
        return self.percentile_duration(99)

    @property
    def rank_imbalance(self) -> float:
        """Maximum per-rank total divided by the mean per-rank total.

        A value of 1.0 means perfectly balanced ranks. Values are ``0.0``
        when no calls were recorded or only one rank has timing data.
        """
        totals = [region.total_duration for region in self._regions.values()]
        totals = [total for total in totals if total > 0]
        if len(totals) < 2:
            return 0.0
        return float(max(totals) / np.mean(totals))

    @property
    def rank_imbalance_pct(self) -> float:
        """Excess of the slowest rank over the mean, as a percentage."""
        ratio = self.rank_imbalance
        return (ratio - 1.0) * 100.0 if ratio else 0.0

    @property
    def min_duration(self) -> float:
        """
        Get the minimum duration across all ranks.

        Returns
        -------
        float
            The minimum duration among all ranks, in seconds.
        """
        values = self.durations
        if values.size:
            return float(np.min(values))
        minimums = [
            region.min_duration for region in self._regions.values() if region.num_calls
        ]
        return min(minimums) if minimums else 0.0

    @property
    def max_duration(self) -> float:
        """
        Get the maximum duration across all ranks.

        Returns
        -------
        float
            The maximum duration among all ranks, in seconds.
        """
        values = self.durations
        if values.size:
            return float(np.max(values))
        maximums = [
            region.max_duration for region in self._regions.values() if region.num_calls
        ]
        return max(maximums) if maximums else 0.0

    @property
    def first_duration(self) -> float:
        """
        Get the duration of the call that started earliest across all ranks.

        Returns
        -------
        float
            Duration in seconds of the chronologically first call, or 0.0 if
            no rank recorded timing.
        """
        timed = [region for region in self._regions.values() if region.has_timing]
        if not timed:
            return 0.0
        return min(timed, key=lambda region: region.first_start_time).first_duration

    @property
    def last_duration(self) -> float:
        """
        Get the duration of the call that ended latest across all ranks.

        Returns
        -------
        float
            Duration in seconds of the chronologically last call, or 0.0 if no
            rank recorded timing.
        """
        timed = [region for region in self._regions.values() if region.has_timing]
        if not timed:
            return 0.0
        return max(timed, key=lambda region: region.last_end_time).last_duration

    @property
    def first_start_time(self) -> float:
        """
        Get the earliest start time across all ranks.

        Returns
        -------
        float
            The earliest start time among all ranks, in seconds, or 0.0 if no
            rank recorded timing.
        """
        starts = [
            region.first_start_time
            for region in self._regions.values()
            if region.has_timing
        ]
        return min(starts) if starts else 0.0

    @property
    def last_end_time(self) -> float:
        """
        Get the latest end time across all ranks.

        Returns
        -------
        float
            The latest end time among all ranks, in seconds, or 0.0 if no rank
            recorded timing.
        """
        ends = [
            region.last_end_time
            for region in self._regions.values()
            if region.has_timing
        ]
        return max(ends) if ends else 0.0

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
        try:
            return self._regions[rank]
        except KeyError:
            raise KeyError(
                f"Region {self._name!r} has no data for rank {rank}. "
                f"Available ranks: {self.ranks}"
            ) from None

    def __iter__(self) -> Iterator[int]:
        """Iterate over the ranks that recorded this region."""
        return iter(self.ranks)

    def __contains__(self, rank: int) -> bool:
        """Whether this region has data for ``rank``."""
        return rank in self._regions

    def __len__(self) -> int:
        """Number of ranks that recorded this region."""
        return len(self._regions)

    def __repr__(self) -> str:
        """Return a one-line summary of the region across ranks."""
        return (
            f"<MPIRegion {self._name!r}: {self.num_calls} calls on "
            f"{len(self._regions)} rank(s), "
            f"total {self.total_duration:.6g} s, avg {self.average_duration:.6g} s>"
        )
