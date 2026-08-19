"""Data container for a single profiling region loaded from HDF5."""

from typing import Any, Dict, List

import numpy as np

NS_PER_SECOND = 1e9


class Region:
    """Timing data for one region on one rank.

    All duration and timestamp properties are reported in **seconds**;
    the underlying HDF5 data is stored in nanoseconds.
    """

    def __init__(
        self,
        start_times: np.ndarray,
        end_times: np.ndarray,
        source_file: str | None = None,
        source_lineno: int | None = None,
        source_text: str | None = None,
    ) -> None:
        """
        Initialize a Region with timing information for multiple calls.

        Parameters
        ----------
        start_times : np.ndarray
            Start times of all calls in nanoseconds.
        end_times : np.ndarray
            End times of all calls in nanoseconds.
        source_file, source_lineno, source_text : optional
            Where this region is defined in user code, and its source text
            (the ``with`` block or decorated function). None when not
            captured, e.g. files written before this was recorded, or a
            region created only via the recursive tracer.
        """
        self._start_times = start_times
        self._end_times = end_times
        self._durations = end_times - start_times
        self._num_calls = len(self._durations)
        self._source_file = source_file
        self._source_lineno = source_lineno
        self._source_text = source_text

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the region's statistics as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing statistics: num_calls, total_duration,
            average_duration, min_duration, max_duration, first_duration,
            last_duration, and std_duration. Durations are in seconds.
        """
        return {
            "num_calls": self.num_calls,
            "total_duration": self.total_duration,
            "average_duration": self.average_duration,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "first_duration": self.first_duration,
            "last_duration": self.last_duration,
            "std_duration": self.std_duration,
        }

    def events(self, origin: float = 0.0) -> List[Dict[str, Any]]:
        """
        Return one dict per recorded call.

        Parameters
        ----------
        origin : float, optional
            Seconds subtracted from every timestamp, so passing
            ``results.minimum_start_time`` yields a timeline starting at zero
            (default: 0.0, i.e. raw timestamps).

        Returns
        -------
        list of dict
            One entry per call with keys ``call_index``, ``start``, ``end``
            and ``duration``, in seconds and in recorded order.
        """
        starts = self.start_times - origin
        ends = self.end_times - origin
        return [
            {
                "call_index": index,
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
            }
            for index, (start, end) in enumerate(zip(starts, ends))
        ]

    @property
    def has_timing(self) -> bool:
        """Whether this region recorded any calls at all."""
        return len(self._durations) > 0

    @property
    def has_source(self) -> bool:
        """Whether this region's call-site source was captured."""
        return self._source_text is not None

    @property
    def source_file(self) -> str | None:
        """Path of the file this region is defined in, or None if not captured."""
        return self._source_file

    @property
    def source_lineno(self) -> int | None:
        """Line the region's call site starts at, or None if not captured."""
        return self._source_lineno

    @property
    def source_text(self) -> str | None:
        """Source text of the region's ``with`` block or decorated function.

        None if it was not captured -- either the file it came from is no
        longer readable, or the file predates this being recorded.
        """
        return self._source_text

    @property
    def start_times_ns(self) -> np.ndarray:
        """Start times of all calls in nanoseconds, exactly as stored."""
        return self._start_times

    @property
    def end_times_ns(self) -> np.ndarray:
        """End times of all calls in nanoseconds, exactly as stored."""
        return self._end_times

    @property
    def durations_ns(self) -> np.ndarray:
        """Duration of all calls in nanoseconds, as integers."""
        return self._durations

    @property
    def start_times(self) -> np.ndarray:
        """Start times of all calls in seconds."""
        return self._start_times / NS_PER_SECOND

    @property
    def first_start_time(self) -> float:
        """First start time in seconds."""
        return (
            float(np.min(self._start_times)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def last_end_time(self) -> float:
        """Last end time in seconds."""
        return (
            float(np.max(self._end_times)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def end_times(self) -> np.ndarray:
        """End times of all calls in seconds."""
        return self._end_times / NS_PER_SECOND

    @property
    def durations(self) -> np.ndarray:
        """Duration of all calls in seconds."""
        return self._durations / NS_PER_SECOND

    @property
    def num_calls(self) -> int:
        """Number of recorded calls."""
        return self._num_calls

    @property
    def total_duration(self) -> float:
        """Total time spent in this region in seconds (sum of all durations)."""
        return (
            float(np.sum(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def average_duration(self) -> float:
        """Average duration per call in seconds."""
        return (
            float(np.mean(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def min_duration(self) -> float:
        """Minimum duration among all calls in seconds."""
        return (
            float(np.min(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def max_duration(self) -> float:
        """Maximum duration among all calls in seconds."""
        return (
            float(np.max(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def first_duration(self) -> float:
        """Duration of the first recorded call, in seconds."""
        return float(self._durations[0]) / NS_PER_SECOND if self.has_timing else 0.0

    @property
    def last_duration(self) -> float:
        """Duration of the last recorded call, in seconds."""
        return float(self._durations[-1]) / NS_PER_SECOND if self.has_timing else 0.0

    @property
    def std_duration(self) -> float:
        """Standard deviation of durations in seconds."""
        return (
            float(np.std(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    def __len__(self) -> int:
        """Number of recorded calls."""
        return self.num_calls

    def __repr__(self) -> str:
        """
        Return a string representation of the region's statistics.

        Returns
        -------
        str
            Formatted string with region statistics.
        """
        _out = "-" * 60 + "\n"
        stats = self.get_summary()
        for key, value in stats.items():
            unit = "" if key == "num_calls" else " s"
            _out += f"  {key:>18}: {value}{unit}\n"
        _out += "-" * 60 + "\n\n"
        return _out
