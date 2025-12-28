from typing import Any, Dict

import numpy as np


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
