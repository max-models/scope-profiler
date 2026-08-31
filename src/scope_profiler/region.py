"""Data container for a single profiling region loaded from HDF5."""

from typing import Any

import numpy as np

NS_PER_SECOND = 1e9


class EventDataUnavailableError(RuntimeError):
    """Raised when a summary-only result is asked for per-call event data."""


class Region:
    """Timing data for one region on one rank.

    All duration and timestamp properties are reported in **seconds**;
    the underlying HDF5 data is stored in nanoseconds.
    """

    def __init__(
        self,
        start_times: np.ndarray,
        end_times: np.ndarray,
        gpu_durations: np.ndarray | None = None,
        call_ids: np.ndarray | None = None,
        parent_ids: np.ndarray | None = None,
        source_file: str | None = None,
        source_lineno: int | None = None,
        source_text: str | None = None,
        tags=(),
        aggregate: dict | None = None,
        event_data_available: bool = True,
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
        self._gpu_durations = (
            None if gpu_durations is None else np.asarray(gpu_durations, dtype=np.int64)
        )
        self._call_ids = (
            None if call_ids is None else np.asarray(call_ids, dtype=np.int64)
        )
        self._parent_ids = (
            None if parent_ids is None else np.asarray(parent_ids, dtype=np.int64)
        )
        # A Region does not know about other regions, so until it is attached
        # to ProfilingResults exclusive time defaults to inclusive time. The
        # array is built on first use: reconstructing the nesting is by far
        # the most expensive part of loading a run, and most callers
        # (durations, timelines, diffs) never ask for exclusive time at all.
        self._exclusive_durations = None
        self._exclusive_resolver = None
        # Total exclusive nanoseconds computed by the writer, when the file
        # recorded one. Saves reconstructing the nesting for the common case
        # of reporting a region's exclusive time without its per-call values.
        self._exclusive_total_ns = None
        self._aggregate = aggregate
        self._event_data_available = bool(event_data_available)
        self._num_calls = (
            int(aggregate.get("count", 0))
            if aggregate is not None
            else len(self._durations)
        )
        self._source_file = source_file
        self._source_lineno = source_lineno
        self._source_text = source_text
        self._tags = tuple(tags)

    def get_summary(self) -> dict[str, Any]:
        """
        Return a summary of the region's statistics as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing statistics: num_calls, total_duration,
            average_duration, min_duration, max_duration, first_duration,
            last_duration, and std_duration. Durations are in seconds.
        """
        summary = {
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

    def _set_exclusive_durations(self, durations: np.ndarray) -> None:
        """Set exclusive durations after nesting has been reconstructed."""
        if len(durations) != self.num_calls:
            raise ValueError("exclusive durations must match the call count")
        self._exclusive_resolver = None
        self._exclusive_durations = np.asarray(durations, dtype=self._durations.dtype)

    def _attach_exclusive_resolver(self, resolver) -> None:
        """Register the callback that reconstructs this region's nesting.

        Called by :class:`~scope_profiler.results.ProfilingResults`, which is
        the only object that can see the other regions a call is nested in.
        It runs at most once, the first time any region asks for exclusive
        time, and fills in every region of the result set at once.

        Any previously computed exclusive durations are dropped, including a
        total the writer supplied: a region can be put into a second result
        set (``merge_results`` reuses the region objects), and against a
        different set of neighbours the same calls have different exclusive
        time. The newest owner therefore wins, as it did when every result set
        recomputed this eagerly on construction, and it is the owner that
        re-supplies a stored total via :meth:`_set_exclusive_total`.
        """
        self._exclusive_durations = None
        self._exclusive_total_ns = None
        self._exclusive_resolver = resolver

    def _set_exclusive_total(self, total_ns) -> None:
        """Adopt an exclusive total the run itself computed, in nanoseconds.

        Only the result set that owns this region may call this: the value is
        only meaningful against the region set it was computed with. Left
        alone, the per-call durations remain lazy -- asking for those still
        reconstructs the nesting, and then wins over this.
        """
        self._exclusive_total_ns = None if total_ns is None else int(total_ns)

    def _exclusive_buffer(self) -> np.ndarray:
        """The writable exclusive-duration array, without resolving nesting.

        For the resolver itself: it fills this in place, per call index.
        Defaults to the inclusive durations, which is what a region with no
        nested calls ends up with anyway.
        """
        if self._exclusive_durations is None:
            self._exclusive_durations = self._durations.copy()
        return self._exclusive_durations

    def _resolved_exclusive_durations(self) -> np.ndarray:
        """Exclusive durations in nanoseconds, reconstructing nesting if needed."""
        if self._exclusive_durations is None:
            resolver = self._exclusive_resolver
            if resolver is not None:
                self._exclusive_resolver = None
                resolver()
            if self._exclusive_durations is None:
                self._exclusive_durations = self._durations.copy()
        return self._exclusive_durations

    def events(self, origin: float = 0.0) -> list[dict[str, Any]]:
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
        if not self._event_data_available:
            raise EventDataUnavailableError(
                "per-call events are unavailable on summary-only results; "
                "load the profile with read_h5()"
            )
        starts = self.start_times - origin
        ends = self.end_times - origin
        events = []
        gpu_durations = self.gpu_durations
        for index, (start, end) in enumerate(zip(starts, ends)):
            event = {
                "call_index": index,
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
            }
            if gpu_durations is not None:
                event["gpu_duration"] = float(gpu_durations[index])
            if self._call_ids is not None:
                event["call_id"] = int(self._call_ids[index])
                event["parent_id"] = int(self._parent_ids[index])
            events.append(event)
        return events

    @property
    def call_ids(self):
        """Explicit call ids, or None for legacy profiles."""
        return self._call_ids

    @property
    def parent_ids(self):
        """Explicit parent ids, or None for legacy profiles."""
        return self._parent_ids

    @property
    def has_timing(self) -> bool:
        """Whether this region recorded any calls at all."""
        return self.num_calls > 0

    @property
    def has_event_data(self) -> bool:
        """Whether per-call timestamps were loaded for this region."""
        return self._event_data_available

    @property
    def has_source(self) -> bool:
        """Whether this region's call-site source was captured."""
        return self._source_file is not None

    @property
    def has_gpu_timing(self) -> bool:
        """Whether this region has CUDA-event elapsed timings."""
        if self._aggregate is not None:
            return int(self._aggregate.get("gpu_count", 0)) > 0
        return self._gpu_durations is not None and len(self._gpu_durations) > 0

    @property
    def stored_summary(self) -> dict | None:
        """Fixed-size statistics used by aggregate and summary-only results."""
        return self._aggregate

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
    def tags(self) -> tuple[str, ...]:
        """User-defined tags attached to this region."""
        return self._tags

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
    def gpu_durations_ns(self) -> np.ndarray | None:
        """CUDA-event elapsed device times in nanoseconds, or None if absent."""
        return self._gpu_durations

    @property
    def inclusive_durations_ns(self) -> np.ndarray:
        """Inclusive duration of every call in nanoseconds."""
        return self._durations

    @property
    def exclusive_durations_ns(self) -> np.ndarray:
        """Exclusive duration of every call in nanoseconds."""
        return self._resolved_exclusive_durations()

    @property
    def start_times(self) -> np.ndarray:
        """Start times of all calls in seconds."""
        return self._start_times / NS_PER_SECOND

    @property
    def first_start_time(self) -> float:
        """First start time in seconds."""
        if self._aggregate is not None:
            return self._aggregate.get("start_minimum", 0) / NS_PER_SECOND
        return (
            float(np.min(self._start_times)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def last_end_time(self) -> float:
        """Last end time in seconds."""
        if self._aggregate is not None:
            return self._aggregate.get("end_maximum", 0) / NS_PER_SECOND
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
    def gpu_durations(self) -> np.ndarray | None:
        """CUDA-event elapsed device times in seconds, or None if absent."""
        if self._gpu_durations is None:
            return None
        return self._gpu_durations / NS_PER_SECOND

    @property
    def inclusive_durations(self) -> np.ndarray:
        """Inclusive duration of every call in seconds."""
        return self.durations

    @property
    def exclusive_durations(self) -> np.ndarray:
        """Exclusive duration of every call in seconds."""
        return self._resolved_exclusive_durations() / NS_PER_SECOND

    @property
    def num_calls(self) -> int:
        """Number of recorded calls."""
        return self._num_calls

    @property
    def total_duration(self) -> float:
        """Total time spent in this region in seconds (sum of all durations)."""
        if self._aggregate is not None:
            return self._aggregate["total"] / NS_PER_SECOND
        return (
            float(np.sum(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def gpu_total_duration(self) -> float | None:
        """Total CUDA-event elapsed device time in seconds, or None if absent."""
        if self._aggregate is not None:
            if not self._aggregate.get("gpu_count", 0):
                return None
            return self._aggregate.get("gpu_total", 0) / NS_PER_SECOND
        if self._gpu_durations is None:
            return None
        return float(np.sum(self._gpu_durations)) / NS_PER_SECOND

    @property
    def inclusive_duration(self) -> float:
        """Total inclusive time, including nested regions, in seconds."""
        return self.total_duration

    @property
    def total_exclusive_duration(self) -> float:
        """Total time excluding nested regions, in seconds."""
        if self._aggregate is not None:
            return self._aggregate["exclusive"] / NS_PER_SECOND
        if not self.has_timing:
            return 0.0
        if self._exclusive_durations is None and self._exclusive_total_ns is not None:
            return self._exclusive_total_ns / NS_PER_SECOND
        return float(np.sum(self._resolved_exclusive_durations())) / NS_PER_SECOND

    @property
    def exclusive_duration(self) -> float:
        """Alias for :attr:`total_exclusive_duration`."""
        return self.total_exclusive_duration

    @property
    def average_duration(self) -> float:
        """Average duration per call in seconds."""
        if self._aggregate is not None:
            return self.total_duration / self.num_calls if self.num_calls else 0.0
        return (
            float(np.mean(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def gpu_average_duration(self) -> float | None:
        """Average CUDA-event elapsed device time in seconds, or None if absent."""
        if self._aggregate is not None:
            count = int(self._aggregate.get("gpu_count", 0))
            return (
                self._aggregate.get("gpu_total", 0) / count / NS_PER_SECOND
                if count
                else None
            )
        if self._gpu_durations is None or len(self._gpu_durations) == 0:
            return None
        return float(np.mean(self._gpu_durations)) / NS_PER_SECOND

    @property
    def min_duration(self) -> float:
        """Minimum duration among all calls in seconds."""
        if self._aggregate is not None:
            return self._aggregate["minimum"] / NS_PER_SECOND
        return (
            float(np.min(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def max_duration(self) -> float:
        """Maximum duration among all calls in seconds."""
        if self._aggregate is not None:
            return self._aggregate["maximum"] / NS_PER_SECOND
        return (
            float(np.max(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    @property
    def first_duration(self) -> float:
        """Duration of the first recorded call, in seconds."""
        if self._aggregate is not None:
            return self._aggregate.get("first", 0) / NS_PER_SECOND
        return float(self._durations[0]) / NS_PER_SECOND if self.has_timing else 0.0

    @property
    def last_duration(self) -> float:
        """Duration of the last recorded call, in seconds."""
        if self._aggregate is not None:
            return self._aggregate.get("last", 0) / NS_PER_SECOND
        return float(self._durations[-1]) / NS_PER_SECOND if self.has_timing else 0.0

    @property
    def std_duration(self) -> float:
        """Standard deviation of durations in seconds."""
        if self._aggregate is not None:
            count = int(self._aggregate.get("count", 0))
            m2 = self._aggregate.get("m2")
            return (
                float(np.sqrt(max(float(m2), 0.0) / count)) / NS_PER_SECOND
                if count and m2 is not None
                else 0.0
            )
        return (
            float(np.std(self._durations)) / NS_PER_SECOND if self.has_timing else 0.0
        )

    def percentile_duration(self, percentile: float) -> float | None:
        """Return a duration percentile in seconds.

        ``percentile`` follows :func:`numpy.percentile` and must be between
        0 and 100. Empty regions return ``0.0`` for consistency with the
        other duration statistics.
        """
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        if self._aggregate is not None and self.num_calls:
            return None
        return (
            float(np.percentile(self._durations, percentile)) / NS_PER_SECOND
            if self.has_timing
            else 0.0
        )

    @property
    def p50_duration(self) -> float:
        """Median duration in seconds."""
        return self.percentile_duration(50)

    @property
    def p95_duration(self) -> float:
        """95th-percentile duration in seconds."""
        return self.percentile_duration(95)

    @property
    def p99_duration(self) -> float:
        """99th-percentile duration in seconds."""
        return self.percentile_duration(99)

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
