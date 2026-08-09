"""Profile region classes implementing the strategy pattern for different profiling modes."""

import functools
from time import perf_counter_ns
from typing import TYPE_CHECKING

import h5py
import numpy as np

from scope_profiler.profile_config import ProfilingConfig

if TYPE_CHECKING:
    pass


def _import_pylikwid():
    """Import and return the pylikwid module.

    This function exists to defer the import of pylikwid until needed,
    preventing unnecessary overhead when LIKWID profiling is disabled. It
    delegates so that regions and the config resolve pylikwid identically,
    including the liblikwid preloading fallback.
    """
    from scope_profiler.profile_config import _import_pylikwid as _import

    return _import()


def _import_line_profiler():
    """Import and return the LineProfiler class from line_profiler.

    Imported lazily: line_profiler is an optional dependency, needed only when
    ``use_line_profiler=True``.
    """
    try:
        from line_profiler import LineProfiler
    except ImportError as exc:
        raise ImportError(
            "Line-by-line profiling requested but line_profiler is not "
            "installed. Install scope-profiler[line-profiler], or "
            "line_profiler directly."
        ) from exc

    return LineProfiler


# Shared read-only stand-in for the timing buffers of regions that never record
# timestamps. Handing out one module-level array keeps `start_times`/`end_times`
# valid attributes (and the derived properties empty) at zero per-region cost.
_EMPTY_TIMES = np.empty(0, dtype=np.int64)
_EMPTY_TIMES.flags.writeable = False


# Base class with common functionality (buffer growth, HDF5 handling)
class BaseProfileRegion:
    """Base class providing shared profiling logic.

    Handles start/end time buffering, call counting, and writing the recorded
    timestamps to HDF5. The buffers grow on demand and are written out once,
    at the end of the run.
    """

    __slots__ = (
        "region_name",
        "config",
        "start_times",
        "end_times",
        "num_calls",
        "ptr",
        "buffer_limit",
        "capacity",
        "group_path",
        "local_file_path",
        "_scope_ptr_stack",
    )

    # Subclasses that never write timestamps set this to False so no per-region
    # buffers are allocated.
    _records_time = True

    def __init__(self, region_name: str, config: ProfilingConfig):
        """Initialize a profiling region.

        Parameters
        ----------
        region_name : str
        Name of the profiled region.
        config : ProfilingConfig
        Profiling configuration containing the initial buffer capacity,
        file paths, and timing reference.
        """
        self.region_name = region_name
        self.config = config
        self.num_calls = 0

        # Preallocate buffers (skipped entirely when no timing is recorded).
        # `buffer_limit` is the *initial* capacity: `_grow` doubles it as
        # needed, so the number of calls a region can record is bounded only
        # by memory.
        #
        # The arrays are int64, so storing a timestamp converts the plain
        # Python int from perf_counter_ns() on assignment. Do not wrap the
        # clock reads in np.int64() to "match" the dtype: building the numpy
        # scalar costs ~190 ns each, twice per call, and stores the same
        # value. See tests/test_overhead.py for the budget this buys.
        self.ptr = 0
        self.buffer_limit = config.buffer_limit
        if self._records_time:
            self.capacity = self.buffer_limit
            self.start_times = np.empty(self.capacity, dtype=np.int64)
            self.end_times = np.empty(self.capacity, dtype=np.int64)
        else:
            self.capacity = 0
            self.start_times = _EMPTY_TIMES
            self.end_times = _EMPTY_TIMES

        # Recursion support: entering a scope reserves its slot (via `ptr`)
        # immediately and remembers it here, so a recursive re-entry before
        # the outer call exits reserves its own slot instead of clobbering
        # the outer one. Only the context-manager form needs the stack; the
        # decorator form keeps its slot in the wrapper's local scope.
        self._scope_ptr_stack = []

        # Setu p paths
        self.group_path = f"regions/{self.region_name}"
        self.local_file_path = self.config._local_file_path

    def _grow(self) -> None:
        """Double the timestamp buffers, preserving already-recorded slots.

        Slot indices are handed out before the profiled call runs and written
        after it returns, so growth must keep every index valid: the contents
        are copied to the same positions in the larger buffers, and the
        deferred writes then land in the new arrays.
        """
        capacity = max(1, self.capacity * 2)
        start_times = np.empty(capacity, dtype=np.int64)
        end_times = np.empty(capacity, dtype=np.int64)
        start_times[: self.capacity] = self.start_times
        end_times[: self.capacity] = self.end_times
        self.start_times = start_times
        self.end_times = end_times
        # Cached so the hot path compares two ints rather than reaching into
        # the array for its size on every call.
        self.capacity = capacity

    def wrap(self, func):
        """Wrap a function for profiling.

        Subclasses must override this method to implement the appropriate
        profiling behavior.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def append(self, start: float, end: float) -> None:
        """Append a start/end time pair to the buffer, growing it if needed."""

        if self.ptr >= self.capacity:
            self._grow()
        self.start_times[self.ptr] = start
        self.end_times[self.ptr] = end
        self.ptr += 1

    def write_to_disk(self):
        """Write the recorded start/end times to the per-rank HDF5 file.

        Called once, at the end of the run. Because the final length is known
        by then, the datasets are created contiguous and exactly sized rather
        than chunked and resizable, which keeps sparse regions small on disk.
        """
        if self.ptr == 0:
            return

        with h5py.File(self.config._local_file_path, "a") as f:
            grp = f.require_group(self.group_path)
            # Never fail on a dataset that is already there: writing twice into
            # the same per-rank file (a second write_to_disk() outside the
            # finalize() path) should replace the data, not raise from h5py.
            for name in ("start_times", "end_times"):
                if name in grp:
                    del grp[name]
            grp.create_dataset("start_times", data=self.start_times[: self.ptr])
            grp.create_dataset("end_times", data=self.end_times[: self.ptr])

    def get_durations_numpy(self) -> np.ndarray:
        """Return durations (end - start) for buffered entries as a NumPy array."""
        return self.end_times[: self.ptr] - self.start_times[: self.ptr]

    def mark_written(self) -> None:
        """Record that everything buffered so far has reached the disk.

        Called by ``finalize()`` once the data is safely written, so that a
        second run in the same process reports only its own events instead of
        re-reporting the first run's. The timestamp buffer rewinds (the arrays
        are reused; anything past ``ptr`` is unread scratch), while
        ``num_calls`` keeps counting for the lifetime of the process — it is
        the in-memory view of the region, which callers inspect after
        ``finalize()``.

        A region that is currently open has a slot reserved in the buffer and
        an index waiting to be popped on exit, so rewinding under it would let
        the next call overwrite a live slot. Such a region is left untouched.
        """
        if self._scope_ptr_stack:
            return
        self.ptr = 0

    def get_end_times_numpy(self) -> np.ndarray:
        """Return end times offset by the run's start time."""
        return self.end_times[: self.ptr] - self.config.start_time_ns

    def get_start_times_numpy(self) -> np.ndarray:
        """Return start times offset by the run's start time."""
        return self.start_times[: self.ptr] - self.config.start_time_ns

    def add_function(self, func) -> None:
        """Register a function for profiling. No-op except in LineProfilerRegion."""
        pass


# Disabled region: does nothing
class DisabledProfileRegion(BaseProfileRegion):
    """Profiling region that performs no measurements.

    Used when profiling is disabled but code paths must remain valid.
    """

    _records_time = False

    def wrap(self, func):
        """Return the original function unchanged — no wrapper, no overhead."""
        return func

    def append(self, start, end):
        """Ignored: no data recorded."""
        pass

    def write_to_disk(self):
        """Ignored: no data recorded."""
        pass

    def get_durations_numpy(self):
        """Return an empty array since nothing is recorded."""
        return np.array([])

    def __enter__(self):
        """Enter a non-operational context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit a non-operational context manager."""
        pass


# Time-only region
class TimeOnlyProfileRegion(BaseProfileRegion):
    """Region that records timing, written to disk once at the end of the run."""

    def wrap(self, func):
        """Wrap a function to measure its execution time."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            # Reserve this call's slot before invoking `func`, so a
            # recursive call re-entering this region gets its own slot
            # instead of overwriting this one.
            if self.ptr >= self.capacity:
                self._grow()
            scope_ptr = self.ptr
            self.ptr += 1
            start = perf_counter_ns()
            try:
                return func(*args, **kwargs)
            finally:
                end = perf_counter_ns()
                self.start_times[scope_ptr] = start
                self.end_times[scope_ptr] = end

        return wrapper

    def __enter__(self):
        """Reserve this scope's slot, record start time, and increment call count."""
        if self.ptr >= self.capacity:
            self._grow()
        scope_ptr = self.ptr
        self.ptr += 1
        self._scope_ptr_stack.append(scope_ptr)
        self.start_times[scope_ptr] = perf_counter_ns()
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record the end time at this scope's reserved slot."""
        scope_ptr = self._scope_ptr_stack.pop()
        self.end_times[scope_ptr] = perf_counter_ns()


# Full region: time + LIKWID
class FullProfileRegion(BaseProfileRegion):
    """Region that records both timing and LIKWID metrics, and writes to HDF5.

    This is the most complete profiling mode: users obtain LIKWID markers,
    nanosecond-resolution timing, and persistent on-disk storage.
    """

    __slots__ = ("likwid_marker_start", "likwid_marker_stop")

    def __init__(self, region_name: str, config: ProfilingConfig):
        """Initialize timing buffers, HDF5 paths, and LIKWID callbacks."""
        super().__init__(region_name, config)
        pylikwid = _import_pylikwid()
        self.likwid_marker_start = pylikwid.markerstartregion
        self.likwid_marker_stop = pylikwid.markerstopregion

    def wrap(self, func):
        """Wrap a function to measure time and collect LIKWID metrics."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            # Reserve this call's slot before invoking `func`, so a
            # recursive call re-entering this region gets its own slot
            # instead of overwriting this one.
            if self.ptr >= self.capacity:
                self._grow()
            scope_ptr = self.ptr
            self.ptr += 1
            start = perf_counter_ns()
            self.likwid_marker_start(self.region_name)
            try:
                return func(*args, **kwargs)
            finally:
                self.likwid_marker_stop(self.region_name)
                end = perf_counter_ns()
                self.start_times[scope_ptr] = start
                self.end_times[scope_ptr] = end

        return wrapper

    def __enter__(self):
        """Reserve this scope's slot, record start time, and start LIKWID region."""
        self.num_calls += 1
        if self.ptr >= self.capacity:
            self._grow()
        scope_ptr = self.ptr
        self.ptr += 1
        self._scope_ptr_stack.append(scope_ptr)
        self.start_times[scope_ptr] = perf_counter_ns()
        self.likwid_marker_start(self.region_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record the end time at this scope's slot and stop the LIKWID region."""
        self.likwid_marker_stop(self.region_name)
        scope_ptr = self._scope_ptr_stack.pop()
        self.end_times[scope_ptr] = perf_counter_ns()


# Line profiler region: time + line_profiler
class LineProfilerRegion(BaseProfileRegion):
    """Region that records timing and line-by-line profiling via line_profiler.

    Uses line_profiler to collect per-line execution statistics for decorated
    functions. Also records nanosecond timestamps and flushes to HDF5.

    Line-by-line profiling is most useful with the decorator (``wrap``) path,
    which automatically registers the function with the line profiler.  When
    used as a context manager, the profiler is enabled/disabled around the
    block - any functions previously added via the decorator path will be
    profiled while the context is active.
    """

    __slots__ = ("_line_profiler",)

    def __init__(self, region_name: str, config: ProfilingConfig):
        """Initialize timing buffers and line_profiler instance."""
        super().__init__(region_name, config)
        LineProfiler = _import_line_profiler()
        self._line_profiler = LineProfiler()

    def wrap(self, func):
        """Wrap a function to measure execution time and collect line-by-line stats."""
        self._line_profiler.add_function(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            # Reserve this call's slot before invoking `func`, so a
            # recursive call re-entering this region gets its own slot
            # instead of overwriting this one.
            if self.ptr >= self.capacity:
                self._grow()
            scope_ptr = self.ptr
            self.ptr += 1
            start = perf_counter_ns()
            self._line_profiler.enable_by_count()
            try:
                return func(*args, **kwargs)
            finally:
                self._line_profiler.disable_by_count()
                end = perf_counter_ns()
                self.start_times[scope_ptr] = start
                self.end_times[scope_ptr] = end

        return wrapper

    def __enter__(self):
        """Reserve this scope's slot, record start time, and enable line profiler."""
        self.num_calls += 1
        if self.ptr >= self.capacity:
            self._grow()
        scope_ptr = self.ptr
        self.ptr += 1
        self._scope_ptr_stack.append(scope_ptr)
        self.start_times[scope_ptr] = perf_counter_ns()
        self._line_profiler.enable_by_count()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Disable the line profiler and record the end time at this scope's slot."""
        self._line_profiler.disable_by_count()
        scope_ptr = self._scope_ptr_stack.pop()
        self.end_times[scope_ptr] = perf_counter_ns()

    def add_function(self, func) -> None:
        """Register a function for line-by-line profiling."""
        self._line_profiler.add_function(func)

    def print_stats(self):
        """Print line-by-line profiling statistics."""
        self._line_profiler.print_stats()

    def get_stats(self):
        """Return the line_profiler stats object."""
        return self._line_profiler.get_stats()
