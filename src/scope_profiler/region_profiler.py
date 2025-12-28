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
    preventing unnecessary overhead when LIKWID profiling is disabled.
    """
    import pylikwid

    return pylikwid


# Base class with common functionality (flush, append, HDF5 handling)
class BaseProfileRegion:
    """Base class providing shared profiling logic.

    Handles start/end time buffering, call counting, lazy HDF5 dataset
    initialization, and flushing data to disk when buffers fill.
    """

    __slots__ = (
        "region_name",
        "config",
        "start_times",
        "end_times",
        "num_calls",
        "ptr",
        "buffer_limit",
        "group_path",
        "local_file_path",
        "hdf5_initialized",
    )

    def __init__(self, region_name: str, config: ProfilingConfig):
        """Initialize a profiling region.

        Parameters
        ----------
        region_name : str
        Name of the profiled region.
        config : ProfilingConfig
        Profiling configuration containing buffer limits,
        file paths, and timing reference.
        """
        self.region_name = region_name
        self.config = config
        self.num_calls = 0

        # Preallocate buffers
        self.ptr = 0
        self.buffer_limit = config.buffer_limit
        self.start_times = np.empty(self.buffer_limit, dtype=np.int64)
        self.end_times = np.empty(self.buffer_limit, dtype=np.int64)

        # Setu p paths
        self.group_path = f"regions/{self.region_name}"
        self.local_file_path = self.config._local_file_path
        self.hdf5_initialized = False

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
        """Append a start/end time pair to the buffer.

        Automatically triggers a flush if the buffer becomes full.
        """

        self.start_times[self.ptr] = start
        self.end_times[self.ptr] = end
        self.ptr += 1
        if self.ptr >= self.buffer_limit:
            self.flush()

    def flush(self):
        """Flush buffered start/end times to the HDF5 file.

        Lazily initializes datasets on the first flush.
        Subsequent flushes append to the existing datasets.
        """
        if self.ptr == 0:
            return

        if not self.hdf5_initialized:
            with h5py.File(self.config._local_file_path, "a") as f:
                grp = f.require_group(f"regions/{self.region_name}")
                for name in ("start_times", "end_times"):
                    if name not in grp:
                        grp.create_dataset(
                            name, shape=(0,), maxshape=(None,), dtype="i8", chunks=True
                        )
            self.hdf5_initialized = True

        with h5py.File(self.config._local_file_path, "a") as f:
            grp = f[f"regions/{self.region_name}"]
            for name, data in [
                ("start_times", self.start_times[: self.ptr]),
                ("end_times", self.end_times[: self.ptr]),
            ]:
                ds = grp[name]
                old_size = ds.shape[0]
                new_size = old_size + self.ptr
                ds.resize((new_size,))
                ds[old_size:new_size] = data

        self.ptr = 0

    def get_durations_numpy(self) -> np.ndarray:
        """Return durations (end - start) for buffered entries as a NumPy array."""
        return self.end_times[: self.ptr] - self.start_times[: self.ptr]

    def get_end_times_numpy(self) -> np.ndarray:
        """Return end times offset by config creation time."""
        return self.end_times[: self.ptr] - self.config.config_creation_time

    def get_start_times_numpy(self) -> np.ndarray:
        """Return start times offset by config creation time."""
        return self.start_times[: self.ptr] - self.config.config_creation_time


# Disabled region: does nothing
class DisabledProfileRegion(BaseProfileRegion):
    """Profiling region that performs no measurements.

    Used when profiling is disabled but code paths must remain valid.
    """

    def wrap(self, func):
        """Return the original function without profiling."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def append(self, start, end):
        """Ignored: no data recorded."""
        pass

    def flush(self):
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


class NCallsOnlyProfileRegion(BaseProfileRegion):
    """Region that records only the number of calls, not timing."""

    def __init__(self, region_name: str, config: ProfilingConfig):
        """Initialize the region without allocating timing buffers."""
        super().__init__(region_name, config)

    def wrap(self, func):
        """Wrap a function and increment the call counter for each invocation."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            out = func(*args, **kwargs)
            return out

        return wrapper

    def append(self, start, end):
        """Ignored: timing information is not stored."""
        pass

    def flush(self):
        """Ignored: no data to flush."""
        pass

    def get_durations_numpy(self):
        """Return an empty array because no timing data is collected."""
        return np.array([])

    def __enter__(self):
        """Increment the call counter when entering the context."""
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context exit does nothing."""
        pass


# Time-only region
class TimeOnlyProfileRegionNoFlush(BaseProfileRegion):
    """Region that records timing but never flushes to disk.

    Used for lightweight profiling where in-memory results are sufficient.
    """

    def wrap(self, func):
        """Wrap a function to measure start and end time without flushing."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            start = np.int64(perf_counter_ns())
            out = func(*args, **kwargs)
            end = np.int64(perf_counter_ns())
            self.start_times[self.ptr] = start
            self.end_times[self.ptr] = end
            self.ptr += 1
            return out

        return wrapper

    def __enter__(self):
        """Record the start time on context entry."""
        self.num_calls += 1
        self.start_times[self.ptr] = np.int64(perf_counter_ns())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record the end time on context exit."""
        self.end_times[self.ptr] = np.int64(perf_counter_ns())
        self.ptr += 1


class TimeOnlyProfileRegion(BaseProfileRegion):
    """Region that records timing and flushes to disk when buffers fill."""

    def wrap(self, func):
        """Wrap a function to measure execution time and flush when needed."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            start = np.int64(perf_counter_ns())
            out = func(*args, **kwargs)
            end = np.int64(perf_counter_ns())
            self.start_times[self.ptr] = start
            self.end_times[self.ptr] = end
            self.ptr += 1
            if self.ptr >= self.buffer_limit:
                self.flush()
            return out

        return wrapper

    def __enter__(self):
        """Record start time and increment call count."""
        self.start_times[self.ptr] = np.int64(perf_counter_ns())
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record end time and flush if needed."""
        self.end_times[self.ptr] = np.int64(perf_counter_ns())
        self.ptr += 1
        if self.ptr >= self.buffer_limit:
            self.flush()


# LIKWID-only region
class LikwidOnlyProfileRegion(BaseProfileRegion):
    """Region that wraps a LIKWID marker region without recording timing.

    This region enables hardware performance counter collection using LIKWID,
    but does not store timing or write data to HDF5. Useful when the user only
    wants LIKWID metrics while still using the unified region API.
    """

    __slots__ = ("likwid_marker_start", "likwid_marker_stop")

    def __init__(self, region_name: str, config: ProfilingConfig):
        """Initialize LIKWID marker callbacks."""
        super().__init__(region_name, config)
        pylikwid = _import_pylikwid()
        self.likwid_marker_start = pylikwid.markerstartregion
        self.likwid_marker_stop = pylikwid.markerstopregion

    def wrap(self, func):
        """Wrap a function to enclose it in a LIKWID marker region."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            self.likwid_marker_start(self.region_name)
            out = func(*args, **kwargs)
            self.likwid_marker_stop(self.region_name)
            return out

        return wrapper

    def __enter__(self):
        """Record start time and increment call count."""
        self.num_calls += 1
        self.likwid_marker_start(self.region_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the LIKWID marker region on context exit."""
        self.likwid_marker_stop(self.region_name)


# Full region: time + LIKWID
class FullProfileRegionNoFlush(BaseProfileRegion):
    """Region that records both timing and LIKWID metrics, without flushing.

    Useful for high-frequency profiling where the user retrieves metrics only
    from in-memory buffers. No HDF5 writes occur.
    """

    __slots__ = ("likwid_marker_start", "likwid_marker_stop")

    def __init__(self, region_name: str, config: ProfilingConfig):
        """Initialize timing buffers and LIKWID marker callbacks."""
        super().__init__(region_name, config)
        pylikwid = _import_pylikwid()
        self.likwid_marker_start = pylikwid.markerstartregion
        self.likwid_marker_stop = pylikwid.markerstopregion

    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            start = np.int64(perf_counter_ns())
            self.likwid_marker_start(self.region_name)
            out = func(*args, **kwargs)
            self.likwid_marker_stop(self.region_name)
            end = np.int64(perf_counter_ns())
            self.start_times[self.ptr] = start
            self.end_times[self.ptr] = end
            self.ptr += 1
            return out

        return wrapper

    def __enter__(self):
        """Start LIKWID region and record start time and increase num_calls by 1."""
        self.likwid_marker_start(self.region_name)
        self.start_times[self.ptr] = np.int64(perf_counter_ns())
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record end time and stop LIKWID region."""
        self.likwid_marker_stop(self.region_name)
        self.end_times[self.ptr] = np.int64(perf_counter_ns())
        self.ptr += 1


class FullProfileRegion(BaseProfileRegion):
    """Region that records both timing and LIKWID metrics, and flushes to HDF5.

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
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            start = np.int64(perf_counter_ns())
            self.likwid_marker_start(self.region_name)
            out = func(*args, **kwargs)
            self.likwid_marker_stop(self.region_name)
            end = np.int64(perf_counter_ns())
            self.start_times[self.ptr] = start
            self.end_times[self.ptr] = end
            self.ptr += 1
            if self.ptr >= self.buffer_limit:
                self.flush()
            return out

        return wrapper

    def __enter__(self):
        """Start LIKWID region and record start time."""
        self.num_calls += 1
        self.start_times[self.ptr] = np.int64(perf_counter_ns())
        self.likwid_marker_start(self.region_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record end time, stop LIKWID region, and flush if needed."""
        self.likwid_marker_stop(self.region_name)
        self.end_times[self.ptr] = np.int64(perf_counter_ns())
        self.ptr += 1
        if self.ptr >= self.buffer_limit:
            self.flush()
