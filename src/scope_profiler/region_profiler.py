import functools
from time import perf_counter_ns
from typing import TYPE_CHECKING

import h5py
import numpy as np

from scope_profiler.profile_config import ProfilingConfig

if TYPE_CHECKING:
    from mpi4py.MPI import Intercomm


def _import_pylikwid():
    import pylikwid

    return pylikwid


# Base class with common functionality (flush, append, HDF5 handling)
class BaseProfileRegion:
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
        """Override this in subclasses."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def append(self, start: float, end: float) -> None:
        self.start_times[self.ptr] = start
        self.end_times[self.ptr] = end
        self.ptr += 1
        if self.ptr >= self.buffer_limit:
            self.flush()

    def flush(self):
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
        return self.end_times[: self.ptr] - self.start_times[: self.ptr]

    def get_end_times_numpy(self) -> np.ndarray:
        return self.end_times[: self.ptr] - self.config.config_creation_time

    def get_start_times_numpy(self) -> np.ndarray:
        return self.start_times[: self.ptr] - self.config.config_creation_time


# Disabled region: does nothing
class DisabledProfileRegion(BaseProfileRegion):
    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def append(self, start, end):
        pass

    def flush(self):
        pass

    def get_durations_numpy(self):
        return np.array([])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class NCallsOnlyProfileRegion(BaseProfileRegion):
    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            out = func(*args, **kwargs)
            return out

        return wrapper

    def __init__(self, region_name: str, config: ProfilingConfig):
        super().__init__(region_name, config)

    def append(self, start, end):
        pass

    def flush(self):
        pass

    def get_durations_numpy(self):
        return np.array([])

    def __enter__(self):
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


# Time-only region
class TimeOnlyProfileRegionNoFlush(BaseProfileRegion):
    def wrap(self, func):
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
        self.num_calls += 1
        self.start_times[self.ptr] = np.int64(perf_counter_ns())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_times[self.ptr] = np.int64(perf_counter_ns())
        self.ptr += 1


class TimeOnlyProfileRegion(BaseProfileRegion):
    def wrap(self, func):
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
        self.start_times[self.ptr] = np.int64(perf_counter_ns())
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_times[self.ptr] = np.int64(perf_counter_ns())
        self.ptr += 1
        if self.ptr >= self.buffer_limit:
            self.flush()


# LIKWID-only region
class LikwidOnlyProfileRegion(BaseProfileRegion):
    __slots__ = ("likwid_marker_start", "likwid_marker_stop")

    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            self.likwid_marker_start(self.region_name)
            out = func(*args, **kwargs)
            self.likwid_marker_stop(self.region_name)
            return out

        return wrapper

    def __init__(self, region_name: str, config: ProfilingConfig):
        super().__init__(region_name, config)
        pylikwid = _import_pylikwid()
        self.likwid_marker_start = pylikwid.markerstartregion
        self.likwid_marker_stop = pylikwid.markerstopregion

    def __enter__(self):
        self.num_calls += 1
        self.likwid_marker_start(self.region_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.likwid_marker_stop(self.region_name)


# Full region: time + LIKWID
class FullProfileRegionNoFlush(BaseProfileRegion):
    __slots__ = ("likwid_marker_start", "likwid_marker_stop")

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

    def __init__(self, region_name: str, config: ProfilingConfig):
        super().__init__(region_name, config)
        pylikwid = _import_pylikwid()
        self.likwid_marker_start = pylikwid.markerstartregion
        self.likwid_marker_stop = pylikwid.markerstopregion

    def __enter__(self):
        self.likwid_marker_start(self.region_name)
        self.start_times[self.ptr] = np.int64(perf_counter_ns())
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.likwid_marker_stop(self.region_name)
        self.end_times[self.ptr] = np.int64(perf_counter_ns())
        self.ptr += 1


class FullProfileRegion(BaseProfileRegion):
    __slots__ = ("likwid_marker_start", "likwid_marker_stop")

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

    def __init__(self, region_name: str, config: ProfilingConfig):
        super().__init__(region_name, config)
        pylikwid = _import_pylikwid()
        self.likwid_marker_start = pylikwid.markerstartregion
        self.likwid_marker_stop = pylikwid.markerstopregion

    def __enter__(self):
        self.num_calls += 1
        self.start_times[self.ptr] = np.int64(perf_counter_ns())
        self.likwid_marker_start(self.region_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.likwid_marker_stop(self.region_name)
        self.end_times[self.ptr] = np.int64(perf_counter_ns())
        self.ptr += 1
        if self.ptr >= self.buffer_limit:
            self.flush()
