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
        "group_path",
        "local_file_path",
        "hdf5_initialized",
    )

    def __init__(self, region_name: str, config: ProfilingConfig):
        self.region_name = region_name
        self.config = config
        self.num_calls = 0
        self.start_times = []
        self.end_times = []
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
        self.start_times.append(start)
        self.end_times.append(end)
        if (
            self.config.flush_to_disk
            and len(self.start_times) >= self.config.buffer_limit
        ):
            self.flush()

    def flush(self):
        if not self.start_times:
            return

        with h5py.File(self.local_file_path, "a") as f:
            grp = f.require_group(self.group_path)

            if not self.hdf5_initialized:
                # Only create datasets once
                for name in ("start_times", "end_times"):
                    if name not in grp:
                        grp.create_dataset(
                            name, shape=(0,), maxshape=(None,), dtype="i8", chunks=True
                        )
                self.hdf5_initialized = True

            ds_start = grp["start_times"]
            ds_end = grp["end_times"]

            old_size = ds_start.shape[0]
            new_size = old_size + len(self.start_times)
            ds_start.resize((new_size,))
            ds_end.resize((new_size,))
            ds_start[old_size:new_size] = np.array(self.start_times, dtype=int)
            ds_end[old_size:new_size] = np.array(self.end_times, dtype=int)

        self.start_times.clear()
        self.end_times.clear()

    def get_durations_numpy(self) -> np.ndarray:
        return self.get_end_times_numpy() - self.get_start_times_numpy()

    def get_end_times_numpy(self) -> np.ndarray:
        return np.array(self.end_times, dtype=int) - self.config.config_creation_time

    def get_start_times_numpy(self) -> np.ndarray:
        return np.array(self.start_times, dtype=int) - self.config.config_creation_time


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
            start = perf_counter_ns()
            out = func(*args, **kwargs)
            end = perf_counter_ns()
            self.start_times.append(start)
            self.end_times.append(end)
            return out

        return wrapper

    def __enter__(self):
        self.start_times.append(perf_counter_ns())
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_times.append(perf_counter_ns())


class TimeOnlyProfileRegion(BaseProfileRegion):
    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            start = perf_counter_ns()
            out = func(*args, **kwargs)
            end = perf_counter_ns()
            self.start_times.append(start)
            self.end_times.append(end)
            if (
                self.config.flush_to_disk
                and len(self.start_times) >= self.config.buffer_limit
            ):
                self.flush()
            return out

        return wrapper

    def __enter__(self):
        self.start_times.append(perf_counter_ns())
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_times.append(perf_counter_ns())
        if (
            self.config.flush_to_disk
            and len(self.start_times) >= self.config.buffer_limit
        ):
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
            start = perf_counter_ns()
            self.likwid_marker_start(self.region_name)
            out = func(*args, **kwargs)
            self.likwid_marker_stop(self.region_name)
            end = perf_counter_ns()
            self.start_times.append(start)
            self.end_times.append(end)
            return out

        return wrapper

    def __init__(self, region_name: str, config: ProfilingConfig):
        super().__init__(region_name, config)
        pylikwid = _import_pylikwid()
        self.likwid_marker_start = pylikwid.markerstartregion
        self.likwid_marker_stop = pylikwid.markerstopregion

    def __enter__(self):
        self.likwid_marker_start(self.region_name)
        self.start_times.append(perf_counter_ns())
        self.num_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.likwid_marker_stop(self.region_name)
        self.end_times.append(perf_counter_ns())


class FullProfileRegion(BaseProfileRegion):
    __slots__ = ("likwid_marker_start", "likwid_marker_stop")

    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.num_calls += 1
            start = perf_counter_ns()
            self.likwid_marker_start(self.region_name)
            out = func(*args, **kwargs)
            self.likwid_marker_stop(self.region_name)
            end = perf_counter_ns()
            self.start_times.append(start)
            self.end_times.append(end)
            if (
                self.config.flush_to_disk
                and len(self.start_times) >= self.config.buffer_limit
            ):
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
        self.start_times.append(perf_counter_ns())
        self.likwid_marker_start(self.region_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.likwid_marker_stop(self.region_name)
        self.end_times.append(perf_counter_ns())
        if (
            self.config.flush_to_disk
            and len(self.start_times) >= self.config.buffer_limit
        ):
            self.flush()
