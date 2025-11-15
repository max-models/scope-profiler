"""
profiling.py

This module provides a centralized profiling configuration and management system
using LIKWID markers. It includes:
- A singleton class for managing profiling configuration.
- A context manager for profiling specific code regions.
- Initialization and cleanup functions for LIKWID markers.
- Convenience functions for setting and getting the profiling configuration.

LIKWID is imported only when profiling is enabled to avoid unnecessary overhead.
"""

import functools
import inspect
import os
import tempfile
import time

# Import the profiling configuration class and context manager
from functools import lru_cache
from typing import Callable, Dict

import h5py
import numpy as np
from mpi4py.MPI import Comm


@lru_cache(maxsize=None)  # Cache the import result to avoid repeated imports
def _import_pylikwid():
    import pylikwid

    return pylikwid


class ProfilingConfig:
    """Singleton class for managing global profiling configuration."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Default values
            cls._instance.comm = None
            cls._instance.profiling_activated = True
            cls._instance.use_likwid = False
            cls._instance.time_trace = False
            cls._instance.flush_to_disk = True
            cls._instance.buffer_limit = 10_000
            cls._instance.file_path = "profiling_data.h5"
        return cls._instance

    def __init__(
        self,
        comm: Comm | None = None,
        profiling_activated: bool = True,
        use_likwid: bool = False,
        time_trace: bool = True,
        flush_to_disk: bool = True,
        buffer_limit: int = 10_000,
        file_path: str = "profiling_data.h5",
    ):
        if self._initialized:
            return

        self.config_creation_time = time.perf_counter_ns()

        # Only update if value provided
        self.comm = comm
        self.profiling_activated = profiling_activated
        self.use_likwid = use_likwid
        self.time_trace = time_trace
        self.flush_to_disk = flush_to_disk
        self.buffer_limit = buffer_limit
        self.file_path = file_path

        comm = self.comm  # TODO, just use MPI.COMM_WORLD
        self._rank = 0 if comm is None else comm.Get_rank()
        self._size = 1 if comm is None else comm.Get_size()

        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="profile_h5_")
        self.temp_dir = self.temp_dir_obj.name

        self._global_file_path = self.file_path

        # Temporary file with rank-specific timings
        self._local_file_path = self.get_local_filepath(self._rank)

        self._pylikwid = None
        if self.use_likwid:
            try:
                self._pylikwid = _import_pylikwid()
                self.pylikwid_markerinit()
            except ImportError as e:
                raise ImportError(
                    "LIKWID profiling requested but pylikwid module not installed"
                ) from e
        self._initialized = True

    def get_local_filepath(self, rank):
        return os.path.join(self.temp_dir, f"rank_{rank}.h5")

    @classmethod
    def reset(cls):
        """Reset the singleton so it can be reinitialized."""
        cls._instance = None
        cls._initialized = False

    def pylikwid_markerinit(self):
        """Initialize LIKWID profiling markers."""
        if self.use_likwid and self._pylikwid:
            self._pylikwid.markerinit()

    def pylikwid_markerclose(self):
        """Close LIKWID profiling markers."""
        if self.use_likwid and self._pylikwid:
            self._pylikwid.markerclose()

    @property
    def comm(self) -> Comm | None:
        return self._comm

    @comm.setter
    def comm(self, value: Comm | None) -> None:
        assert value is None or isinstance(value, Comm)
        self._comm = value

    @property
    def profiling_activated(self) -> bool:
        return self._profiling_activated

    @profiling_activated.setter
    def profiling_activated(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._profiling_activated = value

    @property
    def buffer_limit(self) -> int:
        return self._buffer_limit

    @buffer_limit.setter
    def buffer_limit(self, value: int) -> None:
        assert isinstance(value, int)
        self._buffer_limit = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        assert isinstance(value, str)
        self._file_path = value

    @property
    def use_likwid(self) -> bool:
        return self._likwid

    @use_likwid.setter
    def use_likwid(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._likwid = value

    @property
    def flush_to_disk(self) -> bool:
        return self._flush_to_disk

    @flush_to_disk.setter
    def flush_to_disk(self, value) -> None:
        if not isinstance(value, bool):
            raise TypeError("flush_to_disk must be a bool")
        self._flush_to_disk = value

    @property
    def time_trace(self) -> bool:
        return self._time_trace

    @time_trace.setter
    def time_trace(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._time_trace = value

    @property
    def config_creation_time(self) -> int:
        return self._config_creation_time

    @config_creation_time.setter
    def config_creation_time(self, value: int) -> None:
        assert isinstance(value, int)
        self._config_creation_time = value


class MockProfileRegion:
    """A dummy ProfileRegion that does nothing, used when profiling is disabled."""

    def __init__(self, region_name, config=None):
        self._region_name = region_name
        self._ncalls = 0
        self._started = False

    def append(self, start, end):
        pass

    def flush(self):
        pass

    @property
    def region_name(self):
        return self._region_name

    @property
    def num_calls(self):
        return self._ncalls

    @property
    def started(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class ProfileRegion:
    """Context manager for profiling specific code regions using LIKWID markers."""

    def __init__(
        self,
        region_name: str,
        config: ProfilingConfig,
    ):
        self._region_name = region_name
        self._config = config

        self._comm = self.config.comm
        self._time_trace = self.config.time_trace
        self._buffer_limit = self.config.buffer_limit

        self._flush_to_disk = self.config.flush_to_disk
        self._profiling_activated = self.config.profiling_activated

        # Timer data
        self._ncalls = 0
        self._start_times = []
        self._end_times = []
        self._duration = 0.0
        self._started = False

        # Construct per-rank filename
        region_group = f"regions/{self._region_name}"
        with h5py.File(self.config._local_file_path, "a") as f:
            grp = f.require_group(region_group)
            for name in ("start_times", "end_times", "durations"):
                if name not in grp:
                    grp.create_dataset(
                        name,
                        shape=(0,),
                        maxshape=(None,),
                        dtype="i8",
                        chunks=True,
                        # compression="gzip",
                    )

        if self.config.use_likwid:
            self._likwid_marker_start = _import_pylikwid().markerstartregion
            self._likwid_marker_stop = _import_pylikwid().markerstopregion
        else:
            self._likwid_marker_start = None
            self._likwid_marker_stop = None

    def append(self, start: float, end: float) -> None:
        """Append a timing directly (used by decorator for speed)."""
        if not self._profiling_activated or not self._time_trace:
            return
        self._start_times.append(start)
        self._end_times.append(end)
        if self._flush_to_disk and len(self._start_times) >= self._buffer_limit:
            self.flush()

    def flush(self) -> None:
        """Append buffered profiling data to the HDF5 file and clear memory."""
        if not self._start_times:
            return

        starts = self.start_times
        ends = self.end_times
        durations = self.durations

        with h5py.File(self.config._local_file_path, "a") as f:
            grp = f.require_group(f"regions/{self._region_name}")
            for name, data in [
                ("start_times", starts),
                ("end_times", ends),
                ("durations", durations),
            ]:
                if name in grp:
                    ds = grp[name]
                    old_size = ds.shape[0]
                    new_size = old_size + len(data)
                    ds.resize((new_size,))
                    ds[old_size:new_size] = data
                else:
                    grp.create_dataset(
                        name,
                        data=data,
                        maxshape=(None,),
                        chunks=True,
                        dtype="i8",
                        # compression="gzip",  # optional
                    )

        self._start_times.clear()
        self._end_times.clear()

    @property
    def comm(self) -> Comm | None:
        return self._comm

    @property
    def profiling_activated(self) -> bool:
        return self._profiling_activated

    @property
    def config(self) -> ProfilingConfig:
        return self._config

    @property
    def durations(self) -> np.ndarray:
        return self.end_times - self.start_times

    @property
    def end_times(self) -> np.ndarray:
        return np.array(self._end_times, dtype=int) - self.config.config_creation_time

    @property
    def flush_to_disk(self) -> bool:
        return self._flush_to_disk

    @property
    def num_calls(self) -> int:
        return self._ncalls

    @property
    def region_name(self) -> str:
        return self._region_name

    @property
    def start_times(self) -> np.ndarray:
        return np.array(self._start_times, dtype=int) - self.config.config_creation_time

    @property
    def started(self) -> bool:
        return self._started

    def __enter__(self):
        if not self.profiling_activated:
            return self

        # Pylikwid markerstartregion
        if self._likwid_marker_start:
            self._likwid_marker_start(self.region_name)

        if self._time_trace:
            self._start_time = time.perf_counter_ns()
            self._start_times.append(self._start_time)
            self._started = True

        self._ncalls += 1

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.profiling_activated:
            return

        # Pylikwid markerstartregion
        if self._likwid_marker_stop:
            self._likwid_marker_stop(self.region_name)

        if self._time_trace and self.started:
            end_time = time.perf_counter_ns()
            self._end_times.append(end_time)
            self._started = False

            if self.flush_to_disk and len(self._start_times) >= self._buffer_limit:
                self.flush()


class ProfileManager:
    """
    Singleton class to manage and track all ProfileRegion instances.
    """

    _regions = {}
    _config = ProfilingConfig()
    _region_cls = ProfileRegion if _config.profiling_activated else MockProfileRegion

    @classmethod
    def reset(cls) -> None:
        cls._regions = {}
        cls._config = ProfilingConfig()
        cls._region_cls = (
            ProfileRegion if cls._config.profiling_activated else MockProfileRegion
        )

    @classmethod
    def profile_region(cls, region_name) -> ProfileRegion | MockProfileRegion:
        """
        Get an existing ProfileRegion by name, or create a new one if it doesn't exist.

        Parameters
        ----------
        region_name: str
            The name of the profiling region.

        Returns
        -------
        ProfileRegion | MockProfileRegion: The ProfileRegion instance.
        """

        return cls._regions.setdefault(
            region_name,
            cls._region_cls(region_name, config=cls._config),
        )

    @classmethod
    def profile(cls, region_name: str | None = None) -> Callable:
        """
        Decorator factory for profiling a function.
        """

        def decorator(func: Callable) -> Callable:
            name = region_name or func.__name__
            # ALWAYS create the region object in the dictionary
            region = cls.profile_region(name)
            config = cls.get_config()

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    # print(f"Calling wrapped function {name}")
                    if config.profiling_activated:
                        region._ncalls += 1
                        if config.time_trace:
                            start = time.perf_counter_ns()
                            try:
                                return await func(*args, **kwargs)
                            finally:
                                end = time.perf_counter_ns()
                                region.append(start, end)
                        else:
                            return await func(*args, **kwargs)

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    # print(f"Calling wrapped function {name}")
                    if config.profiling_activated:
                        region._ncalls += 1
                        if config.time_trace:
                            start = time.perf_counter_ns()
                            try:
                                return func(*args, **kwargs)
                            finally:
                                end = time.perf_counter_ns()
                                region.append(start, end)
                        else:
                            return func(*args, **kwargs)

                return sync_wrapper

        # Support @ProfileManager.profile without parentheses
        if callable(region_name):
            func = region_name
            region_name = None  # reset, so decorator picks func.__name__
            return decorator(func)

        return decorator

    @classmethod
    def finalize(cls) -> None:
        config = cls.get_config()
        comm = config.comm
        rank = config._rank
        size = config._size

        # 1. Flush all buffered regions to per-rank files
        if config.flush_to_disk:
            for region in cls.get_all_regions().values():
                region.flush()

        # 2. Barrier to ensure all ranks finished flushing
        if comm is not None:
            comm.Barrier()

        # 3. Only rank 0 performs the merge
        if rank == 0:
            merged_file_path = config.file_path
            with h5py.File(merged_file_path, "w") as fout:
                for r in range(size):
                    rank_file = config.get_local_filepath(rank)
                    if not os.path.exists(rank_file):
                        # print("warning: Profiling file is missing!")
                        continue
                    with h5py.File(rank_file, "r") as fin:
                        # Copy all groups from the rank file under /rank<r>
                        fout.copy(fin, f"rank{r}")

        if config.use_likwid:
            config.pylikwid_markerclose()

    @classmethod
    def get_region(cls, region_name) -> ProfileRegion:
        """
        Get a registered ProfileRegion by name.

        Parameters
        ----------
        region_name: str
            The name of the profiling region.

        Returns
        -------
        ProfileRegion or None: The registered ProfileRegion instance or None if not found.
        """
        return cls._regions.get(region_name)

    @classmethod
    def get_all_regions(cls) -> Dict[str, "ProfileRegion"]:
        """
        Get all registered ProfileRegion instances.

        Returns
        -------
        dict: Dictionary of all registered ProfileRegion instances.
        """
        return cls._regions

    @classmethod
    def print_summary(cls) -> None:
        """
        Print a summary of the profiling data for all regions.
        """

        if not cls._config.time_trace:
            print(
                "time_trace is not set to True --> Time traces are not measured --> Skip printing summary...",
            )
            return

        print("Profiling Summary:")
        print("=" * 40)
        for name, region in cls._regions.items():
            if region.num_calls > 0:
                total_duration = sum(region.durations) / 1e9
                average_duration = (total_duration / region.num_calls) / 1e9
                min_duration = min(region.durations) / 1e9
                max_duration = max(region.durations) / 1e9
                std_duration = np.std(region.durations) / 1e9
            else:
                total_duration = average_duration = min_duration = max_duration = (
                    std_duration
                ) = 0

            print(f"Region: {name}")
            print(f"  Number of Calls: {region.num_calls}")
            print(f"  Total Duration: {total_duration} seconds")
            print(f"  Average Duration: {average_duration} seconds")
            print(f"  Min Duration: {min_duration} seconds")
            print(f"  Max Duration: {max_duration} seconds")
            print(f"  Std Deviation: {std_duration} seconds")
            print("-" * 40)

    @classmethod
    def set_config(cls, config: ProfilingConfig) -> None:
        cls._config = config

    @classmethod
    def get_config(cls) -> ProfilingConfig:
        return cls._config
