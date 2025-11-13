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
import time

# Import the profiling configuration class and context manager
from functools import lru_cache
from typing import Callable, Dict

import h5py
import numpy as np


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
            cls._instance.profiling_activated = True
            cls._instance.use_likwid = False
            cls._instance.time_trace = False
            cls._instance.flush_to_disk = False
        return cls._instance

    def __init__(
        self,
        profiling_activated: bool = True,
        use_likwid: bool = False,
        time_trace: bool = True,
        flush_to_disk: bool = False,
    ):

        if self._initialized:
            return

        # Only update if value provided
        self.profiling_activated = profiling_activated
        self.use_likwid = use_likwid
        self.time_trace = time_trace
        self.flush_to_disk = flush_to_disk

        self._pylikwid = None
        if self.use_likwid:
            try:
                import pylikwid

                self._pylikwid = pylikwid
            except ImportError as e:
                raise ImportError(
                    "LIKWID profiling requested but pylikwid module not installed"
                ) from e
        self._initialized = True

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
    def profiling_activated(self) -> bool:
        return self._profiling_activated

    @profiling_activated.setter
    def profiling_activated(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._profiling_activated = value

    @property
    def use_likwid(self) -> bool:
        return self._likwid

    @use_likwid.setter
    def use_likwid(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._likwid = value

    @property
    def time_trace(self) -> bool:
        return self._time_trace

    @property
    def flush_to_disk(self) -> bool:
        return self._flush_to_disk

    @flush_to_disk.setter
    def flush_to_disk(self, value) -> None:
        if not isinstance(value, bool):
            raise TypeError("flush_to_disk must be a bool")
        self._flush_to_disk = value

    @time_trace.setter
    def time_trace(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._time_trace = value


class ProfileRegion:
    """Context manager for profiling specific code regions using LIKWID markers."""

    def __init__(
        self,
        region_name: str,
        time_trace: bool = False,
        buffer_limit: int = 100000,
        file_path: str | None = None,
        flush_to_disk: bool = False,
        profiling_activated: bool = True,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._config = ProfilingConfig()
        self._region_name = region_name
        self._time_trace = time_trace
        self._buffer_limit = buffer_limit
        self._file_path = file_path or "profiling_data.h5"
        self._flush_to_disk = flush_to_disk
        self._profiling_activated = profiling_activated

        # Timer data
        self._ncalls = 0
        self._start_times = []
        self._end_times = []
        self._duration = 0.0
        self._started = False

        # Create file and datasets if not existing
        if self.flush_to_disk and self._time_trace:
            with h5py.File(self._file_path, "a") as f:
                grp = f.require_group(f"regions/{self._region_name}")
                for name in ["start_times", "end_times", "durations"]:
                    if name not in grp:
                        grp.create_dataset(
                            name,
                            shape=(0,),
                            maxshape=(None,),
                            dtype="f8",
                            chunks=True,
                            compression="gzip",
                        )

    def __enter__(self):
        if not self.profiling_activated:
            return self
        if self.config.use_likwid:
            self._pylikwid().markerstartregion(self.region_name)

        if self._time_trace:

            self._start_time = time.perf_counter()
            self._start_times.append(self._start_time)
            self._started = True

        self._ncalls += 1

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.profiling_activated:
            return
        if self.config.use_likwid:
            self._pylikwid().markerstopregion(self.region_name)
        if self._time_trace and self.started:
            end_time = time.perf_counter()
            self._end_times.append(end_time)
            self._started = False

            if self.flush_to_disk and len(self._start_times) >= self._buffer_limit:
                self.flush()

    def flush(self) -> None:
        """Append buffered profiling data to the HDF5 file and clear memory."""
        if not self._start_times:
            return

        starts = self.start_times  # np.array(self._start_times, dtype=np.float64)
        ends = self.end_times  # np.array(self._end_times, dtype=np.float64)
        durations = self.durations

        with h5py.File(self._file_path, "a") as f:
            grp = f.require_group(f"regions/{self._region_name}")
            for name, data in [
                ("start_times", starts),
                ("end_times", ends),
                ("durations", durations),
            ]:
                ds = grp[name]
                old_size = ds.shape[0]
                new_size = old_size + len(data)
                ds.resize((new_size,))
                ds[old_size:new_size] = data

        self._start_times.clear()
        self._end_times.clear()

    def _pylikwid(self):
        return _import_pylikwid()

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
        return np.array(self._end_times)

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
        return np.array(self._start_times)

    @property
    def started(self) -> bool:
        return self._started


class ProfileManager:
    """
    Singleton class to manage and track all ProfileRegion instances.
    """

    _regions = {}

    @classmethod
    def reset(cls) -> None:
        cls._regions = {}

    @classmethod
    def profile_region(cls, region_name) -> ProfileRegion:
        """
        Get an existing ProfileRegion by name, or create a new one if it doesn't exist.

        Parameters
        ----------
        region_name: str
            The name of the profiling region.

        Returns
        -------
        ProfileRegion: The ProfileRegion instance.
        """
        if region_name in cls._regions:
            # print(f"Using existing region '{region_name}'...")
            return cls._regions[region_name]
        else:
            # print(f"Creating new region '{region_name}'...")
            # Create and register a new ProfileRegion
            cls._regions[region_name] = ProfileRegion(
                region_name,
                time_trace=ProfilingConfig().time_trace,
                flush_to_disk=ProfilingConfig().flush_to_disk,
                profiling_activated=ProfilingConfig().profiling_activated,
            )
            return cls._regions[region_name]

    @classmethod
    def profile(cls, region_name: str | None = None) -> Callable:
        """
        Decorator factory for profiling a function.

        Usage:
          @ProfileManager.profile               # region name defaults to function.__name__
          def foo(...): ...

          @ProfileManager.profile("myregion")
          def bar(...): ...
        """

        def decorator(func: Callable) -> Callable:
            # Default to function.__name__ if region_name is None
            name = region_name or func.__name__
            region = cls.profile_region(name)

            if inspect.iscoroutinefunction(func):
                # async function wrapper
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with region:
                        return await func(*args, **kwargs)

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with region:
                        return func(*args, **kwargs)

                return sync_wrapper

        # If decorator used without parentheses: @ProfileManager.profile
        # Python will pass the function directly to the decorator factory call,
        # but because this is a factory we should allow that too:
        if callable(region_name):
            # invoked as @ProfileManager.profile with no args
            func = region_name
            region_name = None
            return decorator(func)

        return decorator

    @classmethod
    def finalize(cls) -> None:
        if ProfilingConfig().flush_to_disk:
            for name, region in cls.get_all_regions().items():
                region.flush()

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

        _config = ProfilingConfig()
        if not _config.time_trace:
            print(
                "time_trace is not set to True --> Time traces are not measured --> Skip printing summary...",
            )
            return

        print("Profiling Summary:")
        print("=" * 40)
        for name, region in cls._regions.items():
            if region.num_calls > 0:
                total_duration = sum(region.durations)
                average_duration = total_duration / region.num_calls
                min_duration = min(region.durations)
                max_duration = max(region.durations)
                std_duration = np.std(region.durations)
            else:
                total_duration = average_duration = min_duration = max_duration = (
                    std_duration
                ) = 0

            print(f"Region: {name}")
            print(f"  Number of Calls: {region.num_calls}")
            print(f"  Total Duration: {total_duration:.6f} seconds")
            print(f"  Average Duration: {average_duration:.6f} seconds")
            print(f"  Min Duration: {min_duration:.6f} seconds")
            print(f"  Max Duration: {max_duration:.6f} seconds")
            print(f"  Std Deviation: {std_duration:.6f} seconds")
            print("-" * 40)
