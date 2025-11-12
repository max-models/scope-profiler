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

import os
import pickle
import time

# Import the profiling configuration class and context manager
from functools import lru_cache
from typing import Dict

import numpy as np


@lru_cache(maxsize=None)  # Cache the import result to avoid repeated imports
def _import_pylikwid():
    import pylikwid

    return pylikwid


class ProfilingConfig:
    """Singleton class for managing global profiling configuration."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Default values
            cls._instance.use_likwid = False
            cls._instance.simulation_label = ""
            cls._instance.sample_duration = 1.0
            cls._instance.sample_interval = 1.0
            cls._instance.time_trace = False
        return cls._instance

    def __init__(
        self,
        use_likwid: bool = False,
        simulation_label: str = "",
        sample_duration: float | int = 1.0,
        sample_interval: float | int = 1.0,
        time_trace: bool = True,
    ):
        # Only update if value provided
        self.use_likwid = use_likwid
        self.simulation_label = simulation_label
        self.sample_duration = sample_duration
        self.sample_interval = sample_interval
        self.time_trace = time_trace

        self._pylikwid = None
        if self.use_likwid:
            try:
                import pylikwid

                self._pylikwid = pylikwid
            except ImportError as e:
                raise ImportError(
                    "LIKWID profiling requested but pylikwid module not installed"
                ) from e

    def pylikwid_markerinit(self):
        """Initialize LIKWID profiling markers."""
        if self.use_likwid and self._pylikwid:
            self._pylikwid.markerinit()

    def pylikwid_markerclose(self):
        """Close LIKWID profiling markers."""
        if self.use_likwid and self._pylikwid:
            self._pylikwid.markerclose()

    @property
    def use_likwid(self) -> bool:
        return self._likwid

    @use_likwid.setter
    def use_likwid(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._likwid = value

    @property
    def simulation_label(self) -> str:
        return self._simulation_label

    @simulation_label.setter
    def simulation_label(self, value: str) -> None:
        assert isinstance(value, str)
        self._simulation_label = value

    @property
    def sample_duration(self) -> float:
        return self._sample_duration

    @sample_duration.setter
    def sample_duration(self, value) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("sample_duration must be a float")
        self._sample_duration = float(value)

    @property
    def sample_interval(self) -> float:
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, value) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("sample_interval must be a float")
        self._sample_interval = float(value)

    @property
    def time_trace(self) -> bool:
        return self._time_trace

    @time_trace.setter
    def time_trace(self, value: bool) -> None:
        assert isinstance(value, bool)
        if value:
            assert (
                self.sample_interval is not None
            ), "sample_interval must be set first!"
            assert (
                self.sample_duration is not None
            ), "sample_duration must be set first!"
        self._time_trace = value


class ProfileRegion:
    """Context manager for profiling specific code regions using LIKWID markers."""

    def __init__(self, region_name: str, time_trace: bool = False):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._config = ProfilingConfig()
        self._region_name = self.config.simulation_label + region_name
        self._time_trace = time_trace
        self._ncalls = 0
        self._start_times = np.empty(1, dtype=float)
        self._end_times = np.empty(1, dtype=float)
        self._duration = 0.0
        self._started = False

    def __enter__(self):

        if self.config.use_likwid:
            self._pylikwid().markerstartregion(self.region_name)

        if self._time_trace:
            if self._ncalls == len(self._start_times):
                self._start_times = np.append(
                    self._start_times,
                    np.zeros_like(self._start_times),
                )
                self._end_times = np.append(
                    self._end_times, np.zeros_like(self._end_times)
                )

            self._start_time = time.perf_counter()
            if (
                self._start_time % self.config.sample_interval
                < self.config.sample_duration
                or self._ncalls == 0
            ):
                self._start_times[self._ncalls] = self._start_time
                self._started = True

        self._ncalls += 1

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.config.use_likwid:
            self._pylikwid().markerstopregion(self.region_name)
        if self._time_trace and self.started:
            end_time = time.perf_counter()
            self._end_times[self._ncalls - 1] = end_time
            self._started = False

    def _pylikwid(self):
        return _import_pylikwid()

    @property
    def config(self) -> ProfilingConfig:
        return self._config

    @property
    def durations(self) -> np.ndarray:
        return self.end_times - self.start_times

    @property
    def end_times(self) -> np.ndarray:
        return self._end_times

    @property
    def num_calls(self) -> int:
        return self._ncalls

    @property
    def region_name(self) -> str:
        return self._region_name

    @property
    def start_times(self) -> np.ndarray:
        return self._start_times

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
            return cls._regions[region_name]
        else:
            # Create and register a new ProfileRegion
            cls._regions[region_name] = ProfileRegion(
                region_name,
                time_trace=ProfilingConfig().time_trace,
            )
            return cls._regions[region_name]

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
    def save_to_pickle(cls, file_path) -> None:
        """
        Save profiling data to a single file using pickle and NumPy arrays in parallel.

        Parameters
        ----------
        file_path: str
            Path to the file where data will be saved.
        """

        _config = ProfilingConfig()
        if not _config.time_trace:
            print(
                "time_trace is not set to True --> Time traces are not measured --> Skip saving...",
            )
            return

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # size = comm.Get_size()

        # Prepare the data to be gathered
        local_data = {}
        for name, region in cls._regions.items():
            local_data[name] = {
                "num_calls": region.num_calls,
                "durations": np.array(region.durations, dtype=np.float64),
                "start_times": np.array(region.start_times, dtype=np.float64),
                "end_times": np.array(region.end_times, dtype=np.float64),
                "config": {
                    "likwid": region.config.likwid,
                    "simulation_label": region.config.simulation_label,
                    "sample_duration": region.config.sample_duration,
                    "sample_interval": region.config.sample_interval,
                },
            }

        # Gather all data at the root process (rank 0)
        all_data = comm.gather(local_data, root=0)

        # Save the likwid configuration data
        likwid_data = {}
        if ProfilingConfig().likwid:
            pylikwid = _import_pylikwid()

            # Gather LIKWID-specific information
            pylikwid.inittopology()
            likwid_data["cpu_info"] = pylikwid.getcpuinfo()
            likwid_data["cpu_topology"] = pylikwid.getcputopology()
            pylikwid.finalizetopology()

            likwid_data["numa_info"] = pylikwid.initnuma()
            pylikwid.finalizenuma()

            likwid_data["affinity_info"] = pylikwid.initaffinity()
            pylikwid.finalizeaffinity()

            pylikwid.initconfiguration()
            likwid_data["configuration"] = pylikwid.getconfiguration()
            pylikwid.destroyconfiguration()

            likwid_data["groups"] = pylikwid.getgroups()

        if rank == 0:
            # Combine the data from all processes
            combined_data = {
                "config": None,
                "rank_data": {f"rank_{i}": data for i, data in enumerate(all_data)},
            }

            # Add the likwid data
            if likwid_data:
                combined_data["config"] = likwid_data

            # Convert the file path to an absolute path
            absolute_path = os.path.abspath(file_path)

            # Save the combined data using pickle
            with open(absolute_path, "wb") as file:
                pickle.dump(combined_data, file)

            print(f"Data saved to {absolute_path}")

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
