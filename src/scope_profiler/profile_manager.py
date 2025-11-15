import functools
import inspect
import os
import time
from typing import TYPE_CHECKING, Callable, Dict

import h5py
import numpy as np

from scope_profiler.profile_config import ProfilingConfig
from scope_profiler.region_profiler import MockProfileRegion, ProfileRegion


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
