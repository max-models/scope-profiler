import functools
import inspect
import os
from time import perf_counter_ns
from typing import TYPE_CHECKING, Callable, Dict

import h5py
import numpy as np

from scope_profiler.profile_config import ProfilingConfig
from scope_profiler.region_profiler import (
    BaseProfileRegion,
    DisabledProfileRegion,
    FullProfileRegion,
    FullProfileRegionNoFlush,
    LikwidOnlyProfileRegion,
    NCallsOnlyProfileRegion,
    TimeOnlyProfileRegion,
    TimeOnlyProfileRegionNoFlush,
)


class ProfileManager:
    """
    Singleton class to manage and track all ProfileRegion instances.
    """

    _regions = {}
    _config = ProfilingConfig()
    _region_cls = DisabledProfileRegion

    @classmethod
    def _update_region_cls(cls):
        cfg = cls._config
        if not cfg.profiling_activated:
            cls._region_cls = DisabledProfileRegion
        elif cfg.time_trace and cfg.use_likwid:
            if cfg.flush_to_disk:
                cls._region_cls = FullProfileRegion
            else:
                cls._region_cls = FullProfileRegionNoFlush
        elif cfg.time_trace:
            if cfg.flush_to_disk:
                cls._region_cls = TimeOnlyProfileRegion
            else:
                cls._region_cls = TimeOnlyProfileRegionNoFlush
        elif cfg.use_likwid:
            cls._region_cls = LikwidOnlyProfileRegion
        else:
            cls._region_cls = NCallsOnlyProfileRegion

    @classmethod
    def profile_region(cls, region_name) -> BaseProfileRegion:
        """
        Get an existing ProfileRegion by name, or create a new one if it doesn't exist.

        Parameters
        ----------
        region_name: str
            The name of the profiling region.

        Returns
        -------
        ProfileRegion : The ProfileRegion instance.
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

        def decorator(func):
            name = region_name or func.__name__
            region = cls.profile_region(name)
            return region.wrap(func)

        # Support @ProfileManager.profile without parentheses
        if callable(region_name):
            func = region_name
            region_name = None  # reset, so decorator picks func.__name__
            return decorator(func)

        return decorator

    @classmethod
    def finalize(
        cls,
        verbose: bool = True,
    ) -> None:
        config = cls.get_config()

        if not config.profiling_activated:
            return

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
                    rank_file = config.get_local_filepath(r)
                    if not os.path.exists(rank_file):
                        # print("warning: Profiling file is missing!")
                        continue
                    with h5py.File(rank_file, "r") as fin:
                        # Copy all groups from the rank file under /rank<r>
                        fout.copy(fin, f"rank{r}")

                if verbose:
                    # 4. Gather statistics for printing
                    for region_name, region in cls.get_all_regions().items():
                        all_starts = []
                        all_ends = []
                        # Collect from each rank's file
                        for r in range(size):
                            rank_file = config.get_local_filepath(r)
                            if not os.path.exists(rank_file):
                                continue
                            with h5py.File(rank_file, "r") as fin:
                                grp = fin[f"regions/{region_name}"]
                                starts = grp["start_times"][:]
                                ends = grp["end_times"][:]
                                all_starts.append(starts)
                                all_ends.append(ends)

                        if all_starts:
                            starts = np.concatenate(all_starts)
                            ends = np.concatenate(all_ends)
                            durations = ends - starts
                            total_calls = round(len(durations) / size)
                            if total_calls > 0:
                                total_time = durations.sum() / 1e9
                                avg_time = durations.mean() / 1e9
                                min_time = durations.min() / 1e9
                                max_time = durations.max() / 1e9
                                std_time = durations.std() / 1e9
                            else:
                                total_time = avg_time = min_time = max_time = (
                                    std_time
                                ) = 0.0

                            print(f"Region: {region_name}")
                            print(f"  Total Calls : {total_calls}")
                            print(f"  Total Time  : {total_time} s")
                            print(f"  Avg Time    : {avg_time} s")
                            print(f"  Min Time    : {min_time} s")
                            print(f"  Max Time    : {max_time} s")
                            print(f"  Std Dev     : {std_time} s")
                            print("-" * 40)
        if config.use_likwid:
            config.pylikwid_markerclose()

    @classmethod
    def get_region(cls, region_name) -> BaseProfileRegion:
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
    def get_all_regions(cls) -> Dict[str, "BaseProfileRegion"]:
        """
        Get all registered ProfileRegion instances.

        Returns
        -------
        dict: Dictionary of all registered ProfileRegion instances.
        """
        return cls._regions

    @classmethod
    def setup(
        cls,
        profiling_activated: bool = True,
        use_likwid: bool = False,
        time_trace: bool = True,
        flush_to_disk: bool = True,
        buffer_limit: int = 100_000,
        file_path: str = "profiling_data.h5",
    ):
        ProfilingConfig().reset()
        config = ProfilingConfig(
            profiling_activated=profiling_activated,
            use_likwid=use_likwid,
            time_trace=time_trace,
            flush_to_disk=flush_to_disk,
            buffer_limit=buffer_limit,
            file_path=file_path,
        )
        cls.set_config(config=config)

    @classmethod
    def set_config(cls, config: ProfilingConfig) -> None:
        cls._regions.clear()  # Clear old regions
        cls._config = config  # Update the config
        cls._update_region_cls()  # Set the proper region class

    @classmethod
    def get_config(cls) -> ProfilingConfig:
        return cls._config

    @classmethod
    def _reset_regions(cls) -> None:
        cls._regions = {}

    @classmethod
    def _reset_config(cls) -> None:
        ProfilingConfig().reset()
        cls._config = ProfilingConfig()

    @classmethod
    def _reset(cls) -> None:
        cls._reset_regions()
        cls._reset_config()
        cls._update_region_cls()
