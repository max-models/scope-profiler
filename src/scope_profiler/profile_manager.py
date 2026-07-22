"""Singleton manager for creating, configuring, and finalizing profiling regions."""

import functools
import os
import runpy
import site
import sys
import sysconfig
import threading
from types import FrameType
from typing import Callable, Dict

import h5py
import numpy as np

from scope_profiler.profile_config import ProfilingConfig
from scope_profiler.region_profiler import (
    BaseProfileRegion,
    DisabledProfileRegion,
    FullProfileRegion,
    FullProfileRegionNoFlush,
    LikwidOnlyProfileRegion,
    LineProfilerRegion,
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
    _decorators: Dict[str, list] = {}  # name -> [(func, _bound), ...]
    _decorated_codes = set()
    _recursive_state = threading.local()
    _user_code_cache: Dict[object, bool] = {}
    _system_prefixes = None
    _internal_modules = {
        "scope_profiler.profile_manager",
        "scope_profiler.region_profiler",
        "scope_profiler.profile_config",
    }

    @classmethod
    def _is_internal_frame(cls, frame: FrameType) -> bool:
        module_name = frame.f_globals.get("__name__", "")
        return module_name in cls._internal_modules

    @classmethod
    def _frame_region_name(cls, frame: FrameType) -> str:
        module_name = frame.f_globals.get("__name__", "<unknown>")
        qualname = frame.f_code.co_qualname
        return f"{module_name}.{qualname}"

    @classmethod
    def _system_path_prefixes(cls):
        """Realpaths of the stdlib and installed-package directories.

        Computed once and cached; used by ``_is_user_code`` to skip
        instrumenting non-user code when tracing a whole script.
        """
        if cls._system_prefixes is None:
            prefixes = set()
            try:
                paths = sysconfig.get_paths()
                for key in ("stdlib", "platstdlib", "purelib", "platlib"):
                    path = paths.get(key)
                    if path:
                        prefixes.add(os.path.realpath(path))
            except Exception:
                pass
            try:
                for path in site.getsitepackages():
                    prefixes.add(os.path.realpath(path))
            except Exception:
                pass
            try:
                path = site.getusersitepackages()
                if path:
                    prefixes.add(os.path.realpath(path))
            except Exception:
                pass
            cls._system_prefixes = tuple(sorted(prefixes))
        return cls._system_prefixes

    @classmethod
    def _is_user_code(cls, code) -> bool:
        """Whether a code object belongs to user code (not stdlib/site-packages).

        Results are memoized per code object, so the (relatively) expensive
        path check only ever runs once per distinct function traced.
        """
        cached = cls._user_code_cache.get(code)
        if cached is not None:
            return cached

        filename = code.co_filename
        if not filename or filename[0] == "<":
            # e.g. "<frozen importlib._bootstrap>", "<string>": not real user files.
            result = False
        else:
            real_path = os.path.realpath(filename)
            result = not real_path.startswith(cls._system_path_prefixes())

        cls._user_code_cache[code] = result
        return result

    @classmethod
    def _get_recursive_tracer(
        cls,
        root_frame: FrameType,
        prev_profiler,
        only_user_code: bool = False,
    ):
        active_calls = {}

        def tracer(frame: FrameType, event: str, arg):
            if event == "call":
                if frame is root_frame:
                    pass
                elif cls._is_internal_frame(frame):
                    pass
                elif frame.f_code in cls._decorated_codes:
                    # Skip functions that already have explicit decorators to
                    # avoid counting the same call in two regions.
                    pass
                elif only_user_code and not cls._is_user_code(frame.f_code):
                    pass
                else:
                    region = cls.profile_region(cls._frame_region_name(frame))
                    region.__enter__()
                    active_calls[frame] = region
            elif event == "return":
                region = active_calls.pop(frame, None)
                if region is not None:
                    region.__exit__(None, None, None)

            if prev_profiler is not None:
                prev_profiler(frame, event, arg)
            return tracer

        return tracer

    @classmethod
    def _update_region_cls(cls):
        """
        Update the active region class based on current configuration settings.

        Selects the appropriate ProfileRegion subclass based on profiling options
        including time tracing, LIKWID hardware counters, and disk flushing.
        """
        cfg = cls._config
        if not cfg.profiling_activated:
            cls._region_cls = DisabledProfileRegion
        elif cfg.use_line_profiler:
            cls._region_cls = LineProfilerRegion
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
    def profile_region(cls, region_name, functions=None) -> BaseProfileRegion:
        """
        Get an existing ProfileRegion by name, or create a new one if it doesn't exist.

        Parameters
        ----------
        region_name: str
            The name of the profiling region.
        functions : list of callable, optional
            Functions to register for line-by-line profiling. Only has an
            effect when ``use_line_profiler=True``. Useful when using the
            context manager form, since the decorator form (``wrap``) registers
            functions automatically::

                with ProfileManager.profile_region("my_region", functions=[my_func]):
                    my_func()

        Returns
        -------
        ProfileRegion : The ProfileRegion instance.
        """

        region = cls._regions.setdefault(
            region_name,
            cls._region_cls(region_name, config=cls._config),
        )
        if functions is not None:
            for func in functions:
                region.add_function(func)
        return region

    @classmethod
    def profile(
        cls,
        region_name: str | None = None,
        recursive: bool | None = None,
    ) -> Callable:
        """
        Decorator factory for profiling a function.

        Parameters
        ----------
        region_name : str, optional
            Name for the profiling region. If not provided, uses the decorated
            function's name. Supports being used with or without parentheses.
        recursive : bool, optional
            If True, also profiles Python function calls made by the decorated
            function (excluding scope-profiler internals). If None, falls back
            to ``ProfileManager.setup(recursive_profile=...)``.

        Returns
        -------
        Callable
            Decorated function wrapped with profiling instrumentation.

        Notes
        -----
        The decorated function is registered so that calling
        ``ProfileManager.setup()`` after decoration re-binds the wrapper to
        the new region class at zero per-call cost.  This means
        ``@ProfileManager.profile`` can be applied at class-definition time
        even when ``setup()`` is called later.
        """

        def decorator(func):
            name = region_name or func.__name__
            # _bound[1] is the inner callable produced by region.wrap(func).
            # It is replaced (without touching the outer wrapper) whenever
            # set_config() is called, so there is no per-call rebind check.
            _bound = [None, None]  # [region, wrapped_func]
            recursive_override = recursive

            region = cls.profile_region(name)
            _bound[0] = region
            _bound[1] = region.wrap(func)
            cls._decorated_codes.add(func.__code__)

            # Register so set_config() can rebind without a per-call check.
            cls._decorators.setdefault(name, []).append((func, _bound))

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                recursive_enabled = cls._config.recursive_profile
                if recursive_override is not None:
                    recursive_enabled = recursive_override

                if not recursive_enabled:
                    return _bound[1](*args, **kwargs)

                state = cls._recursive_state
                depth = getattr(state, "depth", 0)
                state.depth = depth + 1
                if depth > 0:
                    try:
                        return _bound[1](*args, **kwargs)
                    finally:
                        state.depth -= 1

                prev_profiler = sys.getprofile()
                tracer = cls._get_recursive_tracer(
                    root_frame=sys._getframe(), prev_profiler=prev_profiler
                )
                sys.setprofile(tracer)
                try:
                    return _bound[1](*args, **kwargs)
                finally:
                    sys.setprofile(prev_profiler)
                    state.depth -= 1

            return wrapper

        # Support @ProfileManager.profile without parentheses
        if callable(region_name):
            func = region_name
            region_name = None  # reset, so decorator picks func.__name__
            return decorator(func)

        return decorator

    @classmethod
    def run_script(
        cls,
        script_path: str,
        script_args: list | None = None,
        region_name: str | None = None,
        only_user_code: bool = True,
    ) -> None:
        """
        Run a script under recursive profiling, similar to ``python -m cProfile``.

        Instruments every Python function call made while the script runs
        and records each as its own region, without requiring any
        decorators or context managers in the script itself. Intended to be
        called after ``ProfileManager.setup()`` and followed by
        ``ProfileManager.finalize()``; see ``python -m scope_profiler`` for
        the CLI wrapper around this.

        Parameters
        ----------
        script_path : str
            Path to the script to execute.
        script_args : list of str, optional
            Arguments exposed to the script as ``sys.argv[1:]``.
        region_name : str, optional
            Name for the region wrapping the whole script's execution
            (default: the script's basename).
        only_user_code : bool, optional
            If True (default), skip instrumenting standard-library and
            installed-package frames, tracing only the script's own code.
            This keeps overhead low and the output focused. Set to False to
            trace everything, including third-party and stdlib calls.
        """
        script_path = os.path.abspath(script_path)
        region_name = region_name or os.path.basename(script_path)

        sys.argv = [script_path, *(script_args or [])]
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        region = cls.profile_region(region_name)
        prev_profiler = sys.getprofile()
        tracer = cls._get_recursive_tracer(
            root_frame=sys._getframe(),
            prev_profiler=prev_profiler,
            only_user_code=only_user_code,
        )
        sys.setprofile(tracer)
        try:
            with region:
                runpy.run_path(script_path, run_name="__main__")
        finally:
            sys.setprofile(prev_profiler)

    @classmethod
    def finalize(
        cls,
        verbose: bool = True,
    ) -> None:
        """
        Finalize profiling and merge results from all MPI ranks.

        Flushes buffered profiling data to disk, synchronizes across MPI ranks,
        and merges per-rank profiling files into a single output file. Optionally
        prints profiling statistics for each region.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints profiling statistics for each region (default: True).
        """
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
                # Global environment metadata, gathered from rank 0 only.
                meta_grp = fout.create_group("metadata")
                for key, value in config.metadata.items():
                    meta_grp.attrs[key] = value

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
                                region_path = f"regions/{region_name}"
                                if region_path not in fin:
                                    # Region was created but never flushed
                                    # (e.g. finalize() called from inside the
                                    # profiled function before it returned).
                                    continue
                                grp = fin[region_path]
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

        if config.use_line_profiler and verbose:
            for region in cls.get_all_regions().values():
                if isinstance(region, LineProfilerRegion):
                    region.print_stats()

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
        use_line_profiler: bool = False,
        recursive_profile: bool = False,
        time_trace: bool = True,
        flush_to_disk: bool = True,
        buffer_limit: int = 100_000,
        file_path: str = "profiling_data.h5",
    ):
        """
        Initialize and configure the profiling system.

        Parameters
        ----------
        profiling_activated : bool, optional
            Enable or disable profiling (default: True).
        use_likwid : bool, optional
            Enable LIKWID hardware counter collection (default: False).
        use_line_profiler : bool, optional
            Enable line-by-line profiling via line_profiler (default: False).
        recursive_profile : bool, optional
            Enable recursive profiling for all decorated functions by default
            (default: False). This can be overridden per decorator with
            ``@ProfileManager.profile(..., recursive=...)``.
        time_trace : bool, optional
            Enable timing trace collection (default: True).
        flush_to_disk : bool, optional
            Enable flushing profiling data to disk (default: True).
        buffer_limit : int, optional
            Maximum number of profiling events per buffer before flushing (default: 100_000).
        file_path : str, optional
            Path to the output profiling data file (default: "profiling_data.h5").
        """
        ProfilingConfig().reset()
        config = ProfilingConfig(
            profiling_activated=profiling_activated,
            use_likwid=use_likwid,
            use_line_profiler=use_line_profiler,
            recursive_profile=recursive_profile,
            time_trace=time_trace,
            flush_to_disk=flush_to_disk,
            buffer_limit=buffer_limit,
            file_path=file_path,
        )
        cls.set_config(config=config)

    @classmethod
    def set_config(cls, config: ProfilingConfig) -> None:
        """
        Set a new profiling configuration and update the region class.

        Parameters
        ----------
        config : ProfilingConfig
            The new profiling configuration to apply.
        """
        cls._regions.clear()  # Clear old regions
        cls._config = config  # Update the config
        cls._update_region_cls()  # Set the proper region class
        # Rebind all registered decorator wrappers to the new region class.
        # This is the only place rebinding happens — there is no per-call check.
        for name, entries in cls._decorators.items():
            for func, _bound in entries:
                region = cls.profile_region(name)
                _bound[0] = region
                _bound[1] = region.wrap(func)

    @classmethod
    def get_config(cls) -> ProfilingConfig:
        """
        Get the current profiling configuration.

        Returns
        -------
        ProfilingConfig
            The current profiling configuration.
        """
        return cls._config

    @classmethod
    def _reset_regions(cls) -> None:
        """
        Clear all registered profiling regions.
        """
        cls._regions = {}

    @classmethod
    def _reset_config(cls) -> None:
        """
        Reset the profiling configuration to its default state.
        """
        ProfilingConfig().reset()
        cls._config = ProfilingConfig()

    @classmethod
    def _reset(cls) -> None:
        cls._reset_regions()
        cls._reset_config()
        cls._update_region_cls()
        cls._decorators.clear()
