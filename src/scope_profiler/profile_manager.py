"""Singleton manager for creating, configuring, and finalizing profiling regions."""

import functools
import os
import runpy
import site
import sys
import sysconfig
import threading
from types import FrameType
from typing import TYPE_CHECKING, Callable, Dict

import h5py
import numpy as np

from scope_profiler.profile_config import ProfilingConfig
from scope_profiler.region_profiler import (
    BaseProfileRegion,
    DisabledProfileRegion,
    FullProfileRegion,
    LikwidOnlyProfileRegion,
    LineProfilerRegion,
    NCallsOnlyProfileRegion,
    TimeOnlyProfileRegion,
)

if TYPE_CHECKING:  # imported lazily in read_results() to keep imports cheap
    from scope_profiler.h5reader import ProfilingH5Reader


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
        # co_qualname is Python 3.11+; on 3.10 fall back to the plain function
        # name, which loses the enclosing class but keeps recursive profiling
        # working. Without this, recursive_profile=True and `scope-profiler
        # run` raise AttributeError on 3.10.
        qualname = getattr(frame.f_code, "co_qualname", None) or frame.f_code.co_name
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

        Selects the appropriate ProfileRegion subclass based on profiling
        options: time tracing, LIKWID hardware counters and line profiling.
        ``flush_to_disk`` does not affect the choice -- recording is identical
        either way, it only decides whether finalize() writes the data out.
        """
        cfg = cls._config
        if not cfg.profiling_activated:
            cls._region_cls = DisabledProfileRegion
        elif cfg.use_line_profiler:
            cls._region_cls = LineProfilerRegion
        elif cfg.time_trace and cfg.use_likwid:
            cls._region_cls = FullProfileRegion
        elif cfg.time_trace:
            cls._region_cls = TimeOnlyProfileRegion
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

        # Deliberately not `setdefault`: it evaluates its default eagerly, so
        # every lookup of an existing region would construct (and discard) a
        # full region object, including its preallocated timing buffers. This
        # runs per call event under recursive profiling.
        region = cls._regions.get(region_name)
        if region is None:
            region = cls._region_cls(region_name, config=cls._config)
            cls._regions[region_name] = region
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
    def _snapshot_regions(cls) -> Dict[str, tuple]:
        """Copy every region's buffered timestamps out of the live buffers.

        Taken before ``finalize()`` writes anything, because writing rewinds
        the buffers (see ``BaseProfileRegion.mark_written``) and the arrays are
        then reused by any later call. The copies cover exactly the calls this
        run would write to disk, so in-memory results and the output file agree.

        Returns
        -------
        dict
            Region name -> ``(start_times, end_times, num_calls)``, with the
            timestamps in nanoseconds and ``num_calls`` None for regions whose
            count is implied by their timestamps.
        """
        snapshot = {}
        for name, region in cls.get_all_regions().items():
            # Only this run's calls, matching the slice write_to_disk() and
            # _write_num_calls() persist.
            num_calls = region.num_calls - region._num_calls_written
            if region.ptr == 0 and num_calls <= 0:
                continue
            snapshot[name] = (
                np.array(region.start_times[: region.ptr]),
                np.array(region.end_times[: region.ptr]),
                # Recorded timestamps carry the count with them; a region that
                # has none (count-only, LIKWID-only) needs it stated, which is
                # exactly the split between datasets and attribute on disk.
                None if region.ptr else num_calls,
            )
        return snapshot

    @classmethod
    def _build_results(cls, snapshot, likwid_results):
        """Assemble a ProfilingResults from this rank's snapshot.

        Under MPI the snapshots are gathered on rank 0, which returns the
        results for the whole run while the other ranks return None - the same
        split as the merged output file, which only rank 0 writes.
        """
        from scope_profiler.mpi_region import MPIRegion
        from scope_profiler.region import Region
        from scope_profiler.results import ProfilingResults

        config = cls.get_config()
        comm = config.comm
        payload = (snapshot, {result.tag: result for result in likwid_results})

        if comm is None:
            gathered = [payload]
        else:
            gathered = comm.gather(payload, root=0)
            if config._rank != 0:
                return None

        per_region: Dict[str, dict] = {}
        likwid = {}
        for rank, (rank_snapshot, rank_likwid) in enumerate(gathered):
            if rank_likwid:
                likwid[rank] = rank_likwid
            for name, (starts, ends, num_calls) in rank_snapshot.items():
                per_region.setdefault(name, {})[rank] = Region(
                    starts, ends, num_calls=num_calls
                )

        return ProfilingResults(
            {
                name: MPIRegion(name=name, regions=regions)
                for name, regions in per_region.items()
            },
            metadata=config.metadata,
            num_ranks=config._size,
            likwid=likwid,
            file_path=config.file_path,
        )

    @classmethod
    def finalize(
        cls,
        verbose: bool = True,
        return_results: bool = False,
    ):
        """
        Finalize profiling and merge results from all MPI ranks.

        Flushes buffered profiling data to disk, synchronizes across MPI ranks,
        and merges per-rank profiling files into a single output file. Optionally
        prints profiling statistics for each region.

        With ``use_likwid=True`` this is also where the LIKWID markers are
        closed and every marker region of the run is read back and stored in
        the output file under ``rank<r>/likwid/regions/<tag>``; see
        :meth:`~scope_profiler.results.ProfilingResults.get_likwid_regions`
        for reading it back.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints profiling statistics for each region (default: True).
        return_results : bool, optional
            If True, return the run's data as a
            :class:`~scope_profiler.results.ProfilingResults` - the same
            post-processing API :class:`~scope_profiler.h5reader.ProfilingH5Reader`
            provides, built straight from the in-memory buffers instead of by
            reading the output file back::

                results = ProfileManager.finalize(return_results=True)
                results.print_summary()
                df = results.to_dataframe()

            This works with ``flush_to_disk=False``, where no file is written
            at all. Under MPI the per-rank data is gathered on rank 0, which is
            collective: every rank must pass the same value.

        Returns
        -------
        ProfilingResults or None
            The run's profiling data when ``return_results=True`` - on rank 0
            only, like the merged output file; other ranks get None. None when
            ``return_results`` is False.
        """
        config = cls.get_config()

        if not config.profiling_activated:
            if return_results:
                from scope_profiler.results import ProfilingResults

                return ProfilingResults({}, file_path=config.file_path)
            return None

        # Snapshot before anything is written: writing rewinds the buffers.
        snapshot = cls._snapshot_regions() if return_results else None
        likwid_results = []

        comm = config.comm
        rank = config._rank
        size = config._size

        # 0. Discard any per-rank file left by an earlier finalize() on this
        # config. Nothing else writes it, so its presence means a previous run
        # in this process already finalized, and its regions would otherwise be
        # merged into this run's output alongside the current ones.
        stale_file = config._local_file_path
        if os.path.exists(stale_file):
            os.remove(stale_file)

        # 1. Write every region's buffered timestamps to its per-rank file.
        # Regions that record no timestamps write only their call count, which
        # is cheap and has nothing to do with timing buffers, so they write
        # even when flush_to_disk is off. Otherwise their counts would be lost.
        # Once written, a region is marked so finalize() acts as a run
        # boundary: a second run in the same process (e.g. a restart) writes
        # only its own events. Regions that were not written keep their
        # buffers, because with flush_to_disk off those are the only copy.
        for region in cls.get_all_regions().values():
            if config.flush_to_disk or not region._records_time:
                region.write_to_disk()
                region.mark_written()

        # 2. Close the LIKWID markers and store this rank's hardware counters
        # alongside its timings. This has to happen before the merge below,
        # not after it, or the counters would miss the copy into the output
        # file. Closing the markers here also means the marker file exists in
        # time to be read back; see ProfilingConfig.collect_likwid_results.
        if config.use_likwid:
            from scope_profiler.likwid_data import write_likwid_results

            likwid_results = config.collect_likwid_results(cls.get_all_regions().keys())
            if likwid_results:
                with h5py.File(config._local_file_path, "a") as f:
                    write_likwid_results(
                        f, likwid_results, environment=config.likwid_environment()
                    )

        # 3. Barrier to ensure all ranks finished writing
        if comm is not None:
            comm.Barrier()

        # 4. Only rank 0 performs the merge
        if rank == 0:
            merged_file_path = config.file_path
            with h5py.File(merged_file_path, "w") as fout:
                # Global environment metadata, gathered from rank 0 only.
                meta_grp = fout.create_group("metadata")
                for key, value in config.metadata.items():
                    if isinstance(value, (list, tuple)):
                        # h5py cannot infer a dtype for an empty list, and
                        # would store a non-empty one as fixed-width bytes;
                        # be explicit so list-valued metadata (e.g. the loaded
                        # modules) always round-trips as strings.
                        meta_grp.attrs.create(
                            key, list(value), dtype=h5py.string_dtype()
                        )
                    else:
                        meta_grp.attrs[key] = value

                for r in range(size):
                    rank_file = config.get_local_filepath(r)
                    if not os.path.exists(rank_file):
                        # print("warning: Profiling file is missing!")
                        continue
                    with h5py.File(rank_file, "r") as fin:
                        # Copy all groups from the rank file under /rank<r>
                        fout.copy(fin, f"rank{r}")

            # 5. Summarize the merged file, using the same table that
            # `scope-profiler inspect` and ProfilingH5Reader.print_summary()
            # render. Reading it back keeps the merge above a plain copy and
            # leaves one implementation of the statistics.
            if verbose:
                from scope_profiler.h5reader import ProfilingH5Reader

                ProfilingH5Reader(merged_file_path).print_summary(
                    title=f"{merged_file_path}  ({size} rank(s))"
                )

        if config.use_line_profiler and verbose:
            for region in cls.get_all_regions().values():
                if isinstance(region, LineProfilerRegion):
                    region.print_stats()

        if return_results:
            return cls._build_results(snapshot, likwid_results)
        return None

    @classmethod
    def read_results(cls) -> "ProfilingH5Reader":
        """
        Open the merged profiling file this run wrote, for post-processing.

        Convenience for analysing results in the same script that produced
        them::

            ProfileManager.finalize()
            reader = ProfileManager.read_results()
            reader.print_summary()

        Returns
        -------
        ProfilingH5Reader
            Reader for the file at ``config.file_path``.

        Raises
        ------
        FileNotFoundError
            If the merged file does not exist yet. It is written by
            :meth:`finalize`, and only on rank 0 - guard the call with
            ``if ProfileManager.get_config()._rank == 0`` under MPI.
        """
        from scope_profiler.h5reader import ProfilingH5Reader

        return ProfilingH5Reader(cls.get_config().file_path)

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
        buffer_limit: int = 1024,
        file_path: str = "profiling_data.h5",
        use_mpi: bool | None = None,
        start_time_ns: int | None = None,
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
            Write the recorded data to disk at finalize() (default: True).
            When False, results stay in memory for the process to read.
        buffer_limit : int, optional
            Initial number of profiling events preallocated per region
            (default: 1024). Buffers grow on demand, so this is a starting
            size rather than a limit; raise it for very hot regions to avoid
            repeated reallocation.
        file_path : str, optional
            Path to the output profiling data file (default: "profiling_data.h5").
        use_mpi : bool or None, optional
            Whether to use MPI collectives (default: None). None auto-detects:
            MPI is used only when the process was launched by mpirun/mpiexec/
            srun or an equivalent launcher, so a plain ``python script.py``
            run never imports mpi4py or calls into MPI. True forces MPI on,
            False forces it off.
        start_time_ns : int or None, optional
            The instant the run started, as a ``time.perf_counter_ns()``
            value (default: the moment ``setup()`` is called). Post-processing
            measures its relative timeline from here, so pass a value captured
            earlier to account for work that happened before the profiler was
            configured::

                from time import perf_counter_ns

                T0 = perf_counter_ns()      # first line of the program
                ...                         # imports, input parsing, ...
                ProfileManager.setup(start_time_ns=T0)

            The value is stored in the output file as the ``start_time_ns``
            metadata field.
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
            use_mpi=use_mpi,
            start_time_ns=start_time_ns,
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
