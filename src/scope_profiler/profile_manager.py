"""Singleton manager for creating, configuring, and finalizing profiling regions."""

import functools
import os
import runpy
import site
import sys
import sysconfig
import threading
from time import perf_counter_ns
from types import FrameType
from typing import TYPE_CHECKING, Callable, Dict, NamedTuple

import numpy as np

from scope_profiler.profile_config import ProfilingConfig
from scope_profiler.region_profiler import (
    BaseProfileRegion,
    DisabledProfileRegion,
    FullProfileRegion,
    LineProfilerRegion,
    NVTXProfileRegion,
    TimeOnlyProfileRegion,
    call_site_source,
    function_source,
)

if TYPE_CHECKING:  # imported lazily in read_results() to keep imports cheap
    from scope_profiler.results import ProfilingResults

# Tag for the payload messages, on a communicator of our own (see finalize).
_PAYLOAD_TAG = 0x5C09


class RankPayload(NamedTuple):
    """Everything one rank has to hand to rank 0 at ``finalize()``.

    This is what crosses the wire under MPI, and it is also what rank 0 writes
    into the output file and folds into the returned results -- one transport
    feeding both, so they cannot disagree. A NamedTuple because it pickles as a
    plain tuple, which is what mpi4py's ``send``/``recv`` use.
    """

    regions: dict
    """Region name -> ``(start_times, end_times)`` int64 arrays, nanoseconds."""

    likwid: dict
    """Region tag -> :class:`~scope_profiler.likwid_data.LikwidRegionResult`."""

    likwid_environment: dict
    """This rank's ``LIKWID_*`` environment, stored with its counters."""

    sources: dict | None = None
    """Region name -> ``(source_file, source_lineno, source_text)``.

    Only present for regions whose call site could be captured (see
    ``ProfileManager._capture_region_source``); a name missing here simply
    has no recorded source, e.g. one created only by the recursive tracer.

    Defaults to None rather than ``{}``: a NamedTuple's default is built once
    and shared by every instance that omits the argument, so a mutable
    default here would hand every such payload the *same* dict object.
    Nothing mutates it in place today, but callers should read it via
    ``payload.sources or {}`` rather than relying on that.
    """

    tags: dict | None = None
    """Region name -> tuple of user-defined string tags."""

    line_profile: list | None = None
    """Line-profiler records for this rank, when line profiling is enabled."""


class _ProfilingSession:
    """Context manager backing :meth:`ProfileManager.session`."""

    def __init__(self, manager, setup_kwargs, verbose, return_results, native_traces):
        self._manager = manager
        self._setup_kwargs = setup_kwargs
        self._verbose = verbose
        self._return_results = return_results
        self._native_traces = native_traces
        self.results = None

    def __enter__(self):
        self._manager.setup(**self._setup_kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.results = self._manager.finalize(
            verbose=self._verbose,
            return_results=self._return_results,
            native_traces=self._native_traces,
        )
        return False


class ProfileManager:
    """
    Singleton class to manage and track all ProfileRegion instances.
    """

    _regions = {}
    # Resolved on first use, never at import. Building a ProfilingConfig reads
    # the communicator, which imports mpi4py (i.e. calls MPI_Init) whenever the
    # process looks like an MPI rank -- so constructing one here would mean
    # that merely importing scope_profiler joins the MPI job. That bites any
    # child process of a rank that happens to import the library, including
    # the one LIKWID's counter read-back forks. See get_config().
    _config: ProfilingConfig | None = None
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

        Every active region records timestamps; the remaining options decide
        what it records *on top* of them. ``deactivate_file_output`` does not
        affect the choice -- recording is identical either way, it only
        decides whether finalize() writes the data out.
        """
        cfg = cls._config
        if cfg.deactivate_profiling:
            cls._region_cls = DisabledProfileRegion
        elif cfg.use_line_profiler:
            cls._region_cls = LineProfilerRegion
        elif cfg.use_likwid:
            cls._region_cls = FullProfileRegion
        elif cfg.use_nvtx:
            cls._region_cls = NVTXProfileRegion
        else:
            cls._region_cls = TimeOnlyProfileRegion

    @classmethod
    def profile_region(
        cls, region_name, functions=None, tags=None
    ) -> BaseProfileRegion:
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

        tags : iterable of str, optional
            User-defined labels persisted with the region. Reusing a region
            name with a different non-None tag set raises ``ValueError``.

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
            # Keep the overwhelmingly common untagged lookup on the original
            # hot path: tags are metadata, not per-event work.
            normalized_tags = () if tags is None else tuple(tags)
            region = cls._region_cls(
                region_name, config=cls.get_config(), tags=normalized_tags or ()
            )
            cls._regions[region_name] = region
            cls._capture_region_source(region)
        elif tags is not None:
            normalized_tags = tuple(tags)
            if region.tags != normalized_tags:
                raise ValueError(
                    f"region {region_name!r} already has tags {region.tags!r}; "
                    f"cannot reuse it with {normalized_tags!r}"
                )
        if functions is not None:
            for func in functions:
                region.add_function(func)
        return region

    @classmethod
    def _capture_region_source(cls, region: BaseProfileRegion) -> None:
        """Record a freshly created region's call site, if it has one.

        Runs exactly once per region name, at creation, so it never touches
        the per-call hot path. Only meaningful for a direct
        ``with ProfileManager.profile_region(...):`` call: internal callers
        (the decorator, the recursive tracer, ``run_script``) are skipped by
        the module check, since their own frame is inside scope_profiler
        itself rather than user code. The decorator path instead records the
        decorated function's source directly (see ``profile``), which is
        richer than its one-line decoration site.

        Also skipped for a disabled region: ``deactivate_profiling=True``
        promises near-zero setup cost, and the source of a region that will
        never report any data is not worth even a one-time AST parse. Same
        for ``capture_region_source=False`` (see ``setup()``): both skip this
        before it ever reads a file from disk.

        This assumes the call site is exactly two frames up. A user helper
        that itself wraps ``profile_region(...)`` (rather than calling it
        directly in a ``with``) shifts that: the captured location becomes
        the helper's own call to ``profile_region``, not the ``with`` at the
        helper's call site. There is no reliable way to see through an
        arbitrary wrapper from here, so this is a known limitation of the
        direct-call form, same as e.g. the stdlib ``logging`` module's
        caller detection.
        """
        if (
            isinstance(region, DisabledProfileRegion)
            or not cls._config.capture_region_source
        ):
            return
        frame = sys._getframe(2)  # profile_region() -> here -> caller
        if cls._is_internal_frame(frame):
            return
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        region.set_source(filename, lineno, call_site_source(filename, lineno))

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

            cls._bind_decorated_region(name, func, _bound)
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

        Taken before ``finalize()`` marks the run boundary, because that
        rewinds the buffers (see ``BaseProfileRegion.mark_written``) and the
        arrays are then reused by any later call. These copies are what gets
        written *and* what gets returned, so the output file and the in-memory
        results cannot disagree.

        Returns
        -------
        dict
            Region name -> ``(start_times, end_times)``, in nanoseconds.
        """
        snapshot = {}
        for name, region in cls.get_all_regions().items():
            # Only this run's calls: mark_written() rewinds ptr at the end of
            # each finalize().
            if region.ptr == 0:
                continue
            snapshot[name] = (
                np.array(region.start_times[: region.ptr]),
                np.array(region.end_times[: region.ptr]),
            )
        return snapshot

    @classmethod
    def _snapshot_sources(cls, names) -> Dict[str, tuple]:
        """Call-site source of every named region that captured one.

        Parameters
        ----------
        names : iterable of str
            Region names to look up (normally ``_snapshot_regions()``'s keys).

        Returns
        -------
        dict
            Region name -> ``(source_file, source_lineno, source_text)``,
            omitting names with no captured source.
        """
        sources = {}
        for name in names:
            region = cls._regions.get(name)
            if region is not None and region.source_text is not None:
                sources[name] = (
                    region.source_file,
                    region.source_lineno,
                    region.source_text,
                )
        return sources

    @classmethod
    def _snapshot_tags(cls, names) -> Dict[str, tuple]:
        """Tags of every named region, including explicitly empty tag sets."""
        return {
            name: tuple(cls._regions[name].tags)
            for name in names
            if name in cls._regions
        }

    class _ResultAccumulator:
        """Builds a ProfilingResults from per-rank payloads, one at a time.

        Rank 0 feeds every payload it writes to the output file through here as
        well, so the returned results and the file are assembled from the same
        bytes and cannot disagree.
        """

        def __init__(self, config) -> None:
            """Start an empty accumulation for the run described by ``config``."""
            self._config = config
            self._per_region: Dict[str, dict] = {}
            self._likwid: Dict[int, dict] = {}
            self._line_profile: Dict[int, list] = {}

        def add(self, rank: int, payload: "RankPayload") -> None:
            """Fold one rank's payload into the result set."""
            from scope_profiler.region import Region

            if payload.likwid:
                self._likwid[rank] = payload.likwid
            if payload.line_profile:
                self._line_profile[rank] = payload.line_profile
            sources = payload.sources or {}
            tags = payload.tags or {}
            for name, (starts, ends) in payload.regions.items():
                source_file, source_lineno, source_text = sources.get(
                    name, (None, None, None)
                )
                self._per_region.setdefault(name, {})[rank] = Region(
                    starts,
                    ends,
                    source_file=source_file,
                    source_lineno=source_lineno,
                    source_text=source_text,
                    tags=tags.get(name, ()),
                )

        def build(self):
            """Return the assembled :class:`ProfilingResults`.

            The per-rank entries are sorted by rank: payloads arrive in
            whatever order the ranks send them, and pooled statistics sum the
            per-rank arrays in dict order, so leaving arrival order in place
            would make averages differ in their last bits from run to run --
            and from the file, which is read back in rank order.
            """
            from scope_profiler.mpi_region import MPIRegion
            from scope_profiler.results import ProfilingResults

            return ProfilingResults(
                {
                    name: MPIRegion(name=name, regions=dict(sorted(regions.items())))
                    for name, regions in self._per_region.items()
                },
                metadata=self._config.metadata,
                num_ranks=self._config._size,
                likwid=self._likwid,
                line_profile=self._line_profile,
                file_path=self._config.file_path,
            )

    @classmethod
    def _merge_native_snapshot(cls, snapshot: dict, traces, config) -> dict:
        """Add this rank's Fortran regions to its snapshot.

        Only the trace whose rank matches this one is taken, so under MPI every
        rank folds in its own and the merge downstream is unchanged.

        Raises
        ------
        ValueError
            If a region name was recorded on both sides: merging them would
            silently double-count a Python wrapper and the native region
            inside it.
        """
        from scope_profiler.native_trace import find_traces, read_trace

        merged = dict(snapshot)
        for path in find_traces(traces):
            rank, regions = read_trace(path)
            if rank != config._rank:
                continue
            for name, arrays in regions.items():
                if name in merged:
                    raise ValueError(
                        f"region {name!r} was recorded by both the Python API "
                        f"and the Fortran trace {path}; merging them would "
                        f"double-count it. Give the regions distinct names (a "
                        f"'fortran:' prefix, say)."
                    )
                merged[name] = arrays
        return merged

    @classmethod
    def _collect_payloads(cls, payload, write_file: bool, need_results: bool):
        """Move every rank's payload to rank 0 and consume it there.

        Rank 0 takes one payload at a time -- its own first, then one per
        remaining rank -- writes it into the output file, folds it into the
        results, and drops it before taking the next. Peak memory on rank 0 is
        therefore one rank's data plus the open file, not the whole job's,
        which is what makes this scale to thousands of ranks. (With
        ``return_results=True`` the assembled results are of course the whole
        run: that is the object being returned.)

        Parameters
        ----------
        payload : RankPayload
            This rank's data.
        write_file : bool
            Whether rank 0 writes the output file.
        need_results : bool
            Whether a :class:`ProfilingResults` has to be assembled.

        Returns
        -------
        ProfilingResults or None
            The run's results on rank 0, an empty non-root set elsewhere, and
            None when no results were asked for.

        Notes
        -----
        Collective. Every rank must reach this with the same ``write_file`` and
        ``need_results``, and a rank that dies before sending leaves rank 0
        waiting in ``recv``.

        Receives run in rank order rather than by arrival. Pooled statistics
        sum the per-rank arrays in the order the ranks were added, so a fixed
        order is what keeps a run's averages reproducible and identical to the
        ones read back from the file. It also keeps this loop free of any
        mpi4py import, so the whole thing can be exercised with a stand-in
        communicator.

        The messages go over the run's own communicator, tagged with
        ``_PAYLOAD_TAG``, rather than over a private duplicate of it. A
        duplicate would be tidier -- it could not be intercepted by an
        application posting ``recv(ANY_SOURCE, ANY_TAG)`` -- but ``MPI_Comm_dup``
        is a collective that allocates a new context id, and by this point
        every rank may have forked a child process: ``use_likwid=True`` reads
        the counters back in a subprocess (see
        ``collect_marker_results_isolated``). Open MPI does not support forking
        from a rank using its shared-memory transport, and the duplicate
        reliably segfaulted there. Point-to-point traffic survives it, as the
        barrier this replaced always did.
        """
        from scope_profiler.h5writer import ProfilingWriter

        config = cls.get_config()
        comm = config.comm

        if comm is not None and config._rank != 0:
            comm.send(payload, dest=0, tag=_PAYLOAD_TAG)
            del payload
            return cls._empty_results() if need_results else None

        accumulator = cls._ResultAccumulator(config) if need_results else None
        writer = (
            ProfilingWriter(config.file_path, config.metadata) if write_file else None
        )
        try:
            for source in range(config._size):
                # Rank 0's own data needs no message.
                incoming = (
                    payload
                    if source == 0
                    else comm.recv(source=source, tag=_PAYLOAD_TAG)
                )
                if writer is not None:
                    writer.write_rank(source, incoming)
                if accumulator is not None:
                    accumulator.add(source, incoming)
                # Drop it before taking the next, so only one rank's data is
                # held at a time.
                del incoming
            del payload
        finally:
            if writer is not None:
                writer.close()

        return accumulator.build() if accumulator is not None else None

    @classmethod
    def _empty_results(cls):
        """The result set a non-root rank gets back from ``finalize()``.

        Empty and flagged ``is_root=False`` rather than None, so that a
        parallel script can go on calling print_summary(), the plot functions
        and the exporters without a rank guard: those do nothing for a
        non-root result set.
        """
        from scope_profiler.results import ProfilingResults

        config = cls.get_config()
        return ProfilingResults(
            {},
            metadata=config.metadata,
            num_ranks=config._size,
            file_path=config.file_path,
            is_root=False,
        )

    @classmethod
    def _snapshot_line_profile(cls) -> list:
        """Copy line-profiler timings into MPI/HDF5-safe plain records."""
        records = []
        for region_name, region in cls.get_all_regions().items():
            if not isinstance(region, LineProfilerRegion):
                continue
            stats = region.get_stats()
            unit = float(getattr(stats, "unit", 1.0))
            for (filename, first_lineno, function), timings in stats.timings.items():
                records.append(
                    {
                        "region": region_name,
                        "filename": str(filename),
                        "function": str(function),
                        "first_lineno": int(first_lineno),
                        "line_numbers": np.asarray(
                            [int(row[0]) for row in timings], dtype=np.int64
                        ),
                        "hits": np.asarray(
                            [int(row[1]) for row in timings], dtype=np.int64
                        ),
                        "times": np.asarray(
                            [float(row[2]) for row in timings], dtype=float
                        ),
                        "unit": unit,
                    }
                )
        return records

    @classmethod
    def finalize(
        cls,
        verbose: bool = True,
        return_results: bool = False,
        native_traces=None,
    ):
        """
        Finalize profiling and write the run's data to a single output file.

        Copies each region's buffered timestamps out, moves every rank's copy
        to rank 0, and has rank 0 write them into one HDF5 file. Nothing is
        staged on the filesystem, so no shared ``$TMPDIR`` is needed. Optionally
        prints profiling statistics for each region.

        Under MPI this is **collective**: every rank must call it, with the
        same arguments, or the job hangs -- rank 0 waits for a payload from
        every other rank. A rank that dies before reaching it therefore leaves
        the job waiting rather than silently dropping that rank's data.

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
            post-processing API :func:`~scope_profiler.h5reader.read_h5`
            gives back, built straight from the in-memory buffers instead of
            by reading the output file back::

                results = ProfileManager.finalize(return_results=True)
                results.print_summary()
                df = results.to_dataframe()

            This works with ``deactivate_file_output=True``, where no file is
            written at all. Under MPI the per-rank data is gathered on rank 0,
            which is collective: every rank must pass the same value.

        native_traces : path or sequence of paths, optional
            Trace files (or directories of them) written by the Fortran region
            API in this same process, to fold into this run's output. Each
            rank picks up the trace matching its own rank, so a mixed-language
            MPI run still produces one file::

                kernels.stop_profiling()            # Fortran sp_finalize()
                ProfileManager.finalize(native_traces=".")

            Call the Fortran side's ``sp_finalize()`` first: its trace has to
            exist by the time this reads it. A region name recorded on both
            sides raises, rather than silently double-counting.

        Returns
        -------
        ProfilingResults or None
            The run's profiling data when ``return_results=True``, and None
            otherwise. Under MPI rank 0 gets the whole run, like the merged
            output file; the other ranks get an empty result set for which
            ``print_summary()``, the ``plot_*`` functions and the exporters do
            nothing, so the script above needs no rank guard. See
            :attr:`~scope_profiler.results.ProfilingResults.is_root`.
        """
        config = cls.get_config()

        # Read on the same clock as start_time_ns, and as the very first
        # thing here, so total_time (see ProfilingResults.total_time) reports
        # the program's own setup()-to-finalize() span rather than including
        # whatever this call itself goes on to spend collecting and writing
        # the run's data.
        config.metadata["finalize_time_ns"] = perf_counter_ns()

        if config.deactivate_profiling:
            if return_results:
                from scope_profiler.results import ProfilingResults

                return ProfilingResults({}, file_path=config.file_path)
            return None

        rank = config._rank
        size = config._size

        # These three decide whether this rank communicates, so every one of
        # them must depend only on the config (identical on all ranks, from the
        # same setup()) and on this call's arguments (documented as collective).
        # Nothing rank-local may gate a send or a receive, or the job deadlocks.
        write_file = not config.deactivate_file_output
        need_results = return_results or (verbose and not write_file)
        need_payload = write_file or need_results

        # 1. Copy this run's timestamps out of the live buffers. The copy is
        # both what gets written and what gets returned, so the file and the
        # in-memory results are assembled from the same bytes.
        snapshot = cls._snapshot_regions() if need_payload else {}
        sources = cls._snapshot_sources(snapshot) if need_payload else {}
        tags = cls._snapshot_tags(snapshot) if need_payload else {}
        line_profile = cls._snapshot_line_profile() if need_payload else None

        # The data is safely copied, so the run boundary can be marked now: a
        # second finalize() in this process then reports only its own events.
        # Not when nothing is written, though -- there the buffers are the only
        # copy the caller has left.
        if write_file:
            for region in cls.get_all_regions().values():
                region.mark_written()

        # 2. Close the LIKWID markers and pick up this rank's counters, which
        # travel with the timings. Closing here also means the marker file
        # exists in time to be read back; see collect_likwid_results.
        likwid_results = []
        likwid_environment = {}
        if config.use_likwid:
            likwid_results = config.collect_likwid_results(cls.get_all_regions().keys())
            if likwid_results:
                likwid_environment = config.likwid_environment()

        # 3. Fold in the regions a Fortran (or other native) part of this
        # process recorded for itself. Each rank picks up its own trace, so
        # the transport below needs no special case: by the time anything is
        # written or gathered, a mixed-language run looks like a single-
        # language one.
        if native_traces is not None and need_payload:
            snapshot = cls._merge_native_snapshot(snapshot, native_traces, config)

        payload = RankPayload(
            regions=snapshot,
            likwid={result.tag: result for result in likwid_results},
            likwid_environment=likwid_environment,
            sources=sources,
            tags=tags,
            line_profile=line_profile,
        )

        # 3. Move every rank's payload to rank 0, which writes it straight into
        # the single output file. Nothing is staged on the filesystem, so no
        # shared $TMPDIR is required, and a rank that never reports is a hang
        # rather than a silently missing group.
        results = None
        if need_payload:
            results = cls._collect_payloads(payload, write_file, need_results)

        # 4. Summarize. With a file, it is read back so that the table has one
        # implementation; without one, the same table comes from the results.
        # Non-root ranks hold an empty result set, for which this does nothing.
        if verbose and rank == 0 and write_file:
            from scope_profiler.h5reader import read_h5

            read_h5(config.file_path).print_summary(
                title=f"{config.file_path}  ({size} rank(s))"
            )
        elif verbose and not write_file:
            results.print_summary(
                title=f"{results.display_label}  (in memory, {size} rank(s))"
            )

        if config.use_line_profiler and verbose:
            for region in cls.get_all_regions().values():
                if isinstance(region, LineProfilerRegion):
                    region.print_stats()

        if return_results:
            return results
        return None

    @classmethod
    def read_results(cls) -> "ProfilingResults":
        """
        Open the merged profiling file this run wrote, for post-processing.

        Convenience for analysing results in the same script that produced
        them::

            ProfileManager.finalize()
            results = ProfileManager.read_results()
            results.print_summary()

        Returns
        -------
        ProfilingResults
            The data in the file at ``config.file_path``.

        Raises
        ------
        FileNotFoundError
            If the merged file does not exist yet. It is written by
            :meth:`finalize`, and only on rank 0 - guard the call with
            ``if ProfileManager.get_config()._rank == 0`` under MPI.
        """
        from scope_profiler.h5reader import read_h5

        return read_h5(cls.get_config().file_path)

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
        deactivate_profiling: bool = False,
        deactivate_file_output: bool = False,
        use_likwid: bool = False,
        use_line_profiler: bool = False,
        use_nvtx: bool = False,
        recursive_profile: bool = False,
        buffer_limit: int = 1024,
        file_path: str = "profiling_data.h5",
        label: str | None = None,
        capture_region_source: bool = False,
    ):
        """
        Initialize and configure the profiling system.

        Parameters
        ----------
        deactivate_profiling : bool, optional
            Turn profiling off entirely (default: False). Every region
            becomes a no-op, so the instrumentation can stay in the code at
            near-zero cost instead of being removed.
        deactivate_file_output : bool, optional
            Write no HDF5 file at all (default: False), not even the run
            metadata. Use it with
            ``finalize(return_results=True)`` to analyse a run entirely in
            memory::

                ProfileManager.setup(deactivate_file_output=True)
                ...
                results = ProfileManager.finalize(return_results=True)

        use_likwid : bool, optional
            Enable LIKWID hardware counter collection (default: False).
        use_line_profiler : bool, optional
            Enable line-by-line profiling via line_profiler (default: False).
        use_nvtx : bool, optional
            Add NVTX ranges to profiled regions for NVIDIA Nsight tools
            (default: False). Requires ``scope-profiler[nvtx]``.
        recursive_profile : bool, optional
            Enable recursive profiling for all decorated functions by default
            (default: False). This can be overridden per decorator with
            ``@ProfileManager.profile(..., recursive=...)``.
        buffer_limit : int, optional
            Initial number of profiling events preallocated per region
            (default: 1024). Buffers grow on demand, so this is a starting
            size rather than a limit; raise it for very hot regions to avoid
            repeated reallocation.
        file_path : str, optional
            Path to the output profiling data file (default: "profiling_data.h5").
        label : str or None, optional
            Short name for this run (default: None, i.e. the output file's
            stem). Post-processing uses it wherever a run has to be named --
            chart legends, the summary heading, ``scope-profiler inspect``,
            the JSON statistics -- which is what makes several runs
            distinguishable when they are compared::

                ProfileManager.setup(file_path="run_a.h5", label="128 ranks")

            It is stored in the output file as the ``label`` metadata field,
            so it survives into every later post-processing step.
        capture_region_source : bool, optional
            Record where each region is defined -- the ``with`` block or the
            decorated function -- once per distinct source file, the first
            time any of its regions is created (default: False). See
            :attr:`~scope_profiler.region.Region.source_text`. Off by
            default because the cost, while cheap for a typical file, is not
            always: it is one ``ast.parse`` + tree walk of that file, so it
            tracks the file's total size, not the size or number of the
            regions in it -- under a millisecond for a typical few-hundred-
            line file, but tenths of a second per rank for one containing
            thousands of lines across many regions. Every rank pays that
            independently, so it can compound to whole seconds under
            contention on a job with more ranks than idle cores (measured:
            ~0.3s/rank at 8 ranks, ~2.9s/rank at 64, for a single
            ~10,000-line file, on a shared/oversubscribed node)::

                ProfileManager.setup(capture_region_source=True)

        Notes
        -----
        The run's start time is the moment ``setup()`` is called; it is stored
        as the ``start_time_ns`` metadata field and is the origin of the
        relative timeline in post-processing. MPI is not configurable either:
        collectives are used exactly when the process was started by an MPI
        launcher, so a plain ``python script.py`` never imports mpi4py. See
        :mod:`scope_profiler.mpi_launch` for the detection and its
        ``SCOPE_PROFILER_MPI`` override.
        """
        ProfilingConfig.reset()
        config = ProfilingConfig(
            deactivate_profiling=deactivate_profiling,
            deactivate_file_output=deactivate_file_output,
            use_likwid=use_likwid,
            use_line_profiler=use_line_profiler,
            use_nvtx=use_nvtx,
            recursive_profile=recursive_profile,
            buffer_limit=buffer_limit,
            file_path=file_path,
            label=label,
            capture_region_source=capture_region_source,
        )
        cls.set_config(config=config)

    @classmethod
    def session(
        cls,
        *,
        verbose: bool = True,
        return_results: bool = False,
        native_traces=None,
        **setup_kwargs,
    ):
        """Return a context manager that sets up and finalizes profiling.

        All keyword arguments other than ``verbose``, ``return_results`` and
        ``native_traces`` are passed to :meth:`setup`. Finalization runs even
        when the profiled block raises; the original exception is preserved.

        When ``return_results=True``, the context object exposes the finalized
        :class:`~scope_profiler.results.ProfilingResults` as ``results``::

            with ProfileManager.session(return_results=True, verbose=False) as run:
                with ProfileManager.profile_region("solve"):
                    solve()
            results = run.results
        """
        return _ProfilingSession(
            cls, setup_kwargs, verbose, return_results, native_traces
        )

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
                cls._bind_decorated_region(name, func, _bound)

    @classmethod
    def _bind_decorated_region(cls, name: str, func, _bound: list) -> BaseProfileRegion:
        """Resolve ``name``'s region, capture ``func``'s source, and bind it into ``_bound``.

        Shared between the initial ``@ProfileManager.profile`` decoration and
        ``set_config()``'s rebind, since a config change replaces every region
        object (see ``set_config``) and the new one starts with no source of
        its own -- skipping this on rebind would silently drop it.
        """
        region = cls.profile_region(name)
        if cls._config.capture_region_source and not isinstance(
            region, DisabledProfileRegion
        ):
            source = function_source(func)
            if source is not None:
                region.set_source(*source)
        _bound[0] = region
        _bound[1] = region.wrap(func)
        return region

    @classmethod
    def get_config(cls) -> ProfilingConfig:
        """
        Get the current profiling configuration, creating a default one if
        ``setup()`` has not been called.

        This is the only place a configuration comes into being outside
        ``setup()``, and it is deliberately lazy: constructing one resolves the
        MPI communicator, which imports mpi4py and therefore calls
        ``MPI_Init`` in any process the launcher marked as a rank. Doing that
        at import time would mean ``import scope_profiler`` silently joins the
        MPI job -- fatal in a process forked from a rank, which is exactly
        what the LIKWID counter read-back does.

        Returns
        -------
        ProfilingConfig
            The current profiling configuration.
        """
        if cls._config is None:
            cls._config = ProfilingConfig()
            # Direct attribute read below, not get_config(): _config is set.
            cls._update_region_cls()
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
        Drop the profiling configuration.

        The next ``get_config()`` builds a fresh default one; nothing is
        constructed here, so a reset cannot pull MPI in either.
        """
        ProfilingConfig.reset()
        cls._config = None

    @classmethod
    def _reset(cls) -> None:
        cls._reset_regions()
        cls._reset_config()
        # Back to the state a fresh import leaves behind: no configuration,
        # and regions disabled until setup() or get_config() says otherwise.
        cls._region_cls = DisabledProfileRegion
        cls._decorators.clear()
