"""Managers for creating, configuring, and finalizing profiling regions."""

import atexit
import fnmatch
import functools
import inspect
import os
import runpy
import site
import subprocess
import sys
import sysconfig
import threading
import warnings
from collections.abc import Callable
from contextlib import ContextDecorator, contextmanager
from contextvars import ContextVar
from enum import Enum, auto
from time import perf_counter_ns
from types import FrameType
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import numpy as np

from scope_profiler.call_stack import (
    NestingError,
    build_call_arrays,
    regions_from_snapshot,
)
from scope_profiler.profile_config import (
    _CONFIG_FIELDS,
    ProfilingConfig,
    ProfilingOptions,
    _unknown_setting_error,
    load_profiling_config,
)
from scope_profiler.region_profiler import (
    AggregateProfileRegion,
    BaseProfileRegion,
    CUDATimingNVTXProfileRegion,
    CUDATimingProfileRegion,
    DisabledProfileRegion,
    FullProfileRegion,
    LineProfilerRegion,
    NVTXProfileRegion,
    PerfEventProfileRegion,
    ThreadedProfileRegion,
    TimeOnlyProfileRegion,
    call_site_source,
    function_source,
)

if TYPE_CHECKING:  # imported lazily in read_results() to keep imports cheap
    # Unpack is 3.11+, so the setup()/session() annotations using it are
    # written as strings and never evaluated on the 3.10 floor.
    from typing import Unpack

    from scope_profiler.profile_config import SetupOptions
    from scope_profiler.results import ProfilingResults

# Tag for the payload messages, on a communicator of our own (see finalize).
_PAYLOAD_TAG = 0x5C09
_WRITE_TOKEN_TAG = 0x5C0A
_UNSET = object()


class _WriteToken(NamedTuple):
    """Ownership of the output file, as it is relayed from rank to rank.

    A NamedTuple so it pickles as a plain tuple for mpi4py, and so a receiver
    can index it positionally without importing this module.
    """

    ok: bool

    message: str

    index: dict | None
    """False once any rank has failed to write; later ranks then stand down."""
    """Why the write failed, reported by rank 0 at the end of the relay."""
    """:meth:`~scope_profiler.h5writer.ColumnarIndex.state` of the file so far."""


class RankPayload(NamedTuple):
    """Everything one rank has to hand to rank 0 at ``finalize()``.

    This is what crosses the wire under MPI, and it is also what rank 0 writes
    into the output file and folds into the returned results -- one transport
    feeding both, so they cannot disagree. A NamedTuple because it pickles as a
    plain tuple, which is what mpi4py's ``send``/``recv`` use.
    """

    regions: dict

    likwid: dict

    likwid_environment: dict

    perf_events: dict | None = None

    sources: dict | None = None

    tags: dict | None = None

    event_metadata: dict | None = None

    line_profile: list | None = None

    exclusive_totals: dict | None = None

    aggregate_stats: dict | None = None

    lanes: dict | None = None
    """Region name -> timing arrays, in nanoseconds.

    Values are ``(start_times, end_times)`` for CPU timing only, or
    ``(start_times, end_times, gpu_durations)`` when CUDA-event timing is
    enabled. CPU timestamps still measure enqueue-side region duration.
    """
    """Region tag -> :class:`~scope_profiler.likwid_data.LikwidRegionResult`."""
    """This rank's ``LIKWID_*`` environment, stored with its counters."""
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
    """Region name -> tuple of user-defined string tags."""
    """Line-profiler records for this rank, when line profiling is enabled."""
    """Region name -> total exclusive nanoseconds on this rank.

    Computed here rather than by whoever reads the run back: a rank holds its
    whole region set in memory at finalize(), which is exactly the set
    exclusive time is defined against, and reconstructing the nesting from the
    events afterwards is the most expensive part of loading a profile. Every
    rank does its own, so the cost is spread over the job.
    """
    """Region name -> aggregate counters for aggregation mode."""
    """This rank's thread and task tables, as columns.

    ``{"threads": {...}, "tasks": {...}}`` exactly as
    :meth:`~scope_profiler.concurrency.ConcurrencyTracker.snapshot` builds it,
    or None when the run did not track threads. The per-call ``thread_ids``
    and ``task_ids`` columns index into these.
    """


class _LifecycleState(Enum):
    """Internal lifecycle state, independent of lazy config resolution."""

    INACTIVE = auto()
    ACTIVE = auto()
    FINALIZED = auto()


class ProfilingRun:
    """Handle returned by :meth:`ProfileManager.setup`."""

    def __init__(self, manager, config, output=None) -> None:
        self._manager = manager
        self._config = config
        self._file_path = output or config.file_path
        self.results = None

    @property
    def file_path(self):
        """Configured output path for this run."""
        return self._file_path

    @property
    def is_active(self) -> bool:
        """Whether this run's manager currently records calls."""
        return self._manager._config is self._config and self._manager.is_active()

    def finalize(self, **kwargs):
        """Finalize through the owning manager and retain the results."""
        if self._manager._config is not self._config:
            raise RuntimeError("this profiling run has been replaced")
        kwargs.setdefault("return_results", True)
        self.results = self._manager.finalize(**kwargs)
        return self.results


class RegionHandle:
    """Persistent region definition that follows manager reconfiguration."""

    def __init__(self, manager, name, functions=None, tags=None) -> None:
        self._manager = manager
        self.name = name
        self._functions = None if functions is None else tuple(functions)
        self._tags = None if tags is None else tuple(tags)
        self._stack = ContextVar(f"scope_profiler_region_{id(self)}", default=())

    def __enter__(self):
        region = self._manager.region(
            self.name,
            functions=self._functions,
            tags=self._tags,
        )
        entered = region.__enter__()
        self._stack.set((*self._stack.get(), region))
        return entered

    def __exit__(self, exc_type, exc_value, traceback):
        stack = self._stack.get()
        region = stack[-1]
        self._stack.set(stack[:-1])
        return region.__exit__(exc_type, exc_value, traceback)


class RegionFactory:
    """Callable region definition for repeated scopes with metadata."""

    def __init__(self, manager, name, tags=None):
        self._manager = manager
        self.name = name
        self._tags = tags

    def __call__(self, **metadata):
        @contextmanager
        def scope():
            with (
                self._manager.metadata(**metadata),
                self._manager.region(
                    self.name,
                    tags=self._tags,
                ),
            ):
                yield

        return scope()


class _ProfilingSession(ContextDecorator):
    """Context manager backing :meth:`ProfileManager.session`."""

    ROOT_REGION_NAME = "scope_profiler.session"

    def __init__(
        self,
        manager,
        setup_kwargs,
        verbose,
        verbose_line_profiler,
        return_results,
        native_traces,
    ):
        self._manager = manager
        self._setup_kwargs = setup_kwargs
        self._verbose = verbose
        self._verbose_line_profiler = verbose_line_profiler
        self._return_results = return_results
        self._native_traces = native_traces
        self.results = None
        self._root_region = None

    def _recreate_cm(self):
        """Give every decorated invocation its own session state."""
        return type(self)(
            self._manager,
            self._setup_kwargs,
            self._verbose,
            self._verbose_line_profiler,
            self._return_results,
            self._native_traces,
        )

    def __enter__(self):
        self._manager.setup(**self._setup_kwargs)
        # Keep every region created in the session under one interval.  Apart
        # from making the total elapsed time explicit, this gives call-graph
        # consumers a single root instead of a forest whose display order can
        # be mistaken for execution order (notably by SnakeViz).
        self._root_region = self._manager.profile_region(self.ROOT_REGION_NAME)
        self._root_region.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._root_region.__exit__(exc_type, exc_value, traceback)
        finally:
            try:
                self.results = self._manager.finalize(
                    verbose=self._verbose,
                    verbose_line_profiler=self._verbose_line_profiler,
                    return_results=self._return_results,
                    native_traces=self._native_traces,
                )
            finally:
                # The session is the profiling window, so its end is where
                # the thread, asyncio and greenlet hooks come out again --
                # setup() installed them, and leaving them behind would have
                # a later, unrelated event loop still feeding a finished run.
                # (setup()/finalize() used directly instead keep them until
                # the next setup(), since finalize() there can be a
                # checkpoint in the middle of a run.)
                config = self._manager._config
                if config is not None and config.tracker is not None:
                    config.tracker.uninstall()
                self._manager._stop_mpi_call_profiling()
        return False


class ProfileManager:
    """Manage and track a set of profiling regions.

    The class methods remain the process-wide default API. Instantiating the
    class creates an independent manager with its own configuration, regions,
    decorators, and call-id space, allowing multiple profiling sessions to be
    active at the same time::

        cpu = ProfileManager()
        io = ProfileManager()

        with cpu.session(file_path="cpu.h5"):
            with io.session(file_path="io.h5"):
                with cpu.profile_region("compute"), io.profile_region("write"):
                    work()
    """

    class _ResultAccumulator:
        """Builds a ProfilingResults from per-rank payloads, one at a time.

        Rank 0 feeds every payload it writes to the output file through here as
        well, so the returned results and the file are assembled from the same
        bytes and cannot disagree.
        """

        def __init__(self, config) -> None:
            """Start an empty accumulation for the run described by ``config``."""
            self._config = config
            self._per_region: dict[str, dict] = {}
            self._likwid: dict[int, dict] = {}
            self._line_profile: dict[int, list] = {}
            self._perf_events: dict[int, dict] = {}
            self._exclusive_totals: dict[str, dict] = {}
            self._threads: dict[int, list] = {}
            self._tasks: dict[int, list] = {}

        def add(self, rank: int, payload: "RankPayload") -> None:
            """Fold one rank's payload into the result set."""
            from scope_profiler.region import Region

            if payload.likwid:
                self._likwid[rank] = payload.likwid
            if payload.line_profile:
                self._line_profile[rank] = payload.line_profile
            if payload.perf_events:
                self._perf_events[rank] = payload.perf_events
            if payload.lanes:
                from scope_profiler.concurrency import lane_tables_from_columns

                threads, tasks = lane_tables_from_columns(rank, payload.lanes)
                if threads:
                    self._threads[rank] = threads
                if tasks:
                    self._tasks[rank] = tasks
            sources = payload.sources or {}
            tags = payload.tags or {}
            exclusive_totals = payload.exclusive_totals or {}
            for name, arrays in payload.regions.items():
                starts, ends = arrays[:2]
                gpu_durations = arrays[2] if len(arrays) > 2 else None
                call_ids = arrays[3] if len(arrays) > 3 else None
                parent_ids = arrays[4] if len(arrays) > 4 else None
                thread_ids = arrays[5] if len(arrays) > 5 else None
                task_ids = arrays[6] if len(arrays) > 6 else None
                await_times = arrays[7] if len(arrays) > 7 else None
                source_file, source_lineno, source_text = sources.get(
                    name,
                    (None, None, None),
                )
                self._per_region.setdefault(name, {})[rank] = Region(
                    starts,
                    ends,
                    gpu_durations=gpu_durations,
                    call_ids=call_ids,
                    parent_ids=parent_ids,
                    thread_ids=thread_ids,
                    task_ids=task_ids,
                    await_times=await_times,
                    event_metadata=(payload.event_metadata or {}).get(name),
                    source_file=source_file,
                    source_lineno=source_lineno,
                    source_text=source_text,
                    tags=tags.get(name, ()),
                )
                if name in exclusive_totals:
                    self._exclusive_totals.setdefault(name, {})[rank] = (
                        exclusive_totals[name]
                    )
            for name, aggregate in (payload.aggregate_stats or {}).items():
                self._per_region.setdefault(name, {})[rank] = Region(
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    aggregate=aggregate,
                    source_file=sources.get(name, (None, None, None))[0],
                    source_lineno=sources.get(name, (None, None, None))[1],
                    source_text=sources.get(name, (None, None, None))[2],
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
                perf_events=self._perf_events,
                file_path=self._config.file_path,
                exclusive_totals=self._exclusive_totals,
                threads=self._threads,
                tasks=self._tasks,
            )

    _regions: ClassVar[dict] = {}
    # Next call id to hand out, so ids stay unique across the repeated
    # finalize() calls of one run. Reset by setup(), which starts a new run.
    _next_call_id = 0
    # Resolved on first use, never at import. Building a ProfilingConfig reads
    # the communicator, which imports mpi4py (i.e. calls MPI_Init) whenever the
    # process looks like an MPI rank -- so constructing one here would mean
    # that merely importing scope_profiler joins the MPI job. That bites any
    # child process of a rank that happens to import the library, including
    # the one LIKWID's counter read-back forks. See get_config().
    _config: ProfilingConfig | None = None
    _configured = False
    _lifecycle_state = _LifecycleState.INACTIVE
    _auto_finalize_callback = None
    _last_results = None
    _requested_output = None
    _metadata_context = ContextVar("scope_profiler_metadata", default=None)
    _metadata_scopes: ClassVar[list] = []
    _mpi_profile_context = None
    _region_cls = DisabledProfileRegion
    _decorators: ClassVar[dict[str, list]] = {}  # name -> [(func, _bound), ...]
    _region_definitions: ClassVar[dict[str, tuple]] = {}
    _decorated_codes: ClassVar[set] = set()
    _recursive_state = threading.local()
    _user_code_cache: ClassVar[dict[object, bool]] = {}
    _system_prefixes = None
    _internal_modules: ClassVar[set[str]] = {
        "scope_profiler.profile_manager",
        "scope_profiler.region_profiler",
        "scope_profiler.profile_config",
    }

    @classmethod
    def _is_internal_frame(cls, frame: FrameType) -> bool:
        module_name = frame.f_globals.get("__name__", "")
        return module_name in cls._internal_modules

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
        if isinstance(region, DisabledProfileRegion):
            return
        frame = sys._getframe(2)  # profile_region() -> here -> caller
        if cls._is_internal_frame(frame):
            return
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        region.set_source(
            filename,
            lineno,
            (
                call_site_source(filename, lineno)
                if cls._config.capture_region_source
                else None
            ),
        )

    @classmethod
    def region(
        cls,
        region_name,
        functions=None,
        tags=None,
    ) -> BaseProfileRegion:
        """
        Get the profiling region named ``region_name``, creating it if needed.

        The returned region is a context manager, which is how a block of
        code is timed::

            with ProfileManager.region("solve"):
                solve()

        Parameters
        ----------
        region_name: str
            The name of the profiling region.
        functions : list of callable, optional
            Functions to register for line-by-line profiling. Only has an
            effect when ``use_line_profiler=True``. Useful when using the
            context manager form, since the decorator form (``wrap``) registers
            functions automatically::

                with ProfileManager.region("my_region", functions=[my_func]):
                    my_func()

        tags : iterable of str, optional
            User-defined labels persisted with the region. Reusing a region
            name with a different non-None tag set raises ``ValueError``.

        Returns
        -------
        ProfileRegion : The ProfileRegion instance.

        Notes
        -----
        ``profile_region`` is the original name for this method and remains
        available as an alias; the two are the same object.
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
                region_name,
                config=cls.get_config(),
                tags=normalized_tags or (),
            )
            cls._regions[region_name] = region
            cls._capture_region_source(region)
        elif tags is not None:
            normalized_tags = tuple(tags)
            if region.tags != normalized_tags:
                raise ValueError(
                    f"region {region_name!r} already has tags {region.tags!r}; "
                    f"cannot reuse it with {normalized_tags!r}",
                )
        if functions is not None:
            for func in functions:
                region.add_function(func)
        return region

    #: Original name for :meth:`region`, kept so existing instrumentation
    #: keeps working unchanged. Same object, not a forwarding wrapper: this
    #: is on the per-event hot path.
    profile_region = region

    def __new__(cls):
        """Create a manager whose classmethod-backed state is isolated.

        The original API intentionally uses class methods so instrumentation
        can be imported anywhere without passing an object around. A private
        subclass per instance preserves that low-overhead dispatch while
        giving each instance a separate set of class attributes.
        """
        if cls is not ProfileManager:
            return object.__new__(cls)
        isolated_cls = type(
            "ProfileManagerInstance",
            (cls,),
            {
                "_regions": {},
                "_next_call_id": 0,
                "_config": None,
                "_configured": False,
                "_lifecycle_state": _LifecycleState.INACTIVE,
                "_auto_finalize_callback": None,
                "_last_results": None,
                "_requested_output": None,
                "_metadata_context": ContextVar(
                    f"scope_profiler_metadata_{id(object())}",
                    default=None,
                ),
                "_metadata_scopes": [],
                "_mpi_profile_context": None,
                "_region_cls": DisabledProfileRegion,
                "_decorators": {},
                "_region_definitions": {},
                "_decorated_codes": set(),
                "_recursive_state": threading.local(),
                "__module__": cls.__module__,
            },
        )
        return object.__new__(isolated_cls)

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
        active_calls: dict | None = None,
    ):
        active_calls = {} if active_calls is None else active_calls

        def tracer(frame: FrameType, event: str, arg):
            if event == "call":
                if frame is root_frame or cls._is_internal_frame(frame):
                    pass
                elif frame.f_code in cls._decorated_codes:
                    # Skip functions that already have explicit decorators to
                    # avoid counting the same call in two regions.
                    pass
                elif only_user_code and not cls._is_user_code(frame.f_code):
                    pass
                else:
                    region = cls.profile_region(cls._frame_region_name(frame))
                    if isinstance(region, LineProfilerRegion):
                        region.enter_timing_only()
                    else:
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
        elif cfg.perf_events:
            cls._region_cls = PerfEventProfileRegion
        elif cfg.use_nvtx and cfg.use_gpu_timing:
            cls._region_cls = CUDATimingNVTXProfileRegion
        elif cfg.use_gpu_timing:
            cls._region_cls = CUDATimingProfileRegion
        elif cfg.use_nvtx:
            cls._region_cls = NVTXProfileRegion
        elif cfg.aggregation_mode:
            cls._region_cls = AggregateProfileRegion
        elif cfg.track_threads:
            cls._region_cls = ThreadedProfileRegion
        else:
            cls._region_cls = TimeOnlyProfileRegion

    @classmethod
    def _bind_decorated_region(cls, name: str, func, _bound: list) -> BaseProfileRegion:
        """Resolve ``name``'s region, capture ``func``'s source, and bind it into ``_bound``.

        Shared between the initial ``@ProfileManager.profile`` decoration and
        ``set_config()``'s rebind, since a config change replaces every region
        object (see ``set_config``) and the new one starts with no source of
        its own -- skipping this on rebind would silently drop it.
        """
        region = cls.profile_region(name)
        if not isinstance(region, DisabledProfileRegion):
            if cls._config.capture_region_source:
                source = function_source(func)
                if source is not None:
                    region.set_source(*source)
            else:
                code = getattr(func, "__code__", None)
                if code is not None:
                    region.set_source(code.co_filename, code.co_firstlineno, None)
        _bound[0] = region
        _bound[1] = region.wrap(func)
        return region

    @classmethod
    def profile(
        cls,
        region_name: str | None = None,
        recursive: bool | None = None,
        metadata=None,
        record_outcome: bool = False,
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
        metadata : callable or dict, optional
            Metadata to attach to each call. A callable receives the decorated
            function's ``(*args, **kwargs)`` and must return a mapping.
        record_outcome : bool, optional
            Add ``status`` and, for failures, ``exception`` metadata to each
            decorated call (default: False).

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

            def invoke(*args, **kwargs):
                if metadata is None:
                    values = {}
                else:
                    values = (
                        metadata(*args, **kwargs) if callable(metadata) else metadata
                    )
                if not record_outcome:
                    return (
                        _bound[1](*args, **kwargs)
                        if not values
                        else _invoke_with_metadata(values, args, kwargs)
                    )
                with cls.metadata(**values) as event_metadata:
                    try:
                        result = _bound[1](*args, **kwargs)
                    except BaseException as error:
                        event_metadata.update(
                            status="error",
                            exception=type(error).__name__,
                        )
                        raise
                    event_metadata["status"] = "ok"
                    return result

            def _invoke_with_metadata(values, args, kwargs):
                with cls.metadata(**values):
                    return _bound[1](*args, **kwargs)

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
                    return invoke(*args, **kwargs)

                state = cls._recursive_state
                depth = getattr(state, "depth", 0)
                state.depth = depth + 1
                if depth > 0:
                    try:
                        return invoke(*args, **kwargs)
                    finally:
                        state.depth -= 1

                prev_profiler = sys.getprofile()
                tracer = cls._get_recursive_tracer(
                    root_frame=sys._getframe(),
                    prev_profiler=prev_profiler,
                )
                sys.setprofile(tracer)
                try:
                    return invoke(*args, **kwargs)
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
    def run(cls, func, *args, output=None, **kwargs):
        """Run a callable inside a managed profiling session."""
        session_options = dict(kwargs)
        if output is not None:
            session_options["output"] = output
        session_options.setdefault("verbose", False)
        session_options.setdefault("return_results", True)
        with cls.session(**session_options) as run:
            value = func(*args)
        cls._last_results = run.results
        return value

    @classmethod
    def profile_script(cls, script_path, *, script_args=None, output=None, **kwargs):
        """Profile a script from Python using the CLI runner semantics."""
        setup_kwargs = dict(kwargs)
        if output is not None:
            setup_kwargs["output"] = output
        cls.setup(**setup_kwargs)
        try:
            cls.run_script(
                script_path,
                script_args=script_args,
                recursive=cls.get_config().recursive_profile,
                only_user_code=True,
            )
        finally:
            results = cls.finalize(verbose=False, return_results=True)
        return results

    @classmethod
    def profile_command(
        cls,
        command,
        *,
        output=None,
        recursive=False,
        all=False,
        mpi_calls=False,
    ):
        """Run a command under ``scope-profiler run`` and return its result."""
        command = [os.fspath(value) for value in command]
        if command and (
            command[0] == sys.executable
            or os.path.basename(command[0]) in {"python", "python3"}
        ):
            command = command[1:]
        if not command:
            raise ValueError("profile_command() requires a Python script path")
        output_path = os.fspath(output) if output is not None else "profiling_data.h5"
        options = []
        if recursive or all:
            options.append("--recursive")
        if all:
            options.append("--all")
        if mpi_calls:
            options.append("--mpi-calls")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scope_profiler",
                "run",
                "-q",
                "-o",
                output_path,
                *options,
                *command,
            ],
            check=False,
        )
        completed.profile_results = None
        if completed.returncode == 0:
            from scope_profiler.profile_io import FORMAT_HTML, load, profile_format

            if (
                os.path.exists(output_path)
                and profile_format(output_path) != FORMAT_HTML
            ):
                completed.profile_results = load(output_path)
        return completed

    @classmethod
    def instrument(cls, module, *, include=("*",), exclude=(), prefix=None):
        """Decorate matching module-level functions without editing source."""
        changed = {}
        module_name = getattr(module, "__name__", "module")
        for name, function in list(vars(module).items()):
            if not inspect.isfunction(function):
                continue
            if not any(fnmatch.fnmatch(name, pattern) for pattern in include):
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in exclude):
                continue
            region_name = f"{prefix}.{name}" if prefix else f"{module_name}.{name}"
            wrapped = cls.profile(region_name)(function)
            setattr(module, name, wrapped)
            changed[name] = wrapped
        return changed

    @classmethod
    def region_factory(cls, region_name, *, tags=None) -> RegionFactory:
        """Create reusable regions whose calls may carry dynamic metadata."""
        cls.define_region(region_name, tags=tags)
        return RegionFactory(cls, region_name, tags=tags)

    @classmethod
    def setup_from_env(cls, **overrides):
        """Configure profiling from ``SCOPE_PROFILER_*`` environment variables."""

        def env_bool(name):
            value = os.environ.get(name)
            return (
                None if value is None else value.lower() in {"1", "true", "yes", "on"}
            )

        config_path = os.environ.get("SCOPE_PROFILER_CONFIG")
        output = os.environ.get("SCOPE_PROFILER_OUTPUT")
        if output is not None:
            overrides.setdefault("output", output)
        recursive = env_bool("SCOPE_PROFILER_RECURSIVE")
        auto_finalize = env_bool("SCOPE_PROFILER_AUTO_FINALIZE")
        if recursive is not None:
            overrides.setdefault("recursive_profile", recursive)
        if auto_finalize is not None:
            overrides.setdefault("auto_finalize", auto_finalize)
        return cls.setup(config_path=config_path, **overrides)

    @classmethod
    def run_script(
        cls,
        script_path: str,
        script_args: list | None = None,
        region_name: str | None = None,
        only_user_code: bool = True,
        recursive: bool = True,
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
        recursive : bool, optional
            Trace Python function calls and add a region around the script
            (default: True). When False, run only explicitly instrumented
            ``profile`` decorators and ``region`` blocks.
        """
        script_path = os.path.abspath(script_path)
        region_name = region_name or os.path.basename(script_path)

        sys.argv = [script_path, *(script_args or [])]
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        if not recursive:
            runpy.run_path(script_path, run_name="__main__")
            return

        region = cls.profile_region(region_name)
        prev_profiler = sys.getprofile()
        prev_tracer = sys.gettrace()
        active_calls = {}
        tracer = cls._get_recursive_tracer(
            root_frame=sys._getframe(),
            prev_profiler=prev_profiler,
            only_user_code=only_user_code,
            active_calls=active_calls,
        )
        line_states = {}

        def flush_line(frame, now):
            state = line_states.get(frame)
            region = active_calls.get(frame)
            if state is None or not isinstance(region, LineProfilerRegion):
                return
            lineno, started = state
            region.record_line_timing(frame, lineno, now - started)

        def line_tracer(frame: FrameType, event: str, arg):
            if prev_tracer is not None:
                prev_tracer(frame, event, arg)

            region = active_calls.get(frame)
            if not isinstance(region, LineProfilerRegion):
                return line_tracer

            now = perf_counter_ns()
            if event == "line":
                flush_line(frame, now)
                line_states[frame] = (frame.f_lineno, now)
            elif event in ("return", "exception"):
                flush_line(frame, now)
                line_states.pop(frame, None)
            return line_tracer

        if isinstance(region, LineProfilerRegion):
            sys.settrace(line_tracer)
        sys.setprofile(tracer)
        try:
            if isinstance(region, LineProfilerRegion):
                region.enter_timing_only()
                try:
                    runpy.run_path(script_path, run_name="__main__")
                finally:
                    region.__exit__(None, None, None)
            else:
                with region:
                    runpy.run_path(script_path, run_name="__main__")
        finally:
            sys.setprofile(prev_profiler)
            if isinstance(region, LineProfilerRegion):
                sys.settrace(prev_tracer)

    @classmethod
    def _snapshot_regions(cls) -> dict[str, tuple]:
        """Copy every region's buffered timestamps out of the live buffers.

        Taken before ``finalize()`` marks the run boundary, because that
        rewinds the buffers (see ``BaseProfileRegion.mark_written``) and the
        arrays are then reused by any later call. These copies are what gets
        written *and* what gets returned, so the output file and the in-memory
        results cannot disagree.

        A region that is still open has a slot reserved but no end timestamp
        written yet, so that call is left out entirely rather than snapshotted
        half-finished. It stays in the buffer -- ``mark_written()`` refuses to
        rewind under an open scope -- and is picked up whole by the next
        finalize().

        Returns
        -------
        dict
            Region name -> a tuple of nanosecond arrays, positionally
            ``(start_times, end_times, gpu_durations, call_ids, parent_ids,
            thread_ids, task_ids, await_ns)``. It is truncated after the last
            column the run actually has, and any column may be None -- a
            reader must check both. The call graph columns are filled in
            afterwards by :meth:`_snapshot_call_graph`.
        """
        snapshot = {}
        for name, region in cls.get_all_regions().items():
            # A thread-aware region keeps one buffer per thread and hands
            # them over already concatenated, with the lane columns attached.
            per_thread = getattr(region, "snapshot_arrays", None)
            if per_thread is not None:
                arrays = per_thread()
                if arrays is not None:
                    snapshot[name] = arrays
                continue
            # Only this run's calls: mark_written() rewinds ptr at the end of
            # each finalize().
            if region.ptr == 0:
                continue
            keep = region.closed_slots()
            if keep is None:
                starts = np.array(region.start_times[: region.ptr])
                ends = np.array(region.end_times[: region.ptr])
            else:
                starts = region.start_times[: region.ptr][keep]
                ends = region.end_times[: region.ptr][keep]
                if not starts.size:
                    continue
            get_gpu_durations = getattr(region, "get_gpu_durations_numpy", None)
            if get_gpu_durations is None:
                snapshot[name] = (starts, ends)
            else:
                gpu_durations = np.array(get_gpu_durations())
                if keep is not None:
                    gpu_durations = gpu_durations[keep]
                snapshot[name] = (starts, ends, gpu_durations)
        return snapshot

    @classmethod
    def _snapshot_call_graph(
        cls,
        snapshot: dict[str, tuple],
    ) -> tuple[dict[str, tuple], dict]:
        """Attach explicit call and parent ids to a timestamp snapshot.

        The ids are assigned once at finalization, when all regions for this
        rank are available. This keeps the per-entry instrumentation path
        unchanged while allowing readers to traverse the saved graph without
        looking at timestamps.

        Column-at-a-time throughout: a long run finalizes tens of millions of
        events per rank, and a dict per call costs ~700 bytes and ~16 us
        against ~80 bytes and ~0.2 us here. Nothing in this path may go
        through :func:`~scope_profiler.call_stack.build_call_stack`.

        A rank whose regions are not properly nested keeps its timings and
        loses only the call graph: raising here would throw away a run that
        has already finished computing.
        """
        if not snapshot:
            return snapshot, {}

        try:
            arrays = build_call_arrays(regions_from_snapshot(snapshot, 0), rank=0)
        except NestingError as error:
            warnings.warn(
                f"call graph not recorded: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return snapshot, None

        # Ids continue where the last finalize() left off. Restarting at zero
        # would hand two different calls the same id in a run that finalizes
        # more than once, which is exactly what a long simulation writing
        # periodic checkpoints does.
        base = cls._next_call_id
        cls._next_call_id = base + len(arrays)
        ids = np.arange(base, base + len(arrays), dtype=np.int64)
        parents = np.where(arrays.parent < 0, -1, arrays.parent + base)
        totals = np.zeros(len(arrays.names), dtype=np.int64)
        np.add.at(totals, arrays.region_index, arrays.exclusive_ns)
        exclusive_totals = dict(zip(arrays.names, totals.tolist()))

        updated = {}
        for row, name in enumerate(arrays.names):
            region_arrays = snapshot[name]
            mine = arrays.region_index == row
            # Scatter back into recording order: call_index says which slot
            # of this region's buffers each sorted call came from.
            slots = arrays.call_index[mine]
            call_ids = np.full(len(region_arrays[0]), -1, dtype=np.int64)
            parent_ids = np.full(len(region_arrays[0]), -1, dtype=np.int64)
            call_ids[slots] = ids[mine]
            parent_ids[slots] = parents[mine]
            updated[name] = (
                region_arrays[0],
                region_arrays[1],
                region_arrays[2] if len(region_arrays) > 2 else None,
                call_ids,
                parent_ids,
                region_arrays[5] if len(region_arrays) > 5 else None,
                region_arrays[6] if len(region_arrays) > 6 else None,
                region_arrays[7] if len(region_arrays) > 7 else None,
            )
        return updated, exclusive_totals

    @classmethod
    def _snapshot_sources(cls, names) -> dict[str, tuple]:
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
            if region is not None and region.source_file is not None:
                sources[name] = (
                    region.source_file,
                    region.source_lineno,
                    region.source_text,
                )
        return sources

    @classmethod
    def _snapshot_tags(cls, names) -> dict[str, tuple]:
        """Tags of every named region, including explicitly empty tag sets."""
        return {
            name: tuple(cls._regions[name].tags)
            for name in names
            if name in cls._regions
        }

    @classmethod
    def _snapshot_event_metadata(cls, snapshot) -> dict[str, list[dict]]:
        """Attach completed metadata scopes to calls wholly inside them."""
        if not cls._metadata_scopes:
            return {}
        scopes = sorted(
            cls._metadata_scopes, key=lambda item: item[1] - item[0], reverse=True
        )
        result = {}
        for name, arrays in snapshot.items():
            starts, ends = arrays[:2]
            rows = []
            any_metadata = False
            for start, end in zip(starts, ends):
                values = {}
                for scope_start, scope_end, metadata in scopes:
                    if int(start) >= scope_start and int(end) <= scope_end:
                        values.update(metadata)
                rows.append(values)
                any_metadata |= bool(values)
            if any_metadata:
                result[name] = rows
        return result

    @classmethod
    def _merge_native_snapshot(cls, snapshot: dict, traces, config) -> dict:
        """Add this rank's C/Fortran regions to its snapshot.

        Only the file whose rank matches this one is taken, so under MPI every
        rank folds in its own and the merge downstream is unchanged. Either
        native format is accepted -- a ``.spt`` trace, or the ``.h5`` an
        ``SP_USE_HDF5`` C build writes -- so a mixed-language run still comes
        out as one file however its C side was compiled.

        Raises
        ------
        ValueError
            If a region name was recorded on both sides: merging them would
            silently double-count a Python wrapper and the native region
            inside it.
        """
        from scope_profiler.native_trace import find_traces, read_native_ranks

        merged = dict(snapshot)
        for path in find_traces(traces):
            ranks, _ = read_native_ranks(path)
            for name, region in ranks.get(config._rank, {}).items():
                if name in merged:
                    raise ValueError(
                        f"region {name!r} was recorded by both the Python API "
                        f"and the native profile {path}; merging them would "
                        f"double-count it. Give the regions distinct names (a "
                        f"'c:' or 'fortran:' prefix, say).",
                    )
                # The snapshot holds plain timing arrays, not Region objects:
                # the call-graph reconstruction and the writer index it
                # positionally, the way the Python side's own entries are.
                merged[name] = (region.start_times_ns, region.end_times_ns)
        return merged

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
            ProfilingWriter(
                config.file_path,
                config.metadata,
                compression=config.hdf5_compression,
                compression_level=config.hdf5_compression_level,
                chunk_size=config.hdf5_chunk_size,
                # This writer receives every rank and publishes the finished
                # profile, so it is the one that gives it its final layout.
                repack=True,
            )
            if write_file
            else None
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
        except Exception:
            if writer is not None:
                writer.close(commit=False)
            raise
        else:
            if writer is not None:
                writer.close()

        return accumulator.build() if accumulator is not None else None

    @classmethod
    def _write_payload_direct(cls, payload) -> None:
        """Let MPI ranks append to one serial-HDF5 file in token order.

        Only the ownership token crosses MPI; every rank writes its own arrays
        and closes the file before handing ownership to the next rank.  Rank 0
        creates a temporary file and publishes it atomically after the last
        rank reports completion.

        The token carries the file's :class:`~scope_profiler.h5writer.ColumnarIndex`
        state -- the region names already assigned an id, and the ranks already
        written -- so each rank appends without first reading back index
        columns that grow with every rank before it. That read made the whole
        relay quadratic in the rank count; the state itself is a few ints and
        the region names, which do not grow with the job.
        """
        from scope_profiler.h5writer import (
            ColumnarIndex,
            ProfilingWriter,
            atomic_publish,
        )

        config = cls.get_config()
        comm = config.comm
        rank = config._rank
        size = config._size
        final_path = os.fspath(config.file_path)
        temp_path = final_path + ".scope-profiler.tmp"

        if rank == 0:
            try:
                with ProfilingWriter(
                    temp_path,
                    config.metadata,
                    compression=config.hdf5_compression,
                    compression_level=config.hdf5_compression_level,
                    chunk_size=config.hdf5_chunk_size,
                ) as writer:
                    writer.write_rank(0, payload)
                    token = _WriteToken(True, "", writer.index_state.state())
            except Exception as exc:
                token = _WriteToken(
                    False,
                    f"rank 0 could not write profiling data: {exc}",
                    None,
                )

            if size == 1:
                if not token.ok:
                    raise OSError(token.message)
                atomic_publish(
                    temp_path,
                    final_path,
                    repack=True,
                    compression=config.hdf5_compression,
                    compression_level=config.hdf5_compression_level,
                    chunk_size=config.hdf5_chunk_size,
                )
                return

            comm.send(token, dest=1, tag=_WRITE_TOKEN_TAG)
            token = comm.recv(source=size - 1, tag=_WRITE_TOKEN_TAG)
            if not token[0]:
                raise OSError(token[1])
            atomic_publish(
                temp_path,
                final_path,
                repack=True,
                compression=config.hdf5_compression,
                compression_level=config.hdf5_compression_level,
                chunk_size=config.hdf5_chunk_size,
            )
            return

        token = comm.recv(source=rank - 1, tag=_WRITE_TOKEN_TAG)
        if token[0]:
            index_state = ColumnarIndex(**token[2]) if token[2] is not None else None
            try:
                with ProfilingWriter.open_existing(
                    temp_path,
                    index_state=index_state,
                    compression=config.hdf5_compression,
                    compression_level=config.hdf5_compression_level,
                    chunk_size=config.hdf5_chunk_size,
                ) as writer:
                    writer.write_rank(rank, payload)
                    token = _WriteToken(True, "", writer.index_state.state())
            except Exception as exc:
                token = _WriteToken(
                    False,
                    f"rank {rank} could not write profiling data: {exc}",
                    None,
                )
        destination = rank + 1 if rank + 1 < size else 0
        comm.send(token, dest=destination, tag=_WRITE_TOKEN_TAG)

    @classmethod
    def _write_payload_file(cls, payload) -> None:
        """Choose the MPI single-file backend and write this rank's payload."""
        from scope_profiler.h5writer import (
            atomic_publish,
            compression_filter_available,
            parallel_hdf5_available,
            write_parallel_payload,
        )

        config = cls.get_config()
        # Base this only on the shared configuration, never on rank-local
        # payload contents: choosing different backends on different ranks
        # would deadlock the collective parallel-HDF5 path.
        # Thread tracking joins LIKWID here: the collective writer lays the
        # file out from a shape-only description of each rank's payload, which
        # carries no lane columns and no lane tables, so a parallel write
        # would silently drop everything track_threads recorded.
        parallel_compatible = not config.use_likwid and not config.track_threads
        requested = config.output_mode
        available = parallel_hdf5_available()
        filter_available = compression_filter_available(config.hdf5_compression)

        if requested == "parallel" and not available:
            raise RuntimeError(
                "output_mode='parallel' requires an h5py build with MPI support",
            )
        if requested == "parallel" and config.track_threads:
            raise RuntimeError(
                "output_mode='parallel' cannot be combined with track_threads: "
                "the collective writer does not carry the per-call thread and "
                "task columns. Use output_mode='direct'.",
            )
        if requested == "parallel" and not parallel_compatible:
            raise RuntimeError(
                "output_mode='parallel' cannot currently be combined with LIKWID: "
                "LIKWID counter collection launches a subprocess after MPI "
                "initialization, while parallel HDF5 requires subsequent MPI "
                "collectives. Use output_mode='direct'.",
            )
        if requested == "parallel" and not filter_available:
            raise RuntimeError(
                f"The parallel HDF5 library does not provide the requested "
                f"{config.hdf5_compression!r} compression filter. Use "
                "output_mode='direct', choose another filter, or rebuild "
                "parallel HDF5 with that filter enabled.",
            )

        use_parallel = (
            requested != "direct"
            and not config.aggregation_mode
            and available
            and parallel_compatible
            and filter_available
        )
        if not use_parallel:
            cls._write_payload_direct(payload)
            return

        temp_path = os.fspath(config.file_path) + ".scope-profiler.tmp"
        write_parallel_payload(
            temp_path,
            config.comm,
            config._rank,
            payload,
            config.metadata,
            compression=config.hdf5_compression,
            compression_level=config.hdf5_compression_level,
            chunk_size=config.hdf5_chunk_size,
        )
        # Closing an MPI-HDF5 file is collective, but implementations need not
        # return from close on every rank simultaneously. Do not let rank 0
        # rename while another rank is still releasing the temporary path.
        config.comm.Barrier()
        if config._rank == 0:
            # Serial, after the barrier: every rank has released the file, so
            # rank 0 alone rewrites and renames it.
            atomic_publish(
                temp_path,
                config.file_path,
                repack=True,
                compression=config.hdf5_compression,
                compression_level=config.hdf5_compression_level,
                chunk_size=config.hdf5_chunk_size,
            )
        # Callers may open the published path immediately after finalize().
        config.comm.Barrier()

    @classmethod
    def _snapshot_line_profile(cls) -> list:
        """Copy line-profiler timings into MPI/HDF5-safe plain records."""
        records = []
        for region_name, region in cls.get_all_regions().items():
            if not isinstance(region, LineProfilerRegion):
                continue
            for record in region.manual_line_records(unit=1e-9):
                records.append({"region": region_name, **record})
            stats = region.get_stats()
            unit = float(getattr(stats, "unit", 1.0))
            for (filename, first_lineno, function), timings in stats.timings.items():
                if not timings:
                    continue
                records.append(
                    {
                        "region": region_name,
                        "filename": str(filename),
                        "function": str(function),
                        "first_lineno": int(first_lineno),
                        "line_numbers": np.asarray(
                            [int(row[0]) for row in timings],
                            dtype=np.int64,
                        ),
                        "hits": np.asarray(
                            [int(row[1]) for row in timings],
                            dtype=np.int64,
                        ),
                        "times": np.asarray(
                            [float(row[2]) for row in timings],
                            dtype=float,
                        ),
                        "unit": unit,
                    },
                )
        return records

    @classmethod
    def finalize(
        cls,
        verbose: bool = True,
        return_results: bool = False,
        native_traces=None,
        verbose_line_profiler: bool = False,
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
            If True, prints the concise profiling summary (default: True).
        verbose_line_profiler : bool, optional
            If True, prints detailed line-profiler tables when line profiling
            is enabled (default: False).
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
            Files (or directories of them) written by the C or Fortran region
            API in this same process, to fold into this run's output -- either
            native format, a ``.spt`` trace or the ``.h5`` an ``SP_USE_HDF5``
            C build writes. Each rank picks up the file matching its own rank,
            so a mixed-language MPI run still produces one file::

                kernels.stop_profiling()            # native sp_finalize()
                ProfileManager.finalize(native_traces=".")

            Call the native side's ``sp_finalize()`` first: its output has to
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
        # Decorators resolve a default config while they are declared, so
        # config presence is not evidence that setup() started a run.
        if not cls._configured:
            if return_results:
                from scope_profiler.results import ProfilingResults

                file_path = cls._config.file_path if cls._config is not None else ""
                cls._last_results = ProfilingResults({}, file_path=file_path)
                return cls._last_results
            return None

        config = cls.get_config()

        # Read on the same clock as start_time_ns, and as the very first
        # thing here, so total_time (see ProfilingResults.total_time) reports
        # the program's own setup()-to-finalize() span rather than including
        # whatever this call itself goes on to spend collecting and writing
        # the run's data.
        config.metadata["finalize_time_ns"] = perf_counter_ns()

        # End the process-wide allocation capture before finalization itself
        # allocates buffers, writes HDF5, or gathers MPI payloads.
        config.stop_memory_profiling()

        if config.deactivate_profiling:
            cls._lifecycle_state = _LifecycleState.FINALIZED
            cls._cancel_auto_finalize()
            if return_results:
                from scope_profiler.results import ProfilingResults

                cls._last_results = ProfilingResults({}, file_path=config.file_path)
                return cls._last_results
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
        aggregate_stats = (
            {
                name: region.aggregate_snapshot()
                for name, region in cls._regions.items()
                if region.num_calls
            }
            if need_payload and config.aggregation_mode
            else None
        )
        snapshot = (
            cls._snapshot_regions()
            if need_payload and not config.aggregation_mode
            else {}
        )
        source_names = snapshot if not config.aggregation_mode else aggregate_stats
        sources = cls._snapshot_sources(source_names) if need_payload else {}
        tags = cls._snapshot_tags(source_names) if need_payload else {}
        event_metadata = (
            cls._snapshot_event_metadata(snapshot)
            if need_payload and not config.aggregation_mode
            else {}
        )
        line_profile = cls._snapshot_line_profile() if need_payload else None
        tracker = config.tracker
        lanes = (
            tracker.snapshot(config.start_time_ns)
            if need_payload and tracker is not None
            else None
        )

        # The data is safely copied, so the run boundary can be marked now: a
        # second finalize() in this process then reports only its own events.
        # Not when nothing is written, though -- there the buffers are the only
        # copy the caller has left.
        if write_file:
            for region in cls.get_all_regions().values():
                region.mark_written()
            cls._metadata_scopes = []

        # 2. Close the LIKWID markers and pick up this rank's counters, which
        # travel with the timings. Closing here also means the marker file
        # exists in time to be read back; see collect_likwid_results.
        likwid_results = []
        likwid_environment = {}
        if config.use_likwid:
            likwid_results = config.collect_likwid_results(cls.get_all_regions().keys())
            if likwid_results:
                likwid_environment = config.likwid_environment()
        perf_events = (
            {
                name: region.perf_event_totals()
                for name, region in cls.get_all_regions().items()
                if hasattr(region, "perf_event_totals") and region.num_calls
            }
            if config.perf_events
            else None
        )

        # 3. Fold in the regions a Fortran (or other native) part of this
        # process recorded for itself. Each rank picks up its own trace, so
        # the transport below needs no special case: by the time anything is
        # written or gathered, a mixed-language run looks like a single-
        # language one.
        if native_traces is not None and need_payload:
            snapshot = cls._merge_native_snapshot(snapshot, native_traces, config)

        # 4. Reconstruct this rank's nesting, now that its region set is
        # complete (native traces included). Done here, once, rather than by
        # every later reader of the file, and as columns rather than a dict
        # per call: see ProfileManager._snapshot_call_graph.
        if need_payload and not config.aggregation_mode:
            snapshot, exclusive_totals = cls._snapshot_call_graph(snapshot)
        else:
            exclusive_totals = None

        payload = RankPayload(
            regions=snapshot,
            likwid={result.tag: result for result in likwid_results},
            likwid_environment=likwid_environment,
            perf_events=perf_events,
            sources=sources,
            tags=tags,
            event_metadata=event_metadata,
            line_profile=line_profile,
            exclusive_totals=exclusive_totals,
            aggregate_stats=aggregate_stats,
            lanes=lanes,
        )

        # 5. Move every rank's payload to rank 0, which writes it straight into
        # the single output file. Nothing is staged on the filesystem, so no
        # shared $TMPDIR is required, and a rank that never reports is a hang
        # rather than a silently missing group.
        results = None
        if need_payload:
            if write_file and config.comm is not None:
                cls._write_payload_file(payload)
                if need_results:
                    if rank == 0:
                        from scope_profiler.h5reader import read_h5

                        results = read_h5(config.file_path)
                    else:
                        results = cls._empty_results()
            else:
                results = cls._collect_payloads(payload, write_file, need_results)

        rendered_results = None
        requested_output = cls._requested_output
        if (
            write_file
            and rank == 0
            and requested_output is not None
            and requested_output != config.file_path
        ):
            from pathlib import Path

            from scope_profiler.profile_io import read_profile, write_profile

            rendered_results = results or read_profile(config.file_path)
            rendered_results._file_path = Path(requested_output)
            write_profile(rendered_results, requested_output)
            os.remove(config.file_path)
            if results is not None:
                results = rendered_results

        # 6. Summarize. With a file, it is read back so that the table has one
        # implementation; without one, the same table comes from the results.
        # Non-root ranks hold an empty result set, for which this does nothing.
        if verbose and rank == 0 and write_file:
            if rendered_results is not None:
                rendered_results.print_summary()
            else:
                from scope_profiler.h5reader import read_h5

                read_h5(config.file_path).print_summary()
        elif verbose and not write_file:
            rank_label = "rank" if size == 1 else "ranks"
            results.print_summary(
                title=f"{results.display_label} (in memory, {size} {rank_label})",
            )

        if verbose and rank == 0 and config.memory_profile_path is not None:
            rank_note = " (one capture per rank)" if size > 1 else ""
            print(
                f"\nwrote Memray allocation profile to {config.memory_profile_path}{rank_note}",
            )

        if config.use_line_profiler and verbose_line_profiler:
            for region in cls.get_all_regions().values():
                if isinstance(region, LineProfilerRegion):
                    region.print_stats()

        # finalize() is intentionally still a checkpoint: existing callers
        # may record more events and finalize again without another setup().
        cls._lifecycle_state = _LifecycleState.FINALIZED
        cls._cancel_auto_finalize()
        if return_results:
            cls._last_results = results
            return results
        return None

    @classmethod
    def pause(cls) -> None:
        """Temporarily stop recording new profiling scopes.

        Pause/resume is intended for boundaries between simulation phases. An
        open scope cannot be split safely without changing its call identity,
        so pausing while a region is active raises a clear error.
        """
        if cls._config is None:
            raise RuntimeError("ProfileManager.setup() must be called before pause()")
        if cls._config.paused:
            return
        active = []
        for name, region in cls._regions.items():
            # The session envelope deliberately spans the entire session and
            # is allowed to include paused time. User scopes must still be
            # closed at a pause boundary so their intervals stay meaningful.
            if name == _ProfilingSession.ROOT_REGION_NAME:
                continue
            open_slots = getattr(region, "open_slots", None)
            if (
                open_slots is not None
                and len(open_slots())
                or getattr(region, "_stack", None)
            ):
                active.append(name)
        if active:
            names = ", ".join(repr(name) for name in active[:3])
            suffix = "..." if len(active) > 3 else ""
            raise RuntimeError(
                "cannot pause while profiling scopes are active: " + names + suffix,
            )
        cls._config._paused = True

    @classmethod
    def resume(cls) -> None:
        """Resume recording after :meth:`pause`.

        Calling ``resume`` before ``pause`` is harmless, which makes it safe
        to place in conditional simulation control paths.
        """
        if cls._config is None:
            raise RuntimeError("ProfileManager.setup() must be called before resume()")
        cls._config._paused = False

    @classmethod
    @contextmanager
    def sample_every(cls, every: int, start: int = 0):
        """Profile selected timesteps in a loop.

        The yielded callable is a context manager. Pass it the timestep
        number at the boundary of each iteration::

            with ProfileManager.sample_every(10) as profile_step:
                for timestep in range(num_steps):
                    with profile_step(timestep):
                        advance_simulation()

        Timestep ``start`` and then every ``every``-th timestep are profiled;
        all other iterations run while profiling is paused. The timestep
        context must surround the profiled work and must not be entered while
        another profiled scope is active. On exit, the previous pause state is
        restored.
        """
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError("every must be a positive integer")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("start must be an integer")
        config = cls.get_config()
        was_paused = config.paused

        @contextmanager
        def profile_step(timestep: int):
            if isinstance(timestep, bool) or not isinstance(timestep, int):
                raise TypeError("timestep must be an integer")
            selected = timestep >= start and (timestep - start) % every == 0
            if selected:
                cls.resume()
            else:
                cls.pause()
            try:
                yield selected
            finally:
                # The next step chooses its own state. This also makes a
                # manually used final profile_step() leave the manager in a
                # predictable state before the outer context restores it.
                pass

        try:
            yield profile_step
        finally:
            if was_paused:
                cls.pause()
            else:
                cls.resume()

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
        from scope_profiler.profile_io import read_profile

        return read_profile(cls._requested_output or cls.get_config().file_path)

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
    def get_all_regions(cls) -> dict[str, "BaseProfileRegion"]:
        """
        Get all registered ProfileRegion instances.

        Returns
        -------
        dict: Dictionary of all registered ProfileRegion instances.
        """
        return cls._regions

    @classmethod
    def define_region(cls, region_name, functions=None, tags=None) -> RegionHandle:
        """Define a reusable region handle that survives later setup calls."""
        definition = (
            None if functions is None else tuple(functions),
            None if tags is None else tuple(tags),
        )
        previous = cls._region_definitions.get(region_name)
        if previous is not None and previous != definition:
            raise ValueError(f"region {region_name!r} was already defined differently")
        cls._region_definitions[region_name] = definition
        cls.region(region_name, functions=functions, tags=tags)
        return RegionHandle(cls, region_name, functions=functions, tags=tags)

    @classmethod
    @contextmanager
    def metadata(cls, **values):
        """Attach key/value metadata to calls completed inside this scope."""
        if cls.is_active() and cls._config is not None and cls._config.aggregation_mode:
            raise RuntimeError(
                "scoped metadata requires per-call data and cannot be used in "
                "aggregation mode",
            )
        current = cls._metadata_context.get() or {}
        merged = {**current, **values}
        token = cls._metadata_context.set(merged)
        start = perf_counter_ns()
        try:
            yield merged
        finally:
            end = perf_counter_ns()
            cls._metadata_context.reset(token)
            if cls.is_active() and cls._config is not None:
                cls._metadata_scopes.append((start, end, merged))

    @classmethod
    def registered_regions(cls) -> tuple[str, ...]:
        """Names of all instrumentation points known to this manager."""
        return tuple(cls._regions)

    @classmethod
    def recorded_regions(cls) -> tuple[str, ...]:
        """Names of regions that have recorded at least one call."""
        return tuple(
            name for name, region in cls._regions.items() if region.num_calls > 0
        )

    @classmethod
    def is_configured(cls) -> bool:
        """Whether explicit setup has configured this manager."""
        return cls._configured

    @classmethod
    def is_active(cls) -> bool:
        """Whether instrumentation currently records calls."""
        return (
            cls._configured
            and cls._config is not None
            and not cls._config.deactivate_profiling
            and cls._lifecycle_state
            in {_LifecycleState.ACTIVE, _LifecycleState.FINALIZED}
        )

    @classmethod
    def last_results(cls):
        """Most recent in-memory results requested from this manager, if any."""
        return cls._last_results

    @classmethod
    def _cancel_auto_finalize(cls) -> None:
        """Remove this manager's pending process-exit finalizer, if any."""
        callback = cls._auto_finalize_callback
        cls._auto_finalize_callback = None
        if callback is not None:
            atexit.unregister(callback)

    @classmethod
    def _register_auto_finalize(cls) -> None:
        """Arrange one guarded finalization at normal interpreter exit."""
        cls._cancel_auto_finalize()

        def finalize_at_exit():
            # Clear first so finalize() knows it is running from the hook and
            # cannot unregister or repeat itself.
            cls._auto_finalize_callback = None
            if cls._configured:
                cls.finalize()

        cls._auto_finalize_callback = finalize_at_exit
        atexit.register(finalize_at_exit)

    @classmethod
    def _stop_mpi_call_profiling(cls) -> None:
        """Restore mpi4py globals if this manager installed their proxies."""
        context = cls._mpi_profile_context
        cls._mpi_profile_context = None
        if context is not None:
            context.__exit__(None, None, None)

    @classmethod
    def setup(
        cls,
        options: ProfilingOptions | None = None,
        *,
        auto_finalize: bool = False,
        output=_UNSET,
        replace: bool = False,
        config_path: str | os.PathLike[str] | None = None,
        **overrides: "Unpack[SetupOptions]",
    ):
        """
        Initialize and configure the profiling system.

        Parameters
        ----------
        options : ProfilingOptions, optional
            A :class:`~scope_profiler.profile_config.ProfilingOptions` bag
            holding any of the settings below, for reuse across calls or
            construction away from the call site::

                options = ProfilingOptions(use_likwid=True, file_path="run.h5")
                ProfileManager.setup(options=options)

            An explicit keyword argument passed alongside ``options`` wins
            over the same field on ``options``, which in turn wins over
            ``config_path`` and the defaults below.
        config_path : str or os.PathLike, optional
            TOML file containing a ``[profiling]`` table with these settings.
            Values passed directly to ``setup()`` take precedence. See
            :func:`~scope_profiler.profile_config.load_profiling_config`.
        auto_finalize : bool, optional
            Finalize once at normal interpreter exit, unless ``finalize()``
            is called explicitly first (default: False). This is convenient
            for scripts that want setup without a surrounding session. Avoid
            it when MPI ranks may not exit together.
        output : str, os.PathLike or None, optional
            Direct spelling of the output destination. ``None`` keeps results
            in memory and writes no file. This is an alias for ``file_path``
            and ``deactivate_file_output``; do not combine those spellings.
        replace : bool, optional
            Replace an active configuration (default: False). Without this,
            accidental setup of an already active manager raises.
        **overrides
            Any of the settings below, passed as keyword arguments::

                ProfileManager.setup(file_path="run.h5", use_likwid=True)

            They are the fields of
            :class:`~scope_profiler.profile_config.ProfilingOptions`, which
            is where they are declared once and typed; an unrecognised name
            raises ``TypeError`` naming the closest match. Prefixed settings
            can also be given as groups on ``options`` -- see
            :class:`~scope_profiler.profile_config.MemrayOptions`,
            :class:`~scope_profiler.profile_config.GPUOptions` and
            :class:`~scope_profiler.profile_config.HDF5Options`.
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
        use_likwid : bool, optional
            Enable LIKWID hardware counter collection (default: False).
        use_line_profiler : bool, optional
            Enable line-by-line profiling via line_profiler (default: False).
        use_memray : bool, optional
            Record process-wide allocations with Memray (default: False).
            The capture is a separate Memray ``.bin`` file and requires
            the separately installed ``memray`` package.
        memory_profile_path : str, optional
            Memray capture path (default: ``<file-stem>.memray.bin``).
        memray_native_traces, memray_trace_python_allocators, memray_follow_fork : bool, optional
            Memray capture options. Python allocator tracing can create much
            larger traces and has materially higher overhead.
        deactivate_profiling : bool, optional
            Turn profiling off entirely (default: False). Every region
            becomes a no-op, so the instrumentation can stay in the code at
            near-zero cost instead of being removed.
        use_nvtx : bool, optional
            Add NVTX ranges to profiled regions for NVIDIA Nsight tools
            (default: False). Requires ``scope-profiler[nvtx]``.
        use_gpu_timing : bool, optional
            Record CUDA-event elapsed device time for each profiled region
            (default: False). CPU timestamps are still recorded, so the normal
            timeline remains enqueue-side timing.
        gpu_timing_backend : str or object, optional
            CUDA-event backend for ``use_gpu_timing``: ``"auto"``, ``"torch"``,
            ``"cupy"``, or a custom object implementing ``record_event()`` and
            ``elapsed_time_ns(start_event, end_event)``.
        deactivate_file_output : bool, optional
            Write no HDF5 file at all (default: False), not even the run
            metadata. Use it with
            ``finalize(return_results=True)`` to analyse a run entirely in
            memory::

                ProfileManager.setup(deactivate_file_output=True)
                ...
                results = ProfileManager.finalize(return_results=True)

        recursive_profile : bool, optional
            Enable recursive profiling for all decorated functions by default
            (default: False). This can be overridden per decorator with
            ``@ProfileManager.profile(..., recursive=...)``.
        aggregation_mode : bool, optional
            Record only count, inclusive total, minimum, maximum, and
            exclusive total per region. Timeline events are unavailable in
            this mode; it cannot be combined with line, GPU, NVTX, or LIKWID
            profiling.
        profile_mpi_calls : bool, optional
            Profile mpi4py operations made through predefined and derived
            communicators (default: False). mpi4py is imported only if the
            application imports it. With :meth:`session`, obtain
            ``MPI.COMM_WORLD`` inside the session so it receives the proxy.
        track_threads : bool, optional
            Profile every thread (default: False). Each thread gets its own
            buffers and scope stack, so regions entered concurrently no
            longer overwrite one another, and every recorded call carries the
            thread it ran on. The run also reports each thread's name, OS
            ids, lifetime and CPU time -- see
            :attr:`~scope_profiler.results.ProfilingResults.threads`::

                with ProfileManager.session(track_threads=True):
                    ...

            Cannot be combined with line, GPU, NVTX, LIKWID or aggregation
            profiling.
        track_async : bool, optional
            Follow asyncio tasks and greenlets as well (default: False, and
            implies ``track_threads``). Each call additionally carries the
            task it ran in and the time that task spent suspended inside the
            call, so a ``with`` block held across an ``await`` reports its
            await time rather than charging it to the region. Per-task
            running and awaiting totals are available from
            :attr:`~scope_profiler.results.ProfilingResults.tasks`.
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

        buffer_limit : int, optional
            Initial number of profiling events preallocated per region
            (default: 1024). Buffers grow on demand, so this is a starting
            size rather than a limit; raise it for very hot regions to avoid
            repeated reallocation.
        output_mode : {"auto", "direct", "parallel"}, optional
            MPI file writer. ``auto`` prefers MPI-enabled h5py when compatible
            with the active instrumentation and otherwise lets ranks append
            directly to one serial-HDF5 file in token order.
        hdf5_compression : {"gzip", "lzf", "zstd"} or None, optional
            Compression filter for timestamp and GPU-duration datasets.
        hdf5_compression_level : int or None, optional
            GZIP level 0--9 or Zstandard level 1--22.
        hdf5_chunk_size : int or None, optional
            Maximum events per dataset chunk. Enables chunked partial reads
            even without compression.

        Notes
        -----
        The run's start time is the moment ``setup()`` is called; it is stored
        as the ``start_time_ns`` metadata field and is the origin of the
        relative timeline in post-processing. MPI rank detection is separate
        from ``profile_mpi_calls``: collectives are used exactly when the
        process was started by an MPI launcher. See
        :mod:`scope_profiler.mpi_launch` for detection and its
        ``SCOPE_PROFILER_MPI`` override.
        """
        if (
            cls._configured
            and cls._lifecycle_state is _LifecycleState.ACTIVE
            and not replace
        ):
            raise RuntimeError(
                "profiling is already configured; finalize it first or "
                "pass replace=True",
            )

        unknown = set(overrides) - _CONFIG_FIELDS
        if unknown:
            raise TypeError(_unknown_setting_error(unknown))

        if output is not _UNSET:
            conflicts = {
                name
                for name in ("file_path", "deactivate_file_output")
                if overrides.get(name) is not None
            }
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise TypeError(f"output cannot be combined with {names}")
            if output is None:
                overrides["deactivate_file_output"] = True
            else:
                overrides["file_path"] = os.fspath(output)
                overrides["deactivate_file_output"] = False

        # Defaults live in ProfilingConfig.__init__ alone; only settings that
        # were actually asked for are passed on, so precedence is simply the
        # order these three sources are applied in.
        settings: dict = {}
        if config_path is not None:
            settings.update(load_profiling_config(config_path))
        if options is not None:
            settings.update(options.to_kwargs())
        settings.update(
            {key: value for key, value in overrides.items() if value is not None},
        )
        requested_output = None
        if output is not _UNSET and output is not None:
            from scope_profiler.profile_io import FORMAT_HDF5, profile_format

            requested_output = os.fspath(output)
            if profile_format(requested_output) != FORMAT_HDF5:
                settings["file_path"] = requested_output + ".scope-profiler.h5"

        # Restore MPI globals before resolving the next run's native
        # communicator; otherwise get_comm() could retain the previous proxy.
        cls._stop_mpi_call_profiling()

        # Memray permits exactly one active tracker per process. A new setup
        # starts a new run, so close the prior run's capture first.
        if cls._config is not None:
            cls._config.stop_memory_profiling()
        ProfilingConfig.reset()
        config = ProfilingConfig(
            **settings,
        )
        cls.set_config(config=config)
        cls._requested_output = requested_output
        if auto_finalize:
            cls._register_auto_finalize()
        return ProfilingRun(cls, config, requested_output)

    @classmethod
    def session(
        cls,
        options: ProfilingOptions | None = None,
        *,
        verbose: bool = True,
        verbose_line_profiler: bool = False,
        return_results: bool = False,
        native_traces=None,
        output=_UNSET,
        replace: bool = False,
        config_path: str | os.PathLike[str] | None = None,
        **overrides: "Unpack[SetupOptions]",
    ):
        """Return a context manager that sets up and finalizes profiling.

        Every argument other than ``verbose``, ``verbose_line_profiler``,
        ``return_results`` and ``native_traces`` is passed to :meth:`setup`,
        with the same meaning and precedence there -- ``options`` (a
        :class:`~scope_profiler.profile_config.ProfilingOptions`),
        ``config_path``, and any setting as a keyword::

            with ProfileManager.session(options=options) as run:
                ...

            with ProfileManager.session(file_path="run.h5") as run:
                ...

        Finalization runs even when the profiled block raises; the original
        exception is preserved.

        ``native_traces`` is handed to :meth:`finalize`, which folds the
        output of a C or Fortran library profiled in this same process into
        this run's file -- in either native format. See its documentation for
        the two rules that mixing languages imposes.

        When ``return_results=True``, the context object exposes the finalized
        :class:`~scope_profiler.results.ProfilingResults` as ``results``::

            with ProfileManager.session(return_results=True, verbose=False) as run:
                with ProfileManager.profile_region("solve"):
                    solve()
            results = run.results
        """
        setup_kwargs: dict = dict(overrides)
        if output is not _UNSET:
            setup_kwargs["output"] = output
        if replace:
            setup_kwargs["replace"] = True
        if options is not None:
            setup_kwargs["options"] = options
        if config_path is not None:
            setup_kwargs["config_path"] = config_path
        return _ProfilingSession(
            cls,
            setup_kwargs,
            verbose,
            verbose_line_profiler,
            return_results,
            native_traces,
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
        cls._cancel_auto_finalize()
        cls._stop_mpi_call_profiling()
        cls._regions.clear()  # Clear old regions
        # A new run gets a fresh id space; ids stay unique only within one.
        cls._next_call_id = 0
        # The thread, asyncio and greenlet hooks belong to a configuration and
        # live exactly as long as it is the active one, so a run that
        # finalizes periodically keeps following the threads it started with.
        previous = cls._config
        if previous is not None and previous.tracker is not None:
            previous.tracker.uninstall()
        if previous is not None:
            previous.stop_memory_profiling()
        cls._config = config  # Update the config
        cls._last_results = None
        cls._requested_output = None
        cls._metadata_scopes = []
        cls._configured = True
        cls._lifecycle_state = (
            _LifecycleState.INACTIVE
            if config.deactivate_profiling
            else _LifecycleState.ACTIVE
        )
        if config.profile_mpi_calls and not config.deactivate_profiling:
            from scope_profiler.mpi_wrappers import profile_mpi4py

            cls._mpi_profile_context = profile_mpi4py(lazy=True)
            cls._mpi_profile_context.__enter__()
        if config.tracker is not None:
            config.tracker.install()
        cls._update_region_cls()  # Set the proper region class
        # Rebind all registered decorator wrappers to the new region class.
        # This is the only place rebinding happens — there is no per-call check.
        for name, entries in cls._decorators.items():
            for func, _bound in entries:
                cls._bind_decorated_region(name, func, _bound)
        for name, (functions, tags) in cls._region_definitions.items():
            cls.region(name, functions=functions, tags=tags)

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

        Resolving defaults is not itself setup. Regions remain disabled until
        :meth:`setup` (directly or through :meth:`session`) installs a config
        with :meth:`set_config`. This lets applications declare decorators at
        import time without accidentally starting a profiling run.

        Returns
        -------
        ProfilingConfig
            The current profiling configuration.
        """
        if cls._config is None:
            cls._config = ProfilingConfig()
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
        cls._cancel_auto_finalize()
        cls._stop_mpi_call_profiling()
        ProfilingConfig.reset()
        if cls._config is not None and cls._config.tracker is not None:
            cls._config.tracker.uninstall()
        if cls._config is not None:
            cls._config.stop_memory_profiling()
        cls._config = None
        cls._last_results = None
        cls._requested_output = None
        cls._metadata_scopes = []
        cls._configured = False
        cls._lifecycle_state = _LifecycleState.INACTIVE

    @classmethod
    def _reset(cls) -> None:
        cls._reset_regions()
        cls._reset_config()
        # Back to the state a fresh import leaves behind: no configuration,
        # and regions disabled until setup() or get_config() says otherwise.
        cls._region_cls = DisabledProfileRegion
        cls._decorators.clear()
        cls._region_definitions.clear()
