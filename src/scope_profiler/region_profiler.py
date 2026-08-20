"""Profile region classes implementing the strategy pattern for different profiling modes."""

import ast
import functools
import inspect
import linecache
from time import perf_counter_ns
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from scope_profiler.profile_config import ProfilingConfig

if TYPE_CHECKING:
    pass


# Parsed module ASTs, memoized by filename. Capturing a region's source only
# runs once per region name (see BaseProfileRegion.set_source), but a file can
# define many regions, so the parse itself is cached rather than repeated.
_AST_CACHE: Dict[str, Optional[ast.Module]] = {}


def _parsed_module(filename: str) -> Optional[ast.Module]:
    """Parse ``filename`` into an AST, memoized by filename.

    Returns None for anything that cannot be parsed (a REPL frame, a frozen
    module, a file that no longer exists), so callers fall back gracefully
    instead of raising.
    """
    if filename not in _AST_CACHE:
        try:
            text = "".join(linecache.getlines(filename))
            _AST_CACHE[filename] = ast.parse(text, filename=filename) if text else None
        except (OSError, SyntaxError, ValueError):
            _AST_CACHE[filename] = None
    return _AST_CACHE[filename]


# Lineno -> With node, memoized by filename. Keeping this separate from
# _AST_CACHE means a file with many regions costs one ast.walk() in total
# (done the first time any region in it is captured), not one per region: the
# naive approach -- walking the whole tree again for every new region name --
# turned out to cost hundreds of microseconds per region on a file of a few
# hundred lines (measured in test_source_capture_is_a_one_time_per_name_cost),
# which is fine once but not once per distinct name in a busy file.
_WITH_NODE_CACHE: Dict[str, Dict[int, ast.With]] = {}


def _with_nodes(filename: str) -> Dict[int, ast.With]:
    """Every ``with`` statement in ``filename``, indexed by its start line."""
    if filename not in _WITH_NODE_CACHE:
        tree = _parsed_module(filename)
        nodes: Dict[int, ast.With] = {}
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.With):
                    nodes[node.lineno] = node
        _WITH_NODE_CACHE[filename] = nodes
    return _WITH_NODE_CACHE[filename]


def call_site_source(filename: str, lineno: int) -> Optional[str]:
    """Source text of the ``with`` block starting at ``lineno`` in ``filename``.

    Falls back to just the call-site line when the enclosing ``with``
    statement cannot be located (e.g. the file cannot be parsed), so a region
    still gets *some* location information rather than none.
    """
    node = _with_nodes(filename).get(lineno)
    if node is not None:
        end_lineno = getattr(node, "end_lineno", node.lineno)
        lines = linecache.getlines(filename)[node.lineno - 1 : end_lineno]
        if lines:
            return "".join(lines)
    return linecache.getline(filename, lineno) or None


def function_source(func) -> Optional[Tuple[str, int, str]]:
    """Source file, starting line and text of a decorated function.

    Returns None when the source cannot be recovered (e.g. a function defined
    interactively or via ``exec``).
    """
    try:
        lines, first_lineno = inspect.getsourcelines(func)
        filename = inspect.getsourcefile(func) or func.__code__.co_filename
    except (OSError, TypeError):
        return None
    return filename, first_lineno, "".join(lines)


def _import_pylikwid():
    """Import and return the pylikwid module.

    This function exists to defer the import of pylikwid until needed,
    preventing unnecessary overhead when LIKWID profiling is disabled. It
    delegates so that regions and the config resolve pylikwid identically,
    including the liblikwid preloading fallback.
    """
    from scope_profiler.profile_config import _import_pylikwid as _import

    return _import()


def _import_line_profiler():
    """Import and return the LineProfiler class from line_profiler.

    Imported lazily: line_profiler is an optional dependency, needed only when
    ``use_line_profiler=True``.
    """
    try:
        from line_profiler import LineProfiler
    except ImportError as exc:
        raise ImportError(
            "Line-by-line profiling requested but line_profiler is not "
            "installed. Install scope-profiler[line-profiler], or "
            "line_profiler directly."
        ) from exc

    return LineProfiler


def _import_nvtx():
    """Import NVTX lazily, with an actionable optional-dependency error."""
    try:
        import nvtx
    except ImportError as exc:
        raise ImportError(
            "NVTX annotations requested but nvtx is not installed. Install "
            "scope-profiler[nvtx], or nvtx directly."
        ) from exc
    return nvtx


# Shared read-only stand-in for the timing buffers of regions that never record
# timestamps. Handing out one module-level array keeps `start_times`/`end_times`
# valid attributes (and the derived properties empty) at zero per-region cost.
_EMPTY_TIMES = np.empty(0, dtype=np.int64)
_EMPTY_TIMES.flags.writeable = False


# Base class with common functionality (buffer growth, HDF5 handling)
class BaseProfileRegion:
    """Base class providing shared profiling logic.

    Handles start/end time buffering and call counting. The buffers grow on
    demand and are copied out once, at the end of the run, by
    ``ProfileManager.finalize()`` -- regions never touch HDF5 themselves.
    """

    __slots__ = (
        "region_name",
        "config",
        "start_times",
        "end_times",
        "ptr",
        "buffer_limit",
        "capacity",
        "_completed",
        "_scope_ptr_stack",
        "_push_scope",
        "_pop_scope",
        "source_file",
        "source_lineno",
        "source_text",
        "tags",
    )

    # Subclasses that never write timestamps set this to False so no per-region
    # buffers are allocated.
    _records_time = True

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        """Initialize a profiling region.

        Parameters
        ----------
        region_name : str
        Name of the profiled region.
        config : ProfilingConfig
        Profiling configuration containing the initial buffer capacity,
        file paths, and timing reference.
        """
        self.region_name = region_name
        self.config = config
        self.tags = tuple(tags)
        # Calls already copied out by an earlier finalize(); `num_calls` adds
        # this to `ptr` rather than being incremented on every entry, which
        # takes one attribute write out of the hot path.
        self._completed = 0

        # Preallocate buffers (skipped entirely when no timing is recorded).
        # `buffer_limit` is the *initial* capacity: `_grow` doubles it as
        # needed, so the number of calls a region can record is bounded only
        # by memory.
        #
        # The arrays are int64, so storing a timestamp converts the plain
        # Python int from perf_counter_ns() on assignment. Do not wrap the
        # clock reads in np.int64() to "match" the dtype: building the numpy
        # scalar costs ~190 ns each, twice per call, and stores the same
        # value. See tests/test_overhead.py for the budget this buys.
        self.ptr = 0
        self.buffer_limit = config.buffer_limit
        if self._records_time:
            self.capacity = self.buffer_limit
            self.start_times = np.empty(self.capacity, dtype=np.int64)
            self.end_times = np.empty(self.capacity, dtype=np.int64)
        else:
            self.capacity = 0
            self.start_times = _EMPTY_TIMES
            self.end_times = _EMPTY_TIMES

        # Recursion support: entering a scope reserves its slot (via `ptr`)
        # immediately and remembers it here, so a recursive re-entry before
        # the outer call exits reserves its own slot instead of clobbering
        # the outer one. Only the context-manager form needs the stack; the
        # decorator form keeps its slot in the wrapper's local scope.
        #
        # The push/pop are bound once here: resolving `self._scope_ptr_stack`
        # and then its `.append`/`.pop` costs ~10 ns each, per call, forever.
        self._scope_ptr_stack = []
        self._push_scope = self._scope_ptr_stack.append
        self._pop_scope = self._scope_ptr_stack.pop

        # Where this region is defined in user code, set once via
        # set_source() (see ProfileManager._capture_region_source and the
        # decorator path in ProfileManager.profile). None until then, and for
        # regions whose source could not be recovered at all.
        self.source_file = None
        self.source_lineno = None
        self.source_text = None

    def set_source(self, filename, lineno, text) -> None:
        """Record where this region is defined, the first time it is called.

        First writer wins: a region name reused at more than one call site
        keeps only the first location, matching how their timings are already
        pooled together under one name (see issue #161).
        """
        if self.source_text is not None or filename is None:
            return
        self.source_file = filename
        self.source_lineno = lineno
        self.source_text = text

    @property
    def has_source(self) -> bool:
        """Whether this region's call-site source was captured."""
        return self.source_text is not None

    def _grow(self) -> None:
        """Double the timestamp buffers, preserving already-recorded slots.

        Slot indices are handed out before the profiled call runs and written
        after it returns, so growth must keep every index valid: the contents
        are copied to the same positions in the larger buffers, and the
        deferred writes then land in the new arrays.
        """
        capacity = max(1, self.capacity * 2)
        start_times = np.empty(capacity, dtype=np.int64)
        end_times = np.empty(capacity, dtype=np.int64)
        start_times[: self.capacity] = self.start_times
        end_times[: self.capacity] = self.end_times
        self.start_times = start_times
        self.end_times = end_times
        # Cached so the hot path compares two ints rather than reaching into
        # the array for its size on every call.
        self.capacity = capacity

    def wrap(self, func):
        """Wrap a function for profiling.

        Subclasses must override this method to implement the appropriate
        profiling behavior.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def append(self, start: float, end: float) -> None:
        """Append a start/end time pair to the buffer, growing it if needed."""

        if self.ptr >= self.capacity:
            self._grow()
        self.start_times[self.ptr] = start
        self.end_times[self.ptr] = end
        self.ptr += 1

    @property
    def num_calls(self) -> int:
        """Times this region was entered, for the lifetime of the process.

        Derived rather than counted: the slots in use (``ptr``) plus whatever
        earlier finalize() calls already copied out. Keeping it out of
        ``__enter__`` removes an attribute write from every recorded call.
        """
        return self._completed + self.ptr

    def get_durations_numpy(self) -> np.ndarray:
        """Return durations (end - start) for buffered entries as a NumPy array."""
        return self.end_times[: self.ptr] - self.start_times[: self.ptr]

    def mark_written(self) -> None:
        """Record that everything buffered so far has been handed to finalize().

        Called by ``finalize()`` once the data has been copied out, so that a
        second run in the same process reports only its own events instead of
        re-reporting the first run's. The timestamp buffer rewinds (the arrays
        are reused; anything past ``ptr`` is unread scratch), while
        ``num_calls`` keeps counting for the lifetime of the process — it is
        the in-memory view of the region, which callers inspect after
        ``finalize()``.

        A region that is currently open has a slot reserved in the buffer and
        an index waiting to be popped on exit, so rewinding under it would let
        the next call overwrite a live slot. Such a region is left untouched.
        """
        if self._scope_ptr_stack:
            return
        self._completed += self.ptr
        self.ptr = 0

    def get_end_times_numpy(self) -> np.ndarray:
        """Return end times offset by the run's start time."""
        return self.end_times[: self.ptr] - self.config.start_time_ns

    def get_start_times_numpy(self) -> np.ndarray:
        """Return start times offset by the run's start time."""
        return self.start_times[: self.ptr] - self.config.start_time_ns

    def add_function(self, func) -> None:
        """Register a function for profiling. No-op except in LineProfilerRegion."""
        pass


# Disabled region: does nothing
class DisabledProfileRegion(BaseProfileRegion):
    """Profiling region that performs no measurements.

    Used when profiling is disabled but code paths must remain valid.
    """

    _records_time = False

    def wrap(self, func):
        """Return the original function unchanged — no wrapper, no overhead."""
        return func

    def append(self, start, end):
        """Ignored: no data recorded."""
        pass

    def get_durations_numpy(self):
        """Return an empty array since nothing is recorded."""
        return np.array([])

    def __enter__(self):
        """Enter a non-operational context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit a non-operational context manager."""
        pass


# Time-only region
class TimeOnlyProfileRegion(BaseProfileRegion):
    """Region that records timing, collected once at the end of the run."""

    def wrap(self, func):
        """Wrap a function to measure its execution time."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Reserve this call's slot before invoking `func`, so a
            # recursive call re-entering this region gets its own slot
            # instead of overwriting this one.
            scope_ptr = self.ptr
            if scope_ptr >= self.capacity:
                self._grow()
            self.ptr = scope_ptr + 1
            start = perf_counter_ns()
            try:
                return func(*args, **kwargs)
            finally:
                end = perf_counter_ns()
                self.start_times[scope_ptr] = start
                self.end_times[scope_ptr] = end

        return wrapper

    def __enter__(self):
        """Reserve this scope's slot and record the start time."""
        slot = self.ptr
        if slot >= self.capacity:
            self._grow()
        self.ptr = slot + 1
        self._push_scope(slot)
        self.start_times[slot] = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record the end time at this scope's reserved slot."""
        self.end_times[self._pop_scope()] = perf_counter_ns()


# NVTX region: time + NVIDIA Nsight annotation
class NVTXProfileRegion(TimeOnlyProfileRegion):
    """Region that records CPU time and emits an NVTX range.

    NVTX is an annotation API, not a GPU timing API. Nsight Systems and
    Nsight Compute consume the ranges and correlate them with GPU work. The
    normal CPU timestamps are retained in the scope-profiler result.
    """

    __slots__ = ("_nvtx",)

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        super().__init__(region_name, config, tags=tags)
        self._nvtx = _import_nvtx()

    def wrap(self, func):
        """Wrap a function with both CPU timing and an NVTX range."""
        wrapped = super().wrap(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._nvtx.push_range(self.region_name)
            try:
                return wrapped(*args, **kwargs)
            finally:
                self._nvtx.pop_range()

        return wrapper

    def __enter__(self):
        """Record CPU start time and push an NVTX range."""
        self._nvtx.push_range(self.region_name)
        try:
            return super().__enter__()
        except BaseException:
            self._nvtx.pop_range()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """Pop the NVTX range after recording CPU end time."""
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._nvtx.pop_range()


# Full region: time + LIKWID
class FullProfileRegion(BaseProfileRegion):
    """Region that records both timing and LIKWID metrics, and writes to HDF5.

    This is the most complete profiling mode: users obtain LIKWID markers and
    nanosecond-resolution timing.
    """

    __slots__ = ("likwid_marker_start", "likwid_marker_stop")

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        """Initialize timing buffers, HDF5 paths, and LIKWID callbacks."""
        super().__init__(region_name, config, tags=tags)
        pylikwid = _import_pylikwid()
        self.likwid_marker_start = pylikwid.markerstartregion
        self.likwid_marker_stop = pylikwid.markerstopregion

    def wrap(self, func):
        """Wrap a function to measure time and collect LIKWID metrics."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Reserve this call's slot before invoking `func`, so a
            # recursive call re-entering this region gets its own slot
            # instead of overwriting this one.
            scope_ptr = self.ptr
            if scope_ptr >= self.capacity:
                self._grow()
            self.ptr = scope_ptr + 1
            start = perf_counter_ns()
            self.likwid_marker_start(self.region_name)
            try:
                return func(*args, **kwargs)
            finally:
                self.likwid_marker_stop(self.region_name)
                end = perf_counter_ns()
                self.start_times[scope_ptr] = start
                self.end_times[scope_ptr] = end

        return wrapper

    def __enter__(self):
        """Reserve this scope's slot, record start time, and start LIKWID region."""
        slot = self.ptr
        if slot >= self.capacity:
            self._grow()
        self.ptr = slot + 1
        self._push_scope(slot)
        self.start_times[slot] = perf_counter_ns()
        self.likwid_marker_start(self.region_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record the end time at this scope's slot and stop the LIKWID region."""
        self.likwid_marker_stop(self.region_name)
        self.end_times[self._pop_scope()] = perf_counter_ns()


# Line profiler region: time + line_profiler
class LineProfilerRegion(BaseProfileRegion):
    """Region that records timing and line-by-line profiling via line_profiler.

    Uses line_profiler to collect per-line execution statistics for decorated
    functions. Also records nanosecond timestamps.

    Line-by-line profiling is most useful with the decorator (``wrap``) path,
    which automatically registers the function with the line profiler.  When
    used as a context manager, the profiler is enabled/disabled around the
    block - any functions previously added via the decorator path will be
    profiled while the context is active.
    """

    __slots__ = ("_line_profiler",)

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        """Initialize timing buffers and line_profiler instance."""
        super().__init__(region_name, config, tags=tags)
        LineProfiler = _import_line_profiler()
        self._line_profiler = LineProfiler()

    def wrap(self, func):
        """Wrap a function to measure execution time and collect line-by-line stats."""
        self._line_profiler.add_function(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Reserve this call's slot before invoking `func`, so a
            # recursive call re-entering this region gets its own slot
            # instead of overwriting this one.
            scope_ptr = self.ptr
            if scope_ptr >= self.capacity:
                self._grow()
            self.ptr = scope_ptr + 1
            start = perf_counter_ns()
            self._line_profiler.enable_by_count()
            try:
                return func(*args, **kwargs)
            finally:
                self._line_profiler.disable_by_count()
                end = perf_counter_ns()
                self.start_times[scope_ptr] = start
                self.end_times[scope_ptr] = end

        return wrapper

    def __enter__(self):
        """Reserve this scope's slot, record start time, and enable line profiler."""
        slot = self.ptr
        if slot >= self.capacity:
            self._grow()
        self.ptr = slot + 1
        self._push_scope(slot)
        self.start_times[slot] = perf_counter_ns()
        self._line_profiler.enable_by_count()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Disable the line profiler and record the end time at this scope's slot."""
        self._line_profiler.disable_by_count()
        self.end_times[self._pop_scope()] = perf_counter_ns()

    def add_function(self, func) -> None:
        """Register a function for line-by-line profiling."""
        self._line_profiler.add_function(func)

    def print_stats(self):
        """Print line-by-line profiling statistics."""
        self._line_profiler.print_stats()

    def get_stats(self):
        """Return the line_profiler stats object."""
        return self._line_profiler.get_stats()
