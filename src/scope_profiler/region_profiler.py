"""Profile region classes implementing the strategy pattern for different profiling modes."""

import ast
import functools
import inspect
import linecache
import types
from time import perf_counter_ns
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from scope_profiler.gpu_timing import resolve_gpu_timing_backend
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


def _function_for_frame(frame):
    """Build a function object suitable for line_profiler from an active frame.

    A context manager receives a frame rather than a function object. Existing
    module/local references are preferred; the fallback reconstructs a small
    function with the same code and closure values so nested functions work as
    well. line_profiler only uses the function's code metadata when registering
    it, so the reconstructed function is never called.
    """
    code = frame.f_code
    for namespace in (frame.f_locals, frame.f_globals):
        for value in namespace.values():
            if isinstance(value, types.FunctionType) and value.__code__ is code:
                return value

    if code.co_freevars:

        def make_cell(value):
            return (lambda: value).__closure__[0]

        closure = tuple(
            make_cell(frame.f_locals.get(name)) for name in code.co_freevars
        )
    else:
        closure = None
    try:
        return types.FunctionType(code, frame.f_globals, code.co_name, closure=closure)
    except (TypeError, ValueError):
        return None


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

# End-time value of a slot that has been reserved but not yet written: the
# call is still running. Entering a region hands out its slot before the code
# runs, so a finalize() in the middle of a call would otherwise copy out an
# uninitialised end timestamp -- as garbage from `np.empty` in the decorator
# form, which does not touch either buffer until the call returns.
#
# Zero is the marker, and it costs nothing on either axis. A monotonic clock
# reading is never 0 (it counts from boot, not from an epoch the process can
# reach), which is the same reasoning that makes the native implementations
# return a *negative* timestamp for a failed clock read rather than 0. And
# `np.zeros` is calloc, so allocating and growing a buffer stays O(1) --
# `np.full` would make creating a region proportional to its capacity, which
# tests/test_overhead.py budgets against.
_UNCLOSED = 0
_AGGREGATE_STACK = []


class AggregateProfileRegion:
    """Low-memory region that records aggregate timing statistics only."""

    __slots__ = (
        "region_name",
        "config",
        "tags",
        "source_file",
        "source_lineno",
        "source_text",
        "_completed",
        "_count",
        "_total",
        "_minimum",
        "_maximum",
        "_exclusive",
        "_stack",
    )

    def __init__(self, region_name, config, tags=()):
        self.region_name = region_name
        self.config = config
        self.tags = tuple(tags)
        self.source_file = self.source_lineno = self.source_text = None
        self._completed = 0
        self._count = 0
        self._total = 0
        self._minimum = None
        self._maximum = None
        self._exclusive = 0
        self._stack = []

    @property
    def ptr(self):
        return 0

    @property
    def num_calls(self):
        return self._completed + self._count

    @property
    def has_source(self):
        return self.source_file is not None

    def set_source(self, filename, lineno, text):
        if self.source_file is None and filename is not None:
            self.source_file, self.source_lineno, self.source_text = (
                filename,
                lineno,
                text,
            )

    def add_function(self, func):
        pass

    def get_aggregate(self):
        return {
            "count": self._count,
            "total": self._total,
            "minimum": 0 if self._minimum is None else self._minimum,
            "maximum": 0 if self._maximum is None else self._maximum,
            "exclusive": self._exclusive,
        }

    def _enter(self):
        self._stack.append(perf_counter_ns())
        _AGGREGATE_STACK.append(self)

    def _leave(self):
        duration = perf_counter_ns() - self._stack.pop()
        _AGGREGATE_STACK.pop()
        if _AGGREGATE_STACK:
            # Only the direct parent subtracts this duration. Its own
            # inclusive duration is then subtracted from its parent when it
            # exits, preventing nested children from being subtracted twice.
            _AGGREGATE_STACK[-1]._exclusive -= duration
        self._count += 1
        self._total += duration
        self._exclusive += duration
        self._minimum = (
            duration if self._minimum is None else min(self._minimum, duration)
        )
        self._maximum = (
            duration if self._maximum is None else max(self._maximum, duration)
        )
        return duration

    def __enter__(self):
        self._enter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._leave()

    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._enter()
            try:
                return func(*args, **kwargs)
            finally:
                self._leave()

        return wrapper

    def mark_written(self):
        self._completed += self._count
        self._count = self._total = self._exclusive = 0
        self._minimum = self._maximum = None

    def aggregate_snapshot(self):
        return self.get_aggregate()


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
        "_emitted",
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
        # Slots already handed to a finalize() that could not rewind the
        # buffer afterwards, so the next one skips them instead of reporting
        # the same call twice. None whenever the buffer is clean.
        self._emitted = None

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
            self.end_times = np.zeros(self.capacity, dtype=np.int64)
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
        if self.source_file is not None or filename is None:
            return
        self.source_file = filename
        self.source_lineno = lineno
        self.source_text = text

    @property
    def has_source(self) -> bool:
        """Whether this region's call-site source was captured."""
        return self.source_file is not None

    def _grow(self) -> None:
        """Double the timestamp buffers, preserving already-recorded slots.

        Slot indices are handed out before the profiled call runs and written
        after it returns, so growth must keep every index valid: the contents
        are copied to the same positions in the larger buffers, and the
        deferred writes then land in the new arrays.
        """
        capacity = max(1, self.capacity * 2)
        start_times = np.empty(capacity, dtype=np.int64)
        end_times = np.zeros(capacity, dtype=np.int64)
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

    def open_slots(self) -> np.ndarray:
        """Buffer slots whose call is still running, in ascending order.

        A slot is open until its end time is written, which is what
        distinguishes a call in flight from one that has returned. Both region
        forms are covered: the context manager writes its start on entry, the
        decorator writes nothing until the call returns.
        """
        if not self.ptr:
            return _EMPTY_TIMES
        return np.flatnonzero(self.end_times[: self.ptr] == _UNCLOSED)

    def closed_slots(self) -> "np.ndarray | None":
        """Mask of buffered slots this finalize() should copy out.

        A slot qualifies once its end time is written and it has not already
        gone out with an earlier finalize(). None -- the usual case -- means
        every slot qualifies and the caller can skip the mask entirely.
        """
        if not self.ptr:
            return None
        mask = self.end_times[: self.ptr] != _UNCLOSED
        if self._emitted is not None:
            mask[: self._emitted.size] &= ~self._emitted
        return None if mask.all() else mask

    def mark_written(self) -> None:
        """Record that everything buffered so far has been handed to finalize().

        Called by ``finalize()`` once the data has been copied out, so that a
        second run in the same process reports only its own events instead of
        re-reporting the first run's. The timestamp buffer rewinds (the arrays
        are reused; anything past ``ptr`` is unread scratch), while
        ``num_calls`` keeps counting for the lifetime of the process — it is
        the in-memory view of the region, which callers inspect after
        ``finalize()``.

        A call still running has a slot reserved that finalize() did not copy
        out, so it must survive the rewind. It is moved to the front of the
        buffer rather than pinning everything behind it: leaving the whole
        buffer in place would make the next finalize() re-report every
        completed call sitting below it, with a second set of call ids.

        The one case that still cannot rewind is a call entered through the
        decorator form, which keeps its slot index in the wrapper's own frame
        where nothing can remap it. Recognisable because such a slot is open
        without appearing in ``_scope_ptr_stack``. There the buffer stays put
        and the slots already copied out are recorded instead, so the next
        finalize() skips them rather than reporting those calls again.
        """
        open_slots = self.open_slots()
        if open_slots.size == 0:
            if self.ptr:
                self.end_times[: self.ptr] = _UNCLOSED
            self._completed += self.ptr
            self.ptr = 0
            self._emitted = None
            return
        if open_slots.tolist() != sorted(self._scope_ptr_stack):
            self._emitted = self.end_times[: self.ptr] != _UNCLOSED
            return
        self._completed += self.ptr - open_slots.size
        self._compact(open_slots)
        self._emitted = None

    def _compact(self, keep: np.ndarray) -> None:
        """Move the still-open slots to the front and rewind ``ptr`` onto them.

        ``keep`` is ascending, so the destinations never run ahead of the
        sources. The scope stack is rewritten in place, which keeps the
        ``_push_scope``/``_pop_scope`` bindings made in ``__init__`` valid.
        """
        count = keep.size
        self.start_times[:count] = self.start_times[keep]
        # Every slot from here up is unwritten again, including the ones the
        # kept calls vacated: a later finalize() must not read a stale end
        # time as "this call has returned".
        self.end_times[: self.ptr] = _UNCLOSED
        remapped = {int(old): new for new, old in enumerate(keep.tolist())}
        self._scope_ptr_stack[:] = [remapped[slot] for slot in self._scope_ptr_stack]
        self.ptr = count

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


class CUDATimingProfileRegion(TimeOnlyProfileRegion):
    """Region that records CPU timing plus CUDA-event elapsed device time."""

    __slots__ = (
        "_gpu_backend",
        "_gpu_start_events",
        "_gpu_end_events",
        "gpu_durations",
    )

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        super().__init__(region_name, config, tags=tags)
        self._gpu_backend = resolve_gpu_timing_backend(config.gpu_timing_backend)
        self._gpu_start_events = [None] * self.capacity
        self._gpu_end_events = [None] * self.capacity
        self.gpu_durations = np.empty(self.capacity, dtype=np.int64)

    def _grow(self) -> None:
        old_capacity = self.capacity
        super()._grow()
        self._gpu_start_events.extend([None] * (self.capacity - old_capacity))
        self._gpu_end_events.extend([None] * (self.capacity - old_capacity))
        gpu_durations = np.empty(self.capacity, dtype=np.int64)
        gpu_durations[:old_capacity] = self.gpu_durations
        self.gpu_durations = gpu_durations

    def wrap(self, func):
        """Wrap a function to measure CPU enqueue time and device elapsed time."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            scope_ptr = self.ptr
            if scope_ptr >= self.capacity:
                self._grow()
            self.ptr = scope_ptr + 1
            start = perf_counter_ns()
            self._gpu_start_events[scope_ptr] = self._gpu_backend.record_event()
            try:
                return func(*args, **kwargs)
            finally:
                self._gpu_end_events[scope_ptr] = self._gpu_backend.record_event()
                end = perf_counter_ns()
                self.start_times[scope_ptr] = start
                self.end_times[scope_ptr] = end

        return wrapper

    def __enter__(self):
        """Record CPU start time and a CUDA start event."""
        slot = self.ptr
        if slot >= self.capacity:
            self._grow()
        self.ptr = slot + 1
        self._push_scope(slot)
        self.start_times[slot] = perf_counter_ns()
        self._gpu_start_events[slot] = self._gpu_backend.record_event()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record a CUDA end event and CPU end time at this scope's slot."""
        slot = self._pop_scope()
        self._gpu_end_events[slot] = self._gpu_backend.record_event()
        self.end_times[slot] = perf_counter_ns()

    def _compact(self, keep: np.ndarray) -> None:
        """Move the CUDA events and device durations alongside the timestamps."""
        kept = keep.tolist()
        start_events = [self._gpu_start_events[slot] for slot in kept]
        end_events = [self._gpu_end_events[slot] for slot in kept]
        durations = self.gpu_durations[keep].copy()
        super()._compact(keep)
        count = len(kept)
        self._gpu_start_events[:count] = start_events
        self._gpu_end_events[:count] = end_events
        self.gpu_durations[:count] = durations

    def get_gpu_durations_numpy(self) -> np.ndarray:
        """Synchronize recorded CUDA events and return device durations in ns."""
        for index in range(self.ptr):
            start = self._gpu_start_events[index]
            end = self._gpu_end_events[index]
            if start is None or end is None:
                continue
            self.gpu_durations[index] = self._gpu_backend.elapsed_time_ns(start, end)
        return self.gpu_durations[: self.ptr]


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


class CUDATimingNVTXProfileRegion(CUDATimingProfileRegion):
    """Region that records CPU/CUDA-event timing and emits an NVTX range."""

    __slots__ = ("_nvtx",)

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        super().__init__(region_name, config, tags=tags)
        self._nvtx = _import_nvtx()

    def wrap(self, func):
        """Wrap a function with CPU/CUDA timing and an NVTX range."""
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
        """Record CPU/CUDA start markers and push an NVTX range."""
        self._nvtx.push_range(self.region_name)
        try:
            return super().__enter__()
        except BaseException:
            self._nvtx.pop_range()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """Record CPU/CUDA end markers and pop the NVTX range."""
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

    __slots__ = ("_line_profiler", "_registered_codes", "_manual_line_timings")

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        """Initialize timing buffers and line_profiler instance."""
        super().__init__(region_name, config, tags=tags)
        LineProfiler = _import_line_profiler()
        self._line_profiler = LineProfiler()
        self._registered_codes = set()
        self._manual_line_timings = {}

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
        return self.enter_frame(inspect.currentframe().f_back)

    def enter_frame(self, frame):
        """Enter this region while registering ``frame`` with line_profiler."""
        if frame.f_code not in self._registered_codes:
            func = _function_for_frame(frame)
            if func is not None:
                self._line_profiler.add_function(func)
                self._registered_codes.add(frame.f_code)
        slot = self.ptr
        if slot >= self.capacity:
            self._grow()
        self.ptr = slot + 1
        self._push_scope(slot)
        self.start_times[slot] = perf_counter_ns()
        self._line_profiler.enable_by_count()
        return self

    def enter_timing_only(self):
        """Enter this region without registering or enabling line_profiler."""
        slot = self.ptr
        if slot >= self.capacity:
            self._grow()
        self.ptr = slot + 1
        self._push_scope(slot)
        self.start_times[slot] = perf_counter_ns()
        return self

    def record_line_timing(self, frame, lineno: int, duration_ns: int) -> None:
        """Record one line timing sample from recursive CLI tracing."""
        if duration_ns <= 0:
            return
        code = frame.f_code
        key = (
            code.co_filename,
            int(code.co_firstlineno),
            getattr(code, "co_qualname", None) or code.co_name,
        )
        by_line = self._manual_line_timings.setdefault(key, {})
        hits, elapsed = by_line.get(int(lineno), (0, 0.0))
        by_line[int(lineno)] = (hits + 1, elapsed + float(duration_ns))

    def manual_line_records(self, unit: float = 1e-9) -> list:
        """Return manually traced line timings in the persisted record shape."""
        records = []
        for (filename, first_lineno, function), by_line in sorted(
            self._manual_line_timings.items()
        ):
            line_numbers = np.asarray(sorted(by_line), dtype=np.int64)
            hits = np.asarray([by_line[int(line)][0] for line in line_numbers])
            times = np.asarray([by_line[int(line)][1] for line in line_numbers])
            records.append(
                {
                    "filename": str(filename),
                    "function": str(function),
                    "first_lineno": int(first_lineno),
                    "line_numbers": line_numbers,
                    "hits": hits,
                    "times": times,
                    "unit": unit,
                }
            )
        return records

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
