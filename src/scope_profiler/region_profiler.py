"""Profile region classes implementing the strategy pattern for different profiling modes."""

import ast
import functools
import inspect
import linecache
import threading
import types
from time import perf_counter_ns

import numpy as np

from scope_profiler.concurrency import CPU_SAMPLE_MASK, NO_TASK
from scope_profiler.gpu_timing import resolve_gpu_timing_backend
from scope_profiler.perf_events import PerfEventGroup, PerfEventTotals
from scope_profiler.profile_config import ProfilingConfig

# Parsed module ASTs, memoized by filename. Capturing a region's source only
# runs once per region name (see BaseProfileRegion.set_source), but a file can
# define many regions, so the parse itself is cached rather than repeated.
_AST_CACHE: dict[str, ast.Module | None] = {}


def _parsed_module(filename: str) -> ast.Module | None:
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
_WITH_NODE_CACHE: dict[str, dict[int, ast.With]] = {}


def _with_nodes(filename: str) -> dict[int, ast.With]:
    """Every ``with`` statement in ``filename``, indexed by its start line."""
    if filename not in _WITH_NODE_CACHE:
        tree = _parsed_module(filename)
        nodes: dict[int, ast.With] = {}
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.With):
                    nodes[node.lineno] = node
        _WITH_NODE_CACHE[filename] = nodes
    return _WITH_NODE_CACHE[filename]


def call_site_source(filename: str, lineno: int) -> str | None:
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


def function_source(func) -> tuple[str, int, str] | None:
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
            "line_profiler directly.",
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
            "scope-profiler[nvtx], or nvtx directly.",
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


class AggregateProfileRegion:
    """Low-memory region that records aggregate timing statistics only."""

    __slots__ = (
        "_completed",
        "_count",
        "_exclusive",
        "_maximum",
        "_minimum",
        "_paused_contexts",
        "_stack",
        "_total",
        "config",
        "region_name",
        "source_file",
        "source_lineno",
        "source_text",
        "tags",
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
        self._paused_contexts = []

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
        if self.config.paused:
            return
        self._stack.append(perf_counter_ns())
        self.config._aggregate_stack.append(self)

    def _leave(self):
        if self.config.paused:
            return
        duration = perf_counter_ns() - self._stack.pop()
        aggregate_stack = self.config._aggregate_stack
        aggregate_stack.pop()
        if aggregate_stack:
            # Only the direct parent subtracts this duration. Its own
            # inclusive duration is then subtracted from its parent when it
            # exits, preventing nested children from being subtracted twice.
            aggregate_stack[-1]._exclusive -= duration
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

    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.config.paused:
                return func(*args, **kwargs)
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

    def __enter__(self):
        self._paused_contexts.append(self.config.paused)
        if self._paused_contexts[-1]:
            return self
        self._enter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._paused_contexts.pop():
            return
        self._leave()


# Base class with common functionality (buffer growth, HDF5 handling)
class BaseProfileRegion:
    """Base class providing shared profiling logic.

    Handles start/end time buffering and call counting. The buffers grow on
    demand and are copied out once, at the end of the run, by
    ``ProfileManager.finalize()`` -- regions never touch HDF5 themselves.
    """

    __slots__ = (
        "_completed",
        "_emitted",
        "_paused_contexts",
        "_pop_scope",
        "_push_scope",
        "_scope_ptr_stack",
        "buffer_limit",
        "capacity",
        "config",
        "end_times",
        "ptr",
        "region_name",
        "source_file",
        "source_lineno",
        "source_text",
        "start_times",
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
        self._paused_contexts = []
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

    def get_end_times_numpy(self) -> np.ndarray:
        """Return end times offset by the run's start time."""
        return self.end_times[: self.ptr] - self.config.start_time_ns

    def get_start_times_numpy(self) -> np.ndarray:
        """Return start times offset by the run's start time."""
        return self.start_times[: self.ptr] - self.config.start_time_ns

    def add_function(self, func) -> None:
        """Register a function for profiling. No-op except in LineProfilerRegion."""


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

    def get_durations_numpy(self):
        """Return an empty array since nothing is recorded."""
        return np.array([])

    def __enter__(self):
        """Enter a non-operational context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit a non-operational context manager."""


# Time-only region
class TimeOnlyProfileRegion(BaseProfileRegion):
    """Region that records timing, collected once at the end of the run."""

    def wrap(self, func):
        """Wrap a function to measure its execution time."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.config.paused:
                return func(*args, **kwargs)
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
        self._paused_contexts.append(self.config.paused)
        if self._paused_contexts[-1]:
            return self
        slot = self.ptr
        if slot >= self.capacity:
            self._grow()
        self.ptr = slot + 1
        self._push_scope(slot)
        self.start_times[slot] = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record the end time at this scope's reserved slot."""
        if self._paused_contexts.pop():
            return
        self.end_times[self._pop_scope()] = perf_counter_ns()


class PerfEventProfileRegion(TimeOnlyProfileRegion):
    """Timing region which also sums Linux ``perf_event_open`` counters.

    Counters are opened per active invocation. This makes nested and recursive
    regions correct without sharing a mutable counter state, at the intended
    cost of an extra kernel round trip per selected event.
    """

    __slots__ = ("_perf_groups", "_perf_totals")

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        super().__init__(region_name, config, tags=tags)
        self._perf_groups = []
        self._perf_totals = {event: 0 for event in config.perf_events}

    def _start_perf(self) -> None:
        group = PerfEventGroup(self.config.perf_events)
        group.start()
        self._perf_groups.append(group)

    def _stop_perf(self) -> None:
        values = self._perf_groups.pop().stop()
        for event, value in values.items():
            self._perf_totals[event] += value

    def perf_event_totals(self) -> PerfEventTotals:
        """Counter totals accumulated by completed calls in this region."""
        return PerfEventTotals(calls=self.num_calls, values=dict(self._perf_totals))

    def wrap(self, func):
        wrapped = super().wrap(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.config.paused:
                return func(*args, **kwargs)
            self._start_perf()
            try:
                return wrapped(*args, **kwargs)
            finally:
                self._stop_perf()

        return wrapper

    def __enter__(self):
        if self.config.paused:
            self._paused_contexts.append(True)
            return self
        # TimeOnlyProfileRegion owns the pause stack, so start the counter
        # first and let its normal enter path create exactly one stack entry.
        self._start_perf()
        try:
            return super().__enter__()
        except BaseException:
            self._stop_perf()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        paused = bool(self._paused_contexts) and self._paused_contexts[-1]
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            if not paused:
                self._stop_perf()


class CUDATimingProfileRegion(TimeOnlyProfileRegion):
    """Region that records CPU timing plus CUDA-event elapsed device time."""

    __slots__ = (
        "_gpu_backend",
        "_gpu_end_events",
        "_gpu_start_events",
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
            if self.config.paused:
                return func(*args, **kwargs)
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

    def __enter__(self):
        """Record CPU start time and a CUDA start event."""
        self._paused_contexts.append(self.config.paused)
        if self._paused_contexts[-1]:
            return self
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
        if self._paused_contexts.pop():
            return
        slot = self._pop_scope()
        self._gpu_end_events[slot] = self._gpu_backend.record_event()
        self.end_times[slot] = perf_counter_ns()


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
            if self.config.paused:
                return func(*args, **kwargs)
            self._nvtx.push_range(self.region_name)
            try:
                return wrapped(*args, **kwargs)
            finally:
                self._nvtx.pop_range()

        return wrapper

    def __enter__(self):
        """Record CPU start time and push an NVTX range."""
        if self.config.paused:
            self._paused_contexts.append(True)
            return self
        self._nvtx.push_range(self.region_name)
        try:
            return super().__enter__()
        except BaseException:
            self._nvtx.pop_range()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """Pop the NVTX range after recording CPU end time.

        A region entered while profiling was paused pushed no range, so it
        must not pop one either: NVTX keeps a single per-thread range stack,
        and one unmatched pop shifts the nesting of every range emitted
        afterwards, for the rest of the process.
        """
        paused = bool(self._paused_contexts) and self._paused_contexts[-1]
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            if not paused:
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
            if self.config.paused:
                return func(*args, **kwargs)
            self._nvtx.push_range(self.region_name)
            try:
                return wrapped(*args, **kwargs)
            finally:
                self._nvtx.pop_range()

        return wrapper

    def __enter__(self):
        """Record CPU/CUDA start markers and push an NVTX range."""
        if self.config.paused:
            self._paused_contexts.append(True)
            return self
        self._nvtx.push_range(self.region_name)
        try:
            return super().__enter__()
        except BaseException:
            self._nvtx.pop_range()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """Record CPU/CUDA end markers and pop the NVTX range.

        Paused regions pushed no range; see
        :meth:`NVTXProfileRegion.__exit__` for why popping one anyway would
        corrupt the rest of the run's NVTX nesting.
        """
        paused = bool(self._paused_contexts) and self._paused_contexts[-1]
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            if not paused:
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
            if self.config.paused:
                return func(*args, **kwargs)
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
        self._paused_contexts.append(self.config.paused)
        if self._paused_contexts[-1]:
            return self
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
        if self._paused_contexts.pop():
            return
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

    __slots__ = ("_line_profiler", "_manual_line_timings", "_registered_codes")

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
            if self.config.paused:
                return func(*args, **kwargs)
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

    def enter_frame(self, frame):
        """Enter this region while registering ``frame`` with line_profiler."""
        self._paused_contexts.append(self.config.paused)
        if self._paused_contexts[-1]:
            return self
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
        self._paused_contexts.append(self.config.paused)
        if self._paused_contexts[-1]:
            return self
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
            self._manual_line_timings.items(),
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
                },
            )
        return records

    def add_function(self, func) -> None:
        """Register a function for line-by-line profiling."""
        self._line_profiler.add_function(func)

    def print_stats(self):
        """Print line-by-line profiling statistics."""
        self._line_profiler.print_stats()

    def get_stats(self):
        """Return the line_profiler stats object."""
        return self._line_profiler.get_stats()

    def __enter__(self):
        """Reserve this scope's slot, record start time, and enable line profiler."""
        return self.enter_frame(inspect.currentframe().f_back)

    def __exit__(self, exc_type, exc_value, traceback):
        """Disable the line profiler and record the end time at this scope's slot."""
        if self._paused_contexts.pop():
            return
        self._line_profiler.disable_by_count()
        self.end_times[self._pop_scope()] = perf_counter_ns()


class _LaneBuffer:
    """One thread's timestamp buffers for one region.

    Threaded regions never share a buffer between threads: a slot is reserved
    and written by the same thread that owns the arrays, so the hot path needs
    no lock, no atomic, and no retry -- which is what keeps per-call overhead
    in the same order as the single-threaded path. The price is that a
    region's calls arrive as one buffer per thread and are concatenated once,
    at ``finalize()``.

    ``stack_slots``/``stack_tasks`` are the scope stack, split into two
    parallel lists so entering a scope appends two ints rather than
    allocating a tuple. They are parallel because a thread running an event
    loop interleaves several tasks, and their ``with`` blocks are LIFO *per
    task*, not per thread: a call is closed by popping the topmost entry
    belonging to the task that is exiting, which is the last one in the
    overwhelmingly common case of no interleaving at all.
    """

    __slots__ = (
        "await_ns",
        "capacity",
        "emitted",
        "end_times",
        "ptr",
        "record",
        "stack_slots",
        "stack_tasks",
        "start_times",
        "task_ids",
        "track_async",
    )

    def __init__(self, record, capacity: int, track_async: bool) -> None:
        self.record = record
        self.capacity = capacity
        self.ptr = 0
        self.start_times = np.empty(capacity, dtype=np.int64)
        self.end_times = np.zeros(capacity, dtype=np.int64)
        self.track_async = track_async
        if track_async:
            self.task_ids = np.full(capacity, NO_TASK, dtype=np.int64)
            self.await_ns = np.zeros(capacity, dtype=np.int64)
        else:
            self.task_ids = None
            self.await_ns = None
        self.stack_slots: list[int] = []
        self.stack_tasks: list[int] = []
        # Slots already handed to a finalize() that could not rewind this
        # buffer; see BaseProfileRegion.mark_written.
        self.emitted = None

    def grow(self) -> None:
        """Double this lane's buffers, keeping every reserved slot index."""
        capacity = max(1, self.capacity * 2)
        start_times = np.empty(capacity, dtype=np.int64)
        end_times = np.zeros(capacity, dtype=np.int64)
        start_times[: self.capacity] = self.start_times
        end_times[: self.capacity] = self.end_times
        self.start_times = start_times
        self.end_times = end_times
        if self.track_async:
            task_ids = np.full(capacity, NO_TASK, dtype=np.int64)
            await_ns = np.zeros(capacity, dtype=np.int64)
            task_ids[: self.capacity] = self.task_ids
            await_ns[: self.capacity] = self.await_ns
            self.task_ids = task_ids
            self.await_ns = await_ns
        self.capacity = capacity

    def open_slots(self) -> np.ndarray:
        """Slots whose call is still running, ascending."""
        if not self.ptr:
            return _EMPTY_TIMES
        return np.flatnonzero(self.end_times[: self.ptr] == _UNCLOSED)

    def closed_mask(self) -> "np.ndarray | None":
        """Mask of slots this finalize() should copy out, or None for all."""
        if not self.ptr:
            return None
        mask = self.end_times[: self.ptr] != _UNCLOSED
        if self.emitted is not None:
            mask[: self.emitted.size] &= ~self.emitted
        return None if mask.all() else mask

    def mark_written(self) -> int:
        """Rewind onto the still-open calls; return the number retired."""
        open_slots = self.open_slots()
        if open_slots.size == 0:
            retired = self.ptr
            if self.ptr:
                self.end_times[: self.ptr] = _UNCLOSED
            self.ptr = 0
            self.emitted = None
            return retired
        if open_slots.tolist() != sorted(self.stack_slots):
            # A decorator-form call holds its slot in its own frame, where
            # nothing can remap it; leave the buffer put and skip what has
            # already gone out. See BaseProfileRegion.mark_written.
            self.emitted = self.end_times[: self.ptr] != _UNCLOSED
            return 0
        retired = self.ptr - open_slots.size
        count = open_slots.size
        self.start_times[:count] = self.start_times[open_slots]
        if self.track_async:
            self.task_ids[:count] = self.task_ids[open_slots]
            self.await_ns[:count] = self.await_ns[open_slots]
        self.end_times[: self.ptr] = _UNCLOSED
        remapped = {int(old): new for new, old in enumerate(open_slots.tolist())}
        self.stack_slots[:] = [
            remapped[slot] if slot >= 0 else slot for slot in self.stack_slots
        ]
        self.ptr = count
        self.emitted = None
        return retired


class ThreadedProfileRegion(BaseProfileRegion):
    """Region that records which thread -- and which task -- each call ran on.

    Selected by ``track_threads=True``. Two things change against
    :class:`TimeOnlyProfileRegion`:

    * every thread gets its own buffers and its own scope stack, so
      concurrent calls no longer overwrite each other's reserved slots or
      close each other's scopes;
    * each recorded call carries the lane it ran on, so the nesting
      reconstruction can group calls into stacks that really are stacks (see
      :func:`~scope_profiler.concurrency.lane_ids`).

    With ``track_async=True`` a call additionally carries the id of the
    asyncio task or greenlet it ran in, and the time that task spent suspended
    *inside* the call -- the await time of that one call, as opposed to the
    task's total.

    The per-call cost over :class:`TimeOnlyProfileRegion` is one thread-local
    lookup plus, in async mode, two integer reads off the current task record.
    Nothing here allocates or locks per call.
    """

    __slots__ = (
        "_async",
        "_lane_local",
        "_lanes",
        "_lanes_lock",
        "_retired",
        "_tracker",
    )

    # The region's own buffers stay unallocated: every timestamp lives in a
    # lane buffer instead.
    _records_time = False

    def __init__(self, region_name: str, config: ProfilingConfig, tags=()):
        super().__init__(region_name, config, tags=tags)
        self._tracker = config.tracker
        self._async = config.track_async
        self._lanes: dict[int, _LaneBuffer] = {}
        self._lanes_lock = threading.Lock()
        self._lane_local = threading.local()
        # Calls already copied out by an earlier finalize(), summed over lanes.
        self._retired = 0

    def _new_lane(self) -> _LaneBuffer:
        """Create this thread's buffer for this region, once per thread."""
        record = self._tracker.current_thread()
        lane = _LaneBuffer(record, self.buffer_limit, self._async)
        with self._lanes_lock:
            self._lanes[record.index] = lane
        self._lane_local.lane = lane
        return lane

    @property
    def num_calls(self) -> int:
        """Times this region was entered, over every thread, for the process."""
        with self._lanes_lock:
            return self._retired + sum(lane.ptr for lane in self._lanes.values())

    @property
    def threads(self) -> list:
        """Thread records that have entered this region, in index order."""
        with self._lanes_lock:
            lanes = sorted(self._lanes.items())
        return [lane.record for _, lane in lanes]

    def _enter(self):
        try:
            lane = self._lane_local.lane
        except AttributeError:
            lane = self._new_lane()
        task = lane.record.task
        task_index = NO_TASK if task is None else task.index
        if self.config.paused:
            # A paused scope still has to be tracked, or its __exit__ would
            # close whichever call happens to be open on this lane. -1 is the
            # slot that means "recorded nothing".
            lane.stack_slots.append(-1)
            lane.stack_tasks.append(task_index)
            return
        slot = lane.ptr
        if slot >= lane.capacity:
            lane.grow()
        lane.ptr = slot + 1
        lane.stack_slots.append(slot)
        lane.stack_tasks.append(task_index)
        if self._async and task is not None:
            lane.task_ids[slot] = task_index
            # Negated base: __exit__ adds the task's counter back, leaving the
            # suspension that happened inside this call. One array, not two.
            lane.await_ns[slot] = -task.suspended_ns
        lane.start_times[slot] = perf_counter_ns()

    def _leave(self):
        end = perf_counter_ns()
        try:
            lane = self._lane_local.lane
        except AttributeError:  # __exit__ without a matching __enter__
            return
        slots = lane.stack_slots
        if not slots:
            return
        tasks = lane.stack_tasks
        task = lane.record.task
        task_index = NO_TASK if task is None else task.index
        if tasks[-1] == task_index:
            slot = slots.pop()
            tasks.pop()
        else:
            # Another task on this thread entered a scope of this region and
            # has not left it yet. Close ours, not theirs.
            position = len(tasks) - 1
            while position >= 0 and tasks[position] != task_index:
                position -= 1
            if position < 0:
                return
            slot = slots.pop(position)
            tasks.pop(position)
        if slot < 0:  # entered while paused
            return
        lane.end_times[slot] = end
        if self._async and task is not None:
            lane.await_ns[slot] += task.suspended_ns
        if not (slot & CPU_SAMPLE_MASK):
            # Cheap enough to do on the hot path, and the only way a thread
            # still alive at finalize() reports any CPU time at all.
            lane.record.sample_cpu()

    def wrap(self, func):
        """Wrap a function so its calls are recorded on the caller's thread."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.config.paused:
                return func(*args, **kwargs)
            self._enter()
            try:
                return func(*args, **kwargs)
            finally:
                self._leave()

        return wrapper

    # -- collection -----------------------------------------------------
    def snapshot_arrays(self):
        """This run's completed calls, concatenated over every thread.

        Returns
        -------
        tuple or None
            ``(start_times, end_times, None, None, None, thread_ids,
            task_ids, await_ns)`` in nanoseconds, laid out on the snapshot
            tuple :meth:`ProfileManager._snapshot_regions` documents, or None
            when this region recorded nothing since the last ``finalize()``.
            ``task_ids`` and ``await_ns`` are None unless async tracking is on.
        """
        with self._lanes_lock:
            lanes = sorted(self._lanes.items())
        starts, ends, threads, tasks, awaits = [], [], [], [], []
        for index, lane in lanes:
            if not lane.ptr:
                continue
            keep = lane.closed_mask()
            lane_starts = lane.start_times[: lane.ptr]
            lane_ends = lane.end_times[: lane.ptr]
            if keep is not None:
                lane_starts = lane_starts[keep]
                lane_ends = lane_ends[keep]
            if not lane_starts.size:
                continue
            starts.append(np.array(lane_starts))
            ends.append(np.array(lane_ends))
            threads.append(np.full(lane_starts.size, index, dtype=np.int64))
            if self._async:
                lane_tasks = lane.task_ids[: lane.ptr]
                lane_awaits = lane.await_ns[: lane.ptr]
                if keep is not None:
                    lane_tasks = lane_tasks[keep]
                    lane_awaits = lane_awaits[keep]
                tasks.append(np.array(lane_tasks))
                awaits.append(np.array(lane_awaits))
        if not starts:
            return None
        return (
            np.concatenate(starts),
            np.concatenate(ends),
            None,
            None,
            None,
            np.concatenate(threads),
            np.concatenate(tasks) if tasks else None,
            np.concatenate(awaits) if awaits else None,
        )

    def open_slots(self) -> np.ndarray:
        """Slots still running, over every lane. For diagnostics only."""
        with self._lanes_lock:
            lanes = list(self._lanes.values())
        return np.concatenate([lane.open_slots() for lane in lanes] or [_EMPTY_TIMES])

    def closed_slots(self):
        """Not meaningful per lane; :meth:`snapshot_arrays` applies the mask.

        None, as the base class means it: every slot qualifies, and there is
        no mask for a caller to apply. A threaded region keeps no buffers of
        its own, so there is nothing here to mask in the first place.
        """
        return

    def mark_written(self) -> None:
        """Rewind every lane, keeping calls that are still running."""
        with self._lanes_lock:
            lanes = list(self._lanes.values())
        for lane in lanes:
            self._retired += lane.mark_written()

    def __enter__(self):
        self._enter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._leave()
