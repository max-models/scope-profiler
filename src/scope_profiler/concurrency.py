"""Thread, asyncio-task and greenlet identity for the region profiler.

A region records a ``(start, end)`` pair and nothing else, which is enough
only while one call stack owns the process. With several threads -- or several
coroutines interleaved on one thread -- the recorded intervals of two lanes
overlap without nesting, and the whole downstream stack (flame chart,
exclusive time, ``.prof`` export) has no call graph to reconstruct. See
:class:`~scope_profiler.call_stack.NestingError`.

This module supplies the missing axis: every recorded call is stamped with the
*lane* it ran on, and the lanes themselves are described in two small tables
that travel with the run.

**Threads.** :class:`ThreadRecord` carries a dense per-rank index, the OS
identifiers, the thread's name, when it started and ended, and the CPU time it
burned. The index is what the event column stores; everything else is looked
up once, per thread, not per call.

**Tasks.** An asyncio task is a lane of its own: it runs on a thread, but so
do the other tasks that interleave with it, and a ``with`` block held across
an ``await`` covers wall time the task did not spend running. Instrumenting
the task -- rather than the loop -- gives both the identity and the split:
:class:`TaskRecord` accumulates running and suspended nanoseconds, and a
region reads the suspended counter at entry and exit to report the await time
of that one call. Greenlets are the same shape (cooperative lanes on one
thread) and share :class:`TaskRecord`, tagged ``kind="greenlet"``.

Nothing here is installed unless ``track_threads`` is set: the profiler's
default single-lane path is untouched, hot loop included.
"""

from __future__ import annotations

import os
import sys
import threading
import weakref
from time import perf_counter_ns, thread_time_ns
from typing import Any

import numpy as np

#: Task index of a call that ran outside any task or greenlet.
NO_TASK = -1

#: ``end_ns``/``cpu_ns`` of a thread that was still running at ``finalize()``,
#: and ``done_ns`` of a task that had not finished. Negative because every
#: real value on these clocks is positive, so a reader needs no mask column.
UNKNOWN = -1

#: How often a still-running thread refreshes its CPU-time reading, in
#: recorded calls. A thread's exact total is taken when it dies (see
#: :class:`_ThreadExitSentinel`); this is what makes the number roughly right
#: for threads -- the main one, above all -- that are still alive at
#: ``finalize()``. A power of two so the hot path is one mask and a branch.
CPU_SAMPLE_MASK = 0xFF


class ThreadRecord:
    """One thread of the profiled process, as the lane tables describe it.

    ``index`` is dense within a rank and is what the per-call ``thread_ids``
    column stores. ``cpu_ns`` is exact for a thread that has ended and a
    sampled lower bound for one still running; see :data:`CPU_SAMPLE_MASK`.
    """

    __slots__ = (
        "cpu_ns",
        "daemon",
        "end_ns",
        "ident",
        "index",
        "name",
        "native_id",
        "start_ns",
        "task",
    )

    def __init__(self, index: int, thread) -> None:
        self.index = index
        self.ident = int(thread.ident or 0)
        self.native_id = int(getattr(thread, "native_id", 0) or 0)
        self.name = str(thread.name)
        self.daemon = bool(thread.daemon)
        self.start_ns = perf_counter_ns()
        self.end_ns = UNKNOWN
        self.cpu_ns = thread_time_ns()
        # The task or greenlet currently running on this thread, maintained by
        # the step timer and the greenlet trace. Read on the region hot path,
        # which is why it lives on the thread record rather than behind a
        # ``asyncio.current_task()`` call.
        self.task: TaskRecord | None = None

    @property
    def alive(self) -> bool:
        """Whether the thread was still running when the run was collected."""
        return self.end_ns == UNKNOWN

    def sample_cpu(self) -> None:
        """Refresh this thread's CPU time. Only valid on the thread itself."""
        self.cpu_ns = thread_time_ns()

    def __repr__(self) -> str:
        state = "alive" if self.alive else "finished"
        return (
            f"ThreadRecord(index={self.index}, name={self.name!r}, "
            f"ident={self.ident}, {state})"
        )


class TaskRecord:
    """One asyncio task or greenlet, and how its wall time split.

    ``running_ns`` is the time the lane actually held its thread, summed over
    every step; ``suspended_ns`` is the time between steps -- awaiting, or
    switched away from. A region entered inside the lane reads
    ``suspended_ns`` on the way in and out, and the difference is that one
    call's await time.
    """

    __slots__ = (
        "coro_name",
        "created_ns",
        "done_ns",
        "index",
        "kind",
        "last_step_end_ns",
        "name",
        "running_ns",
        "step_started_ns",
        "steps",
        "suspended_ns",
        "thread_index",
    )

    def __init__(
        self,
        index: int,
        kind: str,
        name: str,
        coro_name: str,
        thread_index: int,
    ) -> None:
        self.index = index
        self.kind = kind
        self.name = name
        self.coro_name = coro_name
        self.thread_index = thread_index
        self.created_ns = perf_counter_ns()
        self.done_ns = UNKNOWN
        self.steps = 0
        self.running_ns = 0
        self.suspended_ns = 0
        # perf_counter_ns of the end of the last step, i.e. the moment this
        # lane was suspended. 0 until it has run at all, so the gap before the
        # first step counts as scheduling latency rather than await time.
        self.last_step_end_ns = 0
        # perf_counter_ns of the start of the step currently running.
        self.step_started_ns = 0

    @property
    def done(self) -> bool:
        """Whether the task finished before the run was collected."""
        return self.done_ns != UNKNOWN

    def enter_step(self) -> int:
        """Charge the gap since the last step as suspension; return the clock."""
        now = perf_counter_ns()
        if self.last_step_end_ns:
            self.suspended_ns += now - self.last_step_end_ns
        self.steps += 1
        self.step_started_ns = now
        return now

    def leave_step(self) -> None:
        """Charge the step that is ending as running time, and suspend."""
        end = perf_counter_ns()
        self.running_ns += end - self.step_started_ns
        self.last_step_end_ns = end

    def finish(self) -> None:
        """Record that the lane completed (first call wins)."""
        if self.done_ns == UNKNOWN:
            self.done_ns = perf_counter_ns()

    def __repr__(self) -> str:
        return (
            f"TaskRecord(index={self.index}, kind={self.kind!r}, "
            f"name={self.name!r}, steps={self.steps})"
        )


class _ThreadExitSentinel:
    """Records a thread's end time and total CPU time, from the thread itself.

    Parked in the tracker's :class:`threading.local`, so CPython drops it when
    the thread's state is cleared -- on that thread, at the moment it exits.
    That is the only place ``thread_time_ns()`` can be read for a thread other
    than the one calling ``finalize()``, and it is exact rather than sampled.

    The clocks are bound as attributes because ``__del__`` can also run during
    interpreter shutdown, when module globals may already be gone.
    """

    __slots__ = ("_clock", "_cpu_clock", "_record")

    def __init__(self, record: ThreadRecord) -> None:
        self._record = record
        self._clock = perf_counter_ns
        self._cpu_clock = thread_time_ns

    def __del__(self) -> None:
        try:
            record = self._record
            record.cpu_ns = self._cpu_clock()
            record.end_ns = self._clock()
        except Exception:  # pragma: no cover - interpreter teardown only
            pass


class _StepTimer:
    """The iterator a wrapped coroutine's ``__await__`` hands to the task.

    Every ``send``/``throw`` the event loop performs on the task lands here
    first, so the boundaries of each step are exactly the boundaries of the
    task holding its thread. Delegating rather than subclassing ``Task`` keeps
    this working with the C implementation of ``asyncio.Task``, which never
    calls a Python-level override.

    The thread is resolved per step rather than at task creation: a task can
    be created on one thread and run on the loop's, and it is the thread that
    actually runs a step whose ``task`` pointer the region hot path reads.
    """

    __slots__ = ("_coro", "_record", "_tracker")

    def __init__(self, coro, record: TaskRecord, tracker) -> None:
        self._coro = coro
        self._record = record
        self._tracker = tracker

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        record = self._record
        thread = self._tracker.current_thread()
        record.thread_index = thread.index
        previous = thread.task
        thread.task = record
        record.enter_step()
        try:
            return self._coro.send(value)
        finally:
            record.leave_step()
            thread.task = previous

    def throw(self, *args):
        record = self._record
        thread = self._tracker.current_thread()
        record.thread_index = thread.index
        previous = thread.task
        thread.task = record
        record.enter_step()
        try:
            return self._coro.throw(*args)
        finally:
            record.leave_step()
            thread.task = previous

    def close(self):
        return self._coro.close()


class _TimedAwaitable:
    """Adapter whose ``__await__`` is the step timer."""

    __slots__ = ("_coro", "_record", "_tracker")

    def __init__(self, coro, record: TaskRecord, tracker) -> None:
        self._coro = coro
        self._record = record
        self._tracker = tracker

    def __await__(self):
        return _StepTimer(self._coro, self._record, self._tracker)


async def _timed_coroutine(coro, record: TaskRecord, tracker):
    """Wrap ``coro`` in a real coroutine whose steps are timed.

    ``asyncio.Task`` insists on a genuine coroutine, so the timing cannot live
    in a bare adapter object; one ``await`` of :class:`_TimedAwaitable` gives
    both -- a coroutine for the task and a per-step hook for us. The extra
    frame is entered once per step and does no work of its own.
    """
    return await _TimedAwaitable(coro, record, tracker)


def _coroutine_name(coro) -> str:
    """Best available name for whatever a task was created from."""
    for attribute in ("__qualname__", "__name__"):
        name = getattr(coro, attribute, None)
        if name:
            return str(name)
    code = getattr(coro, "cr_code", None) or getattr(coro, "gi_code", None)
    if code is not None:
        return str(getattr(code, "co_qualname", None) or code.co_name)
    return type(coro).__name__


class ConcurrencyTracker:
    """Per-run registry of threads, asyncio tasks and greenlets.

    One tracker belongs to one :class:`~scope_profiler.profile_config.ProfilingConfig`,
    so two simultaneously active managers never share lane numbering. Creating
    it is free; :meth:`install` is what puts the (few, cheap) hooks in place,
    and :meth:`uninstall` takes them out again.
    """

    def __init__(self, track_async: bool = False) -> None:
        self.track_async = bool(track_async)
        # Guards the registries only. The hot path never takes it: a thread
        # locks once, when it first records anything, and reads its record
        # from thread-local storage forever after.
        self._lock = threading.Lock()
        self._threads: list[ThreadRecord] = []
        self._tasks: list[TaskRecord] = []
        self._local = threading.local()
        self._installed = False
        self._previous_thread_hook: Any = None
        self._previous_run_forever: Any = None
        self._previous_create_task: Any = None
        self._previous_create_task = None
        self._loop_factories: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._greenlet_records: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._greenlet_module = None
        self._previous_greenlet_trace = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def current_thread(self) -> ThreadRecord:
        """This thread's record, registering it on first use.

        On the region hot path. The common case is one failed-then-cached
        attribute lookup on a :class:`threading.local`, which is why the
        registry itself is never consulted here.
        """
        try:
            return self._local.record
        except AttributeError:
            return self._register_thread()

    def _register_thread(self) -> ThreadRecord:
        """Give the calling thread a record, and arrange to close it on exit."""
        thread = threading.current_thread()
        with self._lock:
            record = ThreadRecord(len(self._threads), thread)
            self._threads.append(record)
        self._local.record = record
        # Dropped by the interpreter on this thread, at thread exit; that is
        # where the exact end time and CPU total come from.
        self._local.sentinel = _ThreadExitSentinel(record)
        return record

    @property
    def threads(self) -> list[ThreadRecord]:
        """Every thread that recorded a call, in registration order."""
        with self._lock:
            return list(self._threads)

    @property
    def tasks(self) -> list[TaskRecord]:
        """Every instrumented task and greenlet, in creation order."""
        with self._lock:
            return list(self._tasks)

    def _new_task(self, kind: str, name: str, coro_name: str) -> TaskRecord:
        thread = self.current_thread()
        with self._lock:
            record = TaskRecord(len(self._tasks), kind, name, coro_name, thread.index)
            self._tasks.append(record)
        return record

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------
    def install(self) -> None:
        """Put the thread-start, asyncio and greenlet hooks in place.

        Idempotent, so a second ``setup()`` on the same configuration does not
        stack wrappers.
        """
        if self._installed:
            return
        self._installed = True
        _LIVE_TRACKERS.add(self)
        self._install_thread_hook()
        if self.track_async:
            self._install_asyncio()
            self._install_greenlet()

    def uninstall(self) -> None:
        """Remove every hook and restore what was there before."""
        if not self._installed:
            return
        self._installed = False
        _LIVE_TRACKERS.discard(self)
        self._uninstall_thread_hook()
        self._uninstall_asyncio()
        self._uninstall_greenlet()

    def _adopt_fork(self) -> None:
        """Stand down in a process forked out of someone else's session.

        The child inherited this tracker, its hooks and its registries, but
        not the session that opened them: nothing in the child will ever
        finalize this run, so left alone the hooks would go on appending a
        record per thread and per task to a table no one reads, for as long
        as the child lives. A forked worker running an event loop is exactly
        that shape.

        The child therefore starts untracked, and profiles concurrency only
        once it calls ``setup()`` for itself -- which is the supported way to
        profile a multiprocessing worker in any case, since each process
        writes its own output file.

        The registries are replaced rather than cleared: they describe the
        parent's threads, which do not exist here, and the lock guarding them
        may have been held by a thread that did not survive the fork.
        """
        self._lock = threading.Lock()
        self._local = threading.local()
        self._threads = []
        self._tasks = []
        self._greenlet_records = weakref.WeakKeyDictionary()
        self.uninstall()

    # -- threads --------------------------------------------------------
    def _install_thread_hook(self) -> None:
        """Register threads at their first bytecode, not their first region.

        ``threading.setprofile`` arms a profile function in every thread
        started from here on. Ours runs once -- on the bootstrap frame's call
        event -- registers the thread with an accurate start time, and then
        takes itself out with ``sys.setprofile(None)``, so no profiling
        overhead survives into the thread's actual work.
        """
        self._previous_thread_hook = getattr(threading, "_profile_hook", None)
        threading.setprofile(self._thread_bootstrap)

    def _thread_bootstrap(self, frame, event, arg):
        sys.setprofile(None)
        self.current_thread()
        previous = self._previous_thread_hook
        if previous is not None:
            sys.setprofile(previous)
            return previous(frame, event, arg)
        return None

    def _uninstall_thread_hook(self) -> None:
        threading.setprofile(self._previous_thread_hook)
        self._previous_thread_hook = None

    # -- asyncio --------------------------------------------------------
    def _install_asyncio(self) -> None:
        """Instrument every event loop this process runs from now on.

        ``BaseEventLoop.run_forever`` is the single funnel every way of
        running a loop goes through -- ``asyncio.run``, ``run_until_complete``
        and a bare ``run_forever`` alike -- so wrapping it reaches loops
        created long after ``setup()``. A loop already running (``setup()``
        called from inside a coroutine) is instrumented directly.
        """
        import asyncio

        tracker = self
        previous = asyncio.BaseEventLoop.run_forever
        self._previous_run_forever = previous

        def run_forever(loop):
            tracker.instrument_loop(loop)
            return previous(loop)

        run_forever.__doc__ = previous.__doc__
        asyncio.BaseEventLoop.run_forever = run_forever  # type: ignore[method-assign]

        # asyncio.run() creates the main task before it starts the loop, so
        # run_forever alone would miss exactly the task the program is about.
        previous_create = asyncio.BaseEventLoop.create_task
        self._previous_create_task = previous_create

        def create_task(loop, coro, **kwargs):
            tracker.instrument_loop(loop)
            return previous_create(loop, coro, **kwargs)

        create_task.__doc__ = previous_create.__doc__
        asyncio.BaseEventLoop.create_task = create_task  # type: ignore[method-assign,assignment]

        try:
            self.instrument_loop(asyncio.get_running_loop())
        except RuntimeError:
            pass

    def instrument_loop(self, loop) -> None:
        """Route ``loop``'s task creation through the step timer.

        Public because a loop implementation that never reaches
        ``BaseEventLoop.run_forever`` (uvloop, say) can be handed here
        directly. Chains onto whatever task factory the application already
        installed instead of replacing it.

        Instrumented once per loop, and keyed on the loop rather than on
        whichever factory is currently in place: an application that installs
        its own factory *after* the loop has started displaces ours, and
        re-wrapping to get back on top would make the two call each other
        forever, since its captured "previous" is us. Such a loop keeps
        recording regions and thread ids; the tasks created after the swap
        simply have no lane of their own.
        """
        if loop in self._loop_factories:
            return
        self._loop_factories[loop] = loop.get_task_factory()
        loop.set_task_factory(self._task_factory)

    def _task_factory(self, loop, coro, **kwargs):
        import asyncio

        record = self._new_task("task", "", _coroutine_name(coro))
        wrapped = _timed_coroutine(coro, record, self)
        previous = self._loop_factories.get(loop)
        if previous is not None:
            task = previous(loop, wrapped, **kwargs)
        else:
            task = asyncio.Task(wrapped, loop=loop, **kwargs)
        record.name = task.get_name()
        task.add_done_callback(lambda _task, _record=record: _record.finish())
        return task

    def _uninstall_asyncio(self) -> None:
        if self._previous_run_forever is None:
            return
        import asyncio

        asyncio.BaseEventLoop.run_forever = self._previous_run_forever  # type: ignore[method-assign]
        self._previous_run_forever = None
        if self._previous_create_task is not None:
            asyncio.BaseEventLoop.create_task = (  # type: ignore[method-assign]
                self._previous_create_task
            )
            self._previous_create_task = None
        for loop, factory in list(self._loop_factories.items()):
            try:
                if loop.get_task_factory() is self._task_factory:
                    loop.set_task_factory(factory)
            except RuntimeError:  # pragma: no cover - loop already closed
                pass
        self._loop_factories = weakref.WeakKeyDictionary()

    # -- greenlets ------------------------------------------------------
    def _install_greenlet(self) -> None:
        """Follow greenlet switches, when greenlet is installed at all.

        ``greenlet.settrace`` reports every switch, which is exactly the
        boundary between one cooperative lane holding the thread and the next
        -- the same split the asyncio step timer measures.
        """
        try:
            import greenlet
        except ImportError:
            return
        self._greenlet_module = greenlet
        self._previous_greenlet_trace = greenlet.settrace(self._greenlet_trace)

    def _greenlet_record(self, greenlet_object) -> TaskRecord:
        record = self._greenlet_records.get(greenlet_object)
        if record is None:
            name = getattr(greenlet_object, "name", None) or (
                f"greenlet-{id(greenlet_object):x}"
            )
            target = getattr(greenlet_object, "run", None)
            record = self._new_task("greenlet", str(name), _coroutine_name(target))
            self._greenlet_records[greenlet_object] = record
        return record

    def _greenlet_trace(self, event, args):
        if event in ("switch", "throw"):
            origin, target = args
            thread = self.current_thread()
            running = thread.task
            if running is not None:
                running.leave_step()
            if getattr(origin, "dead", False):
                self._greenlet_record(origin).finish()
            record = self._greenlet_record(target)
            record.enter_step()
            thread.task = record
        previous = self._previous_greenlet_trace
        if previous is not None:
            previous(event, args)

    def _uninstall_greenlet(self) -> None:
        if self._greenlet_module is None:
            return
        self._greenlet_module.settrace(self._previous_greenlet_trace)
        self._greenlet_module = None
        self._previous_greenlet_trace = None
        self._greenlet_records = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def snapshot(self, origin_ns: int = 0) -> dict:
        """The lane tables, as columns, ready to be written or returned.

        Timestamps are shifted by ``origin_ns`` (the run's start), matching
        what the event columns store, so a thread's lifetime and the calls on
        it share one timeline. :data:`UNKNOWN` entries are left alone.

        Returns
        -------
        dict
            ``{"threads": {...}, "tasks": {...}}``, each a dict of equal-length
            numpy arrays. Empty arrays when nothing was tracked.
        """
        # The calling thread's own CPU reading is only refreshed by the
        # sampling in the region hot path, and finalize() usually runs a long
        # way past the last recorded call.
        try:
            self._local.record.sample_cpu()
        except AttributeError:
            pass

        def shift(values):
            values = np.asarray(values, dtype=np.int64)
            return np.where(values == UNKNOWN, UNKNOWN, values - origin_ns)

        threads = self.threads
        tasks = self.tasks
        return {
            "threads": {
                "index": np.array([t.index for t in threads], dtype=np.int64),
                "ident": np.array([t.ident for t in threads], dtype=np.int64),
                "native_id": np.array([t.native_id for t in threads], dtype=np.int64),
                "name": [t.name for t in threads],
                "daemon": np.array([t.daemon for t in threads], dtype=np.int8),
                "start_ns": shift([t.start_ns for t in threads]),
                "end_ns": shift([t.end_ns for t in threads]),
                "cpu_ns": np.array([t.cpu_ns for t in threads], dtype=np.int64),
            },
            "tasks": {
                "index": np.array([t.index for t in tasks], dtype=np.int64),
                "kind": [t.kind for t in tasks],
                "name": [t.name for t in tasks],
                "coro_name": [t.coro_name for t in tasks],
                "thread_index": np.array(
                    [t.thread_index for t in tasks], dtype=np.int64
                ),
                "created_ns": shift([t.created_ns for t in tasks]),
                "done_ns": shift([t.done_ns for t in tasks]),
                "steps": np.array([t.steps for t in tasks], dtype=np.int64),
                "running_ns": np.array([t.running_ns for t in tasks], dtype=np.int64),
                "suspended_ns": np.array(
                    [t.suspended_ns for t in tasks], dtype=np.int64
                ),
            },
        }


# Trackers with hooks currently installed, so a fork can find them from the
# one handler this module is allowed to register (``register_at_fork`` has no
# matching unregister, so it must not be called per tracker). Weak, so a
# tracker that is dropped without being uninstalled does not leak.
_LIVE_TRACKERS: weakref.WeakSet = weakref.WeakSet()


def _stand_down_after_fork() -> None:
    """Detach every installed tracker in a freshly forked child."""
    for tracker in list(_LIVE_TRACKERS):
        tracker._adopt_fork()


if hasattr(os, "register_at_fork"):  # POSIX only; there is no fork elsewhere
    os.register_at_fork(after_in_child=_stand_down_after_fork)


def lane_ids(thread_ids: np.ndarray, task_ids: np.ndarray | None) -> np.ndarray:
    """One id per call identifying the stack it belongs to.

    A call belongs to its task when it ran inside one and to its bare thread
    otherwise. The two id spaces are disjoint by construction: thread lanes
    are mapped to ``-2 - thread``, task lanes stay non-negative, and ``-1`` is
    left for a call whose thread is unknown -- a region recorded by the
    Fortran API, say, folded into an otherwise thread-aware run. This is the
    key :func:`~scope_profiler.call_stack.build_call_arrays` groups by, since
    only calls sharing a lane can nest.
    """
    thread_ids = np.asarray(thread_ids, dtype=np.int64)
    lanes = np.where(thread_ids < 0, -1, -2 - thread_ids)
    if task_ids is None:
        return lanes
    task_ids = np.asarray(task_ids, dtype=np.int64)
    return np.where(task_ids >= 0, task_ids, lanes)


class ThreadInfo:
    """One row of a run's thread table, as post-processing sees it.

    The read-back counterpart of :class:`ThreadRecord`: same fields, but
    durations in seconds and timestamps relative to the start of the run,
    matching every other time reported by
    :class:`~scope_profiler.results.ProfilingResults`.
    """

    __slots__ = (
        "cpu_time",
        "daemon",
        "end_time",
        "ident",
        "index",
        "name",
        "native_id",
        "rank",
        "start_time",
    )

    def __init__(
        self,
        index: int,
        rank: int,
        name: str,
        ident: int,
        native_id: int,
        daemon: bool,
        start_ns: int,
        end_ns: int,
        cpu_ns: int,
    ) -> None:
        self.index = int(index)
        self.rank = int(rank)
        self.name = str(name)
        self.ident = int(ident)
        self.native_id = int(native_id)
        self.daemon = bool(daemon)
        self.start_time = float(start_ns) / 1e9
        self.end_time = None if int(end_ns) == UNKNOWN else float(end_ns) / 1e9
        self.cpu_time = float(cpu_ns) / 1e9

    @property
    def alive(self) -> bool:
        """Whether the thread was still running when the run was collected."""
        return self.end_time is None

    @property
    def wall_time(self) -> float | None:
        """Seconds between the thread starting and ending, or None if alive."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (
            f"ThreadInfo(rank={self.rank}, index={self.index}, name={self.name!r}, "
            f"cpu_time={self.cpu_time:.6f}s, alive={self.alive})"
        )


class TaskInfo:
    """One row of a run's task table: an asyncio task or a greenlet.

    ``running_time`` and ``awaiting_time`` split the lane's wall time into the
    part it held its thread and the part it did not, which is the number
    ``await``-heavy code is usually looking for.
    """

    __slots__ = (
        "awaiting_time",
        "coro_name",
        "created_time",
        "done_time",
        "index",
        "kind",
        "name",
        "rank",
        "running_time",
        "steps",
        "thread_index",
    )

    def __init__(
        self,
        index: int,
        rank: int,
        kind: str,
        name: str,
        coro_name: str,
        thread_index: int,
        created_ns: int,
        done_ns: int,
        steps: int,
        running_ns: int,
        suspended_ns: int,
    ) -> None:
        self.index = int(index)
        self.rank = int(rank)
        self.kind = str(kind)
        self.name = str(name)
        self.coro_name = str(coro_name)
        self.thread_index = int(thread_index)
        self.created_time = float(created_ns) / 1e9
        self.done_time = None if int(done_ns) == UNKNOWN else float(done_ns) / 1e9
        self.steps = int(steps)
        self.running_time = float(running_ns) / 1e9
        self.awaiting_time = float(suspended_ns) / 1e9

    @property
    def done(self) -> bool:
        """Whether the task finished before the run was collected."""
        return self.done_time is not None

    @property
    def wall_time(self) -> float | None:
        """Seconds from creation to completion, or None if it never finished."""
        if self.done_time is None:
            return None
        return self.done_time - self.created_time

    def __repr__(self) -> str:
        return (
            f"TaskInfo(rank={self.rank}, index={self.index}, kind={self.kind!r}, "
            f"name={self.name!r}, running_time={self.running_time:.6f}s, "
            f"awaiting_time={self.awaiting_time:.6f}s)"
        )


def lane_tables_from_columns(rank: int, tables: dict) -> tuple[list, list]:
    """Turn one rank's stored lane columns into :class:`ThreadInfo`/:class:`TaskInfo`.

    The one place the column layout written to HDF5 and returned in memory is
    turned back into objects, so the file and ``finalize(return_results=True)``
    cannot disagree about what a run's lanes were.
    """
    threads = tables.get("threads") or {}
    tasks = tables.get("tasks") or {}

    def text(values, position):
        value = values[position]
        return value.decode() if isinstance(value, bytes) else str(value)

    thread_infos = [
        ThreadInfo(
            threads["index"][row],
            rank,
            text(threads["name"], row),
            threads["ident"][row],
            threads["native_id"][row],
            bool(threads["daemon"][row]),
            threads["start_ns"][row],
            threads["end_ns"][row],
            threads["cpu_ns"][row],
        )
        for row in range(len(threads.get("index", ())))
    ]
    task_infos = [
        TaskInfo(
            tasks["index"][row],
            rank,
            text(tasks["kind"], row),
            text(tasks["name"], row),
            text(tasks["coro_name"], row),
            tasks["thread_index"][row],
            tasks["created_ns"][row],
            tasks["done_ns"][row],
            tasks["steps"][row],
            tasks["running_ns"][row],
            tasks["suspended_ns"][row],
        )
        for row in range(len(tasks.get("index", ())))
    ]
    return thread_infos, task_infos
