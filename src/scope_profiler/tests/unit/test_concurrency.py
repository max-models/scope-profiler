"""The lane bookkeeping behind ``track_threads`` and ``track_async``."""

import asyncio
import threading

import numpy as np
import pytest

from scope_profiler.concurrency import (
    UNKNOWN,
    ConcurrencyTracker,
    TaskInfo,
    TaskRecord,
    ThreadInfo,
    ThreadRecord,
    _coroutine_name,
    _StepTimer,
    _ThreadExitSentinel,
    lane_ids,
    lane_tables_from_columns,
)


def test_thread_record_describes_the_calling_thread():
    record = ThreadRecord(3, threading.current_thread())

    assert record.index == 3
    assert record.name == threading.current_thread().name
    assert record.ident == threading.get_ident()
    assert record.alive
    assert record.cpu_ns > 0
    assert "finished" not in repr(record)

    before = record.cpu_ns
    for _ in range(200000):
        pass
    record.sample_cpu()
    assert record.cpu_ns >= before


def test_the_exit_sentinel_closes_a_thread_record():
    record = ThreadRecord(0, threading.current_thread())
    sentinel = _ThreadExitSentinel(record)

    assert record.alive
    sentinel.__del__()
    assert not record.alive
    assert record.end_ns > 0


def test_task_record_accumulates_running_and_suspended_time():
    record = TaskRecord(0, "task", "Task-1", "job", thread_index=2)

    assert not record.done
    assert record.done_ns == UNKNOWN
    record.enter_step()
    record.leave_step()
    first_run = record.running_ns
    assert first_run > 0
    # The gap before the second step is suspension, not running time.
    record.enter_step()
    record.leave_step()
    assert record.suspended_ns > 0
    assert record.running_ns > first_run
    assert record.steps == 2

    record.finish()
    finished_at = record.done_ns
    record.finish()  # first call wins
    assert record.done_ns == finished_at
    assert record.done
    assert "TaskRecord" in repr(record)


def test_step_timer_drives_the_coroutine_and_times_each_step():
    tracker = ConcurrencyTracker()
    record = TaskRecord(0, "task", "", "", thread_index=0)

    async def job():
        return 42

    timer = _StepTimer(job(), record, tracker)
    assert iter(timer) is timer
    with pytest.raises(StopIteration) as stopped:
        next(timer)
    assert stopped.value.value == 42
    assert record.steps == 1
    assert record.running_ns > 0


def test_step_timer_forwards_throw_and_close():
    tracker = ConcurrencyTracker()
    record = TaskRecord(0, "task", "", "", thread_index=0)

    async def job():
        await asyncio.sleep(0)

    coro = job()
    timer = _StepTimer(coro, record, tracker)
    next(timer)  # start it, so it is suspended at the sleep
    with pytest.raises(ValueError):
        timer.throw(ValueError("stop"))
    assert record.steps == 2

    other = job()
    timer = _StepTimer(other, record, tracker)
    next(timer)
    timer.close()
    assert other.cr_running is False


def test_coroutine_name_falls_back_to_the_code_object():
    async def job():
        pass

    coro = job()
    try:
        assert _coroutine_name(coro).endswith("job")

        class Bare:
            """A coroutine-like object with only a code object to go on."""

            cr_code = coro.cr_code

        assert _coroutine_name(Bare()).endswith("job")
        assert _coroutine_name(object()) == "object"
    finally:
        coro.close()


def test_lane_ids_never_confuses_a_thread_with_a_task():
    threads = np.array([0, 1, 5, -1])
    lanes = lane_ids(threads, None)
    assert lanes.tolist() == [-2, -3, -7, -1]
    # Every thread lane is distinct from every task id, which start at 0.
    assert lanes.max() < 0

    tasks = np.array([0, -1, 3, -1])
    assert lane_ids(threads, tasks).tolist() == [0, -3, 3, -1]


def test_snapshot_shifts_timestamps_but_leaves_unknowns_alone():
    tracker = ConcurrencyTracker()
    record = tracker.current_thread()
    origin = record.start_ns - 1000

    tables = tracker.snapshot(origin_ns=origin)

    assert tables["threads"]["start_ns"].tolist() == [record.start_ns - origin]
    # Still running, so its end stays the sentinel rather than being shifted.
    assert tables["threads"]["end_ns"].tolist() == [UNKNOWN]
    assert tables["tasks"]["index"].size == 0


def test_snapshot_round_trips_through_the_shared_column_layout():
    tracker = ConcurrencyTracker(track_async=True)
    tracker.current_thread()
    record = tracker._new_task("greenlet", "worker", "run")
    record.enter_step()
    record.leave_step()
    record.finish()

    threads, tasks = lane_tables_from_columns(2, tracker.snapshot())

    assert [thread.rank for thread in threads] == [2]
    assert threads[0].name == threading.current_thread().name
    assert threads[0].wall_time is None
    (task,) = tasks
    assert task.done
    assert task.kind == "greenlet"
    assert task.name == "worker"
    assert task.coro_name == "run"
    assert task.steps == 1
    assert task.running_time > 0
    assert task.wall_time > 0


def test_lane_tables_decode_bytes_from_hdf5():
    """h5py hands string columns back as bytes; the tables must cope."""
    columns = {
        "threads": {
            "index": np.array([0]),
            "ident": np.array([1234]),
            "native_id": np.array([99]),
            "daemon": np.array([1]),
            "name": np.array([b"worker-0"]),
            "start_ns": np.array([0]),
            "end_ns": np.array([2_000_000_000]),
            "cpu_ns": np.array([1_000_000_000]),
        }
    }
    (thread,) = lane_tables_from_columns(0, columns)[0]

    assert thread.name == "worker-0"
    assert thread.daemon is True
    assert not thread.alive
    assert thread.wall_time == pytest.approx(2.0)
    assert thread.cpu_time == pytest.approx(1.0)
    assert "ThreadInfo" in repr(thread)


def test_task_info_reports_an_unfinished_task():
    info = TaskInfo(0, 0, "task", "Task-1", "job", 0, 0, UNKNOWN, 3, 10, 20)

    assert not info.done
    assert info.done_time is None
    assert info.wall_time is None


def test_thread_info_reports_an_unknown_end_as_alive():
    info = ThreadInfo(0, 0, "MainThread", 1, 2, False, 0, UNKNOWN, 5)

    assert info.alive
    assert info.end_time is None
    assert info.wall_time is None


def test_uninstall_is_safe_without_greenlet_or_asyncio():
    tracker = ConcurrencyTracker()  # track_async=False: no asyncio hooks

    tracker.install()
    tracker.uninstall()
    tracker.uninstall()

    assert tracker.threads == []
