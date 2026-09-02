"""Asyncio tasks, greenlets, and the await time a region hides.

Coroutines interleave on one thread, so a ``with`` block held across an
``await`` overlaps the blocks of every other task that runs meanwhile, and its
duration counts wall time the task never held the interpreter. ``track_async``
gives each task a lane of its own and splits that duration; these tests pin
both.
"""

import asyncio
import time

import numpy as np
import pytest

from scope_profiler import ProfileManager, read_h5
from scope_profiler.call_stack import build_call_arrays
from scope_profiler.concurrency import ConcurrencyTracker

SPIN_SECONDS = 0.01


def tasks_named(results, name, rank=0):
    """Task rows whose coroutine is ``name``.

    Matched on the tail of the qualified name: a coroutine defined inside a
    test function is recorded as ``test_x.<locals>.job``.
    """
    return [
        task
        for task in results.tasks[rank]
        if task.coro_name.rsplit(".", 1)[-1] == name
    ]


def spin(seconds=SPIN_SECONDS):
    """Burn ``seconds`` of wall time without yielding to the event loop."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def test_each_task_gets_its_own_lane(tmp_path):
    manager = ProfileManager()

    async def job():
        with manager.profile_region("job"):
            await asyncio.sleep(0.02)
            spin()

    async def main():
        await asyncio.gather(job(), job(), job())

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "tasks.h5"),
    ) as run:
        asyncio.run(main())

    job_region = run.results["job"][0]
    assert job_region.num_calls == 3
    # One thread, three distinct tasks.
    assert set(job_region.thread_ids.tolist()) == {0}
    assert len(set(job_region.task_ids.tolist())) == 3
    arrays = build_call_arrays(run.results.get_regions(), rank=0)
    job_rows = arrays.region_index == arrays.names.index("job")
    # Overlapping in time, yet each is a root of its own lane rather than
    # being nested inside whichever task started first.
    assert np.all(arrays.parent[job_rows] < 0)
    assert np.all(arrays.depth[job_rows] == 0)


def test_await_time_is_separated_from_the_work(tmp_path):
    manager = ProfileManager()

    async def job():
        with manager.profile_region("job"):
            spin()
            await asyncio.sleep(0.05)
            spin()

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "await.h5"),
    ) as run:
        asyncio.run(job())

    job_region = run.results["job"][0]
    duration = job_region.durations[0]
    awaited = job_region.await_times[0]
    assert awaited == pytest.approx(0.05, abs=0.03)
    # The rest is time the task actually held the thread: two spins.
    assert duration - awaited == pytest.approx(2 * SPIN_SECONDS, abs=0.01)
    assert job_region.total_await_duration == pytest.approx(awaited)
    assert job_region.await_times_ns[0] == pytest.approx(awaited * 1e9, rel=1e-9)


def test_a_region_outside_any_task_reports_no_await_time(tmp_path):
    manager = ProfileManager()

    async def main():
        with manager.profile_region("inside"):
            await asyncio.sleep(0.01)

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "outside.h5"),
    ) as run:
        with manager.profile_region("outside"):
            time.sleep(0.001)
        asyncio.run(main())

    outside = run.results["outside"][0]
    assert outside.task_ids.tolist() == [-1]
    assert outside.await_times.tolist() == [0.0]
    assert run.results["inside"][0].task_ids[0] >= 0


def test_task_table_splits_running_from_awaiting(tmp_path):
    manager = ProfileManager()

    async def job():
        with manager.profile_region("job"):
            await asyncio.sleep(0.03)
            spin()

    async def main():
        await asyncio.gather(job(), job())

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "task_table.h5"),
    ) as run:
        asyncio.run(main())

    jobs = tasks_named(run.results, "job")
    assert len(jobs) == 2
    for task in jobs:
        assert task.kind == "task"
        assert task.name.startswith("Task-")
        assert task.thread_index == 0
        assert task.steps >= 2
        assert task.running_time == pytest.approx(SPIN_SECONDS, abs=0.01)
        assert task.awaiting_time == pytest.approx(0.03, abs=0.03)
        assert "TaskInfo" in repr(task)
    # gather() runs from a task of its own, which the run also describes.
    assert tasks_named(run.results, "main")


def test_interleaved_tasks_close_their_own_scope(tmp_path):
    """Two tasks inside the same region, exiting out of entry order.

    The scope stack is per thread, and both tasks are on one; the second to
    enter is the first to leave, so a plain LIFO pop would hand each task the
    other's slot and record two wrong durations.
    """
    manager = ProfileManager()
    started = asyncio.Event()

    async def slow():
        with manager.profile_region("both"):
            started.set()
            await asyncio.sleep(0.05)

    async def quick():
        await started.wait()
        with manager.profile_region("both"):
            await asyncio.sleep(0.005)

    async def main():
        await asyncio.gather(slow(), quick())

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "interleaved.h5"),
    ) as run:
        asyncio.run(main())

    both = run.results["both"][0]
    assert both.num_calls == 2
    durations = sorted(both.durations)
    assert durations[0] == pytest.approx(0.005, abs=0.02)
    assert durations[1] == pytest.approx(0.05, abs=0.03)
    assert len(set(both.task_ids.tolist())) == 2


def test_async_data_survives_the_hdf5_round_trip(tmp_path):
    path = tmp_path / "async_round_trip.h5"
    manager = ProfileManager()

    async def job():
        with manager.profile_region("job"):
            await asyncio.sleep(0.01)

    async def main():
        await asyncio.gather(job(), job())

    with manager.session(
        track_async=True, verbose=False, return_results=True, file_path=str(path)
    ) as run:
        asyncio.run(main())

    memory = run.results
    stored = read_h5(path)
    assert np.array_equal(memory["job"][0].task_ids, stored["job"][0].task_ids)
    assert np.array_equal(
        memory["job"][0].await_times_ns, stored["job"][0].await_times_ns
    )
    assert [task.name for task in stored.tasks[0]] == [
        task.name for task in memory.tasks[0]
    ]
    assert [task.coro_name for task in stored.tasks[0]] == [
        task.coro_name for task in memory.tasks[0]
    ]
    assert [task.kind for task in stored.tasks[0]] == ["task"] * len(memory.tasks[0])


def test_events_expose_the_lane_of_every_call(tmp_path):
    manager = ProfileManager()

    async def job():
        with manager.profile_region("job"):
            await asyncio.sleep(0.01)

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "events.h5"),
    ) as run:
        asyncio.run(job())

    event = run.results["job"][0].events()[0]
    assert event["thread"] == 0
    assert event["task"] >= 0
    assert event["await_duration"] == pytest.approx(0.01, abs=0.02)


def test_a_cancelled_task_still_reports_its_lane(tmp_path):
    manager = ProfileManager()

    async def forever():
        with manager.profile_region("forever"):
            await asyncio.sleep(10)

    async def main():
        task = asyncio.ensure_future(forever())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "cancelled.h5"),
    ) as run:
        asyncio.run(main())

    # Cancellation unwinds through the ``with``, so the call is recorded --
    # and its whole duration was spent awaiting, which is what makes an
    # abandoned region distinguishable from a slow one.
    cancelled = tasks_named(run.results, "forever")
    assert len(cancelled) == 1
    assert cancelled[0].awaiting_time == pytest.approx(0.01, abs=0.02)
    region = run.results["forever"][0]
    assert region.num_calls == 1
    assert region.await_times[0] == pytest.approx(region.durations[0], abs=0.005)


def test_tasks_created_on_a_worker_thread_are_attributed_to_it(tmp_path):
    import threading

    manager = ProfileManager()

    async def job():
        with manager.profile_region("job"):
            await asyncio.sleep(0.005)

    def worker():
        asyncio.run(job())

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "worker_loop.h5"),
    ) as run:
        thread = threading.Thread(target=worker, name="loop")
        thread.start()
        thread.join()

    job_region = run.results["job"][0]
    (thread_index,) = set(job_region.thread_ids.tolist())
    assert run.results.threads[0][thread_index].name == "loop"
    jobs = tasks_named(run.results, "job")
    assert [task.thread_index for task in jobs] == [thread_index]


def test_uninstalling_restores_asyncio(tmp_path):
    original_run_forever = asyncio.BaseEventLoop.run_forever
    original_create_task = asyncio.BaseEventLoop.create_task

    tracker = ConcurrencyTracker(track_async=True)
    tracker.install()
    assert asyncio.BaseEventLoop.run_forever is not original_run_forever
    assert asyncio.BaseEventLoop.create_task is not original_create_task
    tracker.install()  # idempotent
    tracker.uninstall()

    assert asyncio.BaseEventLoop.run_forever is original_run_forever
    assert asyncio.BaseEventLoop.create_task is original_create_task
    # A loop run afterwards is untouched, and creates no task records.
    asyncio.run(asyncio.sleep(0))
    tracker.uninstall()  # idempotent


def test_an_existing_task_factory_is_chained(tmp_path):
    manager = ProfileManager()
    seen = []

    async def job():
        with manager.profile_region("job"):
            await asyncio.sleep(0.001)

    async def main():
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()

        def factory(loop_, coro, **kwargs):
            seen.append(coro)
            if previous is None:
                return asyncio.Task(coro, loop=loop_, **kwargs)
            return previous(loop_, coro, **kwargs)

        loop.set_task_factory(factory)
        await job()
        await asyncio.gather(job())

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "chained.h5"),
    ) as run:
        asyncio.run(main())

    assert seen, "the application's own task factory still ran"
    assert run.results["job"][0].num_calls == 2
    # The application installed its factory on top of ours and chains back
    # into it, so its tasks are still described.
    assert tasks_named(run.results, "job")


def test_greenlets_are_followed_when_greenlet_is_installed(tmp_path):
    greenlet = pytest.importorskip("greenlet")
    manager = ProfileManager()

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "greenlets.h5"),
    ) as run:
        other = None

        def second():
            with manager.profile_region("both"):
                main_greenlet.switch()

        def first():
            with manager.profile_region("both"):
                other.switch()

        main_greenlet = greenlet.getcurrent()
        other = greenlet.greenlet(second)
        worker = greenlet.greenlet(first)
        worker.switch()
        other.switch()
        worker.switch()

    both = run.results["both"][0]
    assert both.num_calls == 2
    assert len(set(both.task_ids.tolist())) == 2
    kinds = {task.kind for task in run.results.tasks[0]}
    assert "greenlet" in kinds


def test_lane_label_names_a_task_with_its_thread(tmp_path):
    manager = ProfileManager()

    async def job():
        with manager.profile_region("job"):
            await asyncio.sleep(0.001)

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "labels.h5"),
    ) as run:
        asyncio.run(job())

    results = run.results
    (task,) = tasks_named(results, "job")
    label = results.lane_label(task.index)
    assert label.startswith(task.name)
    assert "(job)" in label
    assert label.endswith("on MainThread")


def test_speedscope_export_gives_each_task_its_own_profile(tmp_path):
    import json

    from scope_profiler.speedscope_export import export_speedscope

    manager = ProfileManager()

    async def job():
        with manager.profile_region("job"):
            await asyncio.sleep(0.01)
            spin(0.002)

    async def main():
        await asyncio.gather(job(), job())

    with manager.session(
        track_async=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "async_lanes.h5"),
    ) as run:
        asyncio.run(main())

    (written,) = export_speedscope(
        run.results, tmp_path / "profile.speedscope.json", verbose=False
    )
    document = json.loads(written.read_text())
    names = [profile["name"] for profile in document["profiles"]]
    # The two interleaved 'job' tasks, plus the main thread's session region.
    assert sum("(job)" in name for name in names) == 2
    for profile in document["profiles"]:
        stamps = [event["at"] for event in profile["events"]]
        assert stamps == sorted(stamps)
