"""Profiling across processes: one run per process, and what a fork inherits.

There is no cross-process merge -- that is what the MPI path is for -- so a
multiprocessing worker profiles itself, into a file of its own. What needs
pinning is the boundary: a child forked out of a parent's *active* session
inherits that session's hooks along with everything else, and must not go on
feeding a run that nothing in the child will ever finalize.
"""

import asyncio
import multiprocessing
import os
import pickle
import sys
import threading
import time

import pytest

from scope_profiler import ProfileManager, read_h5

requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")


def _run_in_fork(body):
    """Run ``body()`` in a forked child and return whatever it gives back.

    The child answers through a pipe and leaves with ``os._exit``, so it never
    unwinds into pytest's own teardown.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - measured in the child, not here
        status = 1
        try:
            os.close(read_fd)
            payload = pickle.dumps(body())
            os.write(write_fd, payload)
            os.close(write_fd)
            status = 0
        finally:
            os._exit(status)
    os.close(write_fd)
    chunks = []
    with os.fdopen(read_fd, "rb") as stream:
        while chunk := stream.read(65536):
            chunks.append(chunk)
    _, exit_status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(exit_status) == 0, "the child failed"
    return pickle.loads(b"".join(chunks))


@requires_fork
def test_a_forked_child_stops_tracking_the_parents_session(tmp_path):
    """A child inherits the hooks but not the session that opened them.

    Left installed, they would append a record per thread and per task to a
    table nothing in the child ever reads -- unbounded, for a long-lived
    forked worker running an event loop.
    """
    manager = ProfileManager()
    stock_create_task = asyncio.BaseEventLoop.create_task

    def in_child():
        async def job(value):
            await asyncio.sleep(0)
            return value

        async def main():
            return await asyncio.gather(*[job(index) for index in range(20)])

        assert asyncio.run(main()) == list(range(20))
        tracker = manager.get_config().tracker
        return {
            "tasks": len(tracker.tasks),
            "threads": len(tracker.threads),
            "restored": asyncio.BaseEventLoop.create_task is stock_create_task,
            "profile_hook": sys.getprofile() is None,
        }

    with manager.session(
        track_async=True,
        verbose=False,
        file_path=str(tmp_path / "parent.h5"),
    ):
        with manager.profile_region("parent_work"):
            pass
        result = _run_in_fork(in_child)

    assert result == {
        "tasks": 0,
        "threads": 0,
        "restored": True,
        "profile_hook": True,
    }
    # The parent's own session is untouched by what the child did.
    assert manager.get_config().tracker.threads


@requires_fork
def test_a_forked_child_profiles_itself_with_its_own_session(tmp_path):
    """The supported pattern: the worker opens a session, into its own file."""
    parent = ProfileManager()
    child_path = tmp_path / "child.h5"

    def in_child():
        manager = ProfileManager()
        with manager.session(
            track_threads=True,
            verbose=False,
            file_path=str(child_path),
        ):

            def work():
                with manager.profile_region("child_work"):
                    time.sleep(0.001)

            workers = [
                threading.Thread(target=work, name=f"child-{index}")
                for index in range(2)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        return True

    with parent.session(
        track_threads=True,
        verbose=False,
        file_path=str(tmp_path / "parent.h5"),
    ):
        with parent.profile_region("parent_work"):
            pass
        assert _run_in_fork(in_child)

    stored = read_h5(child_path)
    assert stored["child_work"][0].num_calls == 2
    # Only the child's own threads, numbered from zero -- the parent's table
    # did not survive into it.
    assert [thread.name for thread in stored.threads[0]] == [
        "MainThread",
        "child-0",
        "child-1",
    ]
    assert sorted(stored["child_work"][0].thread_ids.tolist()) == [1, 2]
    assert "parent_work" not in stored.region_names


def _spawn_worker(path):
    """Body of a spawned worker; importable by name, as spawn requires."""
    manager = ProfileManager()
    with manager.session(
        track_threads=True, verbose=False, file_path=path, return_results=True
    ) as run:

        def work():
            with manager.profile_region("solve"):
                time.sleep(0.001)

        workers = [
            threading.Thread(target=work, name=f"w{index}") for index in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    results = run.results
    return (
        os.getpid(),
        results["solve"][0].num_calls,
        [thread.name for thread in results.threads[0]],
    )


@pytest.mark.slow
@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(),
    reason="the spawn start method is unavailable",
)
def test_spawned_workers_each_record_their_own_threads(tmp_path):
    context = multiprocessing.get_context("spawn")
    paths = [str(tmp_path / f"worker_{index}.h5") for index in range(2)]

    with context.Pool(2) as pool:
        results = pool.map(_spawn_worker, paths)

    pids = {pid for pid, _, _ in results}
    assert len(pids) == 2
    for _, calls, thread_names in results:
        assert calls == 2
        assert thread_names == ["MainThread", "w0", "w1"]
    for path in paths:
        stored = read_h5(path)
        assert stored["solve"][0].num_calls == 2
        assert [thread.name for thread in stored.threads[0]] == [
            "MainThread",
            "w0",
            "w1",
        ]
