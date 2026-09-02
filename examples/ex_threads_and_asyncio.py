"""Profile concurrent threads, and the await time of asyncio tasks.

Run with::

    python examples/ex_threads_and_asyncio.py

``track_threads=True`` gives every thread its own buffers, its own scope
stack, and a lane of its own in the reconstructed call graph, so regions
entered concurrently record correct and separable timelines. ``track_async``
does the same for asyncio tasks and greenlets, and additionally splits each
call's duration into the time its task held the thread and the time it spent
awaiting.
"""

import asyncio
import threading
import time
import warnings

from scope_profiler import ProfileManager

UNTRACKED_PROFILE = "threads_untracked_profile.h5"
THREAD_PROFILE = "threads_profile.h5"
ASYNC_PROFILE = "async_profile.h5"


def solve(manager, steps=3):
    """Work done by every worker thread, under two nested regions."""
    for _ in range(steps):
        with manager.profile_region("solve"):
            time.sleep(0.005)
            with manager.profile_region("assemble"):
                time.sleep(0.002)


def profile_without_tracking():
    """The same workload, profiled as if it were single-threaded.

    Four threads share one buffer and one scope stack, so their intervals
    interleave instead of nesting. finalize() keeps the timings and drops the
    call graph with a warning rather than throwing the run away -- which means
    exclusive time, the flame chart and the call-path exports have nothing to
    work from.
    """
    manager = ProfileManager()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with manager.session(
            file_path=UNTRACKED_PROFILE,
            verbose=False,
            return_results=True,
        ) as run:
            workers = [
                threading.Thread(target=solve, args=(manager,)) for _ in range(4)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

    results = run.results
    solve_region = results["solve"][0]
    print("Without track_threads")
    print(f"  calls recorded:  {solve_region.num_calls}")
    print(f"  thread recorded: {solve_region.has_thread_data}")
    for warning in caught:
        if "call graph" in str(warning.message):
            print(f"  warning:         {str(warning.message).split(':')[0]}")
            break
    else:
        print("  warning:         none (the threads happened not to overlap)")


def profile_threads():
    """Four worker threads inside the same two regions, at the same time."""
    manager = ProfileManager()
    with manager.session(
        file_path=THREAD_PROFILE,
        track_threads=True,
        verbose=False,
        return_results=True,
    ) as run:
        workers = [
            threading.Thread(target=solve, args=(manager,), name=f"worker-{index}")
            for index in range(4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

    results = run.results
    results.print_summary(title="Threaded profiling results")

    print("\nPer thread:")
    for row in results.thread_summary(rank=0):
        state = "alive" if row["alive"] else "finished"
        print(
            f"  {row['name']:<12s} {row['num_calls']:>3d} calls  "
            f"region {row['region_time'] * 1e3:7.2f} ms  "
            f"cpu {row['cpu_time'] * 1e3:7.2f} ms  ({state})"
        )

    solve_region = results["solve"][0]
    print("\nThe same region, split by thread:")
    for index in solve_region.threads:
        per_thread = solve_region.for_thread(index)
        print(
            f"  thread {index}: {per_thread.num_calls} calls, "
            f"{per_thread.average_duration * 1e3:.2f} ms average"
        )


async def fetch(manager, delay):
    """A task that awaits, then does a little work of its own."""
    with manager.profile_region("fetch"):
        await asyncio.sleep(delay)
        deadline = time.perf_counter() + 0.005
        while time.perf_counter() < deadline:
            pass


async def gather(manager):
    return await asyncio.gather(*(fetch(manager, 0.01 * n) for n in range(1, 4)))


def profile_asyncio():
    """Three interleaved tasks, and where their wall time actually went."""
    manager = ProfileManager()
    with manager.session(
        file_path=ASYNC_PROFILE,
        track_async=True,
        verbose=False,
        return_results=True,
    ) as run:
        asyncio.run(gather(manager))

    results = run.results
    fetch_region = results["fetch"][0]
    print("\nPer call of 'fetch':")
    for index, duration in enumerate(fetch_region.durations):
        awaited = fetch_region.await_times[index]
        print(
            f"  call {index}: {duration * 1e3:6.2f} ms total, "
            f"{awaited * 1e3:6.2f} ms awaiting, "
            f"{(duration - awaited) * 1e3:6.2f} ms running"
        )

    print("\nPer task:")
    for task in results.tasks[0]:
        print(
            f"  {task.name:<8s} {task.coro_name.rsplit('.', 1)[-1]:<20s} "
            f"{task.steps:>2d} steps  "
            f"running {task.running_time * 1e3:6.2f} ms  "
            f"awaiting {task.awaiting_time * 1e3:6.2f} ms"
        )


def main():
    profile_without_tracking()
    print()
    profile_threads()
    profile_asyncio()
    print(f"\nWrote {UNTRACKED_PROFILE}, {THREAD_PROFILE} and {ASYNC_PROFILE}")


if __name__ == "__main__":
    main()
