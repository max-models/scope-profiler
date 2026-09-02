"""Profiling regions entered from several threads at once.

Without ``track_threads`` a region has one buffer and one scope stack for the
whole process, so two threads inside it reserve and close each other's slots.
These tests pin the behaviour that replaces that: one buffer per thread, one
lane per thread, and a description of every thread the run touched.
"""

import threading
import time

import numpy as np
import pytest

from scope_profiler import ProfileManager, read_h5
from scope_profiler.call_stack import NestingError, build_call_arrays
from scope_profiler.concurrency import ConcurrencyTracker, lane_ids
from scope_profiler.h5reader import read_h5_summary
from scope_profiler.profile_config import ProfilingConfig
from scope_profiler.region_profiler import ThreadedProfileRegion

BARRIER_TIMEOUT = 10


def run_in_threads(target, count, *, name="w"):
    """Run ``target(index, barrier)`` on ``count`` threads and join them."""
    barrier = threading.Barrier(count, timeout=BARRIER_TIMEOUT)
    threads = [
        threading.Thread(target=target, args=(index, barrier), name=f"{name}{index}")
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return threads


def nested_worker(manager, outer="outer", inner="inner", repeats=3):
    """A worker whose calls really do overlap between threads."""

    def worker(_index, barrier):
        for _ in range(repeats):
            barrier.wait()
            with manager.profile_region(outer):
                with manager.profile_region(inner):
                    time.sleep(0.001)

    return worker


def test_concurrent_threads_record_every_call_once(tmp_path):
    manager = ProfileManager()
    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "threads.h5"),
    ) as run:
        run_in_threads(nested_worker(manager), 4)

    results = run.results
    outer = results["outer"][0]
    assert outer.num_calls == 12
    assert results["inner"][0].num_calls == 12
    # Four workers, plus the main thread that opened the session region.
    assert sorted(outer.threads) == [1, 2, 3, 4]
    assert np.bincount(outer.thread_ids)[1:].tolist() == [3, 3, 3, 3]


def test_nesting_is_reconstructed_per_thread(tmp_path):
    manager = ProfileManager()
    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "nesting.h5"),
    ) as run:
        run_in_threads(nested_worker(manager), 4)

    results = run.results
    outer = results["outer"][0]
    inner = results["inner"][0]
    # Every inner call sits inside exactly one outer call on its own thread,
    # so outer's exclusive time is what is left after inner.
    assert outer.exclusive_duration == pytest.approx(
        outer.total_duration - inner.total_duration, rel=1e-9
    )
    arrays = build_call_arrays(results.get_regions(), rank=0)
    # One lane per thread, plus the session region's lane on the main thread.
    assert len(set(arrays.lane.tolist())) == 5
    inner_rows = arrays.region_index == arrays.names.index("inner")
    # No inner call is a root: each is nested inside its thread's outer call.
    assert not np.any(arrays.parent[inner_rows] < 0)


def test_interleaved_threads_without_tracking_are_rejected():
    """The failure ``track_threads`` exists to fix, as the reader sees it.

    Two overlapping-but-not-nested intervals are exactly what one shared
    buffer records for two threads, and there is no call graph to build from
    them -- which is why this raises rather than inventing one.
    """
    from scope_profiler.call_stack import regions_from_snapshot

    snapshot = {
        "a": (np.array([0, 20]), np.array([30, 50])),
    }
    with pytest.raises(NestingError, match="not properly nested"):
        build_call_arrays(regions_from_snapshot(snapshot, 0), rank=0)


def test_the_same_intervals_nest_once_they_are_on_separate_lanes():
    from scope_profiler.call_stack import regions_from_snapshot

    snapshot = {
        "a": (
            np.array([0, 20]),
            np.array([30, 50]),
            None,
            None,
            None,
            np.array([0, 1]),
        ),
    }
    arrays = build_call_arrays(regions_from_snapshot(snapshot, 0), rank=0)
    assert arrays.parent.tolist() == [-1, -1]
    assert arrays.lane.tolist() == [-2, -3]


def test_thread_table_describes_every_thread(tmp_path):
    manager = ProfileManager()
    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "table.h5"),
    ) as run:
        run_in_threads(nested_worker(manager), 3, name="solver")

    threads = run.results.threads[0]
    assert [thread.name for thread in threads] == [
        "MainThread",
        "solver0",
        "solver1",
        "solver2",
    ]
    main, *workers = threads
    assert main.alive  # still running at finalize()
    assert main.wall_time is None
    for worker in workers:
        # Joined before finalize(), so the exit sentinel closed them exactly.
        assert not worker.alive
        assert worker.wall_time > 0
        assert worker.cpu_time > 0
        assert worker.ident != 0
    assert len({thread.ident for thread in threads}) == 4


def test_thread_summary_reports_wall_cpu_and_calls(tmp_path):
    manager = ProfileManager()
    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "summary.h5"),
    ) as run:
        run_in_threads(nested_worker(manager), 2)

    rows = run.results.thread_summary(0)
    assert [row["name"] for row in rows] == ["MainThread", "w0", "w1"]
    for row in rows[1:]:
        # Three outer calls and three inner ones per worker; only the outer
        # ones are top level, so region_time counts each once.
        assert row["num_calls"] == 6
        assert row["region_time"] > 0
        assert row["region_time"] <= row["wall_time"]
    assert run.results.thread_summary(99) == []


def test_for_thread_slices_a_region_to_one_thread(tmp_path):
    manager = ProfileManager()
    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "slice.h5"),
    ) as run:
        run_in_threads(nested_worker(manager), 3)

    outer = run.results["outer"][0]
    per_thread = [outer.for_thread(index) for index in outer.threads]
    assert [region.num_calls for region in per_thread] == [3, 3, 3]
    assert sum(region.total_duration for region in per_thread) == pytest.approx(
        outer.total_duration
    )
    assert per_thread[0].source_file == outer.source_file


def test_for_thread_without_thread_data_explains_itself(tmp_path):
    manager = ProfileManager()
    with manager.session(
        verbose=False, return_results=True, file_path=str(tmp_path / "plain.h5")
    ) as run:
        with manager.profile_region("solve"):
            pass

    region = run.results["solve"][0]
    assert not region.has_thread_data
    assert region.threads == []
    with pytest.raises(ValueError, match="track_threads=True"):
        region.for_thread(0)


def test_thread_data_survives_the_hdf5_round_trip(tmp_path):
    path = tmp_path / "round_trip.h5"
    manager = ProfileManager()
    with manager.session(
        track_threads=True, verbose=False, return_results=True, file_path=str(path)
    ) as run:
        run_in_threads(nested_worker(manager), 3)

    memory = run.results
    stored = read_h5(path)
    assert np.array_equal(memory["outer"][0].thread_ids, stored["outer"][0].thread_ids)
    assert [thread.name for thread in stored.threads[0]] == [
        thread.name for thread in memory.threads[0]
    ]
    assert [thread.cpu_time for thread in stored.threads[0]] == [
        thread.cpu_time for thread in memory.threads[0]
    ]
    assert stored["outer"][0].exclusive_duration == pytest.approx(
        memory["outer"][0].exclusive_duration
    )
    # The summary reader skips the event columns but still describes the lanes.
    summary = read_h5_summary(path)
    assert [thread.name for thread in summary.threads[0]] == [
        thread.name for thread in memory.threads[0]
    ]


def test_a_run_without_threads_writes_no_lane_columns(tmp_path):
    import h5py

    path = tmp_path / "no_lanes.h5"
    manager = ProfileManager()
    with manager.session(verbose=False, file_path=str(path)):
        with manager.profile_region("solve"):
            pass

    with h5py.File(path, "r") as h5file:
        assert "thread_ids" not in h5file["events"]
        assert "thread_table" not in h5file
    assert read_h5(path).threads == {}


def test_buffers_grow_independently_per_thread(tmp_path):
    manager = ProfileManager()

    def worker(_index, barrier):
        barrier.wait()
        for _ in range(40):
            with manager.profile_region("hot"):
                pass

    with manager.session(
        track_threads=True,
        buffer_limit=2,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "growth.h5"),
    ) as run:
        run_in_threads(worker, 4)

    hot = run.results["hot"][0]
    assert hot.num_calls == 160
    assert np.bincount(hot.thread_ids)[1:].tolist() == [40, 40, 40, 40]
    assert np.all(hot.durations_ns >= 0)


def test_a_paused_scope_closes_itself_and_not_the_call_inside_it(tmp_path):
    """A region entered while paused reserves no slot, and closes none either.

    One scope stack serves every region on a lane, so a paused ``__exit__``
    that popped anyway would close whichever call happened to be open --
    here, the recorded scope nested inside the paused one.
    """
    manager = ProfileManager()
    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "paused.h5"),
    ) as run:
        manager.pause()
        with manager.profile_region("dropped"):
            manager.resume()
            with manager.profile_region("kept"):
                time.sleep(0.001)
            manager.pause()
        manager.resume()

    results = run.results
    assert "dropped" not in results.region_names
    assert results["kept"][0].num_calls == 1
    assert results["kept"][0].durations_ns[0] > 0


def test_threads_record_nothing_while_profiling_is_paused(tmp_path):
    manager = ProfileManager()

    def worker(_index, barrier):
        barrier.wait()
        with manager.profile_region("dropped"):
            pass

    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "paused_threads.h5"),
    ) as run:
        manager.pause()
        run_in_threads(worker, 3)
        manager.resume()
        with manager.profile_region("kept"):
            pass

    results = run.results
    assert "dropped" not in results.region_names
    assert results["kept"][0].num_calls == 1


def test_decorator_form_records_the_calling_thread(tmp_path):
    manager = ProfileManager()

    @manager.profile("decorated")
    def work():
        time.sleep(0.001)

    def worker(_index, barrier):
        barrier.wait()
        work()

    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "decorated.h5"),
    ) as run:
        run_in_threads(worker, 3)

    decorated = run.results["decorated"][0]
    assert decorated.num_calls == 3
    assert sorted(decorated.thread_ids.tolist()) == [1, 2, 3]


def test_a_second_finalize_reports_only_its_own_calls(tmp_path):
    manager = ProfileManager()
    manager.setup(track_threads=True, file_path=str(tmp_path / "first.h5"))

    def worker(_index, barrier):
        barrier.wait()
        with manager.profile_region("step"):
            pass

    run_in_threads(worker, 3)
    first = manager.finalize(verbose=False, return_results=True)
    run_in_threads(worker, 2, name="second")
    second = manager.finalize(verbose=False, return_results=True)

    assert first["step"][0].num_calls == 3
    assert second["step"][0].num_calls == 2
    # Lane indices keep counting across finalize(): the first batch of
    # workers registered as 0-2, so the second batch starts at 3.
    assert sorted(first["step"][0].thread_ids.tolist()) == [0, 1, 2]
    assert sorted(second["step"][0].thread_ids.tolist()) == [3, 4]


def test_num_calls_counts_every_lane(tmp_path):
    manager = ProfileManager()
    manager.setup(track_threads=True, file_path=str(tmp_path / "live.h5"))

    def worker(_index, barrier):
        barrier.wait()
        with manager.profile_region("live"):
            pass

    run_in_threads(worker, 3)
    region = manager.get_region("live")
    assert isinstance(region, ThreadedProfileRegion)
    assert region.num_calls == 3
    assert [thread.name for thread in region.threads] == ["w0", "w1", "w2"]
    assert region.open_slots().size == 0
    manager.finalize(verbose=False)
    assert region.num_calls == 3


@pytest.mark.parametrize(
    "option",
    ["use_line_profiler", "use_gpu_timing", "use_likwid", "use_nvtx"],
)
def test_track_threads_rejects_incompatible_modes(option):
    with pytest.raises(ValueError, match="cannot be combined"):
        ProfilingConfig(track_threads=True, **{option: True})


def test_track_threads_rejects_aggregation_mode():
    with pytest.raises(ValueError, match="cannot be combined"):
        ProfilingConfig(track_threads=True, aggregation_mode=True)


def test_track_async_implies_track_threads():
    config = ProfilingConfig(track_async=True)
    assert config.track_threads
    assert config.track_async
    assert config.tracker is not None
    assert ProfilingConfig().tracker is None


def test_lane_ids_keeps_thread_and_task_spaces_apart():
    threads = np.array([0, 1, 2, -1])
    assert lane_ids(threads, None).tolist() == [-2, -3, -4, -1]
    tasks = np.array([-1, 7, -1, 0])
    assert lane_ids(threads, tasks).tolist() == [-2, 7, -4, 0]


def test_tracker_registers_a_thread_once():
    tracker = ConcurrencyTracker()
    first = tracker.current_thread()
    assert tracker.current_thread() is first
    assert len(tracker.threads) == 1
    assert "index=0" in repr(first)

    seen = []
    run_in_threads(lambda _index, barrier: seen.append(tracker.current_thread()), 3)
    assert len(tracker.threads) == 4
    assert {record.index for record in seen} == {1, 2, 3}


def test_speedscope_export_writes_one_profile_per_lane(tmp_path):
    """An evented profile's timestamps must never go backwards.

    Two threads walked as one call tree emit exactly that, so each lane gets a
    profile of its own -- which is also what speedscope calls a thread.
    """
    import json

    from scope_profiler.speedscope_export import export_speedscope

    manager = ProfileManager()
    path = tmp_path / "lanes.h5"
    with manager.session(
        track_threads=True, verbose=False, return_results=True, file_path=str(path)
    ) as run:
        run_in_threads(nested_worker(manager), 3)

    (written,) = export_speedscope(
        run.results, tmp_path / "profile.speedscope.json", verbose=False
    )
    document = json.loads(written.read_text())
    names = [profile["name"] for profile in document["profiles"]]
    assert sorted(names) == [
        "rank 0 - MainThread",
        "rank 0 - w0",
        "rank 0 - w1",
        "rank 0 - w2",
    ]
    for profile in document["profiles"]:
        stamps = [event["at"] for event in profile["events"]]
        assert stamps == sorted(stamps)


def test_split_by_lane_reindexes_parents():
    from scope_profiler.call_stack import regions_from_snapshot, split_by_lane

    snapshot = {
        "outer": (
            np.array([0, 10]),
            np.array([100, 110]),
            None,
            None,
            None,
            np.array([0, 1]),
        ),
        "inner": (
            np.array([20, 30]),
            np.array([40, 50]),
            None,
            None,
            None,
            np.array([0, 1]),
        ),
    }
    arrays = build_call_arrays(regions_from_snapshot(snapshot, 0), rank=0)
    lanes = split_by_lane(arrays)

    assert [lane for lane, _ in lanes] == [-3, -2]
    for _, calls in lanes:
        assert len(calls) == 2
        # One root and one child of it, addressed within this lane.
        assert sorted(calls.parent.tolist()) == [-1, 0]
        assert calls.names is arrays.names


def test_split_by_lane_leaves_a_single_lane_alone():
    from scope_profiler.call_stack import regions_from_snapshot, split_by_lane

    # No lane column at all: the untracked run's single implicit stack.
    snapshot = {"a": (np.array([0, 20]), np.array([100, 50]))}
    arrays = build_call_arrays(regions_from_snapshot(snapshot, 0), rank=0)
    assert arrays.lane.size == 0
    assert split_by_lane(arrays) == [(-1, arrays)]

    # A lane column that happens to hold one lane: also returned untouched.
    threaded = dict(snapshot)
    threaded["a"] = (*snapshot["a"], None, None, None, np.array([2, 2]))
    one_lane = build_call_arrays(regions_from_snapshot(threaded, 0), rank=0)
    assert split_by_lane(one_lane) == [(-4, one_lane)]

    assert split_by_lane(build_call_arrays([], rank=0)) == []


def test_lane_label_names_threads_and_tasks(tmp_path):
    manager = ProfileManager()
    with manager.session(
        track_threads=True,
        verbose=False,
        return_results=True,
        file_path=str(tmp_path / "labels.h5"),
    ) as run:
        run_in_threads(nested_worker(manager, repeats=1), 2)

    results = run.results
    assert results.lane_label(-2) == "MainThread"
    assert results.lane_label(-3) == "w0"
    assert results.lane_label(-1) == "unknown lane"
    # Lanes the run does not describe still get a usable name.
    assert results.lane_label(-99) == "thread 97"
    assert results.lane_label(7) == "task 7"


def test_an_untracked_run_builds_no_lane_column(tmp_path):
    """The single-lane path must stay free of per-event lane work.

    Materializing a full-length column of ``-1`` to say "one stack" would put
    an allocation, a concatenate and a gather over every event into the
    reconstruction of every single-threaded run -- ~8 ms per two million
    events, for no information.
    """
    manager = ProfileManager()
    with manager.session(
        verbose=False, return_results=True, file_path=str(tmp_path / "single.h5")
    ) as run:
        for _ in range(5):
            with manager.profile_region("outer"):
                with manager.profile_region("inner"):
                    pass

    arrays = build_call_arrays(run.results.get_regions(), rank=0)
    assert len(arrays) == 11  # five outer, five inner, one session
    assert arrays.lane.size == 0
    # Nesting is still reconstructed, and consumers read the empty column as
    # "one stack" rather than tripping over it.
    assert arrays.depth.max() == 2
    calls = run.results.call_stack(rank=0)
    assert {call["lane"] for call in calls} == {-1}
