"""Write random runs through the real writer and read every property back.

The example-based writer tests pin specific layouts. This one generates whole
runs -- several ranks, several regions, varying call counts, tags and source
locations -- and asserts that a round trip through HDF5 preserves *every*
public property of :class:`~scope_profiler.region.Region`, not just the few a
given test happened to look at.

Timestamps are generated so each region occupies its own band on the timeline,
which makes the run properly nested (trivially, with no nesting at all) and so
lets the exclusive-duration properties be compared too.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scope_profiler import read_h5
from scope_profiler.h5writer import ProfilingWriter
from scope_profiler.profile_manager import RankPayload

NS = 1_000_000_000

# Every scalar and array property a caller can read off a Region. Kept as one
# list so a newly added property shows up here as a failure rather than as a
# silently untested one (see test_every_public_property_is_covered).
ARRAY_PROPERTIES = (
    "start_times_ns",
    "end_times_ns",
    "durations_ns",
    "inclusive_durations_ns",
    "exclusive_durations_ns",
    "start_times",
    "end_times",
    "durations",
    "inclusive_durations",
    "exclusive_durations",
    # Lane columns. None on the runs generated here, which is itself worth
    # pinning: a file written without them must not come back with zeros.
    "thread_ids",
    "task_ids",
    "await_times",
    "await_times_ns",
)
SCALAR_PROPERTIES = (
    "has_timing",
    "has_event_data",
    "has_source",
    "has_gpu_timing",
    "source_file",
    "source_lineno",
    "source_text",
    "tags",
    "first_start_time",
    "last_end_time",
    "num_calls",
    "total_duration",
    "inclusive_duration",
    "total_exclusive_duration",
    "exclusive_duration",
    "average_duration",
    "min_duration",
    "max_duration",
    "first_duration",
    "last_duration",
    "std_duration",
    "p50_duration",
    "p95_duration",
    "p99_duration",
    "gpu_durations",
    "gpu_durations_ns",
    "gpu_total_duration",
    "gpu_average_duration",
    "stored_summary",
    "has_thread_data",
    "threads",
    "total_await_duration",
)


@st.composite
def runs(draw):
    """A whole run: ranks -> region name -> (starts, ends) in nanoseconds."""
    num_ranks = draw(st.integers(min_value=1, max_value=3))
    region_names = draw(
        st.lists(
            st.sampled_from(["solve", "assemble", "io", "push", "gather"]),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    run = {}
    for rank in range(num_ranks):
        regions = {}
        for band, name in enumerate(region_names):
            count = draw(st.integers(min_value=1, max_value=6))
            # Each region gets its own band of the timeline, and its calls run
            # back to back inside it: disjoint, ordered, never overlapping.
            base = band * 1_000 * NS + rank
            gaps = draw(
                st.lists(
                    st.integers(min_value=1, max_value=10_000),
                    min_size=2 * count,
                    max_size=2 * count,
                ),
            )
            starts, ends, cursor = [], [], base
            for index in range(count):
                cursor += gaps[2 * index]
                starts.append(cursor)
                cursor += gaps[2 * index + 1]
                ends.append(cursor)
            regions[name] = (
                np.array(starts, dtype=np.int64),
                np.array(ends, dtype=np.int64),
            )
        run[rank] = regions
    return run


def _write(path, run, metadata=None):
    with ProfilingWriter(path, metadata or {"hostname": "node0"}) as writer:
        for rank, regions in run.items():
            writer.write_rank(
                rank,
                RankPayload(
                    regions=regions,
                    likwid={},
                    likwid_environment={},
                    line_profile=None,
                    exclusive_totals=None,
                ),
            )
    return path


def _assert_regions_equal(expected, actual, context):
    for name in ARRAY_PROPERTIES:
        left, right = getattr(expected, name), getattr(actual, name)
        assert np.array_equal(left, right), f"{context}: {name}"
    for name in SCALAR_PROPERTIES:
        left, right = getattr(expected, name), getattr(actual, name)
        if isinstance(left, float) and isinstance(right, float):
            assert left == pytest.approx(
                right,
                rel=1e-12,
                nan_ok=True,
            ), f"{context}: {name}"
        elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            assert np.array_equal(left, right), f"{context}: {name}"
        else:
            assert left == right, f"{context}: {name}"


@given(run=runs())
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_a_written_run_reads_back_identically(run, tmp_path_factory):
    path = tmp_path_factory.mktemp("roundtrip") / "profiling_data.h5"
    _write(path, run)

    results = read_h5(path)

    assert results.num_ranks == len(run)
    assert sorted(results.region_names) == sorted(
        {name for regions in run.values() for name in regions},
    )
    for rank, regions in run.items():
        for name, (starts, ends) in regions.items():
            region = results[name][rank]
            assert region.start_times_ns.tolist() == starts.tolist()
            assert region.end_times_ns.tolist() == ends.tolist()
            assert region.num_calls == len(starts)
            assert region.durations_ns.tolist() == (ends - starts).tolist()
            # No region overlaps another, so nothing is nested and exclusive
            # time is inclusive time.
            assert region.exclusive_durations_ns.tolist() == (ends - starts).tolist()


@given(run=runs())
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_two_write_read_cycles_are_a_fixed_point(run, tmp_path_factory):
    """Reading a file and writing it back out again changes nothing.

    Every public property is compared, so a field that survives the first
    trip but is dropped or re-derived differently on the second shows up here.
    """
    directory = tmp_path_factory.mktemp("fixedpoint")
    first = read_h5(_write(directory / "first.h5", run))

    second_run = {
        rank: {
            name: (
                first[name][rank].start_times_ns,
                first[name][rank].end_times_ns,
            )
            for name in first.region_names
            if rank in first[name].regions
        }
        for rank in range(first.num_ranks)
    }
    second = read_h5(_write(directory / "second.h5", second_run))

    assert first.region_names == second.region_names
    assert first.num_ranks == second.num_ranks
    for name in first.region_names:
        for rank in first[name].regions:
            _assert_regions_equal(
                first[name][rank],
                second[name][rank],
                f"{name}@{rank}",
            )
    assert first.summary() == second.summary()


def test_every_public_property_is_covered():
    """The round trip compares every property a Region exposes.

    A property added to Region without being listed above would otherwise
    never be round-trip tested.
    """
    from scope_profiler.region import Region

    exposed = {
        name
        for name, value in vars(Region).items()
        if isinstance(value, property) and not name.startswith("_")
    }
    compared = set(ARRAY_PROPERTIES) | set(SCALAR_PROPERTIES)
    # call_ids/parent_ids are the recursive tracer's optional columns; they
    # have their own tests and are None for the runs generated here.
    assert exposed - compared == {"call_ids", "parent_ids"}
