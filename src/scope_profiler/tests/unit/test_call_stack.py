"""The nesting contract and the vectorized reconstruction behind it."""

import numpy as np
import pytest

from scope_profiler.call_stack import (
    NestingError,
    build_call_arrays,
    build_call_stack,
    exclusive_totals_ns,
)
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import Region


def _regions(rank=0, **named):
    return [
        MPIRegion(
            name=name,
            regions={
                rank: Region(
                    np.array([s for s, _ in calls], dtype=np.int64),
                    np.array([e for _, e in calls], dtype=np.int64),
                )
            },
        )
        for name, calls in named.items()
    ]


def test_nesting_is_reconstructed_from_containment():
    regions = _regions(
        outer=[(0, 100)],
        inner=[(10, 40), (50, 90)],
        leaf=[(15, 20)],
    )

    arrays = build_call_arrays(regions, rank=0)

    assert arrays.depth.tolist() == [0, 1, 2, 1]
    assert arrays.parent.tolist() == [-1, 0, 1, 0]
    # outer: 100 - (30 + 40); inner[0]: 30 - 5.
    assert arrays.exclusive_ns.tolist() == [30, 25, 5, 40]


def test_siblings_that_touch_are_not_nested():
    """A call ending exactly when the next starts is its sibling, not parent."""
    arrays = build_call_arrays(_regions(a=[(0, 10), (10, 20)]), rank=0)

    assert arrays.depth.tolist() == [0, 0]
    assert arrays.parent.tolist() == [-1, -1]


def test_recursion_nests_a_region_inside_itself():
    arrays = build_call_arrays(_regions(fib=[(0, 100), (10, 90), (60, 80)]), rank=0)

    assert arrays.depth.tolist() == [0, 1, 2]
    assert arrays.parent.tolist() == [-1, 0, 1]


def test_zero_length_call_nests_inside_its_enclosing_region():
    """An instantaneous call must not read as already closed at its own start.

    Two calls can share a start timestamp on a coarse clock; the shorter one
    is the child, and a zero-length one is a child of everything open.
    """
    arrays = build_call_arrays(
        _regions(outer=[(0, 100)], tick=[(0, 0), (50, 50)]), rank=0
    )

    assert arrays.depth.tolist() == [0, 1, 1]
    assert arrays.parent.tolist() == [-1, 0, 0]
    assert arrays.exclusive_ns.tolist() == [100, 0, 0]


def test_equal_starts_put_the_longer_call_outside():
    arrays = build_call_arrays(_regions(outer=[(5, 100)], inner=[(5, 40)]), rank=0)

    assert arrays.names[int(arrays.region_index[0])] == "outer"
    assert arrays.parent.tolist() == [-1, 0]


def test_partial_overlap_is_rejected():
    """Neither containment nor disjointness: there is no stack to rebuild."""
    regions = _regions(short=[(0, 50)], long=[(10, 200)])

    with pytest.raises(NestingError, match="not properly nested"):
        build_call_arrays(regions, rank=0)


def test_partial_overlap_error_names_both_calls():
    regions = _regions(short=[(0, 50)], long=[(10, 200)])

    with pytest.raises(NestingError) as excinfo:
        build_call_arrays(regions, rank=0)

    message = str(excinfo.value)
    assert "'long' call 0 [10, 200] ns" in message
    assert "'short' call 0 [0, 50] ns" in message


def test_call_ending_before_it_starts_is_rejected():
    with pytest.raises(NestingError, match="ends before it starts"):
        build_call_arrays(_regions(broken=[(100, 10)]), rank=0)


def test_arrays_and_dicts_agree():
    """build_call_stack is a view of build_call_arrays, not a second answer."""
    regions = _regions(outer=[(0, 100)], inner=[(10, 40), (50, 90)], leaf=[(15, 20)])

    arrays = build_call_arrays(regions, rank=0)
    calls = build_call_stack(regions, rank=0)

    assert [call["call_id"] for call in calls] == list(range(len(arrays)))
    assert [call["depth"] for call in calls] == arrays.depth.tolist()
    assert [call["parent"] for call in calls] == [
        None if parent < 0 else parent for parent in arrays.parent.tolist()
    ]
    assert [call["exclusive_duration"] for call in calls] == pytest.approx(
        (arrays.exclusive_ns / 1e9).tolist()
    )
    assert [call["name"] for call in calls] == [
        arrays.names[row] for row in arrays.region_index.tolist()
    ]
    assert [call["call_path"] for call in calls] == [
        "outer",
        "outer > inner",
        "outer > inner > leaf",
        "outer > inner",
    ]


def test_exclusive_totals_match_the_per_call_values():
    """The write-side total and the read-side sum are the same integers."""
    snapshot = {
        "outer": (
            np.array([0, 200], dtype=np.int64),
            np.array([100, 300], dtype=np.int64),
        ),
        "inner": (np.array([10], dtype=np.int64), np.array([40], dtype=np.int64)),
    }

    totals = exclusive_totals_ns(snapshot)

    assert totals == {"outer": 170, "inner": 30}


def test_regions_without_data_for_the_rank_are_skipped():
    regions = _regions(here=[(0, 10)]) + _regions(rank=1, elsewhere=[(0, 10)])

    arrays = build_call_arrays(regions, rank=0)

    assert arrays.names == ["here"]
    assert len(arrays) == 1


def test_no_regions_yields_empty_arrays():
    arrays = build_call_arrays([], rank=0)

    assert len(arrays) == 0
    assert arrays.names == []
    assert build_call_stack([], rank=0) == []
