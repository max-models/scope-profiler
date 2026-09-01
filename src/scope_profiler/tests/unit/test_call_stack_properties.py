"""Property-based checks on the nesting reconstruction.

:func:`~scope_profiler.call_stack.build_call_arrays` infers the call tree from
timestamps alone -- no stack is recorded at run time -- using a sort, a
``searchsorted`` and a per-level pass. The example-based tests next door pin
the cases that were reasoned about; these generate call trees instead, and
check the properties the vectorized reconstruction is supposed to guarantee
for *any* properly nested run.

The generator builds a forest by construction (each child strictly inside its
parent, siblings disjoint), so the true depth and parent of every call are
known before the reconstruction ever sees them.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from scope_profiler.call_stack import (
    NestingError,
    build_call_arrays,
    exclusive_totals_ns,
)
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import Region

# Deep recursion in the generator is slow to shrink and adds nothing: the
# reconstruction treats every level alike.
MAX_DEPTH = 4
MAX_CHILDREN = 3


@st.composite
def call_forest(draw):
    """A properly nested forest of calls, with the true tree recorded.

    Returns a list of ``(start, end, parent, depth)`` in creation order, where
    ``parent`` indexes back into the same list (-1 for a root).
    """
    calls: list[tuple[int, int, int, int]] = []

    def build(low: int, high: int, parent: int, depth: int) -> None:
        # Children live strictly inside (low, high), which keeps containment
        # unambiguous: no child can tie with its parent on both endpoints.
        low, high = low + 1, high - 1
        span = high - low
        if depth > MAX_DEPTH or span < 2:
            return
        count = draw(st.integers(min_value=0, max_value=min(MAX_CHILDREN, span // 2)))
        if count == 0:
            return
        width = span // count
        for slot in range(count):
            slot_low = low + slot * width
            slot_high = slot_low + width
            start = draw(st.integers(min_value=slot_low, max_value=slot_high - 1))
            end = draw(st.integers(min_value=start + 1, max_value=slot_high))
            index = len(calls)
            calls.append((start, end, parent, depth))
            build(start, end, index, depth + 1)

    build(0, draw(st.integers(min_value=8, max_value=2048)), -1, 0)
    assume(calls)
    return calls


def _regions_from(calls, names, rank=0):
    """Group generated calls into MPIRegions by name, as a result set would."""
    by_name: dict[str, list[tuple[int, int]]] = {}
    for (start, end, _, _), name in zip(calls, names):
        by_name.setdefault(name, []).append((start, end))
    return [
        MPIRegion(
            name=name,
            regions={
                rank: Region(
                    np.array([s for s, _ in intervals], dtype=np.int64),
                    np.array([e for _, e in intervals], dtype=np.int64),
                )
            },
        )
        for name, intervals in by_name.items()
    ]


def _names_for(calls, draw_names):
    """Assign region names to calls; several calls may share a name."""
    return [draw_names[index % len(draw_names)] for index in range(len(calls))]


def _reconstructed_tree(arrays):
    """Map the reconstruction back onto (start, end) keys for comparison."""
    by_interval = {}
    for index in range(arrays.start_ns.size):
        key = (int(arrays.start_ns[index]), int(arrays.end_ns[index]))
        parent = int(arrays.parent[index])
        parent_key = (
            None
            if parent < 0
            else (int(arrays.start_ns[parent]), int(arrays.end_ns[parent]))
        )
        by_interval[key] = (int(arrays.depth[index]), parent_key)
    return by_interval


SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@given(calls=call_forest(), name_count=st.integers(min_value=1, max_value=4))
@SETTINGS
def test_depth_and_parent_match_the_generated_tree(calls, name_count):
    """Every call is put back at the depth and under the parent it was built with."""
    names = _names_for(calls, [f"r{i}" for i in range(name_count)])
    arrays = build_call_arrays(_regions_from(calls, names), rank=0)

    tree = _reconstructed_tree(arrays)
    # Intervals are unique by construction, so the interval-keyed view
    # loses nothing and every generated call has to appear in it.
    assert len(tree) == len(calls)
    for index, (start, end, parent, depth) in enumerate(calls):
        found_depth, found_parent = tree[(start, end)]
        assert found_depth == depth
        expected_parent = None if parent < 0 else (calls[parent][0], calls[parent][1])
        assert found_parent == expected_parent


@given(calls=call_forest(), name_count=st.integers(min_value=1, max_value=4))
@SETTINGS
def test_exclusive_time_is_never_negative_and_never_exceeds_inclusive(
    calls, name_count
):
    names = _names_for(calls, [f"r{i}" for i in range(name_count)])
    arrays = build_call_arrays(_regions_from(calls, names), rank=0)

    inclusive = arrays.end_ns - arrays.start_ns
    assert (arrays.exclusive_ns >= 0).all()
    assert (arrays.exclusive_ns <= inclusive).all()


@given(calls=call_forest(), name_count=st.integers(min_value=1, max_value=4))
@SETTINGS
def test_exclusive_time_sums_to_the_wall_clock_union_of_the_roots(calls, name_count):
    """No time is counted twice and none is lost.

    Roots are disjoint by construction, so the total exclusive time of every
    call must add up to exactly the time the roots span.
    """
    names = _names_for(calls, [f"r{i}" for i in range(name_count)])
    arrays = build_call_arrays(_regions_from(calls, names), rank=0)

    roots = arrays.parent < 0
    root_span = int((arrays.end_ns[roots] - arrays.start_ns[roots]).sum())
    assert int(arrays.exclusive_ns.sum()) == root_span


@given(calls=call_forest(), name_count=st.integers(min_value=1, max_value=4))
@SETTINGS
def test_a_childs_span_lies_inside_its_parents(calls, name_count):
    names = _names_for(calls, [f"r{i}" for i in range(name_count)])
    arrays = build_call_arrays(_regions_from(calls, names), rank=0)

    nested = np.flatnonzero(arrays.parent >= 0)
    parents = arrays.parent[nested]
    assert (arrays.start_ns[parents] <= arrays.start_ns[nested]).all()
    assert (arrays.end_ns[parents] >= arrays.end_ns[nested]).all()
    assert (arrays.depth[nested] == arrays.depth[parents] + 1).all()


@given(calls=call_forest(), name_count=st.integers(min_value=1, max_value=4))
@SETTINGS
def test_per_region_totals_agree_with_the_per_call_values(calls, name_count):
    """``exclusive_totals_ns`` is the same number, summed a different way."""
    names = _names_for(calls, [f"r{i}" for i in range(name_count)])
    regions = _regions_from(calls, names)
    arrays = build_call_arrays(regions, rank=0)

    from_calls: dict[str, int] = {name: 0 for name in arrays.names}
    for index in range(arrays.start_ns.size):
        from_calls[arrays.names[arrays.region_index[index]]] += int(
            arrays.exclusive_ns[index]
        )

    # exclusive_totals_ns consumes the finalize-time snapshot shape:
    # {region name: (start array, end array, ...)}.
    snapshot = {
        region.name: (
            region.regions[0].start_times_ns,
            region.regions[0].end_times_ns,
        )
        for region in regions
    }
    assert exclusive_totals_ns(snapshot, rank=0) == from_calls


@given(calls=call_forest(), name_count=st.integers(min_value=1, max_value=4))
@SETTINGS
def test_reconstruction_is_independent_of_the_order_regions_are_given_in(
    calls, name_count
):
    names = _names_for(calls, [f"r{i}" for i in range(name_count)])
    regions = _regions_from(calls, names)
    forward = build_call_arrays(regions, rank=0)
    backward = build_call_arrays(list(reversed(regions)), rank=0)

    assert forward.depth.tolist() == backward.depth.tolist()
    assert forward.start_ns.tolist() == backward.start_ns.tolist()
    assert forward.exclusive_ns.tolist() == backward.exclusive_ns.tolist()


@given(
    start=st.integers(min_value=0, max_value=1000),
    length=st.integers(min_value=2, max_value=1000),
    shift=st.integers(min_value=1, max_value=999),
)
@settings(max_examples=100, deadline=None)
def test_partially_overlapping_calls_are_always_rejected(start, length, shift):
    """Two intervals that cross without containment can never be nested."""
    assume(shift < length)
    first = (start, start + length)
    second = (start + shift, start + length + shift)

    regions = _regions_from(
        [(first[0], first[1], -1, 0), (second[0], second[1], -1, 0)], ["a", "b"]
    )
    with pytest.raises(NestingError):
        build_call_arrays(regions, rank=0)


@given(
    start=st.integers(min_value=0, max_value=1000),
    length=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=50, deadline=None)
def test_a_call_that_ends_before_it_starts_is_always_rejected(start, length):
    regions = _regions_from([(start + length, start, -1, 0)], ["a"])
    with pytest.raises(NestingError, match="ends before it starts"):
        build_call_arrays(regions, rank=0)
