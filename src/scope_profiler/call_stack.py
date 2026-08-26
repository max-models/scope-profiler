"""Reconstruct a call stack from the recorded region intervals.

Regions carry no call graph of their own: each call is just a (start, end)
pair. Nesting is therefore recovered from timestamp containment - a call that
starts while another is still open is its child. The flame chart, the ``.prof``
export and the speedscope export all build on this, and so can user code that
wants to draw its own nested view.

**Intervals must be properly nested.** Two calls recorded on one rank either
nest completely or do not overlap at all; a call that starts inside another
and ends after it is rejected rather than reconstructed. That is what a stack
of ``with`` blocks always produces, and assuming it is what makes
:func:`build_call_arrays` a handful of vectorized passes instead of a
per-call Python loop - the reconstruction is O(events) in numpy rather than
O(events) in dicts, which is the difference between 2 s and 160 s on a run
with ten million events. Manual ``sp_begin``/``sp_end`` pairs in native code
are the only realistic way to violate it; see :class:`NestingError`.
"""

from typing import Any, Iterable, List, NamedTuple

import numpy as np


class NestingError(ValueError):
    """Raised when recorded intervals are not properly nested.

    A rank's calls must form a forest: any two intervals are either disjoint
    or one contains the other. Partial overlap has no call stack to
    reconstruct, and every consumer of this module (flame chart, ``.prof``,
    speedscope, exclusive time) would have to invent one.

    In practice this means a mismatched ``sp_begin``/``sp_end`` pair in C or
    Fortran, or region objects driven from several threads, which the region
    buffers do not support anyway.
    """


class CallArrays(NamedTuple):
    """One rank's reconstructed nesting, in start-sorted (``call_id``) order.

    Every array is indexed by ``call_id``, so ``parent`` indexes straight back
    into the same arrays. ``region_index`` and ``call_index`` point at where a
    call came from: region ``names[region_index[i]]``, call number
    ``call_index[i]`` within that region's buffers.
    """

    names: List[str]
    region_index: np.ndarray
    call_index: np.ndarray
    start_ns: np.ndarray
    end_ns: np.ndarray
    depth: np.ndarray
    parent: np.ndarray
    exclusive_ns: np.ndarray

    def __len__(self) -> int:
        return int(self.start_ns.size)


def build_call_arrays(regions: Iterable, rank: int) -> CallArrays:
    """Reconstruct one rank's nesting as numpy arrays.

    The workhorse behind :func:`build_call_stack`, and what anything
    processing a whole run should call: it never builds a dict per call, so
    it stays usable on the tens of millions of events a long simulation
    produces.

    Parameters
    ----------
    regions : iterable of MPIRegion
        Regions to include, e.g. ``results.get_regions()``. Regions with no
        data for ``rank`` are skipped.
    rank : int
        Rank whose calls to reconstruct.

    Returns
    -------
    CallArrays

    Raises
    ------
    NestingError
        If any interval ends before it starts, or two intervals overlap
        without one containing the other.
    """
    names: List[str] = []
    starts: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    region_index: List[np.ndarray] = []
    call_index: List[np.ndarray] = []
    for region in regions:
        if rank not in region.regions:
            continue
        region_data = region.regions[rank]
        region_starts = np.asarray(region_data.start_times_ns, dtype=np.int64)
        region_ends = np.asarray(region_data.end_times_ns, dtype=np.int64)
        names.append(region.name)
        starts.append(region_starts)
        ends.append(region_ends)
        region_index.append(np.full(region_starts.size, len(names) - 1, dtype=np.int64))
        call_index.append(np.arange(region_starts.size, dtype=np.int64))

    if not starts:
        empty = np.empty(0, dtype=np.int64)
        return CallArrays(
            names,
            empty,
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
        )

    start_ns = np.concatenate(starts)
    end_ns = np.concatenate(ends)
    if np.any(end_ns < start_ns):
        raise NestingError("a recorded call ends before it starts")

    # Parents first: earlier start wins, and on a tie the longer interval,
    # which is the one that must enclose the other.
    order = np.lexsort((-end_ns, start_ns))
    start_ns = start_ns[order]
    end_ns = end_ns[order]
    region_of = np.concatenate(region_index)[order]
    call_of = np.concatenate(call_index)[order]
    n = start_ns.size

    depth = _depths(start_ns, end_ns)
    parent = _parents(depth)

    # The one check that proves the assumption: every child lies inside the
    # parent it was assigned. Its start does so by construction (the sort),
    # so only the end can escape.
    nested = np.flatnonzero(parent >= 0)
    escaped = nested[end_ns[parent[nested]] < end_ns[nested]]
    if escaped.size:
        _raise_overlap(escaped[0], names, region_of, call_of, start_ns, end_ns, parent)

    duration = end_ns - start_ns
    covered = np.zeros(n, dtype=np.int64)
    # Properly nested siblings are disjoint, so the union of a call's direct
    # children collapses to their plain sum.
    np.add.at(covered, parent[nested], duration[nested])
    exclusive_ns = duration - covered

    return CallArrays(
        names=names,
        region_index=region_of,
        call_index=call_of,
        start_ns=start_ns,
        end_ns=end_ns,
        depth=depth,
        parent=parent,
        exclusive_ns=exclusive_ns,
    )


def _raise_overlap(call, names, region_of, call_of, start_ns, end_ns, parent) -> None:
    """Report one offending pair by name, so the user can find the region."""
    enclosing = int(parent[call])

    def label(index: int) -> str:
        return (
            f"{names[int(region_of[index])]!r} call {int(call_of[index])} "
            f"[{int(start_ns[index])}, {int(end_ns[index])}] ns"
        )

    raise NestingError(
        "recorded calls are not properly nested: "
        f"{label(call)} starts inside {label(enclosing)} but ends after it. "
        "Regions must nest completely or not overlap at all - check for a "
        "missing or misordered sp_end / __exit__."
    )


def _depths(start_ns: np.ndarray, end_ns: np.ndarray) -> np.ndarray:
    """Stack depth of every call, from sorted intervals.

    A call's depth is the number of calls still open when it starts. Sorted
    by start, every earlier call has started, and a properly nested one has
    closed exactly when its end is at or before this start - so the depth is
    the sort position minus the number of such ends, with no stack to walk.

    Timestamps are doubled and zero-length intervals extended by one tick
    first. A zero-length call would otherwise count as closed at its own
    start, which underestimates its depth and that of any sibling sharing the
    start; the scale factor keeps that adjustment exact in integers.
    """
    n = start_ns.size
    scaled_start = start_ns * 2
    scaled_end = end_ns * 2
    scaled_end[scaled_end == scaled_start] += 1
    closed = np.searchsorted(np.sort(scaled_end), scaled_start, side="right")
    depth = np.arange(n, dtype=np.int64)
    depth -= closed
    if n and depth.min() < 0:
        raise NestingError("recorded calls are not properly nested")
    return depth


def _parents(depth: np.ndarray) -> np.ndarray:
    """Each call's parent id: the nearest preceding call one level shallower.

    Loops over stack depths, not calls - a call stack is a few dozen levels
    deep at worst, so this is a handful of vectorized passes.
    """
    n = depth.size
    parent = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return parent
    index = np.arange(n, dtype=np.int64)
    candidate = np.empty(n, dtype=np.int64)
    for level in range(1, int(depth.max()) + 1):
        np.copyto(candidate, index)
        candidate[depth != level - 1] = -1
        np.maximum.accumulate(candidate, out=candidate)
        np.copyto(parent, candidate, where=depth == level)
    return parent


def build_call_stack(regions: Iterable, rank: int, origin: float = 0.0) -> List[dict]:
    """Reconstruct per-call nesting for one rank, one dict per call.

    A convenience wrapper over :func:`build_call_arrays` for callers that
    want to iterate calls rather than columns. It allocates a dict per call
    (~700 bytes), so prefer the arrays for anything that has to scale with a
    long run.

    Parameters
    ----------
    regions : iterable of MPIRegion
        Regions to include, e.g. ``results.get_regions()``.
    rank : int
        Rank whose calls to reconstruct. Regions with no data for this rank
        are skipped.
    origin : float, optional
        Seconds subtracted from every timestamp, so passing
        ``results.minimum_start_time`` yields a timeline starting at zero
        (default: 0.0, i.e. raw timestamps).

    Returns
    -------
    list of dict
        One dict per call, ordered by start time (parents before children),
        with keys ``call_id``, ``name``, ``start``, ``end``, ``duration`` (the
        inclusive duration), ``inclusive_duration``, ``exclusive_duration``
        (seconds), ``depth`` (0 for a top-level call) and ``parent`` (the
        ``call_id`` of the enclosing call in this same list, or None). A
        ``color`` key carries whatever the plotting code assigned to the
        region. ``call_id`` is stable for the returned list and is unique
        within this rank/stack reconstruction.

    Raises
    ------
    NestingError
        If the intervals are not properly nested.

    Notes
    -----
    Calls are indexed by position in the returned list rather than by name,
    because a region called more than once - or recursively - contributes
    several calls under one name.
    """
    from scope_profiler.region import NS_PER_SECOND

    regions = list(regions)
    arrays = build_call_arrays(regions, rank)
    colors = {region.name: getattr(region, "color", None) for region in regions}

    starts = arrays.start_ns / NS_PER_SECOND - origin
    ends = arrays.end_ns / NS_PER_SECOND - origin
    durations = (arrays.end_ns - arrays.start_ns) / NS_PER_SECOND
    exclusive = arrays.exclusive_ns / NS_PER_SECOND

    calls: List[dict] = []
    for call_id, region_row in enumerate(arrays.region_index.tolist()):
        name = arrays.names[region_row]
        parent = int(arrays.parent[call_id])
        calls.append(
            {
                "call_id": call_id,
                "name": name,
                "call_index": int(arrays.call_index[call_id]),
                "start": float(starts[call_id]),
                "end": float(ends[call_id]),
                "duration": float(durations[call_id]),
                "inclusive_duration": float(durations[call_id]),
                "exclusive_duration": float(exclusive[call_id]),
                "depth": int(arrays.depth[call_id]),
                "parent": None if parent < 0 else parent,
                "color": colors[name],
            }
        )
    return calls


def call_stack_roots(calls: List[dict]) -> List[int]:
    """Indices of the top-level calls in a :func:`build_call_stack` result."""
    return [index for index, call in enumerate(calls) if call["parent"] is None]


def call_stack_children(calls: List[dict]) -> List[List[int]]:
    """Child indices per call, for walking a :func:`build_call_stack` result.

    Returns a list parallel to ``calls``: entry *i* holds the indices of the
    calls directly nested inside call *i*, in start-time order.
    """
    children: List[List[Any]] = [[] for _ in calls]
    for index, call in enumerate(calls):
        parent = call["parent"]
        if parent is not None:
            children[parent].append(index)
    return children


def exclusive_totals_ns(regions: dict, rank: int = 0) -> dict:
    """Total exclusive time per region, in nanoseconds, for one rank's regions.

    The write-side counterpart of
    :meth:`ProfilingResults._populate_exclusive_durations
    <scope_profiler.results.ProfilingResults._populate_exclusive_durations>`.
    Both sum the same int64 nanosecond values produced by
    :func:`build_call_arrays`, so a total stored in the output file is
    bit-identical to the one a reader reconstructs from the events.

    ``finalize()`` calls this because a rank already holds its whole region
    set in memory, which is exactly the set exclusive time is defined against
    -- and reconstructing the nesting is by far the most expensive part of
    reading a run back.

    Parameters
    ----------
    regions : dict
        Region name -> ``(start_times, end_times, ...)`` int64 arrays in
        nanoseconds, as :meth:`ProfileManager._snapshot_regions` returns them.
    rank : int, optional
        Rank to label the temporary regions with; irrelevant to the result.

    Returns
    -------
    dict
        Region name -> total exclusive nanoseconds, for every named region.
    """
    arrays = build_call_arrays(regions_from_snapshot(regions, rank), rank)
    totals = np.zeros(len(arrays.names), dtype=np.int64)
    np.add.at(totals, arrays.region_index, arrays.exclusive_ns)
    by_name = dict(zip(arrays.names, totals.tolist()))
    return {name: by_name.get(name, 0) for name in regions}


def regions_from_snapshot(regions: dict, rank: int) -> list:
    """Present a ``ProfileManager._snapshot_regions()`` dict as MPIRegions.

    The finalize path holds raw ``(starts, ends, ...)`` arrays rather than
    result objects; this is the adapter both it and
    :func:`exclusive_totals_ns` use to reach :func:`build_call_arrays`.
    """
    from scope_profiler.mpi_region import MPIRegion
    from scope_profiler.region import Region

    return [
        MPIRegion(name=name, regions={rank: Region(arrays[0], arrays[1])})
        for name, arrays in regions.items()
    ]
