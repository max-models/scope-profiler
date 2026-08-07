"""Reconstruct a call stack from the recorded region intervals.

Regions carry no call graph of their own: each call is just a (start, end)
pair. Nesting is therefore recovered from timestamp containment - a call that
starts while another is still open is its child. The flame chart, the ``.prof``
export and the speedscope export all build on this, and so can user code that
wants to draw its own nested view.
"""

from typing import Any, Iterable, List


def build_call_stack(regions: Iterable, rank: int, origin: float = 0.0) -> List[dict]:
    """Reconstruct per-call nesting for one rank from region intervals.

    Parameters
    ----------
    regions : iterable of MPIRegion
        Regions to include, e.g. ``reader.get_regions()``.
    rank : int
        Rank whose calls to reconstruct. Regions with no data for this rank
        are skipped.
    origin : float, optional
        Seconds subtracted from every timestamp, so passing
        ``reader.minimum_start_time`` yields a timeline starting at zero
        (default: 0.0, i.e. raw timestamps).

    Returns
    -------
    list of dict
        One dict per call, ordered by start time (parents before children),
        with keys ``name``, ``start``, ``end``, ``duration`` (seconds),
        ``depth`` (0 for a top-level call) and ``parent`` (the index of the
        enclosing call in this same list, or None). A ``color`` key carries
        whatever the plotting code assigned to the region.

    Notes
    -----
    Calls are indexed by position in the returned list rather than by name,
    because a region called more than once - or recursively - contributes
    several calls under one name.
    """
    calls: List[dict] = []
    for region in regions:
        if rank not in region.regions:
            continue
        region_data = region.regions[rank]
        for start, end in zip(region_data.start_times, region_data.end_times):
            start = float(start) - origin
            end = float(end) - origin
            calls.append(
                {
                    "name": region.name,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "color": getattr(region, "color", None),
                }
            )

    calls.sort(key=lambda call: (call["start"], -call["end"]))

    open_stack: List[int] = []
    for index, call in enumerate(calls):
        while open_stack and calls[open_stack[-1]]["end"] <= call["start"]:
            open_stack.pop()
        call["depth"] = len(open_stack)
        call["parent"] = open_stack[-1] if open_stack else None
        open_stack.append(index)

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
