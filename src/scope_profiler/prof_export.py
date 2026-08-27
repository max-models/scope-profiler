"""Export merged HDF5 profiling data to the cProfile ``.prof`` (pstats) format.

A ``.prof`` file is nothing more than a :mod:`marshal` dump of the dict that
``cProfile.Profile.dump_stats`` writes: it maps a ``(filename, lineno,
funcname)`` key to ``(cc, nc, tt, ct, callers)``, where ``callers`` maps a
caller key to its own ``(cc, nc, tt, ct)`` sub-tuple. Writing that dict is
enough for :mod:`pstats`, ``snakeviz`` and friends to read the data.

Regions carry no call graph of their own, so the caller/callee relations are
reconstructed from timestamp containment
(:func:`~scope_profiler.call_stack.build_call_arrays`). The reconstruction is
consumed as columns rather than as one dict per call: a long run can export
millions of events while the pstats dictionary holds one entry per distinct
call path (or, on request, per region name).
"""

from __future__ import annotations

import marshal
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.call_stack import CallArrays, build_call_arrays
from scope_profiler.plotting_scripts import (
    _as_runs,
    _filename_slug,
    _normalize_ranks,
    _unique_labels,
)
from scope_profiler.results import ProfilingResults

# pstats keys are (filename, lineno, funcname) triples, and
# ``pstats.func_std_string`` renders a key starting with ("~", 0) as the bare
# function name - the convention cProfile uses for builtins. Regions have no
# source location, so borrowing it keeps them labelled "solve" rather than
# "profiling_data.h5:0(solve)" in snakeviz.
PSEUDO_FILENAME = "~"
PSEUDO_LINENO = 0

NS_PER_SECOND = 1e9


def _key(name: str) -> tuple[str, int, str]:
    """Build the pstats key identifying a region by name."""
    return (PSEUDO_FILENAME, PSEUDO_LINENO, name)


def _is_recursive(calls: CallArrays) -> np.ndarray:
    """Whether each call has an ancestor of the same region.

    pstats counts cumulative time for the outermost call of a recursive chain
    only, otherwise the same seconds are counted again at every level.

    Intervals are properly nested, so "contained in another call of this
    region" and "has an ancestor of this region" are the same statement. That
    turns the ancestor walk into a running maximum of end times within each
    region: a call is recursive exactly when an earlier call of its own
    region has not ended yet.
    """
    recursive = np.zeros(len(calls), dtype=bool)
    for row in range(len(calls.names)):
        mine = np.flatnonzero(calls.region_index == row)
        if mine.size < 2:
            continue
        ends = calls.end_ns[mine]
        # Largest end time seen among this region's earlier calls; the first
        # call has no predecessor, so it can never be recursive.
        enclosing = np.maximum.accumulate(ends)
        recursive[mine[1:]] = ends[1:] <= enclosing[:-1]
    return recursive


def build_pstats_dict(
    calls: CallArrays,
    root_name: str | None = None,
    call_paths: bool = True,
) -> dict[tuple[str, int, str], tuple]:
    """Turn reconstructed calls into a pstats-format statistics dict.

    Parameters
    ----------
    calls : CallArrays
        Calls as returned by
        :func:`~scope_profiler.call_stack.build_call_arrays`.
    root_name : str, optional
        When given, a synthetic frame of this name is added as the caller of
        every top-level call, so viewers that draw a single tree (snakeviz)
        show the whole run instead of only its largest region.
    call_paths : bool, optional
        Keep a separate pstats entry for each parent/child path, naming it
        ``"parent > child"``. This makes tree viewers such as SnakeViz show
        distinct uses of a region instead of merging every call named
        ``"child"``. Set this to ``False`` for a compact, name-aggregated
        export.

    Returns
    -------
    dict
        Maps ``(filename, lineno, funcname)`` to
        ``(cc, nc, tt, ct, callers)`` with times in seconds.
    """
    num_regions = len(calls.names)
    if len(calls) == 0 or num_regions == 0:
        return {}

    if call_paths:
        return _build_call_path_pstats_dict(calls, root_name=root_name)

    duration = (calls.end_ns - calls.start_ns) / NS_PER_SECOND
    # Exclusive time is already what pstats calls self time, and under the
    # nesting contract it cannot go negative, so nothing needs clamping.
    self_time = calls.exclusive_ns / NS_PER_SECOND
    primitive = ~_is_recursive(calls)
    cumulative = np.where(primitive, duration, 0.0)

    def per_region(weights=None):
        return np.bincount(calls.region_index, weights=weights, minlength=num_regions)

    region_primitive = per_region(primitive.astype(np.float64))
    region_calls = per_region()
    region_self = per_region(self_time)
    region_cumulative = per_region(cumulative)

    # [primitive_calls, total_calls, self_time, cumulative_time, callers]
    stats: dict[tuple[str, int, str], list] = {}
    for row, name in enumerate(calls.names):
        if not region_calls[row]:
            continue
        stats[_key(name)] = [
            int(region_primitive[row]),
            int(region_calls[row]),
            float(region_self[row]),
            float(region_cumulative[row]),
            {},
        ]

    # Caller attribution, one bucket per (region, calling region) pair. The
    # root bucket (column 0) collects the top-level calls.
    root_key = _key(root_name) if root_name is not None else None
    has_parent = calls.parent >= 0
    caller_row = np.where(has_parent, calls.region_index[calls.parent], -1) + 1
    pair = calls.region_index * (num_regions + 1) + caller_row
    size = num_regions * (num_regions + 1)

    def per_pair(weights=None):
        return np.bincount(pair, weights=weights, minlength=size)

    pair_primitive = per_pair(primitive.astype(np.float64))
    pair_calls = per_pair()
    pair_self = per_pair(self_time)
    pair_cumulative = per_pair(cumulative)

    for index in np.flatnonzero(pair_calls):
        row, caller_column = divmod(int(index), num_regions + 1)
        caller = (
            root_key if caller_column == 0 else _key(calls.names[caller_column - 1])
        )
        if caller is None:
            continue
        stats[_key(calls.names[row])][4][caller] = [
            int(pair_primitive[index]),
            int(pair_calls[index]),
            float(pair_self[index]),
            float(pair_cumulative[index]),
        ]

    if root_key is not None:
        stats[root_key] = [1, 1, 0.0, float(duration[~has_parent].sum()), {}]

    return {
        key: (
            value[0],
            value[1],
            value[2],
            value[3],
            {caller: tuple(times) for caller, times in value[4].items()},
        )
        for key, value in stats.items()
    }


def _call_path_ids(calls: CallArrays) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Assign a compact identity to each distinct root-to-call path.

    Calls are start-sorted with parents before children, so a child's path is
    available when it is visited.  The mapping therefore needs one linear
    pass and stores one entry per distinct path rather than one Python object
    per recorded call.
    """
    path_ids = np.empty(len(calls), dtype=np.int64)
    path_names: list[str] = []
    path_parents: list[int] = []
    identities: dict[tuple[int, int], int] = {}

    for call_id, region_row in enumerate(calls.region_index):
        parent_call = int(calls.parent[call_id])
        parent_path = -1 if parent_call < 0 else int(path_ids[parent_call])
        identity = (parent_path, int(region_row))
        path_id = identities.get(identity)
        if path_id is None:
            path_id = len(path_names)
            identities[identity] = path_id
            name = calls.names[int(region_row)]
            path_names.append(
                name if parent_path < 0 else f"{path_names[parent_path]} > {name}"
            )
            path_parents.append(parent_path)
        path_ids[call_id] = path_id

    return path_ids, path_names, np.asarray(path_parents, dtype=np.int64)


def _build_call_path_pstats_dict(
    calls: CallArrays, root_name: str | None
) -> dict[tuple[str, int, str], tuple]:
    """Build a pstats tree whose nodes are reconstructed call paths."""
    path_ids, path_names, path_parents = _call_path_ids(calls)
    path_count = len(path_names)
    duration = (calls.end_ns - calls.start_ns) / NS_PER_SECOND
    self_time = calls.exclusive_ns / NS_PER_SECOND

    def per_path(weights=None):
        return np.bincount(path_ids, weights=weights, minlength=path_count)

    counts = per_path()
    total_self = per_path(self_time)
    total_cumulative = per_path(duration)
    keys = [_key(name) for name in path_names]
    stats: dict[tuple[str, int, str], list] = {
        keys[path_id]: [
            int(counts[path_id]),
            int(counts[path_id]),
            float(total_self[path_id]),
            float(total_cumulative[path_id]),
            {},
        ]
        for path_id in range(path_count)
        if counts[path_id]
    }

    root_key = _key(root_name) if root_name is not None else None
    for path_id, key in enumerate(keys):
        parent_path = int(path_parents[path_id])
        caller = root_key if parent_path < 0 else keys[parent_path]
        if caller is not None:
            stats[key][4][caller] = (
                int(counts[path_id]),
                int(counts[path_id]),
                float(total_self[path_id]),
                float(total_cumulative[path_id]),
            )

    if root_key is not None:
        root_duration = duration[calls.parent < 0].sum()
        stats[root_key] = [1, 1, 0.0, float(root_duration), {}]

    return {
        key: (value[0], value[1], value[2], value[3], value[4])
        for key, value in stats.items()
    }


def write_prof_file(
    filepath: str | Path, stats: dict[tuple[str, int, str], tuple]
) -> Path:
    """Marshal a pstats dict to ``filepath``, as ``cProfile`` would."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        marshal.dump(stats, f)
    return output_path


def export_prof(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    filepath: str | Path,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    call_paths: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Write per-rank ``.prof`` files readable by ``pstats``/``snakeviz``.

    Parameters
    ----------
    profiling_data : ProfilingResults | Sequence[ProfilingResults]
        The run(s) to export: file runs, in-memory results from
        ``ProfileManager.finalize(return_results=True)``, or a mix.
    filepath : str | Path
        Base output path, e.g. ``figures/profile.prof``. A ``_rank<N>`` suffix
        is appended per rank (and the input file's stem too, when more than one
        file is exported), since ``.prof`` has no notion of ranks or runs.
    ranks : list[int] | int, optional
        Ranks to export (default: rank 0 only).
    include, exclude : list[str] | str, optional
        Region name filters, as for the plotting functions.
    call_paths : bool, optional
        Preserve each reconstructed parent/child path as a separate node in
        the exported pstats tree.  This makes SnakeViz distinguish same-named
        regions called below different parents. Set to ``False`` for a
        compact, name-aggregated export.

    Returns
    -------
    list[Path]
        The files written, in the order they were written.
    """
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 writes the files.
        return []

    normalized_ranks = _normalize_ranks(ranks) if ranks is not None else [0]

    labels = _unique_labels([run.display_label for run in runs])

    prepared = []
    for label, run in zip(labels, runs):
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        for rank in normalized_ranks:
            if rank < 0 or rank >= run.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = build_call_arrays(regions, rank)
            if len(calls):
                prepared.append((label, rank, calls))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    base_path = Path(filepath)
    suffix = base_path.suffix or ".prof"
    multiple_files = len(runs) > 1

    written = []
    for label, rank, calls in prepared:
        parts = [base_path.stem]
        if multiple_files:
            parts.append(_filename_slug(label))
        parts.append(f"rank{rank}")
        out_path = base_path.with_name("_".join(parts) + suffix)
        stats = build_pstats_dict(
            calls, root_name=f"<{label} rank {rank}>", call_paths=call_paths
        )
        written.append(write_prof_file(out_path, stats))
        if verbose:
            print(f"Wrote {out_path} (view with: snakeviz {out_path})")

    return written
