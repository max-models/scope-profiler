"""Export merged HDF5 profiling data to the cProfile ``.prof`` (pstats) format.

A ``.prof`` file is nothing more than a :mod:`marshal` dump of the dict that
``cProfile.Profile.dump_stats`` writes: it maps a ``(filename, lineno,
funcname)`` key to ``(cc, nc, tt, ct, callers)``, where ``callers`` maps a
caller key to its own ``(cc, nc, tt, ct)`` sub-tuple. Writing that dict is
enough for :mod:`pstats`, ``snakeviz`` and friends to read the data.

Regions carry no call graph of their own, so the caller/callee relations are
reconstructed from timestamp containment - the same reconstruction the flame
chart uses (:func:`~scope_profiler.call_stack.build_call_stack`).
"""

from __future__ import annotations

import marshal
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from scope_profiler.call_stack import build_call_stack
from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import (
    _as_readers,
    _normalize_ranks,
    _unique_labels,
)

# pstats keys are (filename, lineno, funcname) triples, and
# ``pstats.func_std_string`` renders a key starting with ("~", 0) as the bare
# function name - the convention cProfile uses for builtins. Regions have no
# source location, so borrowing it keeps them labelled "solve" rather than
# "profiling_data.h5:0(solve)" in snakeviz.
PSEUDO_FILENAME = "~"
PSEUDO_LINENO = 0


def _key(name: str) -> tuple[str, int, str]:
    """Build the pstats key identifying a region by name."""
    return (PSEUDO_FILENAME, PSEUDO_LINENO, name)


def _ancestor_names(calls: list[dict], index: int) -> set[str]:
    """Names of all calls enclosing ``calls[index]``."""
    names = set()
    parent = calls[index]["parent"]
    while parent is not None:
        names.add(calls[parent]["name"])
        parent = calls[parent]["parent"]
    return names


def build_pstats_dict(
    calls: list[dict], root_name: str | None = None
) -> dict[tuple[str, int, str], tuple]:
    """Turn reconstructed calls into a pstats-format statistics dict.

    Parameters
    ----------
    calls : list[dict]
        Calls as returned by
        :func:`~scope_profiler.call_stack.build_call_stack`:
        each entry has ``name``, ``start`` and ``end`` in seconds, and
        ``parent`` (an index into this list, or ``None`` for a top-level call).
    root_name : str, optional
        When given, a synthetic frame of this name is added as the caller of
        every top-level call, so viewers that draw a single tree (snakeviz)
        show the whole run instead of only its largest region.

    Returns
    -------
    dict
        Maps ``(filename, lineno, funcname)`` to
        ``(cc, nc, tt, ct, callers)`` with times in seconds.
    """
    child_time: dict[int, float] = defaultdict(float)
    for call in calls:
        if call["parent"] is not None:
            child_time[call["parent"]] += call["end"] - call["start"]

    root_key = _key(root_name) if root_name is not None else None
    # [primitive_calls, total_calls, self_time, cumulative_time, callers]
    stats: dict[tuple[str, int, str], list] = {}
    root_cumulative = 0.0

    for index, call in enumerate(calls):
        duration = call["end"] - call["start"]
        # Regions that only partially overlap their enclosing region are
        # reconstructed as children of it, which can push the parent's self
        # time below zero; pstats has no meaning for a negative tt.
        self_time = max(duration - child_time[index], 0.0)
        # pstats counts cumulative time for the outermost call of a recursive
        # chain only, otherwise the same seconds are counted at every level.
        primitive = call["name"] not in _ancestor_names(calls, index)
        cumulative = duration if primitive else 0.0

        entry = stats.setdefault(_key(call["name"]), [0, 0, 0.0, 0.0, {}])
        entry[0] += int(primitive)
        entry[1] += 1
        entry[2] += self_time
        entry[3] += cumulative

        if call["parent"] is not None:
            caller_key = _key(calls[call["parent"]]["name"])
        else:
            caller_key = root_key
            root_cumulative += duration
        if caller_key is not None:
            caller = entry[4].setdefault(caller_key, [0, 0, 0.0, 0.0])
            caller[0] += int(primitive)
            caller[1] += 1
            caller[2] += self_time
            caller[3] += cumulative

    if root_key is not None and calls:
        stats[root_key] = [1, 1, 0.0, root_cumulative, {}]

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
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    filepath: str | Path,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Write per-rank ``.prof`` files readable by ``pstats``/``snakeviz``.

    Parameters
    ----------
    profiling_data : ProfilingH5Reader | Sequence[ProfilingH5Reader]
        Reader(s) for the merged HDF5 file(s) to export.
    filepath : str | Path
        Base output path, e.g. ``figures/profile.prof``. A ``_rank<N>`` suffix
        is appended per rank (and the input file's stem too, when more than one
        file is exported), since ``.prof`` has no notion of ranks or runs.
    ranks : list[int] | int, optional
        Ranks to export (default: rank 0 only).
    include, exclude : list[str] | str, optional
        Region name filters, as for the plotting functions.

    Returns
    -------
    list[Path]
        The files written, in the order they were written.
    """
    readers = _as_readers(profiling_data)
    if not readers:
        raise ValueError("No profiling data provided.")

    normalized_ranks = _normalize_ranks(ranks) if ranks is not None else [0]

    labels = _unique_labels([reader.file_path.stem for reader in readers])

    prepared = []
    for label, reader in zip(labels, readers):
        regions = reader.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        for rank in normalized_ranks:
            if rank < 0 or rank >= reader.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = build_call_stack(regions, rank)
            if calls:
                prepared.append((label, rank, calls))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    base_path = Path(filepath)
    suffix = base_path.suffix or ".prof"
    multiple_files = len(readers) > 1

    written = []
    for label, rank, calls in prepared:
        parts = [base_path.stem]
        if multiple_files:
            parts.append(label)
        parts.append(f"rank{rank}")
        out_path = base_path.with_name("_".join(parts) + suffix)
        stats = build_pstats_dict(calls, root_name=f"<{label} rank {rank}>")
        written.append(write_prof_file(out_path, stats))
        if verbose:
            print(f"Wrote {out_path} (view with: snakeviz {out_path})")

    return written
