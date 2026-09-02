"""Export merged HDF5 profiling data to the speedscope JSON format.

A speedscope file is a single JSON document holding a table of frames and one
or more profiles referring to it. The profiles written here are *evented*: each
call contributes an open (``O``) and a close (``C``) event carrying a timestamp,
which preserves every individual call rather than the aggregate a ``.prof``
file keeps. Drop the file on https://www.speedscope.app to view it.

Regions carry no call graph of their own, so the caller/callee relations are
reconstructed from timestamp containment - the same reconstruction the flame
chart and the ``.prof`` export use
(:func:`~scope_profiler.call_stack.build_call_arrays`).

An evented profile is two events per call, so the output is inherently
proportional to the run. The reconstruction feeding it is not: it arrives as
columns, and only the events themselves are ever built as dicts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from scope_profiler.call_stack import CallArrays, build_call_arrays, split_by_lane
from scope_profiler.plotting_scripts import (
    _as_runs,
    _filename_slug,
    _normalize_ranks,
    _unique_labels,
)
from scope_profiler.results import ProfilingResults

SCHEMA_URL = "https://www.speedscope.app/file-format-schema.json"

# Timestamps are seconds throughout the run API, so the profiles say so
# rather than converting back to the nanoseconds the HDF5 files store.
TIME_UNIT = "seconds"

NS_PER_SECOND = 1e9


def _exporter_name() -> str:
    """Identify this package (and version) as the producer of the file."""
    from scope_profiler import __version__

    return f"scope-profiler@{__version__}"


def build_speedscope_profile(
    calls: CallArrays,
    name: str,
    frame_indices: dict[str, int],
    frames: list[dict],
    origin: float = 0.0,
) -> dict:
    """Build one evented speedscope profile from reconstructed calls.

    Parameters
    ----------
    calls : CallArrays
        Calls as returned by
        :func:`~scope_profiler.call_stack.build_call_arrays`. Their intervals
        are properly nested, which is exactly what an evented profile needs:
        a frame can only close while it is on top of the stack.
    name : str
        Name of the profile, shown in speedscope's profile selector.
    frame_indices, frames : dict, list
        The document-wide frame table, extended in place: speedscope shares one
        table across all profiles in a file, so profiles of the same run refer
        to the same frames.
    origin : float, optional
        Subtracted from every timestamp. Timestamps come from
        ``perf_counter_ns`` and are large in absolute terms; rebasing them on a
        common origin keeps the values readable (and precise) without shifting
        profiles relative to each other.

    Returns
    -------
    dict
        A profile object as described by the speedscope file format schema.
    """
    starts = (calls.start_ns / NS_PER_SECOND).tolist()
    ends = (calls.end_ns / NS_PER_SECOND).tolist()
    names = [calls.names[row] for row in calls.region_index.tolist()]

    children: dict[int | None, list[int]] = {}
    for index, parent in enumerate(calls.parent.tolist()):
        children.setdefault(None if parent < 0 else parent, []).append(index)

    events = []
    # Depth-first over the reconstructed tree, siblings in start order (which
    # is the order `calls` is already in). Emitting events this way makes them
    # balanced by construction, instead of sorting timestamps and hoping.
    stack: list[tuple[int, bool]] = [
        (index, False) for index in reversed(children.get(None, []))
    ]
    while stack:
        index, closing = stack.pop()
        start, end = starts[index], ends[index]
        frame_name = names[index]
        if closing:
            events.append({"type": "C", "frame": frame_indices[frame_name], "at": end})
            continue

        if frame_name not in frame_indices:
            frame_indices[frame_name] = len(frames)
            frames.append({"name": frame_name})
        events.append({"type": "O", "frame": frame_indices[frame_name], "at": start})
        stack.append((index, True))
        stack.extend((child, False) for child in reversed(children.get(index, [])))

    for event in events:
        event["at"] -= origin

    return {
        "type": "evented",
        "name": name,
        "unit": TIME_UNIT,
        "startValue": events[0]["at"] if events else 0.0,
        "endValue": events[-1]["at"] if events else 0.0,
        "events": events,
    }


def build_speedscope_document(
    named_calls: Sequence[tuple[str, CallArrays]],
    name: str,
) -> dict:
    """Build a full speedscope document holding one profile per entry.

    Parameters
    ----------
    named_calls : sequence of (str, CallArrays)
        Profile name and calls for each profile to include, e.g. one per rank.
    name : str
        Name of the document, shown in speedscope's title bar.

    Returns
    -------
    dict
        The document, ready to be serialized as JSON.
    """
    starts = [
        calls.start_ns[0] / NS_PER_SECOND for _, calls in named_calls if len(calls)
    ]  # calls are start-ordered
    origin = min(starts) if starts else 0.0

    frames: list[dict] = []
    frame_indices: dict[str, int] = {}
    profiles = [
        build_speedscope_profile(
            calls,
            profile_name,
            frame_indices=frame_indices,
            frames=frames,
            origin=origin,
        )
        for profile_name, calls in named_calls
    ]

    return {
        "$schema": SCHEMA_URL,
        "exporter": _exporter_name(),
        "name": name,
        "activeProfileIndex": 0,
        "shared": {"frames": frames},
        "profiles": profiles,
    }


def write_speedscope_file(filepath: str | Path, document: dict) -> Path:
    """Write a speedscope document to ``filepath`` as JSON."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f)
    return output_path


def export_speedscope(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    filepath: str | Path,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Write speedscope JSON files for the selected ranks.

    One file is written per input HDF5 file, holding one profile per rank ---
    or, for a run profiled with ``track_threads``, one per thread and task.
    Unlike ``.prof``, the format carries several profiles per file, and
    speedscope switches between them from its profile selector.

    Parameters
    ----------
    profiling_data : ProfilingResults | Sequence[ProfilingResults]
        The run(s) to export: file runs, in-memory results from
        ``ProfileManager.finalize(return_results=True)``, or a mix.
    filepath : str | Path
        Base output path, e.g. ``figures/profile.speedscope.json``. The input
        file's stem is appended when more than one file is exported.
    ranks : list[int] | int, optional
        Ranks to export (default: rank 0 only).
    include, exclude : list[str] | str, optional
        Region name filters, as for the plotting functions.

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
        named_calls = []
        for rank in normalized_ranks:
            if rank < 0 or rank >= run.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = build_call_arrays(regions, rank)
            # One profile per lane, which is exactly speedscope's model of a
            # thread -- and a necessity rather than a nicety: an evented
            # profile's timestamps must not go backwards, and two interleaved
            # lanes walked as one tree do exactly that.
            for lane, lane_calls in split_by_lane(calls):
                if not len(lane_calls):
                    continue
                profile_label = f"rank {rank}"
                if lane != -1:
                    profile_label = f"{profile_label} - {run.lane_label(lane, rank)}"
                named_calls.append((profile_label, lane_calls))
        if named_calls:
            prepared.append((label, named_calls))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    base_path = Path(filepath)
    # ".speedscope.json" is the conventional extension, and Path.suffix only
    # sees the ".json" half of it, so the whole tail is kept here.
    stem, dot, extension = base_path.name.partition(".")
    suffix = f".{extension}" if dot else ".speedscope.json"
    multiple_files = len(runs) > 1

    written = []
    for label, named_calls in prepared:
        parts = [stem]
        if multiple_files:
            parts.append(_filename_slug(label))
        out_path = base_path.with_name("_".join(parts) + suffix)
        document = build_speedscope_document(named_calls, name=label)
        written.append(write_speedscope_file(out_path, document))
        if verbose:
            print(f"Wrote {out_path} (view at https://www.speedscope.app)")

    return written
