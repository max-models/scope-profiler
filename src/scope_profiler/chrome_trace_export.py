"""Export profiling runs to the Chrome Trace Event format.

The format is understood by ``chrome://tracing`` and Perfetto. Each rank is a
process and each recorded thread or asyncio task is a lane in that process.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from scope_profiler.plotting_scripts import (
    _as_runs,
    _filename_slug,
    _normalize_ranks,
    _unique_labels,
)
from scope_profiler.results import ProfilingResults


def _exporter_name() -> str:
    from scope_profiler import __version__

    return f"scope-profiler@{__version__}"


def _lane_key(event: dict) -> tuple[str, int]:
    task = event.get("task", -1)
    if task is not None and int(task) >= 0:
        return ("task", int(task))
    thread = event.get("thread")
    if thread is not None and int(thread) >= 0:
        return ("thread", int(thread))
    return ("rank", 0)


def build_chrome_trace_document(
    run: ProfilingResults,
    label: str | None = None,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
) -> dict:
    """Build one Chrome Trace Event document from a profiling run.

    Region durations are emitted in microseconds, as required by the Chrome
    Trace Event format. GPU and await measurements are preserved in event
    arguments. Aggregate perf counters are emitted as counter events at the
    end of their rank timeline.
    """
    selected = _normalize_ranks(ranks) if ranks is not None else [0]
    label = label or run.display_label
    trace_events: list[dict] = []

    for rank in selected:
        if rank < 0 or rank >= run.num_ranks:
            raise ValueError(f"Invalid rank requested: {rank}")
        events = run.events(include=include, exclude=exclude, ranks=rank, relative=True)
        lanes = {_lane_key(event) for event in events}
        lane_ids = {lane: index for index, lane in enumerate(sorted(lanes))}

        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": rank,
                "tid": 0,
                "args": {"name": f"rank {rank}"},
            }
        )

        for lane, tid in lane_ids.items():
            kind, index = lane
            if kind == "task":
                name = run.lane_label(index, rank)
                args = {"lane": "async task", "task_index": index}
            elif kind == "thread":
                name = run.lane_label(-2 - index, rank)
                args = {"lane": "thread", "thread_index": index}
                for thread in run.threads.get(rank, []):
                    if thread.index == index:
                        args.update(
                            {
                                "native_id": thread.native_id,
                                "ident": thread.ident,
                                "cpu_time_s": thread.cpu_time,
                            }
                        )
                        break
            else:
                name = f"rank {rank}"
                args = {"lane": "rank"}
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": rank,
                    "tid": tid,
                    "args": {"name": name, **args},
                }
            )

        for event in events:
            region = run[event["name"]]
            args = {
                "rank": rank,
                "call_index": event["call_index"],
                "duration_s": event["duration"],
            }
            for key in ("call_id", "parent_id", "gpu_duration", "await_duration"):
                if key in event:
                    args[key] = event[key]
            if region.source_file is not None:
                args["source_file"] = region.source_file
                if region.source_lineno is not None:
                    args["source_line"] = region.source_lineno
            trace_events.append(
                {
                    "name": event["name"],
                    "cat": "scope-profiler",
                    "ph": "X",
                    "ts": event["start"] * 1_000_000,
                    "dur": event["duration"] * 1_000_000,
                    "pid": rank,
                    "tid": lane_ids[_lane_key(event)],
                    "args": args,
                }
            )

        perf_events = run.get_perf_events(rank)
        end = max((event["end"] for event in events), default=0.0) * 1_000_000
        for region_name, totals in perf_events.items():
            values = {name: int(value) for name, value in totals.values.items()}
            values["calls"] = int(totals.calls)
            trace_events.append(
                {
                    "name": f"perf events: {region_name}",
                    "cat": "perf_events",
                    "ph": "C",
                    "ts": end,
                    "pid": rank,
                    "tid": 0,
                    "args": values,
                }
            )

    return {
        "displayTimeUnit": "ms",
        "metadata": {"name": label, "exporter": _exporter_name()},
        "traceEvents": trace_events,
    }


def write_chrome_trace_file(filepath: str | Path, document: dict) -> Path:
    """Write a Chrome Trace Event document to ``filepath``."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(document, stream, separators=(",", ":"))
    return output_path


def export_chrome_trace(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    filepath: str | Path,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Write Chrome Trace JSON files, one per input profiling run."""
    runs = _as_runs(profiling_data)
    if not runs:
        return []
    labels = _unique_labels([run.display_label for run in runs])
    base_path = Path(filepath)
    stem, dot, extension = base_path.name.partition(".")
    suffix = f".{extension}" if dot else ".trace.json"
    written = []
    for label, run in zip(labels, runs):
        parts = [stem]
        if len(runs) > 1:
            parts.append(_filename_slug(label))
        output = base_path.with_name("_".join(parts) + suffix)
        document = build_chrome_trace_document(
            run, label=label, ranks=ranks, include=include, exclude=exclude
        )
        written.append(write_chrome_trace_file(output, document))
        if verbose:
            print(f"Wrote {output} (open with Perfetto or chrome://tracing)")
    return written
