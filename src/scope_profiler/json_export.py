"""Export and re-read profiling runs as JSON.

The HDF5 file remains the default output: it is compact, appendable rank by
rank, and the only format the parallel writer can produce. This module adds a
second, dependency-free representation of exactly the same run, for the cases
where HDF5 is the wrong shape -- a browser, ``jq``, a notebook on a machine
without h5py, or a diff in code review.

The document is *lossless*: :func:`read_json` rebuilds a
:class:`~scope_profiler.results.ProfilingResults` that is indistinguishable
from the one :func:`~scope_profiler.h5reader.read_h5` gives back for the same
run, per-call timestamps, lane tables, LIKWID counters and line-profiler
records included. Timestamps stay in the integer nanoseconds the run recorded
rather than being converted to seconds, so a round trip is exact.

Files ending in ``.gz`` are written (and read) gzip-compressed, which is what
a real run wants: the event columns are the bulk of the document and they
compress by roughly an order of magnitude.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.likwid_data import LikwidRegionResult
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import Region
from scope_profiler.results import ProfilingResults

#: Identifier stored in every document, so a reader can tell one of these from
#: the speedscope/plot-data JSON the exporters also write.
FORMAT_NAME = "scope-profiler-profile"

#: Bumped only when the layout below changes incompatibly.
FORMAT_VERSION = 1

#: Suffixes recognised as "a profile as JSON", plain or gzip-compressed.
JSON_SUFFIXES = (".json", ".json.gz")


def is_json_path(path) -> bool:
    """Whether ``path`` names a JSON profile rather than an HDF5 one."""
    name = Path(path).name.lower()
    return name.endswith(JSON_SUFFIXES)


def _exporter_name() -> str:
    """Identify this package (and version) as the producer of the file."""
    from scope_profiler import __version__

    return f"scope-profiler@{__version__}"


def _ints(values) -> list:
    """A JSON list of Python ints, for an integer column."""
    return np.asarray(values).astype(np.int64, copy=False).tolist()


def _floats(values) -> list:
    """A JSON list for a float column, with non-finite values as ``null``.

    ``json.dump`` writes ``NaN`` and ``Infinity`` by default, which no other
    JSON parser accepts. LIKWID hands out both, so they travel as ``null``
    and come back as ``nan``/``inf`` in :func:`_float_column`.
    """
    return [
        None if not math.isfinite(value) else value
        for value in np.asarray(values, dtype=float).tolist()
    ]


def _float_column(values, shape=None) -> np.ndarray:
    """Rebuild a float array written by :func:`_floats`."""
    array: np.ndarray = np.asarray(
        [math.nan if value is None else float(value) for value in _flatten(values)],
        dtype=float,
    )
    return array if shape is None else array.reshape(shape)


def _flatten(values):
    """Yield the scalars of a possibly nested list, row-major."""
    for value in values:
        if isinstance(value, list):
            yield from _flatten(value)
        else:
            yield value


def _matrix(array) -> list:
    """A 2-D float array as a list of rows, non-finite values as ``null``."""
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        # Only a default-constructed result's empty placeholder ever lands
        # here; a real one is (nevents, nthreads) or (nmetrics, nthreads).
        array = array.reshape(0, 0)
    return [_floats(row) for row in array]


def _metadata_value(value):
    """Make one metadata entry JSON-safe without changing what it means."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_metadata_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _region_rows(results: ProfilingResults, region_names, ranks) -> list[dict]:
    """One row per (region, rank), carrying that rank's raw columns."""
    rows = []
    for name in region_names:
        region = results[name]
        for rank in sorted(region.regions):
            if ranks is not None and rank not in ranks:
                continue
            rows.append(_region_row(name, rank, region.regions[rank]))
    return rows


def _region_row(name: str, rank: int, region: Region) -> dict:
    """Serialize one rank's share of one region."""
    row: dict = {"name": name, "rank": rank}
    if region.source_file is not None:
        row["source_file"] = region.source_file
    if region.source_lineno is not None:
        row["source_lineno"] = int(region.source_lineno)
    if region.source_text is not None:
        row["source_text"] = region.source_text
    if region.tags:
        row["tags"] = list(region.tags)

    aggregate = region.stored_summary
    if aggregate is not None:
        # An aggregation-mode run keeps fixed-size statistics and no timeline;
        # writing empty event columns for it would claim it recorded none. The
        # same branch carries a summary-only read, whose statistics include
        # two floats (``mean``, ``m2``) among the integers -- so the values
        # keep their own type rather than being coerced to int.
        row["aggregate"] = {
            key: int(value) if float(value).is_integer() else float(value)
            for key, value in aggregate.items()
        }
        if not region.has_event_data:
            # A run that recorded a timeline this reader did not load. Unlike
            # aggregation mode, the events exist -- in the file it came from.
            row["event_data_available"] = False
        return row

    row["start_times_ns"] = _ints(region.start_times_ns)
    row["end_times_ns"] = _ints(region.end_times_ns)
    for key, column in (
        ("gpu_durations_ns", region.gpu_durations_ns),
        ("call_ids", region.call_ids),
        ("parent_ids", region.parent_ids),
        ("thread_ids", region.thread_ids),
        ("task_ids", region.task_ids),
        ("await_ns", region.await_times_ns),
    ):
        if column is not None:
            row[key] = _ints(column)
    # Written by the run itself; keeping it saves every reader of the document
    # a call-stack reconstruction, exactly as the HDF5 column does.
    total = getattr(region, "_exclusive_total_ns", None)
    if total is not None:
        row["exclusive_total_ns"] = int(total)
    return row


def _likwid_document(likwid: dict, ranks) -> dict:
    """The run's LIKWID results, rank by rank."""
    document = {}
    for rank, tags in sorted(likwid.items()):
        if ranks is not None and rank not in ranks:
            continue
        document[str(rank)] = {
            tag: {
                "tag": result.tag,
                "group_id": int(result.group_id),
                "group_name": result.group_name,
                "cpus": [int(cpu) for cpu in result.cpus],
                "times": _floats(result.times),
                "call_counts": _ints(result.call_counts),
                "event_names": list(result.event_names),
                "counter_names": list(result.counter_names),
                "events": _matrix(result.events),
                "metric_names": list(result.metric_names),
                "metrics": _matrix(result.metrics),
                "source": result.source,
            }
            for tag, result in sorted(tags.items())
        }
    return document


def _line_profile_document(line_profile: dict, ranks) -> dict:
    """The run's persisted line-profiler records, rank by rank."""
    document = {}
    for rank, records in sorted(line_profile.items()):
        if ranks is not None and rank not in ranks:
            continue
        document[str(rank)] = [
            {
                "region": record.get("region", ""),
                "filename": record.get("filename", ""),
                "function": record.get("function", ""),
                "first_lineno": int(record.get("first_lineno", 0)),
                "line_numbers": _ints(record.get("line_numbers", ())),
                "hits": _ints(record.get("hits", ())),
                "times": _floats(record.get("times", ())),
                "unit": float(record.get("unit", 1.0)),
            }
            for record in records
        ]
    return document


def _lane_document(rows_by_rank: dict, fields, ranks) -> dict:
    """Thread or task tables as row dicts, in the columns' own units."""
    document = {}
    for rank, rows in sorted(rows_by_rank.items()):
        if ranks is not None and rank not in ranks:
            continue
        document[str(rank)] = [
            {name: extract(row) for name, extract in fields} for row in rows
        ]
    return document


def _ns(seconds) -> int:
    """Seconds back to the integer nanoseconds the run recorded."""
    from scope_profiler.concurrency import UNKNOWN

    if seconds is None:
        return UNKNOWN
    return round(float(seconds) * 1e9)


_THREAD_FIELDS = (
    ("index", lambda row: int(row.index)),
    ("name", lambda row: row.name),
    ("ident", lambda row: int(row.ident)),
    ("native_id", lambda row: int(row.native_id)),
    ("daemon", lambda row: bool(row.daemon)),
    ("start_ns", lambda row: _ns(row.start_time)),
    ("end_ns", lambda row: _ns(row.end_time)),
    ("cpu_ns", lambda row: _ns(row.cpu_time)),
)

_TASK_FIELDS = (
    ("index", lambda row: int(row.index)),
    ("kind", lambda row: row.kind),
    ("name", lambda row: row.name),
    ("coro_name", lambda row: row.coro_name),
    ("thread_index", lambda row: int(row.thread_index)),
    ("created_ns", lambda row: _ns(row.created_time)),
    ("done_ns", lambda row: _ns(row.done_time)),
    ("steps", lambda row: int(row.steps)),
    ("running_ns", lambda row: _ns(row.running_time)),
    ("suspended_ns", lambda row: _ns(row.awaiting_time)),
)


def build_json_document(
    results: ProfilingResults,
    include=None,
    exclude=None,
    ranks: list[int] | int | None = None,
) -> dict:
    """Build the JSON document describing one run.

    Parameters
    ----------
    results : ProfilingResults
        The run to serialize, from a file or straight out of
        ``ProfileManager.finalize(return_results=True)``.
    include, exclude : list[str] | str, optional
        Region name filters, as for the plotting functions.
    ranks : list[int] | int, optional
        Ranks to keep (default: every rank the run recorded).

    Returns
    -------
    dict
        A JSON-serializable document; see :func:`write_json`.
    """
    from scope_profiler.plotting_scripts import _normalize_ranks

    selected_ranks = None if ranks is None else set(_normalize_ranks(ranks))
    # Filter through get_regions, but keep the run's own order of appearance:
    # that is the order the HDF5 file stores, and a round trip through JSON
    # should not quietly re-sort a run's regions.
    selected = {region.name for region in results.get_regions(include, exclude)}
    region_names = [name for name in results.region_names if name in selected]

    document = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "exporter": _exporter_name(),
        # Stated once, rather than per column: everything named ``*_ns`` below
        # is integer nanoseconds on the run's own clock.
        "time_unit": "nanoseconds",
        "num_ranks": int(results.num_ranks),
        "event_data_available": bool(results.has_event_data),
        "metadata": {
            key: _metadata_value(value) for key, value in results.metadata.items()
        },
        "regions": _region_rows(results, region_names, selected_ranks),
    }
    for key, rows, fields in (
        ("threads", results.threads, _THREAD_FIELDS),
        ("tasks", results.tasks, _TASK_FIELDS),
    ):
        lanes = _lane_document(rows, fields, selected_ranks)
        if lanes:
            document[key] = lanes
    likwid = _likwid_document(results.get_likwid_regions(), selected_ranks)
    if likwid:
        document["likwid"] = likwid
    line_profile = _line_profile_document(results.line_profile, selected_ranks)
    if any(line_profile.values()):
        document["line_profile"] = line_profile
    return document


def write_json_file(filepath: str | Path, document: dict, indent=None) -> Path:
    """Write ``document`` to ``filepath``, gzip-compressed for a ``.gz`` name."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=indent, allow_nan=False)
    if output_path.name.lower().endswith(".gz"):
        # Neither the name nor the time of writing goes into the header, so
        # two exports of the same run are byte-identical -- what deterministic
        # output means, and what a content hash in CI relies on.
        with (
            open(output_path, "wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f,
        ):
            f.write(text.encode("utf-8"))
    else:
        output_path.write_text(text, encoding="utf-8")
    return output_path


def write_json(
    results: ProfilingResults,
    filepath: str | Path,
    *,
    include=None,
    exclude=None,
    ranks: list[int] | int | None = None,
    indent: int | None = None,
) -> Path:
    """Write one run to ``filepath`` as JSON, and return the path written."""
    return write_json_file(
        filepath,
        build_json_document(results, include=include, exclude=exclude, ranks=ranks),
        indent=indent,
    )


def export_json(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    filepath: str | Path,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    verbose: bool = True,
    indent: int | None = None,
) -> list[Path]:
    """Write one JSON document per run, exactly as ``export_speedscope`` does.

    Parameters
    ----------
    profiling_data : ProfilingResults | Sequence[ProfilingResults]
        The run(s) to export: file runs, in-memory results, or a mix.
    filepath : str | Path
        Base output path, e.g. ``figures/profile.json``. Ending it in ``.gz``
        compresses the output. The run's label is appended to the stem when
        more than one run is exported.
    ranks : list[int] | int, optional
        Ranks to export (default: every rank).
    include, exclude : list[str] | str, optional
        Region name filters, as for the plotting functions.
    indent : int, optional
        Passed to ``json.dumps``; the default writes the document on one line.

    Returns
    -------
    list[Path]
        The files written, in the order they were written.
    """
    from scope_profiler.plotting_scripts import (
        _as_runs,
        _filename_slug,
        _unique_labels,
    )

    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 writes the files.
        return []

    base_path = Path(filepath)
    # ".json.gz" is two suffixes and Path.suffix sees only the ".gz" half, so
    # the whole tail is kept, as the speedscope exporter keeps its own.
    stem, dot, extension = base_path.name.partition(".")
    suffix = f".{extension}" if dot else ".json"
    labels = _unique_labels([run.display_label for run in runs])
    multiple_files = len(runs) > 1

    written = []
    for label, run in zip(labels, runs):
        parts = [stem]
        if multiple_files:
            parts.append(_filename_slug(label))
        out_path = base_path.with_name("_".join(parts) + suffix)
        written.append(
            write_json(
                run,
                out_path,
                include=include,
                exclude=exclude,
                ranks=ranks,
                indent=indent,
            )
        )
        if verbose:
            print(f"Wrote {out_path}")
    return written


class JSONProfileError(ValueError):
    """Raised when a file is not a scope-profiler JSON profile."""


def _read_document(file_path: Path) -> dict:
    """Parse the JSON document at ``file_path``, gzip-compressed or not."""
    if file_path.name.lower().endswith(".gz"):
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            document = json.load(f)
    else:
        with open(file_path, encoding="utf-8") as f:
            document = json.load(f)
    if not isinstance(document, dict) or document.get("format") != FORMAT_NAME:
        raise JSONProfileError(
            f"{file_path} is not a scope-profiler JSON profile (written by "
            "`scope-profiler export json` or `-o <name>.json`)."
        )
    version = int(document.get("format_version", 0))
    if version > FORMAT_VERSION:
        raise JSONProfileError(
            f"{file_path} uses JSON profile format version {version}; this "
            f"package supports versions through {FORMAT_VERSION}. Upgrade "
            "scope-profiler to read this file."
        )
    return document


def _region_from_row(row: dict) -> Region:
    """Rebuild one rank's :class:`~scope_profiler.region.Region` from a row."""
    source_file = row.get("source_file")
    source_lineno = row.get("source_lineno")
    source_text = row.get("source_text")
    tags = tuple(row.get("tags", ()))

    aggregate = row.get("aggregate")
    if aggregate is not None:
        return Region(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            aggregate=dict(aggregate),
            event_data_available=bool(row.get("event_data_available", True)),
            source_file=source_file,
            source_lineno=source_lineno,
            source_text=source_text,
            tags=tags,
        )

    def column(key):
        values = row.get(key)
        return None if values is None else np.asarray(values, dtype=np.int64)

    return Region(
        np.asarray(row.get("start_times_ns", ()), dtype=np.int64),
        np.asarray(row.get("end_times_ns", ()), dtype=np.int64),
        gpu_durations=column("gpu_durations_ns"),
        call_ids=column("call_ids"),
        parent_ids=column("parent_ids"),
        thread_ids=column("thread_ids"),
        task_ids=column("task_ids"),
        await_times=column("await_ns"),
        source_file=source_file,
        source_lineno=source_lineno,
        source_text=source_text,
        tags=tags,
    )


def _lanes_from_document(document: dict) -> tuple[dict, dict]:
    """Rebuild the thread and task tables from their row dicts."""
    from scope_profiler.concurrency import lane_tables_from_columns

    def table(key, fields, position):
        rows_by_rank: dict[int, list] = {}
        for rank, rows in (document.get(key) or {}).items():
            columns = {name: [row.get(name) for row in rows] for name, _ in fields}
            infos = lane_tables_from_columns(int(rank), {key: columns})[position]
            if infos:
                rows_by_rank[int(rank)] = infos
        return rows_by_rank

    return (
        table("threads", _THREAD_FIELDS, 0),
        table("tasks", _TASK_FIELDS, 1),
    )


def _likwid_from_document(document: dict) -> dict:
    """Rebuild the LIKWID results, rank by rank."""
    likwid: dict[int, dict[str, LikwidRegionResult]] = {}
    for rank, tags in (document.get("likwid") or {}).items():
        likwid[int(rank)] = {
            tag: LikwidRegionResult(
                tag=entry.get("tag", tag),
                group_id=int(entry.get("group_id", -1)),
                group_name=entry.get("group_name", ""),
                cpus=[int(cpu) for cpu in entry.get("cpus", ())],
                times=_float_column(entry.get("times", ())),
                call_counts=np.asarray(entry.get("call_counts", ()), dtype=np.int64),
                event_names=list(entry.get("event_names", ())),
                counter_names=list(entry.get("counter_names", ())),
                events=_float_column(
                    entry.get("events", ()),
                    shape=(
                        len(entry.get("event_names", ())),
                        len(entry.get("cpus", ())),
                    ),
                ),
                metric_names=list(entry.get("metric_names", ())),
                metrics=_float_column(
                    entry.get("metrics", ()),
                    shape=(
                        len(entry.get("metric_names", ())),
                        len(entry.get("cpus", ())),
                    ),
                ),
                source=entry.get("source", ""),
            )
            for tag, entry in tags.items()
        }
    return likwid


def _line_profile_from_document(document: dict) -> dict:
    """Rebuild the persisted line-profiler records, rank by rank."""
    return {
        int(rank): [
            {
                "region": record.get("region", ""),
                "filename": record.get("filename", ""),
                "function": record.get("function", ""),
                "first_lineno": int(record.get("first_lineno", 0)),
                "line_numbers": np.asarray(
                    record.get("line_numbers", ()), dtype=np.int64
                ),
                "hits": np.asarray(record.get("hits", ()), dtype=np.int64),
                "times": _float_column(record.get("times", ())),
                "unit": float(record.get("unit", 1.0)),
            }
            for record in records
        ]
        for rank, records in (document.get("line_profile") or {}).items()
    }


def load_json(file_path: str | Path, verbose: bool = False) -> dict:
    """Parse a JSON profile into :class:`ProfilingResults` arguments.

    The JSON twin of :func:`~scope_profiler.h5reader.load_h5`, and the parsing
    half of :func:`read_json`.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON profile not found: {file_path}")
    document = _read_document(file_path)

    per_region: dict[str, dict[int, Region]] = {}
    region_names: list[str] = []
    exclusive_totals: dict[str, dict[int, int]] = {}
    for row in document.get("regions", ()):
        name = row["name"]
        rank = int(row["rank"])
        if verbose:
            print(f"{name = }, {rank = }")
        region_names.append(name)
        per_region.setdefault(name, {})[rank] = _region_from_row(row)
        total = row.get("exclusive_total_ns")
        if total is not None:
            exclusive_totals.setdefault(name, {})[rank] = int(total)

    threads, tasks = _lanes_from_document(document)
    return {
        "regions": {
            name: MPIRegion(name=name, regions=per_region[name])
            for name in dict.fromkeys(region_names)
        },
        "metadata": dict(document.get("metadata") or {}),
        "num_ranks": int(document.get("num_ranks", 0)),
        "likwid": _likwid_from_document(document),
        "line_profile": _line_profile_from_document(document),
        "threads": threads,
        "tasks": tasks,
        "file_path": file_path,
        "exclusive_totals": exclusive_totals,
        "event_data_available": bool(document.get("event_data_available", True)),
    }


def read_json(file_path: str | Path, verbose: bool = False) -> ProfilingResults:
    """Read a JSON profile back into :class:`ProfilingResults`.

    The JSON twin of :func:`~scope_profiler.h5reader.read_h5`::

        results = read_json("profiling_data.json")
        results.print_summary()
    """
    return ProfilingResults(**load_json(file_path, verbose=verbose))
