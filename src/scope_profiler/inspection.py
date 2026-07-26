"""CLI entry point for ``scope-profiler inspect``: summarize a profiling file.

Prints the run metadata in full and a one-line-per-region overview of the
timing data, without producing any plots. The metadata can also be exported
to JSON, either from the CLI (``--export-metadata``) or with
:func:`write_metadata_json`.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.h5reader import ProfilingH5Reader

# Metadata is printed in these groups, in this order, so the fields that
# identify a run come first and the sprawling environment variables last.
# Anything not listed here still gets printed (see _metadata_sections), so new
# metadata fields never silently disappear from the output.
_RUN_FIELDS = (
    "timestamp",
    "user",
    "hostname",
    "working_directory",
    "scope_profiler_version",
    "python_version",
)
_SYSTEM_FIELDS = (
    "platform",
    "uname",
    "chip_information",
)
_PARALLELISM_FIELDS = (
    "mpi_size",
    "omp_num_threads",
    "total_cores",
)

_SLURM_PREFIXES = ("SLURM_", "SLURMD_")

# Long values (PATH, LD_LIBRARY_PATH, ...) are clipped unless --full is given.
_DEFAULT_VALUE_WIDTH = 96
_ELLIPSIS = " […]"

_SORT_KEYS = ("total", "calls", "avg", "max", "name")


def _region_durations(region, ranks=None) -> np.ndarray:
    """Pool every recorded call duration of a region, in seconds.

    Durations come from ``Region.durations`` (seconds) rather than the
    aggregate properties, which report nanoseconds.
    """
    selected = (
        region.regions
        if ranks is None
        else {rank: region.regions[rank] for rank in ranks if rank in region.regions}
    )
    values = [data.durations for data in selected.values() if data.durations.size]
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def _region_row(region, ranks=None) -> dict:
    """Collect the summary statistics shown for one region."""
    if ranks is None:
        per_rank = region.regions
    else:
        per_rank = {
            rank: region.regions[rank] for rank in ranks if rank in region.regions
        }

    durations = _region_durations(region, ranks)
    return {
        "name": region.name,
        "num_ranks": len(per_rank),
        "calls": sum(data.num_calls for data in per_rank.values()),
        "total": float(np.sum(durations)) if durations.size else None,
        "avg": float(np.mean(durations)) if durations.size else None,
        "min": float(np.min(durations)) if durations.size else None,
        "max": float(np.max(durations)) if durations.size else None,
        "std": float(np.std(durations)) if durations.size else None,
    }


def _time_span(reader) -> float | None:
    """Wall-clock seconds between the first region entry and the last exit."""
    starts = []
    ends = []
    for region in reader.get_regions():
        for data in region.regions.values():
            if data.durations.size:
                starts.append(float(np.min(data.start_times)))
                ends.append(float(np.max(data.end_times)))
    if not starts:
        return None
    return max(ends) - min(starts)


def _format_duration(value) -> str:
    """Format a duration in seconds, or a dash when no timing was recorded."""
    return "-" if value is None else f"{value:.6g}"


def _clip(value: str, full: bool) -> str:
    """Shorten an over-long metadata value unless the full text was requested."""
    if full or len(value) <= _DEFAULT_VALUE_WIDTH:
        return value
    return value[: _DEFAULT_VALUE_WIDTH - len(_ELLIPSIS)] + _ELLIPSIS


def _metadata_sections(metadata: dict) -> list:
    """Split metadata into ordered, titled sections.

    Every key is placed in exactly one section; keys that match none of the
    known groups end up under "Other", so the output stays complete as more
    metadata is recorded.
    """
    remaining = dict(metadata)

    def take(names):
        return [(name, remaining.pop(name)) for name in names if name in remaining]

    sections = [
        ("Run", take(_RUN_FIELDS)),
        ("System", take(_SYSTEM_FIELDS)),
        ("Parallelism", take(_PARALLELISM_FIELDS)),
    ]

    modules = remaining.pop("modules", None)

    slurm = sorted(
        (key, remaining.pop(key))
        for key in list(remaining)
        if key.startswith(_SLURM_PREFIXES)
    )
    sections.append(("Slurm", slurm))

    environment = sorted(
        (key, remaining.pop(key)) for key in list(remaining) if key.isupper()
    )
    sections.append(("Environment", environment))
    sections.append(("Other", sorted(remaining.items())))

    return sections, modules


def _print_metadata(metadata: dict, full: bool, stream) -> None:
    """Print the metadata group, one section at a time."""
    if not metadata:
        print("Metadata\n  (none recorded)\n", file=stream)
        return

    sections, modules = _metadata_sections(metadata)
    key_width = max(len(str(key)) for key in metadata)

    print("Metadata", file=stream)
    for title, entries in sections:
        if not entries:
            continue
        print(f"  {title}", file=stream)
        for key, value in entries:
            print(f"    {key:<{key_width}} : {_clip(str(value), full)}", file=stream)

    if modules is not None:
        # `modules` is a list; print one per line rather than as a repr.
        modules = list(modules)
        print(f"  Modules ({len(modules)})", file=stream)
        for module in modules:
            print(f"    {module}", file=stream)
        if not modules:
            print("    (none loaded)", file=stream)
    print(file=stream)


def _print_regions(reader, rows: list, stream) -> None:
    """Print the per-region statistics table."""
    if not rows:
        print("Regions\n  (none recorded)", file=stream)
        return

    formatted = [
        {
            "name": row["name"],
            "ranks": str(row["num_ranks"]),
            "calls": str(row["calls"]),
            "total": _format_duration(row["total"]),
            "avg": _format_duration(row["avg"]),
            "min": _format_duration(row["min"]),
            "max": _format_duration(row["max"]),
            "std": _format_duration(row["std"]),
        }
        for row in rows
    ]

    total_calls = sum(row["calls"] for row in rows)
    timed = [row["total"] for row in rows if row["total"] is not None]
    formatted.append(
        {
            "name": "TOTAL",
            "ranks": "",
            "calls": str(total_calls),
            "total": _format_duration(sum(timed) if timed else None),
            "avg": "",
            "min": "",
            "max": "",
            "std": "",
        }
    )

    headers = {
        "name": "region",
        "ranks": "ranks",
        "calls": "calls",
        "total": "total [s]",
        "avg": "avg [s]",
        "min": "min [s]",
        "max": "max [s]",
        "std": "std [s]",
    }
    widths = {
        column: max(len(header), max(len(row[column]) for row in formatted))
        for column, header in headers.items()
    }

    def render(row, left_align_name=True):
        cells = [
            (
                f"{row['name']:<{widths['name']}}"
                if left_align_name
                else f"{row['name']:>{widths['name']}}"
            )
        ]
        cells += [
            f"{row[column]:>{widths[column]}}"
            for column in ("ranks", "calls", "total", "avg", "min", "max", "std")
        ]
        return "  ".join(cells)

    print(f"Regions ({len(rows)})", file=stream)
    header_line = render(headers)
    print(f"  {header_line}", file=stream)
    print(f"  {'-' * len(header_line)}", file=stream)
    for row in formatted[:-1]:
        print(f"  {render(row)}".rstrip(), file=stream)
    print(f"  {'-' * len(header_line)}", file=stream)
    print(f"  {render(formatted[-1])}".rstrip(), file=stream)

    notes = []
    if len(rows) > 1:
        # Nested regions are counted in both the inner and the outer row, so
        # the summed total legitimately exceeds the run's wall-clock time.
        notes.append(
            "Regions may nest, so the summed total can exceed the wall-clock time."
        )
    if any(row["total"] is None for row in rows):
        notes.append(
            "Regions without timing were profiled with time_trace=False; "
            "only their call counts were recorded."
        )
    for note in notes:
        print(f"\n  {note}", file=stream)


def inspect_file(
    file_path,
    include=None,
    exclude=None,
    ranks=None,
    sort: str = "total",
    show_metadata: bool = True,
    show_regions: bool = True,
    full: bool = False,
    stream=None,
) -> None:
    """Print a summary of one profiling HDF5 file.

    Parameters
    ----------
    file_path : str or Path
        Merged profiling file to inspect.
    include, exclude : list of str or str, optional
        Regex patterns selecting which regions to report.
    ranks : list of int, optional
        Restrict the region statistics to these ranks (default: all).
    sort : str, optional
        Region ordering: one of ``total``, ``calls``, ``avg``, ``max`` (all
        descending) or ``name`` (alphabetical). Default: ``total``.
    show_metadata, show_regions : bool, optional
        Sections to print (both default to True).
    full : bool, optional
        Print long metadata values in full instead of clipping them.
    stream : file-like, optional
        Where to write (default: stdout).
    """
    stream = sys.stdout if stream is None else stream
    reader = ProfilingH5Reader(file_path)

    path = Path(reader.file_path)
    size_mb = path.stat().st_size / 1024**2
    span = _time_span(reader)
    headline = (
        f"{reader.num_ranks} rank(s), {len(reader.get_regions())} region(s), "
        f"{size_mb:.2f} MiB"
    )
    if span is not None:
        headline += f", {span:.6g} s wall clock"

    print("=" * 78, file=stream)
    print(path, file=stream)
    print(headline, file=stream)
    print("=" * 78 + "\n", file=stream)

    if show_metadata:
        _print_metadata(reader.metadata, full=full, stream=stream)

    if not show_regions:
        return

    rows = [
        _region_row(region, ranks)
        for region in reader.get_regions(include=include, exclude=exclude)
    ]

    # Sort by name first so that the stable sort below breaks ties
    # alphabetically rather than by whatever order the file happened to use.
    rows.sort(key=lambda row: row["name"])
    if sort != "name":
        # None (no timing recorded) sorts last.
        rows.sort(key=lambda row: (row[sort] is not None, row[sort] or 0), reverse=True)

    _print_regions(reader, rows, stream=stream)


def _json_safe(value):
    """Convert a metadata value into something ``json.dump`` accepts.

    Values read back from HDF5 arrive as numpy scalars and arrays, which the
    JSON encoder rejects.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def collect_file_metadata(
    profiling_data: ProfilingH5Reader | str | Path | Sequence,
) -> dict:
    """Collect the run metadata of one or more profiling files.

    Parameters
    ----------
    profiling_data : ProfilingH5Reader, path, or sequence of either
        Files to read the metadata from.

    Returns
    -------
    dict
        ``{"files": [{"file_path": ..., "num_ranks": ..., "metadata": {...}}]}``,
        matching the envelope used by
        :func:`~scope_profiler.plotting_scripts.collect_region_statistics`, so
        a single document can describe several runs.
    """
    if isinstance(profiling_data, (ProfilingH5Reader, str, Path)):
        profiling_data = [profiling_data]

    files = []
    for item in profiling_data:
        reader = (
            item if isinstance(item, ProfilingH5Reader) else ProfilingH5Reader(item)
        )
        files.append(
            {
                "file_path": str(Path(reader.file_path).resolve()),
                "num_ranks": reader.num_ranks,
                "metadata": {
                    key: _json_safe(value) for key, value in reader.metadata.items()
                },
            }
        )

    return {"files": files}


def write_metadata_json(
    profiling_data: ProfilingH5Reader | str | Path | Sequence,
    filepath: str | Path,
) -> dict:
    """Write the run metadata of one or more profiling files to JSON.

    Parameters
    ----------
    profiling_data : ProfilingH5Reader, path, or sequence of either
        Files to read the metadata from.
    filepath : str or Path
        Destination JSON file. Parent directories are created as needed.

    Returns
    -------
    dict
        The payload that was written (see :func:`collect_file_metadata`).
    """
    payload = collect_file_metadata(profiling_data)

    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``scope-profiler inspect``."""
    parser = argparse.ArgumentParser(
        prog="scope-profiler inspect",
        description="Print the metadata and region statistics of profiling files.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths or glob patterns for profiling_data.h5 files",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Only report regions whose name matches these regex patterns",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Skip regions whose name matches these regex patterns",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        help="Restrict region statistics to these ranks, e.g. 0 2 or 0-3",
    )
    parser.add_argument(
        "--sort",
        choices=_SORT_KEYS,
        default="total",
        help="Order regions by this column (default: total)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print long metadata values (PATH, LD_LIBRARY_PATH, ...) in full",
    )
    parser.add_argument(
        "--export-metadata",
        metavar="PATH",
        help="Also write the metadata of every inspected file to this JSON file",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the printed summary (useful with --export-metadata)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--metadata-only",
        action="store_true",
        help="Print only the metadata section",
    )
    group.add_argument(
        "--regions-only",
        action="store_true",
        help="Print only the region statistics",
    )
    return parser


def main(argv: list | None = None):
    """Entry point for ``scope-profiler inspect``."""
    from scope_profiler.post_processing import expand_file_patterns, parse_ranks

    parser = build_parser()
    args = parser.parse_args(argv)
    files = expand_file_patterns(args.files, parser)

    ranks = None
    if args.ranks:
        ranks = sorted({rank for spec in args.ranks for rank in parse_ranks(spec)})

    printed = 0
    for file_path in files:
        if args.quiet:
            continue
        if printed:
            print(file=sys.stdout)
        inspect_file(
            file_path,
            include=args.include,
            exclude=args.exclude,
            ranks=ranks,
            sort=args.sort,
            show_metadata=not args.regions_only,
            show_regions=not args.metadata_only,
            full=args.full,
        )
        printed += 1

    if args.export_metadata:
        write_metadata_json(files, args.export_metadata)
        print(f"Metadata written to {args.export_metadata}", file=sys.stdout)
