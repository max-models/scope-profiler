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
from scope_profiler.summary import SORT_KEYS, print_region_table, region_rows

# Metadata is printed in these groups, in this order, so the fields that
# identify a run come first and the sprawling environment variables last.
# Anything not listed here still gets printed (see _metadata_sections), so new
# metadata fields never silently disappear from the output.
_RUN_FIELDS = (
    "timestamp",
    "start_time_ns",
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


def _time_span(reader) -> float | None:
    """Wall-clock seconds of the run, or None when nothing was timed."""
    if not any(region.has_timing for region in reader.get_regions()):
        return None
    return reader.time_span


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

    rows = region_rows(reader, include=include, exclude=exclude, ranks=ranks, sort=sort)
    print_region_table(rows, title=f"Regions ({len(rows)})", stream=stream)


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
        choices=SORT_KEYS,
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
