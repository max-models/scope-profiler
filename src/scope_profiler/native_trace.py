"""Reading the per-rank output written by the native (C and Fortran) region APIs.

The C and Fortran modules shipped in ``scope_profiler/c/`` and
``scope_profiler/fortran/`` record regions with no Python involved, and write
one small file per rank at ``sp_finalize()``.

A C build compiled with ``SP_USE_HDF5`` writes ``<prefix>_rank<NNNNN>.h5``
directly, in the same schema-2 layout :mod:`scope_profiler.h5writer`
produces: one rank's profile, already readable by
:func:`~scope_profiler.read_h5` with no import step. Everything this module
then does for those files is *merging* the ranks of one run into a single
profile.

Every other native build -- Fortran, and C without HDF5 -- writes the compact
binary trace documented below instead, one ``.spt`` per rank. Both write a
*compatible* trace format, so a program built from either (or both) lands in
one profile. This module turns those files into the same
:class:`~scope_profiler.results.ProfilingResults` -- and the same HDF5 layout
-- a Python run produces, so a Fortran run gets the whole post-processing
stack (summaries, plots, exporters, ``plot``) for free. The two kinds of
input mix freely in one import.

Trace layout, little- or big-endian, as written by ``sp_finalize``::

    char[8]   "SCOPEPRF"
    int32     format version (1 or 2)
    int32     rank
    int64     number of regions
    per region:
        int32     length of the name in bytes
        char[]    name
        -- version 2 only --
        int32     length of the source file path in bytes (0 if unknown)
        char[]    source file path
        int32     source line (-1 if unknown)
        -- end version 2 only --
        int64     number of calls
        int64[]   start timestamps, nanoseconds
        int64[]   end timestamps, nanoseconds

Version 1 is what the Fortran API writes, and what older C releases wrote: no
per-region source location. Version 2 is the current C API, which can attach
one via ``sp_region_at()``. Both are readable, per file, so a mixed C/Fortran
run merges normally.

The timestamps come from the same clock as :func:`time.perf_counter_ns`, so
Fortran and Python regions from one process tree share a timeline.
"""

from pathlib import Path

import numpy as np

#: Directory holding the Fortran sources shipped with the package.
FORTRAN_DIR = Path(__file__).resolve().parent / "fortran"

#: Directory holding the C sources shipped with the package.
C_DIR = Path(__file__).resolve().parent / "c"


def fortran_source_path() -> Path:
    """Path to ``scope_profiler.f90``, the module to compile into your program.

    It ships with the package, so this works from an installed wheel::

        gfortran -c $(python -c \
            "import scope_profiler.native_trace as t; print(t.fortran_source_path())")

    Returns
    -------
    Path
        The Fortran module source.
    """
    return FORTRAN_DIR / "scope_profiler.f90"


def c_source_path() -> Path:
    """Path to ``scope_profiler.c``, the implementation entry point to compile.

    Its public and private implementation headers sit next to it;
    :func:`c_include_dir` is what to put on the compiler's include path::

        cc -c $(python -c \
            "import scope_profiler.native_trace as t; print(t.c_source_path())") \
           -I$(python -c \
            "import scope_profiler.native_trace as t; print(t.c_include_dir())")

    Returns
    -------
    Path
        The C source file.
    """
    return C_DIR / "scope_profiler.c"


def c_include_dir() -> Path:
    """Directory holding ``scope_profiler.h``, for the compiler's ``-I``.

    Returns
    -------
    Path
        The include directory.
    """
    return C_DIR


MAGIC = b"SCOPEPRF"
"""First eight bytes of every trace file."""

FORMAT_VERSION = 2
"""Newest layout this module writes metadata for; ``sp_finalize`` writes the
same number. Version 1 (written by the Fortran API, and by older C releases)
has no per-region source location; version 2 (the C API) adds one. Both are
readable -- see :func:`read_trace`."""

KNOWN_FORMAT_VERSIONS = (1, 2)
"""Trace format versions this reader accepts."""

TRACE_SUFFIX = ".spt"
"""Extension ``sp_finalize`` gives a binary trace."""

HDF5_SUFFIX = ".h5"
"""Extension ``sp_finalize`` gives a directly-written HDF5 profile."""

RANK_HDF5_GLOB = f"*_rank[0-9]*{HDF5_SUFFIX}"
"""How :func:`find_traces` recognizes per-rank HDF5 output in a directory.

Matching the ``_rank<NNNNN>`` that ``sp_finalize`` writes, rather than every
``.h5``, keeps a merged profile sitting in the same directory (the output of
a previous import, say) from being swallowed as an input to the next one.
Naming such a file explicitly on the command line still reads it.
"""

_HEADER = np.dtype([("magic", "S8"), ("version", "i4"), ("rank", "i4")])


class TraceFormatError(ValueError):
    """A file is not a scope-profiler Fortran trace, or is truncated."""


class _RegionTrace:
    """Unpacks as ``(start_times, end_times)``, with the region's source
    location (if any) attached as extra attributes -- so existing
    ``starts, ends = regions[name]`` call sites keep working unchanged
    whether or not the trace carried source information.
    """

    __slots__ = ("end_times", "source_file", "source_lineno", "start_times")

    def __init__(
        self,
        start_times: np.ndarray,
        end_times: np.ndarray,
        source_file: str | None = None,
        source_lineno: int | None = None,
    ):
        self.start_times = start_times
        self.end_times = end_times
        self.source_file = source_file
        self.source_lineno = source_lineno

    def __getitem__(self, index):
        return (self.start_times, self.end_times)[index]

    def __iter__(self):
        return iter((self.start_times, self.end_times))

    def __len__(self) -> int:
        return 2


def _byte_order(buffer: bytes, path) -> tuple[str, int]:
    """Return ``("<" or ">", version)`` for the file's endianness and format.

    The magic is byte-order agnostic, so the version field decides: exactly
    one interpretation of it is a version we know.
    """
    for order in ("<", ">"):
        (version,) = np.frombuffer(buffer, dtype=f"{order}i4", count=1, offset=8)
        if int(version) in KNOWN_FORMAT_VERSIONS:
            return order, int(version)

    (little,) = np.frombuffer(buffer, dtype="<i4", count=1, offset=8)
    raise TraceFormatError(
        f"{path}: unsupported trace format version {int(little)} "
        f"(this scope-profiler reads versions {KNOWN_FORMAT_VERSIONS})",
    )


def read_trace(path) -> tuple:
    """Read one rank's trace file.

    Parameters
    ----------
    path : str or Path
        A ``.spt`` file written by ``sp_finalize()``.

    Returns
    -------
    tuple
        ``(rank, regions)``, where ``regions`` maps a region name to
        ``(start_times, end_times)`` int64 arrays in nanoseconds -- exactly the
        shape :class:`~scope_profiler.profile_manager.RankPayload` carries.
        Each value also carries ``.source_file`` / ``.source_lineno``
        attributes (both None on a version-1 trace, or a region registered
        without ``sp_region_at()``), without changing how it unpacks.

    Raises
    ------
    TraceFormatError
        If the file is not a trace, is truncated, or was written by a newer
        format version.
    """
    path = Path(path)
    buffer = path.read_bytes()

    if len(buffer) < _HEADER.itemsize + 8:
        raise TraceFormatError(f"{path}: too short to be a trace file")
    if buffer[:8] != MAGIC:
        raise TraceFormatError(
            f"{path}: not a scope-profiler Fortran trace "
            f"(expected {MAGIC!r}, found {buffer[:8]!r})",
        )

    order, version = _byte_order(buffer, path)
    i4 = np.dtype(f"{order}i4")
    i8 = np.dtype(f"{order}i8")

    rank = int(np.frombuffer(buffer, dtype=i4, count=1, offset=12)[0])
    offset = 16
    (num_regions,) = np.frombuffer(buffer, dtype=i8, count=1, offset=offset)
    offset += 8

    regions = {}
    for _ in range(int(num_regions)):
        try:
            (name_len,) = np.frombuffer(buffer, dtype=i4, count=1, offset=offset)
            offset += 4
            name = buffer[offset : offset + int(name_len)].decode("utf-8")
            offset += int(name_len)

            source_file = None
            source_lineno = None
            if version >= 2:
                (source_len,) = np.frombuffer(buffer, dtype=i4, count=1, offset=offset)
                offset += 4
                if int(source_len):
                    source_file = buffer[offset : offset + int(source_len)].decode(
                        "utf-8"
                    )
                offset += int(source_len)
                (source_line,) = np.frombuffer(buffer, dtype=i4, count=1, offset=offset)
                offset += 4
                if int(source_line) >= 0:
                    source_lineno = int(source_line)

            (num_calls,) = np.frombuffer(buffer, dtype=i8, count=1, offset=offset)
            offset += 8
            count = int(num_calls)
            starts = np.frombuffer(buffer, dtype=i8, count=count, offset=offset)
            offset += 8 * count
            ends = np.frombuffer(buffer, dtype=i8, count=count, offset=offset)
            offset += 8 * count
        except ValueError as exc:
            raise TraceFormatError(f"{path}: truncated trace file ({exc})") from exc

        # Copy out of the read-only buffer, and normalize to native int64 so
        # everything downstream sees the same dtype a Python run produces.
        regions[name] = _RegionTrace(
            np.ascontiguousarray(starts, dtype=np.int64),
            np.ascontiguousarray(ends, dtype=np.int64),
            source_file=source_file,
            source_lineno=source_lineno,
        )

    if offset != len(buffer):
        raise TraceFormatError(
            f"{path}: {len(buffer) - offset} trailing byte(s) after the last region",
        )
    return rank, regions


def find_traces(inputs) -> list:
    """Collect native output files from paths, directories, or a mix of both.

    Parameters
    ----------
    inputs : path or sequence of paths
        Files to read, and/or directories to search (non-recursively) for
        ``*.spt`` traces and ``*_rank<NNNNN>.h5`` per-rank profiles.

    Returns
    -------
    list of Path
        The files, sorted, with duplicates removed.

    Raises
    ------
    FileNotFoundError
        If an input does not exist, or a directory holds neither kind of file.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    found = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            in_dir = sorted(
                [*path.glob(f"*{TRACE_SUFFIX}"), *path.glob(RANK_HDF5_GLOB)],
            )
            if not in_dir:
                raise FileNotFoundError(
                    f"no {TRACE_SUFFIX} traces and no per-rank {HDF5_SUFFIX} "
                    f"profiles in {path}",
                )
            found.extend(in_dir)
        elif path.exists():
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")

    return sorted(set(found))


def read_native_h5(path) -> tuple:
    """Read the per-rank HDF5 profile an ``SP_USE_HDF5`` C build writes.

    The file is an ordinary schema-2 profile holding one rank, so this is
    :func:`~scope_profiler.read_h5` plus a regrouping into the ``(rank,
    regions)`` shape :func:`read_trace` returns -- which is what lets one
    import mix directly-written HDF5 with ``.spt`` traces from other ranks.

    Parameters
    ----------
    path : str or Path
        An ``.h5`` file written by ``sp_finalize()``. A file holding several
        ranks (a merged profile) is accepted too, and contributes all of them.

    Returns
    -------
    tuple
        ``(ranks, metadata)``, where ``ranks`` maps a rank to its
        ``{region name: Region}`` and ``metadata`` is the run metadata stored
        in the file.
    """
    from scope_profiler.h5reader import read_h5

    results = read_h5(path)
    ranks: dict = {}
    for region in results.get_regions():
        for rank, data in region.regions.items():
            ranks.setdefault(int(rank), {})[region.name] = data
    return ranks, dict(results.metadata)


def read_native_ranks(path) -> tuple:
    """Read one native output file, whichever of the two formats it is in.

    The format-agnostic entry point: callers that must accept whatever a
    native build happened to write -- :func:`load_traces` and
    ``ProfileManager.finalize(native_traces=...)`` -- go through this rather
    than choosing :func:`read_trace` or :func:`read_native_h5` by suffix
    themselves.

    Parameters
    ----------
    path : str or Path
        A ``.spt`` trace or an ``.h5`` profile written by ``sp_finalize()``.

    Returns
    -------
    tuple
        ``(ranks, metadata)``, where ``ranks`` maps a rank to its
        ``{region name: Region}``. A ``.spt`` trace contributes exactly one
        rank and no metadata; an ``.h5`` file contributes every rank it holds
        (one, as ``sp_finalize()`` writes it) and the metadata it stores.
    """
    from scope_profiler.region import Region

    if Path(path).suffix == HDF5_SUFFIX:
        return read_native_h5(path)

    rank, regions = read_trace(path)
    return {
        rank: {
            name: Region(
                *region_trace,
                source_file=region_trace.source_file,
                source_lineno=region_trace.source_lineno,
            )
            for name, region_trace in regions.items()
        },
    }, {}


def load_traces(inputs, label: str | None = None):
    """Read native output into the standard post-processing API.

    Both formats are accepted, and mix: ``.spt`` traces from a Fortran or
    plain C build, and the per-rank ``.h5`` profiles an ``SP_USE_HDF5`` C
    build writes. Every rank of a run lands in one result either way.

    Parameters
    ----------
    inputs : path or sequence of paths
        Files and/or directories containing them (see :func:`find_traces`).
    label : str, optional
        Name for the run in summaries, charts and exports.

    Returns
    -------
    ProfilingResults
        The same object a Python run produces, so every summary, plot and
        exporter works on it unchanged.

    Raises
    ------
    TraceFormatError
        If two files claim the same rank.
    """
    from scope_profiler.mpi_region import MPIRegion
    from scope_profiler.results import ProfilingResults

    paths = find_traces(inputs)

    per_region: dict = {}
    seen_ranks: dict = {}
    # A directly-written HDF5 file carries the environment its rank recorded
    # (hostname, timestamp, ...). Keep the lowest rank's, the way a merged
    # run stores rank 0's, and let the fields derived below override it.
    file_metadata: dict = {}
    metadata_rank: int | None = None
    earliest = None
    for path in paths:
        ranks, metadata = read_native_ranks(path)
        for rank, regions in sorted(ranks.items()):
            if rank in seen_ranks:
                raise TraceFormatError(
                    f"{path} and {seen_ranks[rank]} both claim rank {rank}; "
                    f"pass the MPI rank to sp_init() so each rank writes its own",
                )
            seen_ranks[rank] = path
            if metadata and (metadata_rank is None or rank < metadata_rank):
                file_metadata = metadata
                metadata_rank = rank
            for name, region in regions.items():
                per_region.setdefault(name, {})[rank] = region
                starts = region.start_times_ns
                if starts.size:
                    first = int(starts[0])
                    earliest = first if earliest is None else min(earliest, first)

    metadata = dict(file_metadata)
    metadata.update({"source": "native", "trace_format_version": FORMAT_VERSION})
    if earliest is not None:
        # The timeline origin, exactly as a Python run records it at setup().
        metadata["start_time_ns"] = earliest
    if label is not None:
        metadata["label"] = label

    return ProfilingResults(
        {
            name: MPIRegion(name=name, regions=dict(sorted(ranks.items())))
            for name, ranks in per_region.items()
        },
        metadata=metadata,
        num_ranks=len(seen_ranks),
        file_path=label or "native_trace",
    )


def write_results(results, output_path):
    """Write any :class:`ProfilingResults` out as a standard HDF5 file.

    Goes through :class:`~scope_profiler.h5writer.ProfilingWriter`, so the
    result has exactly the layout a Python run produces -- which is what lets
    an imported (or merged) run be read back by
    :func:`~scope_profiler.read_h5` and fed to every plot and exporter.

    Parameters
    ----------
    results : ProfilingResults
        The run to write.
    output_path : str or Path
        HDF5 file to create.

    Returns
    -------
    Path
        The file that was written.
    """
    from scope_profiler.h5writer import ProfilingWriter
    from scope_profiler.profile_manager import RankPayload

    output_path = Path(output_path)

    # Regroup by rank: the writer emits one group per rank, as finalize() does.
    by_rank: dict = {}
    sources_by_rank: dict = {}
    for region in results.get_regions():
        for rank, data in region.regions.items():
            by_rank.setdefault(rank, {})[region.name] = (
                data.start_times_ns,
                data.end_times_ns,
            )
            if data.source_file is not None:
                # Carried through so an import keeps the location a region was
                # registered at (sp_region_at(), or a Python decorator), which
                # `inspect` and the reports show next to the region name.
                sources_by_rank.setdefault(rank, {})[region.name] = (
                    data.source_file,
                    -1 if data.source_lineno is None else data.source_lineno,
                    data.source_text or "",
                )

    likwid = results.get_likwid_regions()
    with ProfilingWriter(output_path, results.metadata) as writer:
        for rank in sorted(set(by_rank) | set(likwid)):
            writer.write_rank(
                rank,
                RankPayload(
                    regions=by_rank.get(rank, {}),
                    likwid=likwid.get(rank, {}),
                    likwid_environment={},
                    sources=sources_by_rank.get(rank),
                ),
            )
    return output_path


def convert_traces(inputs, output_path, label: str | None = None):
    """Convert Fortran traces into a standard scope-profiler HDF5 file.

    The result is indistinguishable from one a Python run wrote, so
    ``scope-profiler plot`` / ``inspect`` and :func:`~scope_profiler.read_h5`
    work on it directly.

    Parameters
    ----------
    inputs : path or sequence of paths
        Trace files and/or directories containing them.
    output_path : str or Path
        HDF5 file to write.
    label : str, optional
        Name for the run; defaults to the output file's stem.

    Returns
    -------
    Path
        The file that was written.
    """
    output_path = Path(output_path)
    return write_results(
        load_traces(inputs, label=label or output_path.stem),
        output_path,
    )
