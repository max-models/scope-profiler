"""Reading the trace files written by the native (C and Fortran) region APIs.

The C and Fortran modules shipped in ``scope_profiler/c/`` and
``scope_profiler/fortran/`` record regions with no HDF5 and no Python
involved, and dump one small binary file per rank at ``sp_finalize()``. Both
write the *same* format, so a program built from either -- or both -- lands in
one profile. This module turns those files into the same
:class:`~scope_profiler.results.ProfilingResults` -- and the same HDF5 layout
-- a Python run produces, so a Fortran run gets the whole post-processing
stack (summaries, plots, exporters, ``plot``) for free.

Trace layout, little- or big-endian, as written by ``sp_finalize``::

    char[8]   "SCOPEPRF"
    int32     format version (1)
    int32     rank
    int64     number of regions
    per region:
        int32     length of the name in bytes
        char[]    name
        int64     number of calls
        int64[]   start timestamps, nanoseconds
        int64[]   end timestamps, nanoseconds

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
    """Path to ``scope_profiler.c``, the implementation to compile in.

    Its header sits next to it; :func:`c_include_dir` is what to put on the
    compiler's include path::

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

FORMAT_VERSION = 1
"""Layout this module reads; ``sp_finalize`` writes the same number."""

TRACE_SUFFIX = ".spt"
"""Extension ``sp_finalize`` gives its output."""

_HEADER = np.dtype([("magic", "S8"), ("version", "i4"), ("rank", "i4")])


class TraceFormatError(ValueError):
    """A file is not a scope-profiler Fortran trace, or is truncated."""


def _byte_order(buffer: bytes, path) -> str:
    """Return ``"<"`` or ``">"`` for the endianness the file was written with.

    The magic is byte-order agnostic, so the version field decides: exactly
    one interpretation of it is the version we know.
    """
    for order in ("<", ">"):
        (version,) = np.frombuffer(buffer, dtype=f"{order}i4", count=1, offset=8)
        if version == FORMAT_VERSION:
            return order

    (little,) = np.frombuffer(buffer, dtype="<i4", count=1, offset=8)
    raise TraceFormatError(
        f"{path}: unsupported trace format version {int(little)} "
        f"(this scope-profiler reads version {FORMAT_VERSION})"
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
            f"(expected {MAGIC!r}, found {buffer[:8]!r})"
        )

    order = _byte_order(buffer, path)
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
        regions[name] = (
            np.ascontiguousarray(starts, dtype=np.int64),
            np.ascontiguousarray(ends, dtype=np.int64),
        )

    if offset != len(buffer):
        raise TraceFormatError(
            f"{path}: {len(buffer) - offset} trailing byte(s) after the last region"
        )
    return rank, regions


def find_traces(inputs) -> list:
    """Collect trace files from paths, directories, or a mix of both.

    Parameters
    ----------
    inputs : path or sequence of paths
        Files to read, and/or directories to search (non-recursively) for
        ``*.spt``.

    Returns
    -------
    list of Path
        The trace files, sorted, with duplicates removed.

    Raises
    ------
    FileNotFoundError
        If an input does not exist, or a directory holds no trace files.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    found = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            in_dir = sorted(path.glob(f"*{TRACE_SUFFIX}"))
            if not in_dir:
                raise FileNotFoundError(f"no {TRACE_SUFFIX} trace files in {path}")
            found.extend(in_dir)
        elif path.exists():
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")

    return sorted(set(found))


def load_traces(inputs, label: str | None = None):
    """Read Fortran traces into the standard post-processing API.

    Parameters
    ----------
    inputs : path or sequence of paths
        Trace files and/or directories containing them (see :func:`find_traces`).
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
        If two trace files claim the same rank.
    """
    from scope_profiler.mpi_region import MPIRegion
    from scope_profiler.region import Region
    from scope_profiler.results import ProfilingResults

    paths = find_traces(inputs)

    per_region: dict = {}
    seen_ranks: dict = {}
    earliest = None
    for path in paths:
        rank, regions = read_trace(path)
        if rank in seen_ranks:
            raise TraceFormatError(
                f"{path} and {seen_ranks[rank]} both claim rank {rank}; "
                f"pass the MPI rank to sp_init() so each rank writes its own"
            )
        seen_ranks[rank] = path
        for name, (starts, ends) in regions.items():
            per_region.setdefault(name, {})[rank] = Region(starts, ends)
            if starts.size:
                first = int(starts[0])
                earliest = first if earliest is None else min(earliest, first)

    metadata = {"source": "native", "trace_format_version": FORMAT_VERSION}
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
    for region in results.get_regions():
        for rank, data in region.regions.items():
            by_rank.setdefault(rank, {})[region.name] = (
                data.start_times_ns,
                data.end_times_ns,
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
        load_traces(inputs, label=label or output_path.stem), output_path
    )
