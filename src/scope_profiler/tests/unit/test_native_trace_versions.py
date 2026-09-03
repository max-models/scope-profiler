"""Format-version handling in native_trace.read_trace(), without a C compiler.

The C API writes format version 2 (adds an optional source file/line per
region); the Fortran API still writes version 1. Both must stay readable from
the same trace directory, and read_trace()'s return value must keep unpacking
as ``(start_times, end_times)`` regardless of which version produced it. These
tests build the bytes by hand instead of compiling and running a program, so
they run everywhere the rest of test_c_api.py is skipped.
"""

import struct

import numpy as np
import pytest

from scope_profiler.native_trace import MAGIC, TraceFormatError, load_traces, read_trace


def _write_v1_trace(path, rank, regions):
    """regions: name -> (starts, ends), version-1 layout (no source fields)."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", rank))
        f.write(struct.pack("<q", len(regions)))
        for name, (starts, ends) in regions.items():
            encoded = name.encode("utf-8")
            f.write(struct.pack("<i", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<q", len(starts)))
            f.write(np.asarray(starts, dtype="<i8").tobytes())
            f.write(np.asarray(ends, dtype="<i8").tobytes())


def _write_v2_trace(path, rank, regions):
    """regions: name -> (starts, ends, source_file, source_line)."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", 2))
        f.write(struct.pack("<i", rank))
        f.write(struct.pack("<q", len(regions)))
        for name, (starts, ends, source_file, source_line) in regions.items():
            encoded = name.encode("utf-8")
            f.write(struct.pack("<i", len(encoded)))
            f.write(encoded)
            if source_file is None:
                f.write(struct.pack("<i", 0))
            else:
                encoded_source = source_file.encode("utf-8")
                f.write(struct.pack("<i", len(encoded_source)))
                f.write(encoded_source)
            f.write(struct.pack("<i", source_line if source_line is not None else -1))
            f.write(struct.pack("<q", len(starts)))
            f.write(np.asarray(starts, dtype="<i8").tobytes())
            f.write(np.asarray(ends, dtype="<i8").tobytes())


def test_version_1_trace_has_no_source_and_still_unpacks(tmp_path):
    path = tmp_path / "v1_rank00000.spt"
    _write_v1_trace(path, rank=0, regions={"step": ([100, 200], [150, 260])})

    rank, regions = read_trace(path)

    assert rank == 0
    starts, ends = regions["step"]  # must still unpack as a 2-tuple
    assert list(starts) == [100, 200]
    assert list(ends) == [150, 260]
    assert regions["step"].source_file is None
    assert regions["step"].source_lineno is None


def test_version_2_trace_carries_source_location(tmp_path):
    path = tmp_path / "v2_rank00000.spt"
    _write_v2_trace(
        path,
        rank=0,
        regions={"solve": ([10], [20], "solver.c", 42)},
    )

    _, regions = read_trace(path)

    starts, ends = regions["solve"]
    assert list(starts) == [10]
    assert list(ends) == [20]
    assert regions["solve"].source_file == "solver.c"
    assert regions["solve"].source_lineno == 42


def test_version_2_trace_with_no_source_reads_as_none(tmp_path):
    path = tmp_path / "v2_rank00000.spt"
    _write_v2_trace(path, rank=0, regions={"plain": ([1], [2], None, None)})

    _, regions = read_trace(path)

    assert regions["plain"].source_file is None
    assert regions["plain"].source_lineno is None


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "future_rank00000.spt"
    _write_v1_trace(path, rank=0, regions={"step": ([1], [2])})
    data = bytearray(path.read_bytes())
    data[8:12] = struct.pack("<i", 99)
    path.write_bytes(bytes(data))

    with pytest.raises(TraceFormatError, match="unsupported trace format version"):
        read_trace(path)


def test_load_traces_carries_source_into_region(tmp_path):
    _write_v2_trace(
        tmp_path / "trace_rank00000.spt",
        rank=0,
        regions={"solve": ([10, 30], [20, 45], "solver.c", 42)},
    )

    results = load_traces(tmp_path)

    region = results["solve"]
    assert region.source_file == "solver.c"
    assert region.source_lineno == 42


def test_mixed_v1_and_v2_traces_merge(tmp_path):
    """A C rank (v2) and a Fortran rank (v1) writing to the same directory."""
    _write_v2_trace(
        tmp_path / "trace_rank00000.spt",
        rank=0,
        regions={"shared": ([0], [10], "c_side.c", 7)},
    )
    _write_v1_trace(
        tmp_path / "trace_rank00001.spt",
        rank=1,
        regions={"shared": ([0], [15])},
    )

    results = load_traces(tmp_path)

    assert results.num_ranks == 2
    assert results["shared"].num_calls == 2
