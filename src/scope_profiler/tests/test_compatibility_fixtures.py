"""Read immutable examples of every supported storage format.

Round-trip tests prove that today's writer and reader agree. These fixtures
pin the stronger compatibility promise: today's reader still understands
bytes that were committed independently of the current writer.
"""

import base64
import gzip
from pathlib import Path

import h5py
import pytest

from scope_profiler import read_h5, read_json
from scope_profiler.native_trace import read_trace

FIXTURES = Path(__file__).with_name("fixtures") / "compatibility"
STARTS = [10, 20, 40]
ENDS = [15, 33, 55]


def _decode_fixture(name: str, destination: Path, *, compressed: bool = False) -> Path:
    payload = base64.b64decode((FIXTURES / name).read_text(encoding="ascii"))
    destination.write_bytes(gzip.decompress(payload) if compressed else payload)
    return destination


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_supported_hdf5_schemas_remain_readable(schema, tmp_path):
    path = _decode_fixture(
        f"hdf5-schema-{schema}.h5.gz.b64",
        tmp_path / f"schema-{schema}.h5",
        compressed=True,
    )

    with h5py.File(path, "r") as handle:
        # Schema 1 predates the attribute; absence is itself part of the
        # historical compatibility contract.
        assert int(handle.attrs.get("scope_profiler_schema", 1)) == schema
        assert ("scope_profiler_schema" in handle.attrs) == (schema >= 2)

    results = read_h5(path)
    region = results["solve"][0]
    assert results.num_ranks == 1
    assert region.start_times_ns.tolist() == STARTS
    assert region.end_times_ns.tolist() == ENDS
    assert region.source_lineno == 7
    assert region.tags == ("golden",)


def test_supported_json_profile_remains_readable():
    results = read_json(FIXTURES / "json-v1.json")
    region = results["solve"][0]

    assert results.num_ranks == 1
    assert region.start_times_ns.tolist() == STARTS
    assert region.end_times_ns.tolist() == ENDS
    assert region.source_file == "solver.py"
    assert region.source_lineno == 7
    assert region.tags == ("golden",)


@pytest.mark.parametrize("version", [1, 2])
def test_supported_native_traces_remain_readable(version, tmp_path):
    path = _decode_fixture(
        f"native-v{version}.spt.b64",
        tmp_path / f"native-v{version}.spt",
    )

    rank, regions = read_trace(path)
    region = regions["solve"]
    assert rank == 0
    assert region.start_times.tolist() == STARTS
    assert region.end_times.tolist() == ENDS
    if version == 2:
        assert region.source_file == "solver.c"
        assert region.source_lineno == 7
    else:
        assert region.source_file is None
        assert region.source_lineno is None
