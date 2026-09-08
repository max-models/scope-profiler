"""Storage layout of the published profile: encoding, packing, compression.

These cover the three things that decide how large an output file is, and the
one thing that matters more than any of them --- that none of it changes what
the file says. Every case here asserts the round trip as well as the size.

Schema-2 files are built by hand rather than by an old writer: what has to
keep working is the *reader*, and a hand-built file pins the old layout down
more precisely than a fixture recorded from a previous release would.
"""

import h5py
import numpy as np
import pytest

from scope_profiler import ProfileManager, load
from scope_profiler.h5writer import (
    AUTO_COMPRESSION_MIN_EVENTS,
    decode_start_deltas,
    encode_durations,
    encode_start_deltas,
    publish_file,
)


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


def _profile(path, regions=1, calls=1, **setup):
    with ProfileManager.session(file_path=str(path), verbose=False, **setup):
        for index in range(regions):
            region = ProfileManager.region(f"r{index}")
            for _ in range(calls):
                with region:
                    pass


# --- the encoding -----------------------------------------------------------


def test_start_deltas_round_trip_exactly():
    starts = np.array([1_700_000_000_000, 1_700_000_000_417, 1_700_000_001_022])

    encoded = encode_start_deltas(starts)

    # First value absolute, the rest gaps -- which is the whole point: the
    # gaps are three orders of magnitude smaller than the timestamps.
    assert encoded.tolist() == [1_700_000_000_000, 417, 605]
    assert decode_start_deltas(encoded).tolist() == starts.tolist()


def test_durations_round_trip_exactly():
    starts = np.array([100, 200, 300])
    ends = np.array([150, 260, 330])

    durations = encode_durations(starts, ends)

    assert durations.tolist() == [50, 60, 30]
    assert (starts + durations).tolist() == ends.tolist()


def test_encoding_an_empty_run_stays_empty():
    empty = np.empty(0, dtype=np.int64)

    assert encode_start_deltas(empty).size == 0
    assert decode_start_deltas(empty).size == 0
    assert encode_durations(empty, empty).size == 0


def test_the_file_stores_the_encoded_columns(tmp_path):
    path = tmp_path / "encoded.h5"
    _profile(path, regions=2, calls=4)

    with h5py.File(path, "r") as handle:
        assert set(handle["events"]) >= {"start_deltas", "durations"}
        assert "start_times" not in handle["events"]
        assert "end_times" not in handle["events"]


def test_every_timestamp_survives_the_encoding(tmp_path):
    """The file must reproduce the in-memory run exactly, not approximately."""
    path = tmp_path / "exact.h5"
    with ProfileManager.session(
        file_path=str(path),
        verbose=False,
        return_results=True,
    ) as run:
        for index in range(3):
            region = ProfileManager.region(f"r{index}")
            for _ in range(25):
                with region:
                    pass

    memory, disk = run.results, load(path)
    assert disk.region_names == memory.region_names
    for name in memory.region_names:
        recorded, restored = memory[name][0], disk[name][0]
        assert restored.start_times_ns.tolist() == recorded.start_times_ns.tolist()
        assert restored.end_times_ns.tolist() == recorded.end_times_ns.tolist()
    assert disk.summary() == memory.summary()


# --- reading the older layout ----------------------------------------------


def _write_schema_two(path):
    """A schema-2 file: absolute timestamps, under the old column names."""
    starts = np.array([10, 20, 40], dtype=np.int64)
    ends = np.array([15, 33, 55], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.attrs["scope_profiler_schema"] = 2
        handle.attrs["storage_layout"] = "columnar"
        handle.create_group("metadata")
        handle.create_dataset(
            "region_table/names",
            data=np.array(["solve"], dtype=h5py.string_dtype()),
        )
        index = handle.create_group("rank_region_index")
        index.create_dataset("region_ids", data=np.array([0], dtype=np.uint32))
        index.create_dataset("ranks", data=np.array([0], dtype=np.uint32))
        index.create_dataset("event_offsets", data=np.array([0], dtype=np.uint64))
        index.create_dataset("event_counts", data=np.array([3], dtype=np.uint64))
        index.create_dataset("source_lines", data=np.array([-1], dtype=np.int64))
        index.create_dataset("exclusive_totals", data=np.array([-1], dtype=np.int64))
        for name in ("source_files", "source_texts", "tags"):
            index.create_dataset(
                name,
                data=np.array([""], dtype=h5py.string_dtype()),
            )
        events = handle.create_group("events")
        events.create_dataset("start_times", data=starts)
        events.create_dataset("end_times", data=ends)
    return starts, ends


def test_a_schema_two_file_still_reads(tmp_path):
    path = tmp_path / "old.h5"
    starts, ends = _write_schema_two(path)

    region = load(path)["solve"][0]

    assert region.start_times_ns.tolist() == starts.tolist()
    assert region.end_times_ns.tolist() == ends.tolist()
    assert region.num_calls == 3


def test_a_schema_two_file_still_reads_summary_only(tmp_path):
    """The summary reader needs the index, which schema 2 already had."""
    from scope_profiler import read_h5_summary

    path = tmp_path / "old.h5"
    _write_schema_two(path)

    # No summary_statistics column in this hand-built file, so the reader
    # falls back to a full read rather than failing on the schema version.
    assert read_h5_summary(path).region_names == ["solve"]


# --- packing and compression ------------------------------------------------


def test_a_small_profile_is_stored_contiguously(tmp_path):
    """The published file pays no per-dataset chunk overhead."""
    path = tmp_path / "small.h5"
    _profile(path)

    with h5py.File(path, "r") as handle:
        chunked = []

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.chunks is not None:
                chunked.append(name)

        handle.visititems(collect)
        assert chunked == []
    # An order of magnitude below the ~190 KiB an unpacked file cost.
    assert path.stat().st_size < 40 * 1024


def test_auto_compression_leaves_a_small_profile_alone(tmp_path):
    """Below the threshold the filter costs more than it saves."""
    path = tmp_path / "small_auto.h5"
    _profile(path, calls=8, hdf5_compression="auto")

    with h5py.File(path, "r") as handle:
        assert handle["events/start_deltas"].compression is None
    assert load(path)["r0"][0].num_calls == 8


def test_auto_compression_compresses_a_large_profile(tmp_path):
    path = tmp_path / "large_auto.h5"
    calls = AUTO_COMPRESSION_MIN_EVENTS + 1
    _profile(path, calls=calls, hdf5_compression="auto")

    with h5py.File(path, "r") as handle:
        dataset = handle["events/start_deltas"]
        assert dataset.compression == "gzip"
        assert dataset.shuffle is True
    assert load(path)["r0"][0].num_calls == calls


def test_an_explicit_filter_still_wins_over_auto(tmp_path):
    path = tmp_path / "explicit.h5"
    _profile(path, calls=64, hdf5_compression="lzf")

    with h5py.File(path, "r") as handle:
        assert handle["events/start_deltas"].compression == "lzf"


def test_auto_is_accepted_as_a_compression_setting():
    ProfileManager.setup(deactivate_file_output=True, hdf5_compression="auto")

    assert ProfileManager.get_config().hdf5_compression == "auto"


def test_an_unknown_compression_name_is_rejected():
    with pytest.raises(ValueError, match="hdf5_compression must be"):
        ProfileManager.setup(deactivate_file_output=True, hdf5_compression="brotli")


def test_publishing_a_missing_file_reports_rather_than_raises(tmp_path):
    assert publish_file(tmp_path / "absent.h5") is False


def test_an_explicit_chunk_size_survives_publication(tmp_path):
    """``hdf5_chunk_size`` asks for chunked storage; packing must not undo it.

    Chunking is what makes partial reads possible without compression, so a
    caller who set it explicitly gets it, even on a file small enough that the
    packing pass would otherwise store it contiguously.
    """
    path = tmp_path / "chunked.h5"
    _profile(path, calls=6, hdf5_chunk_size=2)

    with h5py.File(path, "r") as handle:
        assert handle["events/start_deltas"].chunks == (2,)
    assert load(path)["r0"][0].num_calls == 6


def test_the_chunk_size_reaches_the_index_datasets_too(tmp_path):
    """It used to reach only the event columns."""
    path = tmp_path / "all_chunked.h5"
    _profile(path, calls=6, hdf5_chunk_size=2)

    with h5py.File(path, "r") as handle:
        assert handle["rank_region_index/ranks"].chunks == (2,)
        assert handle["region_table/names"].chunks == (2,)
