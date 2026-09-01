"""What the reader does with a file it cannot trust.

A profiling file is written at the end of a run, which is exactly when a job
gets killed, runs out of disk, or hits its wall-clock limit. The half-written
result then outlives the run, and the next thing that happens is a user
pointing the CLI at it. These tests pin what that user is told.
"""

import h5py
import numpy as np
import pytest

from scope_profiler import read_h5, read_h5_summary
from scope_profiler.h5reader import CorruptProfileError, load_h5
from scope_profiler.h5schema import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_ATTRIBUTE,
    HDF5SchemaError,
    read_schema_version,
)
from scope_profiler.h5writer import ProfilingWriter
from scope_profiler.profile_manager import RankPayload

NS = 1_000_000_000


def _payload(regions):
    return RankPayload(
        regions=regions,
        likwid={},
        likwid_environment={},
        line_profile=None,
        exclusive_totals=None,
    )


@pytest.fixture
def profile(tmp_path):
    """A small, valid two-region profile to damage in different ways."""
    path = tmp_path / "profiling_data.h5"
    with ProfilingWriter(path, {"hostname": "node0"}) as writer:
        writer.write_rank(
            0,
            _payload(
                {
                    "solve": (
                        np.array([0, 10 * NS], dtype=np.int64),
                        np.array([5 * NS, 20 * NS], dtype=np.int64),
                    ),
                    "io": (
                        np.array([21 * NS], dtype=np.int64),
                        np.array([22 * NS], dtype=np.int64),
                    ),
                }
            ),
        )
    return path


def test_a_missing_file_is_named_in_the_error(tmp_path):
    missing = tmp_path / "never_written.h5"
    with pytest.raises(FileNotFoundError, match="never_written.h5"):
        read_h5(missing)


def test_a_truncated_file_is_reported_as_an_interrupted_run(profile, tmp_path):
    """The realistic corruption: the job died mid-write."""
    truncated = tmp_path / "truncated.h5"
    truncated.write_bytes(profile.read_bytes()[: profile.stat().st_size // 3])

    with pytest.raises(CorruptProfileError) as excinfo:
        read_h5(truncated)

    message = str(excinfo.value)
    assert "truncated.h5" in message
    assert "interrupted while writing" in message


def test_a_file_that_is_not_hdf5_at_all_is_rejected(tmp_path):
    not_hdf5 = tmp_path / "notes.h5"
    not_hdf5.write_text("start_times = [1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(CorruptProfileError, match="not a readable HDF5"):
        read_h5(not_hdf5)


def test_an_empty_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.h5"
    empty.write_bytes(b"")

    with pytest.raises(CorruptProfileError, match="not a readable HDF5"):
        read_h5(empty)


def test_a_corrupt_file_still_raises_oserror_for_older_handlers(profile, tmp_path):
    """The new error stays an OSError, so existing `except OSError` keeps working."""
    truncated = tmp_path / "truncated.h5"
    truncated.write_bytes(profile.read_bytes()[:1024])

    with pytest.raises(OSError):
        read_h5(truncated)


def test_the_summary_reader_reports_corruption_the_same_way(profile, tmp_path):
    truncated = tmp_path / "truncated.h5"
    truncated.write_bytes(profile.read_bytes()[:1024])

    with pytest.raises(CorruptProfileError, match="not a readable HDF5"):
        read_h5_summary(truncated)


def test_a_file_from_a_newer_scope_profiler_says_to_upgrade(profile):
    with h5py.File(profile, "r+") as handle:
        handle.attrs[SCHEMA_ATTRIBUTE] = CURRENT_SCHEMA_VERSION + 1

    with pytest.raises(HDF5SchemaError) as excinfo:
        read_h5(profile)

    message = str(excinfo.value)
    assert f"version {CURRENT_SCHEMA_VERSION + 1}" in message
    assert "Upgrade scope-profiler" in message


@pytest.mark.parametrize("version", [0, -1])
def test_a_non_positive_schema_version_is_rejected(profile, version):
    with h5py.File(profile, "r+") as handle:
        handle.attrs[SCHEMA_ATTRIBUTE] = version

    with pytest.raises(HDF5SchemaError, match="must be positive"):
        read_h5(profile)


@pytest.mark.parametrize("version", ["2", 2.0, True])
def test_a_non_integer_schema_version_is_rejected(profile, version):
    with h5py.File(profile, "r+") as handle:
        handle.attrs[SCHEMA_ATTRIBUTE] = version

    with pytest.raises(HDF5SchemaError, match="must be an integer"):
        read_h5(profile)


def test_a_file_with_no_schema_attribute_is_read_as_the_original_layout(tmp_path):
    """Files written before versioning stay readable."""
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        assert SCHEMA_ATTRIBUTE not in handle.attrs
        assert read_schema_version(handle) == 1


def test_the_gpu_column_is_absent_unless_the_run_recorded_one(profile):
    """A CPU-only run writes no GPU column, and the reader must not need one."""
    with h5py.File(profile, "r") as handle:
        assert "gpu_durations" not in handle["events"]

    results = read_h5(profile)

    assert results["solve"][0].gpu_durations is None
    assert results["solve"][0].num_calls == 2


def test_a_deleted_optional_column_still_reads(profile):
    """Call-id/parent-id columns are optional to the reader by design."""
    with h5py.File(profile, "r+") as handle:
        del handle["events/call_ids"]
        del handle["events/parent_ids"]

    results = read_h5(profile)

    assert results["solve"][0].call_ids is None
    assert results["solve"][0].parent_ids is None
    assert results["solve"][0].num_calls == 2


def test_a_missing_exclusive_totals_column_falls_back_to_reconstruction(
    profile, tmp_path
):
    """Older files have no stored totals; the reader derives them instead."""
    with h5py.File(profile, "r+") as handle:
        del handle["rank_region_index/exclusive_totals"]

    results = read_h5(profile)

    # solve's two calls are disjoint, so exclusive time equals inclusive time.
    assert results["solve"][0].exclusive_duration == pytest.approx(15.0)


def test_a_missing_required_group_is_not_silently_read_as_an_empty_run(profile):
    """Losing the event table must fail, not report a run with no calls."""
    with h5py.File(profile, "r+") as handle:
        del handle["events"]

    with pytest.raises((KeyError, OSError, ValueError)):
        load_h5(profile)


def test_an_index_row_pointing_past_the_event_columns_is_rejected(profile):
    """A row claiming more events than the file holds must not read as valid.

    The index and the shared event columns are written together, so a file
    where they disagree is damaged. Reading it anyway would silently give a
    region another region's calls, or drop calls the run recorded -- neither
    of which is visible in the numbers that come out.
    """
    with h5py.File(profile, "r+") as handle:
        counts = handle["rank_region_index/event_counts"][()]
        counts[-1] += 1
        handle["rank_region_index/event_counts"][...] = counts

    with pytest.raises(CorruptProfileError, match="truncated or damaged"):
        read_h5(profile)


def test_overlapping_index_rows_are_rejected(profile):
    """Two regions may not lay claim to the same recorded events."""
    with h5py.File(profile, "r+") as handle:
        counts = handle["rank_region_index/event_counts"][()]
        # Grow the first row over the second one's slice. The total still fits
        # inside the event columns, so only the overlap is wrong.
        counts[0] += 1
        handle["rank_region_index/event_counts"][...] = counts

    with pytest.raises(CorruptProfileError, match="overlapping region rows"):
        read_h5(profile)


def test_a_valid_file_passes_the_index_check(profile):
    """The guard must not reject a healthy file."""
    results = read_h5(profile)

    assert results["solve"][0].num_calls == 2
    assert results["io"][0].num_calls == 1
