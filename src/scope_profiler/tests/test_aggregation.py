import h5py
import pytest

from scope_profiler import ProfileManager, read_h5


def test_aggregation_mode_keeps_statistics_without_events(tmp_path):
    path = tmp_path / "aggregate.h5"
    ProfileManager.setup(file_path=str(path), aggregation_mode=True)
    outer = ProfileManager.profile_region("outer")
    inner = ProfileManager.profile_region("inner")

    for _ in range(3):
        with outer:
            with inner:
                pass

    results = ProfileManager.finalize(verbose=False, return_results=True)
    assert results["outer"][0].num_calls == 3
    assert results["inner"][0].num_calls == 3
    assert results["outer"][0].total_exclusive_duration < results["outer"][0].total_duration
    assert results["inner"][0].total_exclusive_duration == pytest.approx(
        results["inner"][0].total_duration
    )
    assert results["outer"][0].events() == []

    with h5py.File(path, "r") as handle:
        assert handle.attrs["storage_layout"] == "aggregate"
        assert "events" in handle
        assert handle["events/start_times"].shape == (0,)
        assert handle["rank_region_index/aggregate_counts"][()].tolist() == [3, 3]

    loaded = read_h5(path)
    assert loaded["outer"][0].num_calls == 3
    assert loaded["outer"][0].max_duration >= loaded["outer"][0].min_duration
    assert loaded["outer"][0].events() == []


@pytest.mark.parametrize("option", ["use_line_profiler", "use_gpu_timing", "use_likwid", "use_nvtx"])
def test_aggregation_mode_rejects_event_extensions(option):
    with pytest.raises(ValueError, match="aggregation_mode"):
        ProfileManager.setup(aggregation_mode=True, **{option: True})
