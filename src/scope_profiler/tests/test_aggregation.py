import h5py
import pytest

from scope_profiler import ProfileManager, read_h5


def test_aggregation_mode_keeps_statistics_without_events(tmp_path):
    path = tmp_path / "aggregate.h5"
    ProfileManager.setup(file_path=str(path), aggregation_mode=True)
    outer = ProfileManager.profile_region("outer")
    inner = ProfileManager.profile_region("inner")

    for _ in range(3):
        with outer, inner:
            pass

    results = ProfileManager.finalize(verbose=False, return_results=True)
    assert results["outer"][0].num_calls == 3
    assert results["inner"][0].num_calls == 3
    assert (
        results["outer"][0].total_exclusive_duration
        < results["outer"][0].total_duration
    )
    assert results["inner"][0].total_exclusive_duration == pytest.approx(
        results["inner"][0].total_duration,
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


def test_pause_and_resume_exclude_intervening_scopes(tmp_path):
    ProfileManager.setup(file_path=str(tmp_path / "pause.h5"))
    region = ProfileManager.profile_region("work")

    with region:
        pass
    ProfileManager.pause()
    with region:
        pass
    ProfileManager.resume()
    with region:
        pass

    results = ProfileManager.finalize(verbose=False, return_results=True)
    assert results["work"][0].num_calls == 2


def test_pause_is_idempotent_and_rejects_active_scopes(tmp_path):
    ProfileManager.setup(file_path=str(tmp_path / "pause.h5"))
    region = ProfileManager.profile_region("work")
    with region, pytest.raises(RuntimeError, match="scopes are active"):
        ProfileManager.pause()
    ProfileManager.pause()
    ProfileManager.pause()
    ProfileManager.resume()
    ProfileManager.resume()
    ProfileManager.finalize(verbose=False)


def test_pause_before_setup_is_an_error():
    ProfileManager._reset()
    with pytest.raises(RuntimeError, match="setup"):
        ProfileManager.pause()
    with pytest.raises(RuntimeError, match="setup"):
        ProfileManager.resume()


def test_sample_every_profiles_selected_timesteps(tmp_path):
    ProfileManager.setup(file_path=str(tmp_path / "sample.h5"))
    region = ProfileManager.profile_region("step")

    with ProfileManager.sample_every(10) as profile_step:
        for timestep in range(25):
            with profile_step(timestep) as selected:
                with region:
                    pass
                assert selected is (timestep % 10 == 0)

    results = ProfileManager.finalize(verbose=False, return_results=True)
    assert results["step"][0].num_calls == 3


def test_pause_resume_works_inside_session_envelope(tmp_path):
    with ProfileManager.session(
        file_path=str(tmp_path / "session-pause.h5"),
        verbose=False,
        return_results=True,
    ) as run:
        step = ProfileManager.profile_region("step")
        ProfileManager.pause()
        with step:
            pass
        ProfileManager.resume()
        with step:
            pass

    assert run.results["step"][0].num_calls == 1


@pytest.mark.parametrize("every", [0, -1, True, 1.5])
def test_sample_every_validates_interval(every):
    ProfileManager._reset()
    with pytest.raises(ValueError, match="every"):
        with ProfileManager.sample_every(every):
            pass


@pytest.mark.parametrize(
    "option",
    ["use_line_profiler", "use_gpu_timing", "use_likwid", "use_nvtx"],
)
def test_aggregation_mode_rejects_event_extensions(option):
    with pytest.raises(ValueError, match="aggregation_mode"):
        ProfileManager.setup(aggregation_mode=True, **{option: True})
