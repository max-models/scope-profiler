"""Independent profile managers can keep sessions active simultaneously."""

from scope_profiler import ProfileManager, read_h5
from scope_profiler.profile_config import ProfilingConfig


def test_profiling_configs_are_independent(tmp_path):
    first = ProfilingConfig(file_path=str(tmp_path / "first.h5"), label="first")
    second = ProfilingConfig(file_path=str(tmp_path / "second.h5"), label="second")

    assert first is not second
    assert first.file_path.endswith("first.h5")
    assert second.file_path.endswith("second.h5")
    assert first.label == "first"
    assert second.label == "second"


def test_multiple_sessions_can_overlap(tmp_path):
    first = ProfileManager()
    second = ProfileManager()
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"

    with first.session(file_path=str(first_path), label="first", verbose=False):
        with first.profile_region("first-before"):
            pass

        with second.session(file_path=str(second_path), label="second", verbose=False):
            with first.profile_region("first-during"):
                pass
            with second.profile_region("second-only"):
                pass

        with first.profile_region("first-after"):
            pass

    first_results = read_h5(first_path)
    second_results = read_h5(second_path)

    assert set(first_results.region_names) == {
        "scope_profiler.session",
        "first-before",
        "first-during",
        "first-after",
    }
    assert set(second_results.region_names) == {
        "scope_profiler.session",
        "second-only",
    }
    assert first_results.label == "first"
    assert second_results.label == "second"
    assert first.get_config().file_path == str(first_path)
    assert second.get_config().file_path == str(second_path)


def test_instance_is_isolated_from_class_level_default_manager(tmp_path):
    isolated = ProfileManager()
    ProfileManager._reset()

    try:
        with (
            ProfileManager.session(
                file_path=str(tmp_path / "default.h5"),
                deactivate_file_output=True,
                return_results=True,
                verbose=False,
            ) as default_run,
            isolated.session(
                file_path=str(tmp_path / "isolated.h5"),
                deactivate_file_output=True,
                return_results=True,
                verbose=False,
            ) as isolated_run,
        ):
            with ProfileManager.profile_region("default-only"):
                pass
            with isolated.profile_region("isolated-only"):
                pass

        assert "default-only" in default_run.results.region_names
        assert "isolated-only" not in default_run.results.region_names
        assert "isolated-only" in isolated_run.results.region_names
        assert "default-only" not in isolated_run.results.region_names
    finally:
        isolated._reset()
        ProfileManager._reset()


def test_decorators_belong_to_the_manager_that_created_them(tmp_path):
    first = ProfileManager()
    second = ProfileManager()

    @first.profile("first-work")
    def first_work():
        pass

    @second.profile("second-work")
    def second_work():
        pass

    with (
        first.session(
            file_path=str(tmp_path / "first.h5"),
            deactivate_file_output=True,
            return_results=True,
            verbose=False,
        ) as first_run,
        second.session(
            file_path=str(tmp_path / "second.h5"),
            deactivate_file_output=True,
            return_results=True,
            verbose=False,
        ) as second_run,
    ):
        first_work()
        second_work()

    assert first_run.results["first-work"].num_calls == 1
    assert "second-work" not in first_run.results.region_names
    assert second_run.results["second-work"].num_calls == 1
    assert "first-work" not in second_run.results.region_names


def test_aggregation_nesting_is_isolated_between_managers(tmp_path):
    first = ProfileManager()
    second = ProfileManager()

    first.setup(
        file_path=str(tmp_path / "first.h5"),
        aggregation_mode=True,
        deactivate_file_output=True,
    )
    second.setup(
        file_path=str(tmp_path / "second.h5"),
        aggregation_mode=True,
        deactivate_file_output=True,
    )

    with first.profile_region("first"), second.profile_region("second"):
        pass

    first_result = first.finalize(verbose=False, return_results=True)
    second_result = second.finalize(verbose=False, return_results=True)

    assert first_result["first"].total_exclusive_duration > 0
    assert second_result["second"].total_exclusive_duration > 0
