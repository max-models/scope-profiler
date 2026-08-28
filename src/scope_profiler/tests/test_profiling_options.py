import pytest

from scope_profiler import ProfileManager, ProfilingOptions


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


def test_options_are_applied_to_the_config(tmp_path):
    options = ProfilingOptions(
        file_path=str(tmp_path / "run.h5"),
        label="from-options",
        buffer_limit=2048,
        use_nvtx=True,
    )

    ProfileManager.setup(options=options)
    config = ProfileManager.get_config()

    assert config.file_path == str(tmp_path / "run.h5")
    assert config.label == "from-options"
    assert config.buffer_limit == 2048
    assert config.use_nvtx is True


def test_explicit_keyword_overrides_the_same_options_field(tmp_path):
    options = ProfilingOptions(file_path=str(tmp_path / "a.h5"), label="a")

    ProfileManager.setup(options=options, file_path=str(tmp_path / "b.h5"))
    config = ProfileManager.get_config()

    assert config.file_path == str(tmp_path / "b.h5")
    # Fields not overridden still come from options.
    assert config.label == "a"


def test_unset_options_fields_fall_back_to_defaults(tmp_path):
    options = ProfilingOptions(file_path=str(tmp_path / "run.h5"))

    ProfileManager.setup(options=options)
    config = ProfileManager.get_config()

    assert config.use_likwid is False
    assert config.buffer_limit == 1024


def test_options_can_be_reused_across_setup_calls(tmp_path):
    options = ProfilingOptions(deactivate_file_output=True, label="shared")

    ProfileManager.setup(options=options)
    with ProfileManager.profile_region("first"):
        pass
    first = ProfileManager.finalize(verbose=False, return_results=True)

    ProfileManager.setup(options=options)
    with ProfileManager.profile_region("second"):
        pass
    second = ProfileManager.finalize(verbose=False, return_results=True)

    assert first.label == second.label == "shared"
    assert first.get_region("first").num_calls == 1
    assert second.get_region("second").num_calls == 1


def test_session_accepts_options(tmp_path):
    options = ProfilingOptions(deactivate_file_output=True)

    with ProfileManager.session(
        options=options, return_results=True, verbose=False
    ) as run:
        with ProfileManager.profile_region("work"):
            pass

    assert run.results.get_region("work").num_calls == 1


def test_session_records_a_single_root_around_all_regions():
    options = ProfilingOptions(deactivate_file_output=True)

    with ProfileManager.session(
        options=options, return_results=True, verbose=False
    ) as run:
        with ProfileManager.profile_region("first"):
            pass
        with ProfileManager.profile_region("second"):
            pass

    calls = run.results.call_stack()
    root = calls[0]
    assert root["name"] == "scope_profiler.session"
    assert root["parent"] is None
    assert [call["name"] for call in calls[1:]] == ["first", "second"]
    assert all(call["parent"] == 0 for call in calls[1:])


def test_to_kwargs_only_includes_set_fields():
    options = ProfilingOptions(use_likwid=True, buffer_limit=512)

    assert options.to_kwargs() == {"use_likwid": True, "buffer_limit": 512}
