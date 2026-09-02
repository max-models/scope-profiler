"""End-to-end tests for the opt-in pytest plugin."""

from scope_profiler import read_h5

# pytester is bundled with pytest but must be explicitly enabled for its
# temporary-project fixture.
pytest_plugins = ("pytester",)


def test_plugin_profiles_each_selected_test_call(pytester, tmp_path):
    pytester.makepyfile(
        test_sample="""
        def test_fast():
            assert 1 + 1 == 2

        def test_slow():
            sum(range(100))
        """,
    )
    output = tmp_path / "pytest-profile.h5"

    result = pytester.runpytest_subprocess(
        "--scope-profile",
        "--scope-profile-out",
        str(output),
    )

    result.assert_outcomes(passed=2)
    assert output.exists()
    assert read_h5(output).region_names == [
        "pytest::test_sample.py::test_fast::call",
        "pytest::test_sample.py::test_slow::call",
    ]


def test_plugin_can_include_setup_and_teardown(pytester, tmp_path):
    pytester.makepyfile(
        test_sample="""
        import pytest

        @pytest.fixture
        def value():
            yield 3

        def test_value(value):
            assert value == 3
        """,
    )
    output = tmp_path / "phases.h5"

    result = pytester.runpytest_subprocess(
        "--scope-profile",
        "--scope-profile-phases=all",
        "--scope-profile-out",
        str(output),
    )

    result.assert_outcomes(passed=1)
    assert read_h5(output).region_names == [
        "pytest::test_sample.py::test_value::setup",
        "pytest::test_sample.py::test_value::call",
        "pytest::test_sample.py::test_value::teardown",
    ]
