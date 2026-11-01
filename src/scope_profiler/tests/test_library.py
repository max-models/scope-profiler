"""sp.library: a library's opt-in tag, its region prefix, and the test helpers."""

import os
from importlib.metadata import EntryPoint

import pytest

import scope_profiler as sp
from scope_profiler import profile_manager, testing
from scope_profiler.testing import (
    RUN_REGION,
    assert_opt_in_tag,
    fresh_profiler,
    recorded_region_names,
)


@pytest.fixture(autouse=True)
def _fresh_opt_in_tags(monkeypatch):
    """Give every test its own registry; the real one outlives resets."""
    monkeypatch.setattr(profile_manager, "_OPT_IN_TAGS", set())
    monkeypatch.setattr(profile_manager, "_OPT_IN_TAG_MODULES", {})
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)


@pytest.fixture
def declared(monkeypatch):
    """Pretend the installed packages declare these opt-in entry points."""
    points = {}

    def fake_entry_points(group):
        assert group == testing.ENTRY_POINT_GROUP
        return [EntryPoint(name, value, group) for name, value in points.items()]

    monkeypatch.setattr(testing, "entry_points", fake_entry_points)
    return points


def _mylib():
    prof = sp.library("mylib")

    @prof.profile("solve")
    def solve():
        with prof.region("assemble"):
            pass

    @prof.profile
    def bare():
        pass

    class Solver:
        @prof.profile()
        def step(self):
            pass

    def run():
        solve()
        bare()
        Solver().step()

    return prof, run


def test_library_registers_the_tag_with_the_calling_module():
    prof = sp.library("mylib")

    assert prof.tag == "mylib"
    assert prof.tags == ("mylib",)
    assert "mylib" in sp.opt_in_tags()
    assert profile_manager._OPT_IN_TAG_MODULES["mylib"] == __name__
    assert repr(prof) == "LibraryProfiler(tag='mylib')"


def test_register_opt_in_tag_still_records_the_calling_module():
    sp.register_opt_in_tag("otherlib")

    assert profile_manager._OPT_IN_TAG_MODULES["otherlib"] == __name__


def test_library_regions_are_off_by_default():
    _prof, run = _mylib()

    assert recorded_region_names(run) == {RUN_REGION}


@pytest.mark.parametrize("enable", [["mylib"], ["*"]])
def test_library_regions_are_prefixed_and_tagged(enable):
    _prof, run = _mylib()

    names = recorded_region_names(run, enable_tags=enable)

    assert names == {
        RUN_REGION,
        "mylib.solve",
        "mylib.assemble",
        "mylib._mylib.<locals>.bare",
        "mylib._mylib.<locals>.Solver.step",
    }
    tagged = sp.opt_in_tags()["mylib"]
    assert "mylib.solve" in tagged


def test_extra_tags_are_added_after_the_library_tag():
    prof = sp.library("mylib")

    @prof.profile("solve", tags=("detail", "mylib"))
    def solve():
        with prof.region("inner", tags=["detail"]):
            pass

    with fresh_profiler():
        sp.setup(output=None, enable_tags=["mylib"])
        solve()
        results = sp.finalize(verbose=False, return_results=True)

    assert results.get_region("mylib.solve").tags == ("mylib", "detail")
    assert results.get_region("mylib.inner").tags == ("mylib", "detail")


def test_profile_passes_the_other_options_on():
    prof = sp.library("mylib")

    @prof.profile("fail", record_outcome=True)
    def fail():
        raise ValueError

    with fresh_profiler():
        sp.setup(output=None, enable_tags=["mylib"])
        with pytest.raises(ValueError):
            fail()
        results = sp.finalize(verbose=False, return_results=True)

    assert results.get_region("mylib.fail").num_calls == 1


def test_fresh_profiler_hides_and_restores_the_environment(monkeypatch):
    monkeypatch.setenv("SCOPE_PROFILER_ENABLE_TAGS", "mylib")

    with fresh_profiler():
        assert "SCOPE_PROFILER_ENABLE_TAGS" not in os.environ
        assert not sp.is_configured()

    assert os.environ["SCOPE_PROFILER_ENABLE_TAGS"] == "mylib"


def test_the_fresh_profiler_fixture(fresh_profiler):
    assert not sp.is_configured()
    sp.setup(output=None)


def test_assert_opt_in_tag_passes_for_a_well_made_library(declared):
    _prof, run = _mylib()
    declared["mylib"] = "mylib"

    assert_opt_in_tag("mylib", run=run, expected={"mylib.solve", "mylib.assemble"})


def test_assert_opt_in_tag_checks_the_entry_point_module(declared):
    _prof, run = _mylib()
    declared["mylib"] = "mylib.core"

    assert_opt_in_tag("mylib", run=run, expected={"mylib.solve"}, module="mylib.core")
    with pytest.raises(AssertionError, match="no entry point mylib = 'mylib'"):
        assert_opt_in_tag("mylib", run=run, expected={"mylib.solve"})


def test_assert_opt_in_tag_catches_a_region_that_is_always_on(declared):
    declared["mylib"] = "mylib"

    def run():
        with sp.region("mylib.untagged"):
            pass

    with pytest.raises(AssertionError, match="recorded without enabling it"):
        assert_opt_in_tag("mylib", run=run, expected=set())


def test_assert_opt_in_tag_catches_a_missing_region(declared):
    _prof, run = _mylib()
    declared["mylib"] = "mylib"

    with pytest.raises(AssertionError, match=r"\['mylib.missing'\] were not"):
        assert_opt_in_tag("mylib", run=run, expected={"mylib.missing"})


def test_assert_opt_in_tag_rejects_expected_names_of_another_tag(declared):
    _prof, run = _mylib()

    with pytest.raises(AssertionError, match="not named 'mylib.'"):
        assert_opt_in_tag("mylib", run=run, expected={"other.solve"})


def test_assert_opt_in_tag_skips_the_region_checks_off_rank_zero(declared, monkeypatch):
    """Under MPI only rank 0 gets the gathered run back."""
    declared["mylib"] = "mylib"
    monkeypatch.setattr(testing, "recorded_region_names", lambda run, **kw: set())

    assert_opt_in_tag("mylib", run=lambda: None, expected={"mylib.solve"})
