"""Opt-in region tags: library instrumentation that is off unless enabled."""

import sys

import pytest

import scope_profiler as sp
from scope_profiler import ProfileManager, ProfilingOptions, profile_manager, read_h5
from scope_profiler.__main__ import main as cli_main
from scope_profiler.region_profiler import DisabledProfileRegion


@pytest.fixture(autouse=True)
def _fresh_opt_in_tags(monkeypatch):
    """Give every test its own registry; the real one outlives resets."""
    monkeypatch.setattr(profile_manager, "_OPT_IN_TAGS", set())
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)


def _record_library_and_user_regions(**setup_kwargs):
    ProfileManager.setup(output=None, **setup_kwargs)
    with sp.region("mylib.kernel", tags=("mylib",)):
        pass
    with sp.region("user.step"):
        pass
    results = ProfileManager.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_opt_in_regions_are_off_by_default_while_user_regions_record():
    sp.register_opt_in_tag("mylib")

    assert _record_library_and_user_regions() == {"user.step"}


def test_enable_tags_keyword_turns_opt_in_regions_on():
    sp.register_opt_in_tag("mylib")

    names = _record_library_and_user_regions(enable_tags=["mylib"])

    assert names == {"mylib.kernel", "user.step"}


def test_enable_tags_option_field_turns_opt_in_regions_on():
    sp.register_opt_in_tag("mylib")

    names = _record_library_and_user_regions(
        options=ProfilingOptions(enable_tags=("mylib",)),
    )

    assert names == {"mylib.kernel", "user.step"}


def test_enable_tags_from_toml(tmp_path):
    sp.register_opt_in_tag("mylib")
    config = tmp_path / "profiling.toml"
    config.write_text('[profiling]\nenable_tags = ["mylib"]\n', encoding="utf-8")

    names = _record_library_and_user_regions(config_path=config)

    assert names == {"mylib.kernel", "user.step"}


def test_enable_tags_from_environment_adds_to_the_setting(monkeypatch):
    sp.register_opt_in_tag("mylib", "otherlib")
    monkeypatch.setenv("SCOPE_PROFILER_ENABLE_TAGS", " mylib , ")

    ProfileManager.setup(output=None, enable_tags=["otherlib"])
    with sp.region("mylib.kernel", tags=("mylib",)):
        pass
    with sp.region("otherlib.kernel", tags=("otherlib",)):
        pass
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert set(results.region_names) == {"mylib.kernel", "otherlib.kernel"}


def test_star_enables_every_opt_in_tag():
    sp.register_opt_in_tag("mylib")

    names = _record_library_and_user_regions(enable_tags=["*"])

    assert names == {"mylib.kernel", "user.step"}


def test_any_enabled_opt_in_tag_enables_the_region():
    sp.register_opt_in_tag("mylib", "otherlib")
    ProfileManager.setup(output=None, enable_tags=["otherlib"])

    with sp.region("shared.kernel", tags=("mylib", "otherlib", "hot")):
        pass
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results["shared.kernel"].num_calls == 1


def test_region_tags_filter_enables_the_opt_in_tags_it_names():
    sp.register_opt_in_tag("mylib")

    names = _record_library_and_user_regions(region_tags=["mylib"])

    assert names == {"mylib.kernel"}


def test_region_tags_filter_does_not_bypass_an_opt_in_tag():
    sp.register_opt_in_tag("mylib")
    ProfileManager.setup(output=None, region_tags=["hot"])

    assert isinstance(
        sp.region("mylib.kernel", tags=("mylib", "hot")),
        DisabledProfileRegion,
    )


def test_decorated_function_follows_the_setting_across_sessions():
    sp.register_opt_in_tag("mylib")
    calls = []

    @sp.profile("mylib.solve", tags=("mylib",))
    def solve():
        calls.append(1)

    ProfileManager.setup(output=None)
    solve()
    off = ProfileManager.finalize(verbose=False, return_results=True)

    ProfileManager.setup(output=None, enable_tags=["mylib"], replace=True)
    solve()
    on = ProfileManager.finalize(verbose=False, return_results=True)

    assert len(calls) == 2
    assert "mylib.solve" not in off.region_names
    assert on["mylib.solve"].num_calls == 1
    assert on["mylib.solve"].tags == ("mylib",)


def test_instrument_tags_every_wrapped_function(tmp_path, monkeypatch):
    sp.register_opt_in_tag("mylib")
    module_path = tmp_path / "mylib_kernels.py"
    module_path.write_text("def kernel():\n    return 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import mylib_kernels

    sp.instrument(mylib_kernels, tags=("mylib",))
    ProfileManager.setup(output=None, enable_tags=["mylib"])
    mylib_kernels.kernel()
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results["mylib_kernels.kernel"].tags == ("mylib",)


def test_tag_registered_after_setup_still_disables_new_regions():
    ProfileManager.setup(output=None)
    sp.register_opt_in_tag("mylib")

    assert isinstance(
        sp.region("mylib.kernel", tags=("mylib",)),
        DisabledProfileRegion,
    )


def test_unregistered_tags_behave_as_before():
    ProfileManager.setup(output=None)

    with sp.region("solve", tags=("mylib",)):
        pass
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results["solve"].num_calls == 1


def test_register_opt_in_tag_rejects_non_strings():
    with pytest.raises(TypeError, match="non-empty strings"):
        sp.register_opt_in_tag("")


def test_cli_enable_tag(tmp_path):
    script = tmp_path / "library_user.py"
    script.write_text(
        """\
import scope_profiler as sp

sp.register_opt_in_tag("mylib")
with sp.region("mylib.kernel", tags=["mylib"]):
    pass
with sp.region("user.step"):
    pass
""",
        encoding="utf-8",
    )
    off = tmp_path / "off.h5"
    on = tmp_path / "on.h5"

    cli_main(["run", "-q", "-o", str(off), str(script)])
    ProfileManager._reset()
    cli_main(["run", "-q", "--enable-tag", "mylib", "-o", str(on), str(script)])

    assert set(read_h5(off).region_names) == {"user.step"}
    assert set(read_h5(on).region_names) == {"mylib.kernel", "user.step"}


def test_reset_keeps_decorated_functions_for_the_next_setup():
    sp.register_opt_in_tag("mylib")

    @sp.profile("mylib.solve", tags=("mylib",))
    def solve():
        return 1

    sp.reset()
    assert solve() == 1  # calls straight through while nothing is configured
    ProfileManager.setup(output=None, enable_tags=["mylib"])
    solve()
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results["mylib.solve"].num_calls == 1


def test_private_reset_forgets_decorated_functions():
    @sp.profile("solve")
    def solve():
        pass

    ProfileManager._reset()
    ProfileManager.setup(output=None)

    assert "solve" not in sp.registered_regions()


def test_disabled_decorated_function_skips_metadata_and_tracing():
    sp.register_opt_in_tag("mylib")
    metadata_calls = []
    profilers = []

    def metadata(*args, **kwargs):
        metadata_calls.append(args)
        return {"n": 1}

    @sp.profile("mylib.solve", tags=("mylib",), metadata=metadata, recursive=True)
    def solve():
        profilers.append(sys.getprofile())

    before = sys.getprofile()
    ProfileManager.setup(output=None)
    solve()
    ProfileManager.finalize(verbose=False)

    assert metadata_calls == []
    assert profilers == [before]


def test_registering_a_tag_late_disables_existing_instrumentation():
    @sp.profile("mylib.solve", tags=("mylib",))
    def solve():
        pass

    ProfileManager.setup(output=None)
    with sp.region("mylib.kernel", tags=("mylib",)):
        pass
    sp.register_opt_in_tag("mylib")
    solve()
    with sp.region("mylib.kernel", tags=("mylib",)):
        pass
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert "mylib.solve" not in results.region_names
    assert "mylib.kernel" not in results.region_names


def test_region_tags_filter_decorated_functions_and_revert_with_none():
    @sp.profile("core.solve", tags=("core",))
    def solve():
        pass

    @sp.profile("detail.check", tags=("detail",))
    def check():
        pass

    @sp.profile("untagged")
    def untagged():
        pass

    def record(**setup_kwargs):
        ProfileManager.setup(output=None, **setup_kwargs)
        solve()
        check()
        untagged()
        results = ProfileManager.finalize(verbose=False, return_results=True)
        return results

    subset = record(region_tags=["core"])
    assert set(subset.region_names) == {"core.solve"}
    assert subset["core.solve"].tags == ("core",)

    full = record(region_tags=None)
    assert set(full.region_names) == {"core.solve", "detail.check", "untagged"}
    assert full["detail.check"].tags == ("detail",)


def test_decorator_tags_accept_any_iterable():
    @sp.profile("gen.solve", tags=(tag for tag in ["a", "b"]))
    def solve():
        pass

    ProfileManager.setup(output=None, region_tags={"b"})
    solve()
    results = ProfileManager.finalize(verbose=False, return_results=True)
    assert results["gen.solve"].tags == ("a", "b")
