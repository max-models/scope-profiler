"""The recursive tracer's decision logic, called directly.

``recursive_profile=True`` installs :meth:`ProfileManager._get_recursive_tracer`
as a ``sys.setprofile`` callback. That makes the whole-run behaviour easy to
test (see ``test_app.py``) but the individual decisions hard to see: which
frames get a region, which are skipped as internal, as already-decorated, or
as library code. It also makes those decisions invisible to coverage, since
code running inside a profile callback is not attributable to the function it
came from.

So this drives the tracer as a plain function, with real frames, and checks
each rule in isolation.
"""

import os
import sys
import sysconfig

import numpy as np
import pytest

from scope_profiler import ProfileManager
from scope_profiler.region_profiler import (
    DisabledProfileRegion,
    LineProfilerRegion,
    TimeOnlyProfileRegion,
)


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


@pytest.fixture
def manager(tmp_path):
    ProfileManager.setup(
        file_path=str(tmp_path / "profiling_data.h5"),
        deactivate_file_output=True,
    )
    return ProfileManager


def _frame():
    """A frame belonging to this test module."""
    return sys._getframe()


def _run_tracer(tracer, frame):
    """Send one call/return pair through the tracer, as sys.setprofile would."""
    tracer(frame, "call", None)
    tracer(frame, "return", None)


# --------------------------------------------------------------------------
# Region naming and internal-frame detection
# --------------------------------------------------------------------------


def test_a_region_is_named_module_dot_qualname():
    frame = _frame()
    name = ProfileManager._frame_region_name(frame)

    assert name.startswith(__name__ + ".")
    assert name.endswith("_frame")


def test_a_frame_with_no_module_name_is_still_named():
    class FakeCode:
        co_qualname = "work"
        co_name = "work"
        co_filename = __file__

    class FakeFrame:
        f_globals: dict = {}
        f_code = FakeCode()

    assert ProfileManager._frame_region_name(FakeFrame()) == "<unknown>.work"


def test_the_profilers_own_frames_are_internal():
    """Tracing must not recurse into the profiler while the profiler traces."""
    import scope_profiler.profile_manager as pm

    class FakeFrame:
        f_globals = {"__name__": pm.__name__}

    assert ProfileManager._is_internal_frame(FakeFrame())


def test_user_frames_are_not_internal():
    assert not ProfileManager._is_internal_frame(_frame())


# --------------------------------------------------------------------------
# User code vs. library code
# --------------------------------------------------------------------------


def test_this_test_module_counts_as_user_code():
    assert ProfileManager._is_user_code(_frame().f_code)


def test_stdlib_code_is_not_user_code():
    assert not ProfileManager._is_user_code(os.path.join.__code__)


def test_synthetic_filenames_are_not_user_code():
    """`<string>`, `<frozen ...>` and friends are not files anyone wrote."""
    code = compile("x = 1", "<string>", "exec")
    assert not ProfileManager._is_user_code(code)


def test_a_code_object_with_no_filename_is_not_user_code():
    code = compile("x = 1", "", "exec")
    assert not ProfileManager._is_user_code(code)


def test_the_user_code_decision_is_memoized_per_code_object():
    code = _frame().f_code
    ProfileManager._user_code_cache.pop(code, None)

    first = ProfileManager._is_user_code(code)

    assert code in ProfileManager._user_code_cache
    assert ProfileManager._is_user_code(code) is first


def test_the_system_prefixes_include_the_standard_library():
    prefixes = ProfileManager._system_path_prefixes()

    stdlib = os.path.realpath(sysconfig.get_paths()["stdlib"])
    assert stdlib in prefixes


def test_the_system_prefixes_are_computed_once():
    ProfileManager._system_prefixes = None
    first = ProfileManager._system_path_prefixes()
    second = ProfileManager._system_path_prefixes()

    assert first is second


# --------------------------------------------------------------------------
# The tracer itself
# --------------------------------------------------------------------------


def test_a_traced_call_opens_and_closes_a_region(manager):
    frame = _frame()
    tracer = manager._get_recursive_tracer(root_frame=None, prev_profiler=None)

    _run_tracer(tracer, frame)

    name = manager._frame_region_name(frame)
    region = manager.get_all_regions()[name]
    assert region.num_calls == 1
    assert region.get_durations_numpy()[0] >= 0


def test_the_root_frame_is_never_given_a_region(manager):
    """The frame the tracer was installed from is the boundary, not a region."""
    frame = _frame()
    tracer = manager._get_recursive_tracer(root_frame=frame, prev_profiler=None)

    _run_tracer(tracer, frame)

    assert manager.get_all_regions() == {}


def test_the_profilers_own_frames_are_skipped(manager):
    import scope_profiler.profile_manager as pm

    class FakeCode:
        co_qualname = "internal"
        co_name = "internal"
        co_filename = __file__

    class FakeFrame:
        f_globals = {"__name__": pm.__name__}
        f_code = FakeCode()

    tracer = manager._get_recursive_tracer(root_frame=None, prev_profiler=None)
    _run_tracer(tracer, FakeFrame())

    assert manager.get_all_regions() == {}


def test_an_already_decorated_function_is_not_counted_twice(manager):
    """Its decorator already records it; the tracer must not add a second region."""

    @manager.profile("explicit")
    def work():
        return sys._getframe()

    frame = work()
    tracer = manager._get_recursive_tracer(root_frame=None, prev_profiler=None)
    _run_tracer(tracer, frame)

    assert list(manager.get_all_regions()) == ["explicit"]
    assert manager.get_all_regions()["explicit"].num_calls == 1


def test_library_frames_are_skipped_when_only_user_code_is_asked_for(manager):
    class FakeCode:
        co_qualname = "loads"
        co_name = "loads"
        co_filename = os.path.join(sysconfig.get_paths()["stdlib"], "json", "x.py")

    class FakeFrame:
        f_globals = {"__name__": "json"}
        f_code = FakeCode()

    tracer = manager._get_recursive_tracer(
        root_frame=None,
        prev_profiler=None,
        only_user_code=True,
    )
    _run_tracer(tracer, FakeFrame())

    assert manager.get_all_regions() == {}


def test_library_frames_are_traced_when_user_code_is_not_required(manager):
    class FakeCode:
        co_qualname = "loads"
        co_name = "loads"
        co_filename = os.path.join(sysconfig.get_paths()["stdlib"], "json", "x.py")

    class FakeFrame:
        f_globals = {"__name__": "json"}
        f_code = FakeCode()

    tracer = manager._get_recursive_tracer(
        root_frame=None,
        prev_profiler=None,
        only_user_code=False,
    )
    _run_tracer(tracer, FakeFrame())

    assert list(manager.get_all_regions()) == ["json.loads"]


def test_a_return_without_a_matching_call_is_ignored(manager):
    """Frames already running when tracing started have no region to close."""
    tracer = manager._get_recursive_tracer(root_frame=None, prev_profiler=None)

    tracer(_frame(), "return", None)

    assert manager.get_all_regions() == {}


def test_the_previous_profiler_still_sees_every_event(manager):
    """Installing the tracer must not silently displace another profiler."""
    seen = []
    tracer = manager._get_recursive_tracer(
        root_frame=None,
        prev_profiler=lambda f, e, a: seen.append(e),
    )

    _run_tracer(tracer, _frame())

    assert seen == ["call", "return"]


def test_the_tracer_returns_itself_so_tracing_continues(manager):
    tracer = manager._get_recursive_tracer(root_frame=None, prev_profiler=None)

    assert tracer(_frame(), "call", None) is tracer


def test_nested_calls_produce_nested_regions(manager):
    """Two frames entered in order close in reverse, like a real call stack."""

    def outer():
        return sys._getframe()

    def inner():
        return sys._getframe()

    outer_frame, inner_frame = outer(), inner()
    tracer = manager._get_recursive_tracer(root_frame=None, prev_profiler=None)

    tracer(outer_frame, "call", None)
    tracer(inner_frame, "call", None)
    tracer(inner_frame, "return", None)
    tracer(outer_frame, "return", None)

    regions = manager.get_all_regions()
    outer_region = regions[manager._frame_region_name(outer_frame)]
    inner_region = regions[manager._frame_region_name(inner_frame)]
    assert outer_region.num_calls == 1
    assert inner_region.num_calls == 1
    # The inner call really is contained by the outer one.
    assert outer_region.get_start_times_numpy()[0] <= (
        inner_region.get_start_times_numpy()[0]
    )
    assert (
        inner_region.get_end_times_numpy()[0] <= outer_region.get_end_times_numpy()[0]
    )


def test_a_line_profiler_region_is_entered_without_line_registration(manager, tmp_path):
    """Under the line profiler the tracer times the frame but does not register it.

    Registering every traced frame with line_profiler would profile the whole
    program line by line, which is not what recursive region profiling asks
    for.
    """
    ProfileManager._reset()
    ProfileManager.setup(
        file_path=str(tmp_path / "lines.h5"),
        deactivate_file_output=True,
        use_line_profiler=True,
    )
    frame = _frame()
    tracer = ProfileManager._get_recursive_tracer(root_frame=None, prev_profiler=None)

    _run_tracer(tracer, frame)

    region = ProfileManager.get_all_regions()[ProfileManager._frame_region_name(frame)]
    assert isinstance(region, LineProfilerRegion)
    assert region.num_calls == 1
    assert region._registered_codes == set()


def test_active_calls_can_be_shared_with_the_caller(manager):
    """The CLI keeps the open-call map so it can flush line timings on return."""
    active: dict = {}
    frame = _frame()
    tracer = manager._get_recursive_tracer(
        root_frame=None,
        prev_profiler=None,
        active_calls=active,
    )

    tracer(frame, "call", None)
    assert frame in active

    tracer(frame, "return", None)
    assert active == {}


def test_a_disabled_manager_traces_into_disabled_regions(manager, tmp_path):
    ProfileManager._reset()
    ProfileManager.setup(file_path=str(tmp_path / "off.h5"), deactivate_profiling=True)
    frame = _frame()
    tracer = ProfileManager._get_recursive_tracer(root_frame=None, prev_profiler=None)

    _run_tracer(tracer, frame)

    region = ProfileManager.get_all_regions()[ProfileManager._frame_region_name(frame)]
    assert isinstance(region, DisabledProfileRegion)
    assert region.num_calls == 0


def test_the_default_region_class_records_time(manager):
    frame = _frame()
    tracer = manager._get_recursive_tracer(root_frame=None, prev_profiler=None)
    _run_tracer(tracer, frame)

    region = manager.get_all_regions()[manager._frame_region_name(frame)]
    assert isinstance(region, TimeOnlyProfileRegion)
    assert np.all(region.get_durations_numpy() >= 0)
