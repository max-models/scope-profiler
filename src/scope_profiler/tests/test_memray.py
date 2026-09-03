"""Tests for the optional Memray allocation-capture integration."""

import sys
import types

import pytest

from scope_profiler import ProfileManager, ProfilingOptions


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


class _Tracker:
    instances = []

    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        self.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_):
        self.exited = True


def test_memray_capture_is_started_and_finished(tmp_path, monkeypatch):
    fake_memray = types.SimpleNamespace(Tracker=_Tracker)
    monkeypatch.setitem(sys.modules, "memray", fake_memray)
    output = tmp_path / "timing.h5"

    ProfileManager.setup(
        file_path=str(output),
        use_memray=True,
        memray_native_traces=True,
        memray_trace_python_allocators=True,
    )
    config = ProfileManager.get_config()

    assert config.memory_profile_path == tmp_path / "timing.memray.bin"
    assert _Tracker.instances[-1].path == str(tmp_path / "timing.memray.bin")
    assert _Tracker.instances[-1].kwargs == {
        "native_traces": True,
        "trace_python_allocators": True,
        "follow_fork": False,
    }
    with ProfileManager.profile_region("work"):
        pass
    ProfileManager.finalize(verbose=False)
    assert _Tracker.instances[-1].exited is True


def test_memray_missing_dependency_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "memray", None)

    with pytest.raises(ImportError, match=r"scope-profiler\[extras\]"):
        ProfileManager.setup(file_path=str(tmp_path / "timing.h5"), use_memray=True)


def test_memray_options_round_trip(tmp_path):
    options = ProfilingOptions(
        use_memray=True,
        memory_profile_path=str(tmp_path / "allocations.bin"),
    )

    assert options.to_kwargs() == {
        "use_memray": True,
        "memory_profile_path": str(tmp_path / "allocations.bin"),
    }
