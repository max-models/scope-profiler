"""CUDA-event timing backends, exercised against stand-in CUDA modules.

The real backends need a GPU. Everything the profiler asks of them -- event
creation, stream selection, the millisecond-to-nanosecond conversion, and the
resolution rules in :func:`resolve_gpu_timing_backend` -- is provider-agnostic,
so the tests install fake ``torch``/``cupy`` modules and check that contract.
"""

import sys
import types

import pytest

from scope_profiler.gpu_timing import (
    CuPyCUDATimingBackend,
    TorchCUDATimingBackend,
    resolve_gpu_timing_backend,
)


class _FakeEvent:
    """Stands in for a CUDA event, remembering how it was used."""

    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self.recorded_on = None
        self.synchronized = False
        self.elapsed = 2.5

    def record(self, stream=None):
        self.recorded_on = stream

    def synchronize(self):
        self.synchronized = True

    def elapsed_time(self, other):
        return self.elapsed


def _fake_torch(cuda_available=True):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        Event=_FakeEvent,
        current_stream=lambda: "torch-stream",
    )
    return torch


def _fake_cupy(device_count=1, raise_on_count=False, with_get_elapsed_time=True):
    cupy = types.ModuleType("cupy")

    def get_device_count():
        if raise_on_count:
            raise RuntimeError("no driver")
        return device_count

    cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(getDeviceCount=get_device_count),
        Event=lambda: _FakeEvent(),
        get_current_stream=lambda: "cupy-stream",
    )
    if with_get_elapsed_time:
        cuda.get_elapsed_time = lambda start, end: 3.5
    cupy.cuda = cuda
    return cupy


@pytest.fixture
def install_module(monkeypatch):
    """Install a module under a name for the duration of one test."""

    def _install(name, module):
        monkeypatch.setitem(sys.modules, name, module)

    return _install


@pytest.fixture
def hide_module(monkeypatch):
    """Make importing a module raise ImportError, even if it is installed."""

    def _hide(name):
        monkeypatch.setitem(sys.modules, name, None)

    return _hide


def test_torch_backend_records_events_on_the_current_stream(install_module):
    install_module("torch", _fake_torch())
    backend = TorchCUDATimingBackend()

    assert backend.name == "torch"
    event = backend.record_event()
    assert event.enable_timing is True
    assert event.recorded_on == "torch-stream"


def test_torch_backend_converts_milliseconds_to_nanoseconds(install_module):
    install_module("torch", _fake_torch())
    backend = TorchCUDATimingBackend()
    start, end = backend.record_event(), backend.record_event()

    # The fake reports 2.5 ms between the events.
    assert backend.elapsed_time_ns(start, end) == 2_500_000
    assert end.synchronized is True


def test_torch_backend_reports_a_missing_install(hide_module):
    hide_module("torch")
    with pytest.raises(ImportError, match="torch is not installed"):
        TorchCUDATimingBackend()


def test_torch_backend_reports_a_machine_without_cuda(install_module):
    install_module("torch", _fake_torch(cuda_available=False))
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        TorchCUDATimingBackend()


def test_cupy_backend_records_events_on_the_current_stream(install_module):
    install_module("cupy", _fake_cupy())
    backend = CuPyCUDATimingBackend()

    assert backend.name == "cupy"
    assert backend.record_event().recorded_on == "cupy-stream"


def test_cupy_backend_prefers_the_module_level_elapsed_time(install_module):
    install_module("cupy", _fake_cupy())
    backend = CuPyCUDATimingBackend()
    start, end = backend.record_event(), backend.record_event()

    assert backend.elapsed_time_ns(start, end) == 3_500_000


def test_cupy_backend_falls_back_to_the_event_method(install_module):
    """Older CuPy exposes elapsed time only on the event object."""
    install_module("cupy", _fake_cupy(with_get_elapsed_time=False))
    backend = CuPyCUDATimingBackend()
    start, end = backend.record_event(), backend.record_event()

    assert backend.elapsed_time_ns(start, end) == 2_500_000


def test_cupy_backend_reports_a_missing_install(hide_module):
    hide_module("cupy")
    with pytest.raises(ImportError, match="cupy is not installed"):
        CuPyCUDATimingBackend()


def test_cupy_backend_reports_a_failing_driver(install_module):
    install_module("cupy", _fake_cupy(raise_on_count=True))
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        CuPyCUDATimingBackend()


def test_cupy_backend_reports_a_machine_with_no_devices(install_module):
    install_module("cupy", _fake_cupy(device_count=0))
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        CuPyCUDATimingBackend()


@pytest.mark.parametrize("name", ["torch", "pytorch", "TORCH"])
def test_resolve_selects_torch_by_name(install_module, name):
    install_module("torch", _fake_torch())
    assert resolve_gpu_timing_backend(name).name == "torch"


def test_resolve_selects_cupy_by_name(install_module):
    install_module("cupy", _fake_cupy())
    assert resolve_gpu_timing_backend("cupy").name == "cupy"


def test_resolve_auto_prefers_torch(install_module):
    install_module("torch", _fake_torch())
    install_module("cupy", _fake_cupy())
    assert resolve_gpu_timing_backend("auto").name == "torch"


def test_resolve_auto_falls_through_to_cupy(install_module, hide_module):
    hide_module("torch")
    install_module("cupy", _fake_cupy())
    assert resolve_gpu_timing_backend("auto").name == "cupy"


def test_resolve_none_means_auto(install_module, hide_module):
    hide_module("torch")
    install_module("cupy", _fake_cupy())
    assert resolve_gpu_timing_backend(None).name == "cupy"


def test_resolve_auto_reports_every_backend_it_tried(hide_module):
    hide_module("torch")
    hide_module("cupy")
    with pytest.raises(RuntimeError) as excinfo:
        resolve_gpu_timing_backend("auto")

    message = str(excinfo.value)
    assert "no supported GPU backend" in message
    # Both failures are reported, so the user knows neither was silently skipped.
    assert "torch is not installed" in message
    assert "cupy is not installed" in message


def test_resolve_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown gpu_timing_backend"):
        resolve_gpu_timing_backend("rocm")


def test_resolve_accepts_a_custom_backend_object():
    class Custom:
        def record_event(self):
            return object()

        def elapsed_time_ns(self, start, end):
            return 7

    backend = Custom()
    assert resolve_gpu_timing_backend(backend) is backend


@pytest.mark.parametrize("missing", ["record_event", "elapsed_time_ns"])
def test_resolve_rejects_an_incomplete_backend_object(missing):
    class Custom:
        def record_event(self):
            return object()

        def elapsed_time_ns(self, start, end):
            return 7

    backend = Custom()
    setattr(type(backend), missing, None)
    try:
        with pytest.raises(TypeError, match="record_event"):
            resolve_gpu_timing_backend(backend)
    finally:
        delattr(type(backend), missing)
