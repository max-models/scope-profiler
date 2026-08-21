"""Optional CUDA-event timing backends for asynchronous GPU work."""

from __future__ import annotations

from typing import Protocol


class GPUTimingBackend(Protocol):
    """Minimal backend contract used by GPU-timed regions."""

    name: str

    def record_event(self):
        """Record and return an event on the backend's current CUDA stream."""

    def elapsed_time_ns(self, start_event, end_event) -> int:
        """Return elapsed device time between two events in nanoseconds."""


class TorchCUDATimingBackend:
    """CUDA-event timing through PyTorch's current CUDA stream."""

    name = "torch"

    def __init__(self):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch CUDA timing requested but torch is not installed."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch CUDA timing requested but CUDA is not available."
            )
        self._torch = torch

    def record_event(self):
        event = self._torch.cuda.Event(enable_timing=True)
        event.record(self._torch.cuda.current_stream())
        return event

    def elapsed_time_ns(self, start_event, end_event) -> int:
        end_event.synchronize()
        return round(start_event.elapsed_time(end_event) * 1_000_000.0)


class CuPyCUDATimingBackend:
    """CUDA-event timing through CuPy's current CUDA stream."""

    name = "cupy"

    def __init__(self):
        try:
            import cupy
        except ImportError as exc:
            raise ImportError(
                "CuPy CUDA timing requested but cupy is not installed."
            ) from exc
        try:
            device_count = cupy.cuda.runtime.getDeviceCount()
        except Exception as exc:
            raise RuntimeError(
                "CuPy CUDA timing requested but CUDA is not available."
            ) from exc
        if device_count <= 0:
            raise RuntimeError("CuPy CUDA timing requested but CUDA is not available.")
        self._cupy = cupy

    def record_event(self):
        event = self._cupy.cuda.Event()
        event.record(self._cupy.cuda.get_current_stream())
        return event

    def elapsed_time_ns(self, start_event, end_event) -> int:
        end_event.synchronize()
        get_elapsed_time = getattr(self._cupy.cuda, "get_elapsed_time", None)
        if get_elapsed_time is not None:
            milliseconds = get_elapsed_time(start_event, end_event)
        else:
            milliseconds = start_event.elapsed_time(end_event)
        return round(milliseconds * 1_000_000.0)


def resolve_gpu_timing_backend(backend="auto") -> GPUTimingBackend:
    """Resolve a CUDA timing backend name or validate a backend-like object."""
    if backend is None:
        backend = "auto"
    if not isinstance(backend, str):
        for attr in ("record_event", "elapsed_time_ns"):
            if not callable(getattr(backend, attr, None)):
                raise TypeError(
                    "gpu_timing_backend objects must provide record_event() and "
                    "elapsed_time_ns(start_event, end_event)."
                )
        return backend

    normalized = backend.lower()
    choices = {
        "torch": TorchCUDATimingBackend,
        "pytorch": TorchCUDATimingBackend,
        "cupy": CuPyCUDATimingBackend,
    }
    if normalized == "auto":
        errors = []
        for factory in (TorchCUDATimingBackend, CuPyCUDATimingBackend):
            try:
                return factory()
            except (ImportError, RuntimeError) as exc:
                errors.append(str(exc))
        joined = " ".join(errors)
        raise RuntimeError(
            "CUDA-event timing requested but no supported GPU backend is available. "
            "Install PyTorch or CuPy with CUDA support, or pass a custom "
            f"gpu_timing_backend. Tried: {joined}"
        )
    if normalized not in choices:
        raise ValueError(
            "Unknown gpu_timing_backend "
            f"{backend!r}; expected 'auto', 'torch', 'pytorch', 'cupy', or a backend object."
        )
    return choices[normalized]()
