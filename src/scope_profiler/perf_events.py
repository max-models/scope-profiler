"""Small Linux ``perf_event_open`` wrapper used for region counter totals.

The wrapper intentionally uses only the stable hardware and software event
types exposed by ``linux/perf_event.h``.  It has no third-party dependency and
keeps counters attached to the calling thread (``pid=0, cpu=-1``).
"""

import ctypes
import os
import platform
import struct
from dataclasses import dataclass


class PerfEventError(RuntimeError):
    """A requested kernel performance event could not be opened or read."""


# perf_type_id, perf_hw_id / perf_sw_id.  The names deliberately match
# ``perf stat -e`` where possible.
EVENTS = {
    "cycles": (0, 0),
    "instructions": (0, 1),
    "cache-references": (0, 2),
    "cache-misses": (0, 3),
    "branches": (0, 4),
    "branch-misses": (0, 5),
    "cpu-clock": (1, 0),
    "task-clock": (1, 1),
    "page-faults": (1, 2),
    "context-switches": (1, 3),
}

_SYS_PERF_EVENT_OPEN = {"x86_64": 298, "aarch64": 241, "arm64": 241}
_PERF_EVENT_IOC_ENABLE = 0x2400
_PERF_EVENT_IOC_DISABLE = 0x2401
_PERF_EVENT_IOC_RESET = 0x2403
_PY_PERF_EVENT_HARDWARE = {
    "cycles": "CPU_CYCLES",
    "instructions": "INSTRUCTIONS",
    "cache-references": "CACHE_REFERENCES",
    "cache-misses": "CACHE_MISSES",
    "branches": "BRANCH_INSTRUCTIONS",
    "branch-misses": "BRANCH_MISSES",
}


def validate_events(events) -> tuple[str, ...]:
    """Normalize requested event names and reject unsupported names early."""
    if isinstance(events, str):
        events = [events]
    if not isinstance(events, (list, tuple)) or not events:
        raise ValueError("perf_events must be a non-empty list of event names")
    normalized = tuple(str(event).strip().lower() for event in events)
    if not all(normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("perf_events must contain unique, non-empty event names")
    unknown = sorted(set(normalized) - EVENTS.keys())
    if unknown:
        raise ValueError(
            "Unknown perf event(s): "
            + ", ".join(unknown)
            + ". Supported events: "
            + ", ".join(EVENTS)
        )
    if platform.system() != "Linux":
        raise PerfEventError("perf_events is supported only on Linux")
    if platform.machine().lower() not in _SYS_PERF_EVENT_OPEN:
        raise PerfEventError(
            f"perf_events is not implemented for {platform.machine()!r}; "
            "use LIKWID or run on x86_64/aarch64 Linux"
        )
    return normalized


@dataclass
class PerfEventTotals:
    """Aggregated counts for one region."""

    calls: int
    values: dict[str, int]


class PerfEventGroup:
    """Counters opened for one active region invocation.

    When the optional ``py-perf-event`` package can represent every requested
    event, its Rust implementation is used. It surfaces multiplexed/partial
    reads, avoiding understated counts on oversubscribed PMUs. The standard
    library syscall implementation remains the dependency-free fallback and
    also covers the software events not exposed by that package.
    """

    def __init__(self, events: tuple[str, ...]):
        self.events = events
        self.fds: list[int] = []
        self._measure = None

    def start(self) -> None:
        self._measure = _make_py_perf_measure(self.events)
        if self._measure is not None:
            try:
                self._measure.enable()
            except Exception as exc:
                self._measure = None
                raise _py_perf_error("enable", exc) from exc
            return
        try:
            for event in self.events:
                self.fds.append(_open_event(*EVENTS[event]))
            for fd in self.fds:
                _ioctl(fd, _PERF_EVENT_IOC_RESET)
                _ioctl(fd, _PERF_EVENT_IOC_ENABLE)
        except BaseException:
            self.close()
            raise

    def stop(self) -> dict[str, int]:
        if self._measure is not None:
            try:
                read = self._measure.read()
                if read.time_running_ns < read.time_enabled_ns:
                    raise PerfEventError(
                        "perf counters were multiplexed (enabled for "
                        f"{read.time_enabled_ns} ns but running for "
                        f"{read.time_running_ns} ns); reduce perf_events"
                    )
                return dict(zip(self.events, map(int, read.measurements)))
            except PerfEventError:
                raise
            except Exception as exc:
                raise _py_perf_error("read", exc) from exc
            finally:
                try:
                    self._measure.disable()
                finally:
                    self._measure = None
        try:
            for fd in self.fds:
                _ioctl(fd, _PERF_EVENT_IOC_DISABLE)
            return {
                event: struct.unpack("Q", os.read(fd, 8))[0]
                for event, fd in zip(self.events, self.fds)
            }
        finally:
            self.close()

    def close(self) -> None:
        if self._measure is not None:
            try:
                self._measure.disable()
            finally:
                self._measure = None
        while self.fds:
            os.close(self.fds.pop())


def _make_py_perf_measure(events: tuple[str, ...]):
    """Build an optional py-perf-event measurement, if it supports ``events``."""
    if not all(event in _PY_PERF_EVENT_HARDWARE for event in events):
        return None
    try:
        from py_perf_event import Hardware, Measure
    except ImportError:
        return None
    try:
        return Measure(
            [getattr(Hardware, _PY_PERF_EVENT_HARDWARE[event]) for event in events]
        )
    except Exception as exc:
        raise _py_perf_error("create", exc) from exc


def _py_perf_error(action: str, exc: Exception) -> PerfEventError:
    """Add the same actionable permission hint to optional-backend errors."""
    message = str(exc)
    hint = (
        "; kernel policy may forbid this (check /proc/sys/kernel/perf_event_paranoid)"
        if "permission denied" in message.lower() or "os error 13" in message.lower()
        else ""
    )
    return PerfEventError(f"py-perf-event could not {action} counters: {message}{hint}")


def _ioctl(fd: int, request: int) -> None:
    try:
        import fcntl

        fcntl.ioctl(fd, request, 0)
    except OSError as exc:
        raise PerfEventError(f"perf event ioctl failed: {exc}") from exc


def _open_event(event_type: int, config: int) -> int:
    # struct perf_event_attr, zero-filled through ``size``.  The first 40
    # bytes are stable across supported kernels; 120 keeps modern kernels
    # happy while allowing older ones to ignore trailing zero fields.
    attr = (ctypes.c_ubyte * 120)()
    struct.pack_into("IIQ", attr, 0, event_type, len(attr), config)
    # disabled | exclude_kernel | exclude_hv: unprivileged users normally
    # may count user-space code even when kernel/hypervisor counting is barred.
    struct.pack_into("Q", attr, 40, 1 | (1 << 5) | (1 << 6))
    syscall = _SYS_PERF_EVENT_OPEN[platform.machine().lower()]
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(syscall, ctypes.byref(attr), 0, -1, -1, 0)
    if fd < 0:
        err = ctypes.get_errno()
        detail = os.strerror(err)
        hint = (
            "; kernel policy may forbid this (check /proc/sys/kernel/perf_event_paranoid)"
            if err in {1, 13}
            else ""
        )
        raise PerfEventError(f"Could not open perf event: {detail}{hint}")
    return fd
