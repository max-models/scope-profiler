"""Optional Memray allocation-trace integration."""

from pathlib import Path


class MemrayAllocationTracker:
    """Start and stop one optional process-wide Memray allocation capture."""

    def __init__(
        self,
        path: str | Path,
        *,
        native_traces: bool = False,
        trace_python_allocators: bool = False,
        follow_fork: bool = False,
    ) -> None:
        try:
            import memray
        except ImportError as exc:
            raise ImportError(
                "Memory allocation profiling requested but memray is not installed. "
                'Install it with `pip install "scope-profiler[extras]"` '
                "or `pip install memray`.",
            ) from exc

        self.path = Path(path)
        self._tracker = memray.Tracker(
            str(self.path),
            native_traces=native_traces,
            trace_python_allocators=trace_python_allocators,
            follow_fork=follow_fork,
        )
        self._active = False

    def start(self) -> None:
        """Enable allocation tracking for this process."""
        self._tracker.__enter__()
        self._active = True

    def stop(self) -> None:
        """Flush and close the capture. Safe to call more than once."""
        if self._active:
            self._tracker.__exit__(None, None, None)
            self._active = False
