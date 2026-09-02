"""Capture memory allocations with Memray alongside region timings.

Install the optional dependency, then run this example::

    pip install "scope-profiler[memray]"
    python examples/ex_memory_profiling.py

It writes the usual ``memory_allocation_example.h5`` timing profile and a
Memray allocation capture, ``memory_allocation_example.memray.bin``. Render
the latter with Memray's reporters, for example::

    memray flamegraph memory_allocation_example.memray.bin

Enable ``memray_trace_python_allocators`` only when individual Python-object
allocations matter: it provides more detail at a substantial runtime and file
size cost.
"""

from scope_profiler import ProfileManager


def build_payload(size: int) -> list[dict[str, int]]:
    """Allocate a deliberately visible payload inside one timed region."""
    with ProfileManager.profile_region("build-payload"):
        return [{"index": index, "square": index * index} for index in range(size)]


def main() -> None:
    with ProfileManager.session(
        file_path="memory_allocation_example.h5",
        use_memray=True,
        # Set this to True when investigating Python object allocation sites.
        memray_trace_python_allocators=False,
    ):
        payload = build_payload(100_000)
        with ProfileManager.profile_region("transform-payload"):
            # Keep the result alive through finalization so the flame graph
            # has a clear high-watermark allocation to display.
            payload = [item["square"] for item in payload]
        assert len(payload) == 100_000


if __name__ == "__main__":
    main()
