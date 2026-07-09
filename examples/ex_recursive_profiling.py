"""
Recursive function-call profiling with scope-profiler
=====================================================

This example enables recursive profiling so one decorated entrypoint can
capture nested Python function calls automatically.

Run::

    python examples/ex_recursive_profiling.py
"""

from scope_profiler import ProfileManager


ProfileManager.setup(
    use_likwid=False,
    time_trace=True,
    flush_to_disk=True,
    recursive_profile=True,
)


def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def work():
    out = 0
    for n in range(10):
        out += fib(n)
    return out


@ProfileManager.profile("run_workload")
def run_workload():
    return work()


run_workload()
ProfileManager.finalize()
