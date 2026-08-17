"""
Inspecting a region's source code
==================================

Every region remembers where it was defined in your code: the ``with`` block
for the context-manager form, or the whole function body for the decorator
form. It is captured once, when the region is first created, so it costs
nothing on the hot path -- see ``region.source_file``, ``region.source_lineno``
and ``region.source_text`` below.

If the same region name is used at more than one call site, the source of
whichever call site created it first is kept (their timings are pooled
together either way, under that one name).

Run::

    python examples/ex_region_source.py

Then look the same information up from the written file, without re-running
anything::

    scope-profiler inspect profiling_data.h5 --source assemble solve
"""

import math

from scope_profiler import ProfileManager


@ProfileManager.profile("assemble")
def assemble(size):
    """Stand-in for the expensive setup phase of a solver."""
    return [math.sin(i) * math.cos(i) for i in range(size)]


def solve(values):
    with ProfileManager.profile_region("solve"):
        return sum(math.sqrt(abs(v)) + math.log1p(abs(v)) for v in values)


def main():
    ProfileManager.setup()

    values = assemble(20_000)
    solve(values)

    # return_results=True hands back the same data the written file holds,
    # so the source can be printed without a separate read_h5() call.
    results = ProfileManager.finalize(verbose=False, return_results=True)

    for name in ("assemble", "solve"):
        region = results[name]
        print(f"\n{name}  ({region.source_file}:{region.source_lineno})")
        print(f"  {region.num_calls} call(s), {region.total_duration:.4f} s total")
        for line in region.source_text.rstrip("\n").splitlines():
            print(f"    {line}")


if __name__ == "__main__":
    main()
