"""Profile only aggregate timing statistics.

Run with::

    python examples/ex_aggregation.py

Aggregation mode is useful when a detailed event timeline would be too large.
It records count, inclusive total, minimum, maximum, and exclusive total per
region, but does not retain individual events.
"""

from scope_profiler import ProfileManager


def work():
    """A decorated region contributes aggregate statistics."""
    with ProfileManager.profile_region("inner"):
        sum(range(1000))


def main():
    with ProfileManager.session(
        file_path="aggregation_profile.h5",
        aggregation_mode=True,
        verbose=False,
        return_results=True,
    ) as run:
        work_region = ProfileManager.profile("work")(work)
        total = ProfileManager.profile_region("total")
        outer = ProfileManager.profile_region("outer")
        with total:
            for _ in range(10):
                with outer:
                    work_region()

    results = run.results
    results.print_summary(title="Aggregate profiling results")
    for region in results.get_regions():
        assert region[0].events() == []


if __name__ == "__main__":
    main()
