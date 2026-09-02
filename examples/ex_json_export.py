"""
Writing a run as JSON instead of HDF5
=====================================

HDF5 is the default output and stays the better choice for a long run: it is
read back column by column rather than whole, and it is the only format the
parallel and rank-by-rank writers can produce. JSON is the right choice when
the consumer is not this package --- a browser, ``jq``, a notebook on a
machine without h5py, or a diff in code review.

Nothing is lost by choosing it. The document holds every call's timestamps in
integer nanoseconds, along with the thread and task tables, LIKWID counters
and line-profiler records, so ``read_json()`` gives back a ``ProfilingResults``
indistinguishable from the one ``read_h5()`` builds from the HDF5 file --- as
this example asserts.

There are three ways to get one, all shown below:

1. ``ProfileManager.session(file_path="run.json")`` --- ask the run for it.
2. ``write_json(results, ...)`` --- from results you already hold.
3. ``scope-profiler run -o run.json script.py`` or ``scope-profiler export
   json run.h5 -o exports`` --- from the command line, without writing Python.

Run::

    python examples/ex_json_export.py
"""

import math
from pathlib import Path

from scope_profiler import ProfileManager, read_h5, read_json, write_json

OUTPUT_DIR = Path("figures")


@ProfileManager.profile("assemble")
def assemble(size):
    """Stand-in for the expensive setup phase of a solver."""
    return [math.sin(i) * math.cos(i) for i in range(size)]


def simulate(num_iterations=5, size=20_000):
    """Run a few iterations of the toy 'solver', profiling each phase."""
    with ProfileManager.profile_region("simulation"):
        values = assemble(size)
        for _ in range(num_iterations):
            with ProfileManager.profile_region("iteration"):
                sum(math.sqrt(abs(value)) for value in values)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hdf5_path = OUTPUT_DIR / "json_example.h5"

    with ProfileManager.session(
        file_path=str(hdf5_path),
        verbose=False,
        return_results=True,
    ) as run:
        simulate()

    results = run.results
    results.print_summary(title="Run to export")

    # 1. From results you already hold. The extension decides: ".json.gz"
    # writes the same document gzip-compressed, which is what a real run
    # wants --- the event columns are the bulk of it and compress by roughly
    # an order of magnitude.
    json_path = write_json(results, OUTPUT_DIR / "json_example.json", indent=2)
    gzip_path = write_json(results, OUTPUT_DIR / "json_example.json.gz")

    # Do not read a rule off these three numbers: this run records a few
    # dozen events, where the HDF5 container's fixed overhead dominates
    # everything else. At a few hundred thousand events the picture settles
    # down to plain JSON being roughly a third larger than the HDF5 file, and
    # the gzipped document a few times smaller than it.
    print()
    for path, description in (
        (hdf5_path, "HDF5"),
        (json_path, "JSON, indented"),
        (gzip_path, "JSON, gzipped"),
    ):
        size = path.stat().st_size / 1024
        print(f"  {str(path):<34} {size:8.1f} KiB  ({description})")

    # 2. Read it back. Not "close enough": the same numbers, region by region
    # and call by call, as the HDF5 file the same run wrote.
    from_json = read_json(json_path)
    from_hdf5 = read_h5(hdf5_path)
    assert from_json.region_names == from_hdf5.region_names
    assert from_json.summary() == from_hdf5.summary()
    assert from_json.events() == from_hdf5.events()
    assert from_json.call_stack() == from_hdf5.call_stack()
    assert read_json(gzip_path).summary() == from_hdf5.summary()

    from_json.print_summary(title="The same run, read back from JSON")

    # 3. Everything downstream takes it from here: the plotting functions, the
    # exporters, and every CLI subcommand that reads a profile.
    print("  scope-profiler inspect", json_path)
    print("  scope-profiler report", json_path, "-o report.html")
    print("\n  Or skip the HDF5 file entirely:")
    print("  scope-profiler run -o profile.json examples/ex_cli_profiling.py")


if __name__ == "__main__":
    main()
