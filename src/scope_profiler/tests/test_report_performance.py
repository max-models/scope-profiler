"""Time and size limits for HTML report generation.

The shapes are the cases of ``benchmarks/report_workload.py``; run that
benchmark (``scope-profiler benchmark run benchmarks/report.toml``) for real
numbers. Measured on an Apple M1 (warm, best of three, Plotly runtime left
out), the limits below sit about 5x above the time and 2x above the size, so
a slower CI runner or coverage tracing passes while a path that turns
quadratic in the region count -- a stacked durations chart once wrote 2.3
million rows for 1550 regions -- does not:

==============  =======  =======  ==========
case            regions  time     size
==============  =======  =======  ==========
small                93  0.10 s   0.27 MB
nested_1550        1550  1.2 s    2.5 MB
many_calls          310  0.7 s    5.6 MB (31k calls)
four_ranks          310  0.5 s    1.3 MB
compare_1550   2 x 1550  1.9 s    4.0 MB
==============  =======  =======  ==========
"""

import time

import pytest

from scope_profiler.html_report import create_html_report

from .report_profiles import write_nested

pytestmark = pytest.mark.slow

#: name -> (profiles compared, write_nested keyword arguments,
#: seconds allowed, megabytes allowed).
LIMITS = {
    "small": (1, {"phases": 3}, 1.5, 1.0),
    "nested_1550": (1, {"phases": 50}, 8.0, 5.0),
    "many_calls": (1, {"phases": 10, "calls": 100}, 6.0, 10.0),
    "four_ranks": (1, {"phases": 10, "ranks": 4}, 4.0, 3.0),
    "compare_1550": (2, {"phases": 50}, 12.0, 8.0),
}


def _best_of(runs, report, *args, **kwargs):
    """Fastest of ``runs`` reports: the least disturbed by other load."""
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        create_html_report(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return best, report.stat().st_size / 1e6


@pytest.fixture(scope="module")
def warm(tmp_path_factory):
    """Pay the one-off imports (plotly, maxplotlib) outside every timing."""
    pytest.importorskip("plotly")
    directory = tmp_path_factory.mktemp("warm")
    write_nested(directory / "warm.h5", phases=1)
    create_html_report(directory / "warm.h5", directory / "warm.html")


@pytest.mark.parametrize("case", LIMITS)
def test_report_generation_stays_within_its_limits(warm, tmp_path, case):
    count, shape, seconds, megabytes = LIMITS[case]
    paths = []
    for index in range(count):
        path = tmp_path / f"{case}-{index}.h5"
        write_nested(path, **shape)
        paths.append(path)
    report = tmp_path / f"{case}.html"

    elapsed, size = _best_of(
        2,
        report,
        paths if count > 1 else paths[0],
        report,
        individual_reports=False,
        # The embedded Plotly runtime is a fixed ~5 MB; the limit is on what
        # the profile adds.
        charts_cdn=True,
    )

    assert size < megabytes, f"{case}: {size:.1f} MB, limit {megabytes} MB"
    assert elapsed < seconds, f"{case}: {elapsed:.2f} s, limit {seconds} s"


def test_report_time_grows_about_linearly_with_regions(warm, tmp_path):
    # Ten times the regions takes about 12x as long; quadratic work would
    # take about 100x. The ratio holds on any machine, unlike a ceiling.
    small = tmp_path / "small.h5"
    large = tmp_path / "large.h5"
    write_nested(small, phases=5)
    write_nested(large, phases=50)
    report = tmp_path / "report.html"

    small_time, _ = _best_of(3, report, small, report, charts_cdn=True)
    large_time, _ = _best_of(3, report, large, report, charts_cdn=True)

    ratio = large_time / small_time
    assert ratio < 25, (
        f"155 regions: {small_time:.3f} s, 1550: {large_time:.3f} s "
        f"({ratio:.1f}x for 10x the regions)"
    )
