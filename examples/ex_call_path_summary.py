"""Show that same-named regions are summarized separately by call path.

Run with ``python examples/ex_call_path_summary.py``.  The summary contains
two ``c`` rows: one below ``a`` (three short calls) and one below ``b`` (two
longer calls).  Their totals and averages differ even though both scopes are
named ``c``.
"""

import time

from scope_profiler import ProfileManager


@ProfileManager.profile("c")
def c(seconds: float) -> None:
    time.sleep(seconds)


@ProfileManager.profile("a")
def a() -> None:
    for _ in range(3):
        c(0.003)


@ProfileManager.profile("b")
def b() -> None:
    for _ in range(2):
        c(0.015)


# ``verbose=False`` avoids the context manager's automatic table so the
# explicitly titled summary below is the example's only output.
with ProfileManager.session(return_results=True, verbose=False) as run:
    a()
    b()

run.results.print_summary(title="Same region name, separate call paths")
