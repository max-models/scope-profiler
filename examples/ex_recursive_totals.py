"""
Totals of recursive regions and of their children
=================================================

A recursive region is one row in the summary: its call count includes every
level of the recursion, but its total only adds up the outermost calls, since
each call's duration already contains the calls nested inside it. The total
therefore never exceeds the wall-clock time of the region that called it, and
agrees with its ``% session`` share.

The regions called from a recursive function are pooled over the levels of the
recursion too, but they do not call themselves. They are not marked with the
recursive symbol, and their totals add up every call, without overlap.

The script checks all of that against the summary rows.

Run::

    python examples/ex_recursive_totals.py
"""

import time

from scope_profiler import ProfileManager
from scope_profiler.summary import region_rows


@ProfileManager.profile("fibonacci")
def fibonacci(n):
    if n < 2:
        with ProfileManager.region("leaf"):
            time.sleep(0.001)  # stand-in for leaf work
        return n
    with ProfileManager.region("combine"):
        time.sleep(0.0005)  # stand-in for work done at every level
    return fibonacci(n - 1) + fibonacci(n - 2)


def main():
    with ProfileManager.session(
        use_likwid=False,
        deactivate_file_output=True,
        file_path="recursive_totals.h5",
        return_results=True,
        verbose=False,
    ) as run:
        fibonacci(10)

    results = run.results
    results.print_summary()

    rows = {row["name"]: row for row in region_rows(results)}
    session = rows["scope_profiler.session"]
    fib, combine, leaf = rows["fibonacci"], rows["combine"], rows["leaf"]

    # The recursive row: marked, and no more than the time it ran for.
    assert fib["recursive"]
    assert fib["calls"] == 177
    assert fib["total"] <= session["total"]
    assert abs(fib["total"] - fib["coverage"]) < 1e-9

    # Its children: not recursive, and together no more than their parent.
    assert not combine["recursive"] and not leaf["recursive"]
    assert (combine["calls"], leaf["calls"]) == (88, 89)
    assert combine["total"] + leaf["total"] <= fib["total"] + 1e-9

    print(
        f"fibonacci: {fib['total']:.4f} s of a {session['total']:.4f} s session; "
        f"combine + leaf: {combine['total'] + leaf['total']:.4f} s"
    )


if __name__ == "__main__":
    main()
