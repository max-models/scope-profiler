"""
Zero-instrumentation profiling with `python -m scope_profiler`
================================================================

This script contains no scope-profiler imports, decorators, or context
managers -- it looks like ordinary code. Run it under scope-profiler's CLI
the same way you'd run `python -m cProfile`, and every function call it
makes is automatically recorded as its own region.

Run::

    python -m scope_profiler examples/ex_cli_profiling.py

By default only this script's own functions are instrumented (standard
library calls, e.g. `json.dumps` below, are skipped to keep overhead low).
Pass `--all` to also trace standard-library/installed-package calls, and
`-o`/`--outfile` to change where results are written (default:
`profiling_data.h5`)::

    python -m scope_profiler --all -o cli_profile.h5 examples/ex_cli_profiling.py
"""

import json


def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def summarize(values):
    return {"count": len(values), "total": sum(values)}


def main():
    values = [fib(n) for n in range(15)]
    print(json.dumps(summarize(values)))


main()
