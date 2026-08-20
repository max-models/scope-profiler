"""
A deliberately unoptimized numerical pipeline, meant to be handed to an AI
coding agent as the *target* of a measure -> change -> re-measure -> judge
loop, driven through the scope-profiler MCP tools (see ``README.md`` in this
directory for the exact tool calls).

Nothing in this file imports scope_profiler: running it under
``scope-profiler run`` (or the MCP ``run_profile`` tool) auto-instruments
every function call, so each function below becomes its own region with no
decorators needed.

Run directly (no profiling):

    python examples/agent_workflow/optimize_me.py

Run under scope-profiler:

    scope-profiler run examples/agent_workflow/optimize_me.py -o baseline.h5
"""

import math
import random


def smooth(data, iterations=5):
    """Jacobi-style smoother, written as pure-Python loops on purpose."""
    n = len(data)
    for _ in range(iterations):
        new = list(data)
        for i in range(1, n - 1):
            new[i] = 0.5 * (data[i - 1] + data[i + 1])
        data = new
    return data


def norm(data):
    """L2 norm, computed with an explicit loop instead of vectorized ops."""
    total = 0.0
    for x in data:
        total += x * x
    return math.sqrt(total)


def transform(data):
    """Elementwise transform re-deriving ``sqrt(abs(x))`` per element."""
    result = []
    for x in data:
        result.append(math.sin(x) * math.sqrt(abs(x) + 1.0))
    return result


def run_pipeline(n=4000, seed=0):
    rng = random.Random(seed)
    data = [rng.random() for _ in range(n)]

    smoothed = smooth(data, iterations=20)
    transformed = transform(smoothed)
    result_norm = norm(transformed)
    return result_norm


if __name__ == "__main__":
    value = run_pipeline()
    print(f"result_norm = {value}")
