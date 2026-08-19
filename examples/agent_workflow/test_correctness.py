"""
Correctness gate for optimize_me.py.

This is what makes the agent loop safe: a speedup only counts if this still
passes. Reference values were computed from the original, unoptimized
implementation and must not change when the implementation is optimized --
only the runtime should change.

Run:

    pytest examples/agent_workflow/test_correctness.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from optimize_me import run_pipeline  # noqa: E402

# Reference value from the original pure-Python implementation at n=4000,
# seed=0, iterations=20. Any optimized version of run_pipeline / smooth /
# norm / transform must reproduce this to floating-point tolerance.
EXPECTED_RESULT_NORM = 38.446926250156515


def test_pipeline_matches_reference():
    value = run_pipeline(n=4000, seed=0)
    assert math.isclose(value, EXPECTED_RESULT_NORM, rel_tol=1e-9)


def test_pipeline_deterministic():
    a = run_pipeline(n=500, seed=1)
    b = run_pipeline(n=500, seed=1)
    assert a == b
