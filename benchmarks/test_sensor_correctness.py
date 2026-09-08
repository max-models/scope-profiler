import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensor_workload import workload


def test_sensor_workload_is_correct_and_deterministic():
    expected = 450.93046978980857
    first = workload()
    second = workload()
    assert math.isclose(first, expected, rel_tol=1e-12)
    assert first == second
