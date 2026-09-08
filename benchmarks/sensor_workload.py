"""Deterministic, representative workload built from the project's sensor code."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensor import Analyzer, Sensor


def build_sensors(count=64, readings_per_sensor=10000):
    sensors = []
    for sensor_id in range(count):
        readings = [
            ((sensor_id * 17 + index * 13) % 140) - 20
            for index in range(readings_per_sensor)
        ]
        sensors.append(Sensor(f"S{sensor_id}", readings))
    return sensors


def workload():
    scores = Analyzer(build_sensors()).compute_scores()
    return math.fsum(scores.values())


if __name__ == "__main__":
    print(f"sensor_score={workload():.15f}")
