import math


def clamp(value, minimum, maximum):
    """Clamp a value between a minimum and maximum."""
    return max(minimum, min(value, maximum))


def average(values):
    """Compute the average of a list of numbers."""
    return sum(values) / len(values)


class Sensor:
    def __init__(self, name, readings):
        self.name = name
        self.readings = readings

    def calibrated_readings(self):
        return [clamp(x, 0.0, 100.0) for x in self.readings]

    def mean(self):
        return average(self.calibrated_readings())


class Analyzer:
    def __init__(self, sensors):
        self.sensors = sensors

    def compute_scores(self):
        scores = {}
        for sensor in self.sensors:
            scores[sensor.name] = self._score(sensor)
        return scores

    def _score(self, sensor):
        mean = sensor.mean()
        return math.sqrt(mean)


def print_report(scores):
    print("=== Sensor Report ===")
    for name, score in scores.items():
        print(f"{name:10s}: {score:.2f}")


def create_sensors():
    return [
        Sensor("A", [10, 20, 30, 110]),
        Sensor("B", [50, 60, 70]),
        Sensor("C", [-5, 15, 25]),
    ]


def main():
    sensors = create_sensors()
    analyzer = Analyzer(sensors)
    scores = analyzer.compute_scores()
    print_report(scores)


if __name__ == "__main__":
    main()
