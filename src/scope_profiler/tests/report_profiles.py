"""Synthetic nested profiles for the HTML report benchmark and its limits.

Shared by ``benchmarks/report_workload.py`` and ``test_report_performance.py``
so the benchmark and the test limits measure the same shapes.
"""

import h5py
import numpy as np


def write_nested(path, phases, steps=10, kernels=2, calls=1, ranks=1):
    """``phases`` top-level regions, each with ``steps`` children and
    ``kernels`` grandchildren per child: phases * (1 + steps * (1 + kernels))
    regions, every one called ``calls`` times on every rank."""
    names, starts, ends = [], [], []
    width = 1000
    t = 0
    for a in range(phases):
        for _ in range(calls):
            span = steps * (kernels + 1) * width
            entries = [(f"phase_{a:03d}", t, t + span)]
            for b in range(steps):
                step_start = t + b * (kernels + 1) * width
                entries.append(
                    (
                        f"phase_{a:03d}.step_{b}",
                        step_start,
                        step_start + (kernels + 1) * width - 10,
                    )
                )
                for c in range(kernels):
                    kernel_start = step_start + c * width
                    entries.append(
                        (
                            f"phase_{a:03d}.step_{b}.kernel_{c}",
                            kernel_start,
                            kernel_start + width - 10 - a,
                        )
                    )
            for name, start, end in entries:
                names.append(name)
                starts.append(start)
                ends.append(end)
            t += span + 100
    by_region: dict[str, tuple[list, list]] = {}
    for name, start, end in zip(names, starts, ends):
        region = by_region.setdefault(name, ([], []))
        region[0].append(start)
        region[1].append(end)
    with h5py.File(path, "w") as h5file:
        for rank in range(ranks):
            regions = h5file.create_group(f"rank{rank}").create_group("regions")
            for name, (region_starts, region_ends) in by_region.items():
                group = regions.create_group(name)
                group.create_dataset(
                    "start_times", data=np.asarray(region_starts, dtype=np.int64)
                )
                group.create_dataset(
                    "end_times", data=np.asarray(region_ends, dtype=np.int64)
                )
