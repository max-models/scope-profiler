"""Regenerate the checked-in plot-data fixtures.

The figure builders are tested against JSON the profiler really writes, not
hand-typed payloads: every builder that lost the ``file`` column did so while
its hand-written test payload had only one run. Run this after changing an
exporter payload::

    python packages/plotly/test/fixtures/generate_fixtures.py

The runs are synthetic and deterministic -- two files, three ranks, three
regions with several calls each -- so the fixtures stay small and a rerun
produces no diff.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
# Two runs of the same code at different scales, so the multi-run payloads
# (heatmap lanes, imbalance series, histogram bins) carry two files.
RUNS = {
    # Different rank counts, so the scaling payloads have two points to plot
    # and the heatmap has lanes that only one of the runs contributes.
    "run_3ranks": {"ranks": 3, "scale": 1.0},
    "run_6ranks": {"ranks": 6, "scale": 0.55},
}
REGIONS = {
    "setup": (1, 4_000_000),
    "solve": (12, 9_000_000),
    "exchange": (12, 2_500_000),
}
PLOTS = [
    "gantt",
    "density",
    "durations",
    "timeseries",
    "histogram",
    "imbalance",
    "rank_heatmap",
    "flame_chart",
    "flame_graph",
    "callgraph",
    "speedup",
    "weak_scaling",
    "scaling_efficiency",
]


def write_profile(path: Path, ranks: int, scale: float) -> None:
    """Write one HDF5 profile with deterministic nanosecond timestamps."""
    with h5py.File(path, "w") as h5file:
        for rank in range(ranks):
            regions = h5file.create_group(f"rank{rank}").create_group("regions")
            clock = 0
            call_id = 0
            for name, (calls, duration) in REGIONS.items():
                starts, ends = [], []
                for call in range(calls):
                    # A per-rank stretch factor gives the ranks something to be
                    # imbalanced about, without any randomness.
                    length = int(duration * scale * (1.0 + 0.12 * rank + 0.03 * call))
                    starts.append(clock)
                    ends.append(clock + length)
                    clock += length + 1_000_000
                    call_id += 1
                group = regions.create_group(name)
                group.create_dataset(
                    "start_times", data=np.asarray(starts, dtype=np.int64)
                )
                group.create_dataset("end_times", data=np.asarray(ends, dtype=np.int64))


def main() -> int:
    work = HERE / "_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    profiles = []
    for label, spec in RUNS.items():
        path = work / f"{label}.h5"
        write_profile(path, spec["ranks"], spec["scale"])
        profiles.append(str(path))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scope_profiler",
            "export",
            "plot-data",
            *profiles,
            "-o",
            str(work / "out"),
            "--format",
            "json",
            "--plots",
            *PLOTS,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    for source in sorted((work / "out").glob("*.json")):
        document = json.loads(source.read_text(encoding="utf-8"))
        # The statistics document records where each profile lived, which is a
        # machine-specific absolute path; the fixture keeps only the file name.
        for entry in document.get("files", []):
            if "file_path" in entry:
                entry["file_path"] = Path(entry["file_path"]).name
        (HERE / source.name).write_text(
            json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {source.name}")

    shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
