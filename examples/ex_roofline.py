"""Plot LIKWID regions on a hardware roofline.

Install the plotting and LIKWID extras, then run under a FLOPS group that
reports both a FLOP rate and memory bandwidth, for example::

    pip install 'scope-profiler[likwid,pproc]'
    likwid-perfctr -C 0-3 -g FLOPS_DP -m python examples/ex_roofline.py

Supply measured machine ceilings below for a true roofline.  Omitting either
one still plots the observations, but uses clearly labelled empirical
ceilings instead.
"""

from pathlib import Path

import numpy as np

from scope_profiler import ProfileManager, plot_roofline, read_h5

OUTPUT_DIR = Path("figures")
H5_PATH = OUTPUT_DIR / "roofline_profile.h5"

# Replace these with STREAM and compute-benchmark measurements for the machine
# being profiled.  Units are GB/s and GFLOP/s respectively.
PEAK_BANDWIDTH_GBS = 320.0
PEAK_FLOPS_GFLOPS = 2400.0


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    matrix_a = np.random.default_rng(1).random((512, 512))
    matrix_b = np.random.default_rng(2).random((512, 512))
    stream = np.ones(16_000_000)

    with ProfileManager.session(use_likwid=True, file_path=str(H5_PATH)):
        with ProfileManager.region("compute_bound"):
            for _ in range(10):
                matrix_a @ matrix_b
        with ProfileManager.region("memory_stream"):
            for _ in range(10):
                np.add(stream, 1.0, out=stream)

    results = read_h5(H5_PATH)
    if not results.has_likwid:
        print("No LIKWID counters were recorded; run via likwid-perfctr as above.")
        return

    plot_roofline(
        results,
        peak_flops=PEAK_FLOPS_GFLOPS,
        peak_bandwidth=PEAK_BANDWIDTH_GBS,
        filepath=OUTPUT_DIR / "roofline.png",
        data_filepath=OUTPUT_DIR / "roofline_data.json",
        data_format="json",
        include="compute_bound|memory_stream",
    )
    print(f"Wrote {OUTPUT_DIR / 'roofline.png'} and roofline_data.json")


if __name__ == "__main__":
    main()
