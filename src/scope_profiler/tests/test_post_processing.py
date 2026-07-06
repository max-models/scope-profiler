import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_durations, plot_gantt
from scope_profiler.post_processing import main


def _write_sample_h5(path, rank_regions):
    with h5py.File(path, "w") as h5file:
        for rank, regions in rank_regions.items():
            rank_group = h5file.create_group(f"rank{rank}")
            regions_group = rank_group.create_group("regions")
            for region_name, (start_times, end_times) in regions.items():
                region_group = regions_group.create_group(region_name)
                region_group.create_dataset(
                    "start_times",
                    data=np.asarray(start_times, dtype=np.int64),
                )
                region_group.create_dataset(
                    "end_times",
                    data=np.asarray(end_times, dtype=np.int64),
                )


def _sample_file_data(scale):
    return {
        0: {
            "setup": ([0, 20], [10 * scale, 35 * scale]),
            "solve": ([40], [70 * scale]),
        },
        1: {
            "setup": ([5], [15 * scale]),
            "solve": ([45], [95 * scale]),
        },
    }


def test_plot_durations_comparison(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    out_file = tmp_path / "durations_plot.png"

    _write_sample_h5(file_one, _sample_file_data(scale=1))
    _write_sample_h5(file_two, _sample_file_data(scale=2))

    readers = [ProfilingH5Reader(file_one), ProfilingH5Reader(file_two)]

    plot_durations(readers, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_gantt_combined(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    out_file = tmp_path / "gantt_plot.png"

    _write_sample_h5(file_one, _sample_file_data(scale=1))
    _write_sample_h5(file_two, _sample_file_data(scale=2))

    readers = [ProfilingH5Reader(file_one), ProfilingH5Reader(file_two)]

    plot_gantt(readers, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_post_processing_cli_supports_multiple_files(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(scale=1))
    _write_sample_h5(file_two, _sample_file_data(scale=2))

    main([str(file_one), str(file_two), "-o", str(output_dir)])

    durations_plot = output_dir / "durations_plot.png"
    gantt_plot = output_dir / "gantt_plot.png"

    assert durations_plot.exists()
    assert durations_plot.stat().st_size > 0
    assert gantt_plot.exists()
    assert gantt_plot.stat().st_size > 0
