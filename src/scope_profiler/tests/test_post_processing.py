import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_durations, plot_gantt, plot_speedup
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


def _sample_file_data(rank_count, setup_duration, solve_duration):
    return {
        rank: {
            "setup": ([0], [setup_duration]),
            "solve": ([20], [20 + solve_duration]),
        }
        for rank in range(rank_count)
    }


def test_plot_durations_comparison(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    out_file = tmp_path / "durations_plot.png"

    _write_sample_h5(file_one, _sample_file_data(2, 10, 20))
    _write_sample_h5(file_two, _sample_file_data(2, 20, 40))

    readers = [ProfilingH5Reader(file_one), ProfilingH5Reader(file_two)]

    plot_durations(readers, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_gantt_combined(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    out_file = tmp_path / "gantt_plot.png"

    _write_sample_h5(file_one, _sample_file_data(2, 10, 20))
    _write_sample_h5(file_two, _sample_file_data(2, 20, 40))

    readers = [ProfilingH5Reader(file_one), ProfilingH5Reader(file_two)]

    plot_gantt(readers, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_speedup(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    file_four = tmp_path / "run_4.h5"
    out_file = tmp_path / "speedup_plot.png"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))
    _write_sample_h5(file_four, _sample_file_data(4, 25, 50))

    readers = [
        ProfilingH5Reader(file_one),
        ProfilingH5Reader(file_two),
        ProfilingH5Reader(file_four),
    ]

    plot_speedup(readers, filepath=out_file, show=False, verbose=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_post_processing_cli_supports_multiple_files(tmp_path):
    file_one = tmp_path / "run_1.h5"
    file_two = tmp_path / "run_2.h5"
    file_four = tmp_path / "run_4.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))
    _write_sample_h5(file_four, _sample_file_data(4, 25, 50))

    main([str(file_one), str(file_two), str(file_four), "-o", str(output_dir)])

    durations_plot = output_dir / "durations_plot.png"
    gantt_plot = output_dir / "gantt_plot.png"
    speedup_plot = output_dir / "speedup_plot.png"

    assert durations_plot.exists()
    assert durations_plot.stat().st_size > 0
    assert gantt_plot.exists()
    assert gantt_plot.stat().st_size > 0
    assert speedup_plot.exists()
    assert speedup_plot.stat().st_size > 0


def test_post_processing_cli_supports_wildcard_file_patterns(tmp_path):
    file_one = tmp_path / "file_1.h5"
    file_two = tmp_path / "file_2.h5"
    output_dir = tmp_path / "figures"

    _write_sample_h5(file_one, _sample_file_data(1, 100, 200))
    _write_sample_h5(file_two, _sample_file_data(2, 50, 100))

    wildcard_pattern = str(tmp_path / "file_*.h5")
    main([wildcard_pattern, "-o", str(output_dir)])

    durations_plot = output_dir / "durations_plot.png"
    gantt_plot = output_dir / "gantt_plot.png"
    speedup_plot = output_dir / "speedup_plot.png"

    assert durations_plot.exists()
    assert durations_plot.stat().st_size > 0
    assert gantt_plot.exists()
    assert gantt_plot.stat().st_size > 0
    assert speedup_plot.exists()
    assert speedup_plot.stat().st_size > 0
