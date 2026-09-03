"""Tests for ``scope_profiler.mcp_server.tools``.

These test the plain Python functions the MCP server delegates to, without
importing the ``mcp`` package at all -- the ``mcp``-dependent wiring in
``scope_profiler.mcp_server.server`` is covered separately in
``test_mcp_server.py``.
"""

import h5py
import numpy as np
import pytest

from scope_profiler.mcp_server.tools import (
    ToolError,
    compare_profiles,
    inspect_profile,
    plot_profile,
    run_profile,
)

NS = 1_000_000_000


def _write_sample_h5(path, rank_regions, metadata=None):
    with h5py.File(path, "w") as h5file:
        meta_grp = h5file.create_group("metadata")
        for key, value in (metadata or {}).items():
            meta_grp.attrs[key] = value
        for rank, regions in rank_regions.items():
            regions_group = h5file.create_group(f"rank{rank}").create_group("regions")
            for region_name, payload in regions.items():
                region_group = regions_group.create_group(region_name)
                starts, ends = payload
                region_group.create_dataset(
                    "start_times",
                    data=np.asarray(starts, dtype=np.int64),
                )
                region_group.create_dataset(
                    "end_times",
                    data=np.asarray(ends, dtype=np.int64),
                )


@pytest.fixture
def baseline_file(tmp_path):
    """setup: 1 s. solve: 2 calls, 2 s each (4 s). total_time: 6 s."""
    path = tmp_path / "baseline.h5"
    _write_sample_h5(
        path,
        {0: {"setup": ([0], [1 * NS]), "solve": ([1 * NS, 4 * NS], [3 * NS, 6 * NS])}},
        metadata={
            "start_time_ns": 0,
            "finalize_time_ns": 6 * NS,
            "user": "max",
            "hostname": "lrdn1234",
            "mpi_size": 1,
            "omp_num_threads": 4,
            "SLURM_JOB_ID": "1234567",
        },
    )
    return path


@pytest.fixture
def candidate_file(tmp_path):
    """setup: 1 s (unchanged). solve: 2 calls, 1 s each (2 s, -50%). total_time: 3 s."""
    path = tmp_path / "candidate.h5"
    _write_sample_h5(
        path,
        {
            0: {
                "setup": ([0], [1 * NS]),
                "solve": ([1 * NS, 2 * NS], [2 * NS, 3 * NS]),
            },
        },
        metadata={"start_time_ns": 0, "finalize_time_ns": 3 * NS},
    )
    return path


class TestInspectProfile:
    def test_returns_structured_headline_and_regions(self, baseline_file):
        payload = inspect_profile(str(baseline_file))

        assert payload["num_ranks"] == 1
        assert payload["num_regions"] == 2
        assert payload["total_time_seconds"] == pytest.approx(6.0)
        assert payload["regions"]["total_matching_filter"] == 2
        names = {region["name"] for region in payload["regions"]["items"]}
        assert names == {"setup", "solve"}
        # "total" sort (default): solve (4s) ranks above setup (1s).
        assert payload["regions"]["items"][0]["name"] == "solve"

    def test_metadata_is_grouped_and_json_safe(self, baseline_file):
        payload = inspect_profile(str(baseline_file))

        metadata = payload["metadata"]
        assert metadata["parallelism"] == {"mpi_size": 1, "omp_num_threads": 4}
        assert metadata["slurm"] == {"SLURM_JOB_ID": "1234567"}
        assert isinstance(
            metadata["parallelism"]["mpi_size"],
            int,
        )  # not a numpy scalar

    def test_top_n_limits_regions_but_reports_the_true_total(self, baseline_file):
        payload = inspect_profile(str(baseline_file), top_n=1)

        assert payload["regions"]["returned"] == 1
        assert payload["regions"]["total_matching_filter"] == 2

    def test_include_filters_regions(self, baseline_file):
        payload = inspect_profile(str(baseline_file), include=["solve"])

        assert payload["regions"]["total_matching_filter"] == 1
        assert payload["regions"]["items"][0]["name"] == "solve"

    def test_likwid_is_none_without_likwid_data(self, baseline_file):
        payload = inspect_profile(str(baseline_file))

        assert payload["likwid"] is None

    def test_missing_file_raises_tool_error(self, tmp_path):
        with pytest.raises(ToolError, match="not found"):
            inspect_profile(str(tmp_path / "does_not_exist.h5"))

    def test_invalid_sort_raises_tool_error(self, baseline_file):
        with pytest.raises(ToolError, match="sort must be one of"):
            inspect_profile(str(baseline_file), sort="bogus")

    def test_full_metadata_includes_environment_section(self, tmp_path):
        path = tmp_path / "with_env.h5"
        _write_sample_h5(
            path,
            {0: {"setup": ([0], [1 * NS])}},
            metadata={
                "start_time_ns": 0,
                "finalize_time_ns": 1 * NS,
                "MY_ENV_VAR": "x",
            },
        )

        collapsed = inspect_profile(str(path))
        assert "environment" not in collapsed["metadata"]
        assert collapsed["metadata"]["environment_field_count"] == 1

        full = inspect_profile(str(path), full_metadata=True)
        assert full["metadata"]["environment"] == {"MY_ENV_VAR": "x"}


class TestCompareProfiles:
    def test_overall_reports_speedup_and_faster(self, baseline_file, candidate_file):
        payload = compare_profiles(str(baseline_file), str(candidate_file))

        assert payload["overall"]["faster"] is True
        assert payload["overall"]["absolute_diff_seconds"] == pytest.approx(-3.0)
        assert payload["overall"]["relative_change_pct"] == pytest.approx(-50.0)
        assert payload["overall"]["speedup"] == pytest.approx(2.0)

    def test_per_region_deltas(self, baseline_file, candidate_file):
        payload = compare_profiles(str(baseline_file), str(candidate_file))

        by_name = {row["name"]: row for row in payload["regions"]["items"]}
        assert by_name["solve"]["baseline"] == pytest.approx(4.0)
        assert by_name["solve"]["candidate"] == pytest.approx(2.0)
        assert by_name["solve"]["delta"] == pytest.approx(-2.0)
        assert by_name["setup"]["delta"] == pytest.approx(0.0)

    def test_improvements_and_regressions_split_by_threshold(
        self,
        baseline_file,
        candidate_file,
    ):
        payload = compare_profiles(
            str(baseline_file),
            str(candidate_file),
            threshold_pct=5.0,
        )

        assert [row["name"] for row in payload["improvements"]] == ["solve"]
        assert payload["regressions"] == []

    def test_a_slower_candidate_is_reported_as_a_regression(
        self,
        tmp_path,
        baseline_file,
    ):
        slower = tmp_path / "slower.h5"
        _write_sample_h5(
            slower,
            {
                0: {
                    "setup": ([0], [1 * NS]),
                    "solve": ([1 * NS, 4 * NS], [3 * NS, 9 * NS]),
                },
            },
            metadata={"start_time_ns": 0, "finalize_time_ns": 9 * NS},
        )

        payload = compare_profiles(str(baseline_file), str(slower))

        assert payload["overall"]["faster"] is False
        assert payload["overall"]["absolute_diff_seconds"] == pytest.approx(3.0)
        assert [row["name"] for row in payload["regressions"]] == ["solve"]

    def test_invalid_metric_raises_tool_error(self, baseline_file, candidate_file):
        with pytest.raises(ToolError, match="metric must be one of"):
            compare_profiles(str(baseline_file), str(candidate_file), metric="bogus")

    def test_missing_baseline_raises_tool_error(self, tmp_path, candidate_file):
        with pytest.raises(ToolError):
            compare_profiles(str(tmp_path / "missing.h5"), str(candidate_file))


class TestRunProfile:
    def test_runs_a_script_and_returns_a_structured_summary(self, tmp_path):
        script = tmp_path / "bench.py"
        script.write_text(
            "import time\n"
            "from scope_profiler import ProfileManager\n"
            "with ProfileManager.profile_region('work'):\n"
            "    time.sleep(0.001)\n",
        )

        payload = run_profile(str(script), output_path=str(tmp_path / "out.h5"))

        assert payload["num_ranks"] == 1
        names = {region["name"] for region in payload["regions"]["items"]}
        assert "work" in names
        assert (tmp_path / "out.h5").exists()

    def test_missing_script_raises_tool_error(self, tmp_path):
        with pytest.raises(ToolError, match="No such script"):
            run_profile(str(tmp_path / "does_not_exist.py"))

    def test_a_script_that_raises_is_reported_as_a_tool_error(self, tmp_path):
        script = tmp_path / "broken.py"
        script.write_text("raise RuntimeError('boom')\n")

        with pytest.raises(ToolError, match="failed"):
            run_profile(str(script))

    def test_timeout_kills_a_runaway_script(self, tmp_path):
        script = tmp_path / "slow.py"
        script.write_text("import time\ntime.sleep(5)\n")

        with pytest.raises(ToolError, match="exceeded"):
            run_profile(str(script), timeout_seconds=0.2)

    def test_script_args_are_passed_through_without_a_shell(self, tmp_path):
        script = tmp_path / "echo_args.py"
        script.write_text(
            "import sys\n"
            "from scope_profiler import ProfileManager\n"
            "with ProfileManager.profile_region('args'):\n"
            "    pass\n"
            "assert sys.argv[1:] == ['hello world', '$(rm -rf /)']\n",
        )

        # A shell metacharacter-laden argument must reach the script literally,
        # never be interpreted -- proving no shell is involved.
        payload = run_profile(str(script), script_args=["hello world", "$(rm -rf /)"])
        assert payload["num_ranks"] == 1


class TestPlotProfile:
    def test_renders_a_gantt_chart_and_returns_its_path(self, baseline_file, tmp_path):
        payload = plot_profile(
            str(baseline_file),
            plot_type="gantt",
            output_dir=str(tmp_path / "figs"),
        )

        assert payload["plot_type"] == "gantt"
        assert len(payload["paths"]) == 1
        plot_path = payload["paths"][0]
        assert plot_path.endswith(".png")
        import os

        assert os.path.getsize(plot_path) > 0

    def test_unknown_plot_type_raises_tool_error(self, baseline_file):
        with pytest.raises(ToolError, match="plot_type must be one of"):
            plot_profile(str(baseline_file), plot_type="bogus")

    def test_speedup_requires_at_least_two_files(self, baseline_file):
        with pytest.raises(ToolError, match="at least 2"):
            plot_profile(str(baseline_file), plot_type="speedup")

    def test_accepts_a_single_path_as_a_string(self, baseline_file, tmp_path):
        payload = plot_profile(
            str(baseline_file),
            plot_type="durations",
            output_dir=str(tmp_path / "figs"),
        )
        assert payload["paths"]
