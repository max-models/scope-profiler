"""Tests for ``scope-profiler diff``."""

import h5py
import numpy as np
import pytest

from scope_profiler.__main__ import main as cli_main
from scope_profiler.diff import (
    check_files,
    check_main,
    check_rows,
    diff_files,
    diff_rows,
)
from scope_profiler.diff import main as diff_main

NS = 1_000_000_000


def _write_sample_h5(path, rank_regions, label=None):
    with h5py.File(path, "w") as h5file:
        meta_grp = h5file.create_group("metadata")
        if label is not None:
            meta_grp.attrs["label"] = label
        for rank, regions in rank_regions.items():
            regions_group = h5file.create_group(f"rank{rank}").create_group("regions")
            for region_name, payload in regions.items():
                region_group = regions_group.create_group(region_name)
                starts, ends = payload
                region_group.create_dataset(
                    "start_times", data=np.asarray(starts, dtype=np.int64)
                )
                region_group.create_dataset(
                    "end_times", data=np.asarray(ends, dtype=np.int64)
                )


@pytest.fixture
def file_a(tmp_path):
    """setup: 1 call, 1 s. solve: 2 calls, 2 s each (4 s total)."""
    path = tmp_path / "a.h5"
    _write_sample_h5(
        path,
        {0: {"setup": ([0], [1 * NS]), "solve": ([1 * NS, 4 * NS], [3 * NS, 6 * NS])}},
        label="baseline",
    )
    return path


@pytest.fixture
def file_b(tmp_path):
    """setup: 1 call, 1 s (unchanged). solve: 2 calls, 3 s each (6 s total, +50%).

    Also adds a new "teardown" region absent from ``file_a``.
    """
    path = tmp_path / "b.h5"
    _write_sample_h5(
        path,
        {
            0: {
                "setup": ([0], [1 * NS]),
                "solve": ([1 * NS, 5 * NS], [4 * NS, 8 * NS]),
                "teardown": ([0], [2 * NS]),
            }
        },
        label="candidate",
    )
    return path


def test_diff_rows_computes_delta_and_pct(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    rows = diff_rows(read_h5(file_a), read_h5(file_b))
    by_name = {row["name"]: row for row in rows}

    assert by_name["setup"]["a"] == pytest.approx(1.0)
    assert by_name["setup"]["b"] == pytest.approx(1.0)
    assert by_name["setup"]["delta"] == pytest.approx(0.0)
    assert by_name["setup"]["pct"] == pytest.approx(0.0)

    assert by_name["solve"]["a"] == pytest.approx(4.0)
    assert by_name["solve"]["b"] == pytest.approx(6.0)
    assert by_name["solve"]["delta"] == pytest.approx(2.0)
    assert by_name["solve"]["pct"] == pytest.approx(50.0)

    # Region only present in b: no baseline, so no percent change.
    assert by_name["teardown"]["a"] is None
    assert by_name["teardown"]["b"] == pytest.approx(2.0)
    assert by_name["teardown"]["delta"] == pytest.approx(2.0)
    assert by_name["teardown"]["pct"] is None


def test_diff_rows_region_only_in_a(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    # Swap a and b: now "teardown" only exists in a.
    rows = diff_rows(read_h5(file_b), read_h5(file_a))
    by_name = {row["name"]: row for row in rows}

    assert by_name["teardown"]["a"] == pytest.approx(2.0)
    assert by_name["teardown"]["b"] is None
    assert by_name["teardown"]["delta"] == pytest.approx(-2.0)
    # a is a valid nonzero baseline, so a region dropping out is a well
    # defined -100% change.
    assert by_name["teardown"]["pct"] == pytest.approx(-100.0)


def test_diff_rows_metric_calls(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    rows = diff_rows(read_h5(file_a), read_h5(file_b), metric="calls")
    by_name = {row["name"]: row for row in rows}

    assert by_name["solve"]["a"] == 2
    assert by_name["solve"]["b"] == 2
    assert by_name["solve"]["delta"] == 0


def test_diff_rows_percentile_metric(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    rows = diff_rows(read_h5(file_a), read_h5(file_b), metric="p95")
    assert {row["name"] for row in rows} == {"setup", "solve", "teardown"}


def test_diff_rows_sort_by_delta_magnitude(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    rows = diff_rows(read_h5(file_a), read_h5(file_b), sort="delta")
    # |delta|: teardown 2.0, solve 2.0, setup 0.0 -- ties keep alpha order.
    assert [row["name"] for row in rows] == ["solve", "teardown", "setup"]


def test_diff_rows_sort_by_name(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    rows = diff_rows(read_h5(file_a), read_h5(file_b), sort="name")
    assert [row["name"] for row in rows] == ["setup", "solve", "teardown"]


def test_diff_rows_include_exclude(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    rows = diff_rows(read_h5(file_a), read_h5(file_b), include="solve")
    assert [row["name"] for row in rows] == ["solve"]

    rows = diff_rows(read_h5(file_a), read_h5(file_b), exclude="solve")
    assert {row["name"] for row in rows} == {"setup", "teardown"}


def test_diff_rows_threshold_filters_small_changes(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    rows = diff_rows(read_h5(file_a), read_h5(file_b), threshold=10.0)
    names = {row["name"] for row in rows}
    # setup is unchanged (0%) and gets filtered; solve (+50%) and the new
    # teardown region (no baseline, always kept) survive.
    assert names == {"solve", "teardown"}


def test_diff_rows_rejects_bad_metric_or_sort(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    results_a, results_b = read_h5(file_a), read_h5(file_b)
    with pytest.raises(ValueError):
        diff_rows(results_a, results_b, metric="bogus")
    with pytest.raises(ValueError):
        diff_rows(results_a, results_b, sort="bogus")


def test_diff_files_prints_table(file_a, file_b, capsys):
    diff_files(file_a, file_b)
    out = capsys.readouterr().out

    assert "a: baseline" in out
    assert "b: candidate" in out
    assert "Regions (3)" in out
    assert "solve" in out and "+50%" in out
    assert "Only in b: teardown" in out


def test_diff_files_only_in_a_note(file_a, file_b, capsys):
    diff_files(file_b, file_a)
    out = capsys.readouterr().out
    assert "Only in a: teardown" in out


def test_diff_files_no_regions_match(file_a, file_b, capsys):
    diff_files(file_a, file_b, include="nonexistent")
    out = capsys.readouterr().out
    assert "(no regions to compare)" in out


def test_cli_entry_point(file_a, file_b, capsys):
    diff_main([str(file_a), str(file_b)])
    out = capsys.readouterr().out
    assert "Regions (3)" in out


def test_cli_metric_and_sort_flags(file_a, file_b, capsys):
    diff_main([str(file_a), str(file_b), "--metric", "calls", "--sort", "name"])
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip().startswith("solve")]
    assert lines  # solve row present with the calls metric


def test_cli_threshold_flag(file_a, file_b, capsys):
    diff_main([str(file_a), str(file_b), "--threshold", "10"])
    out = capsys.readouterr().out
    assert "setup" not in out.split("Regions")[1]


def test_check_passes_within_budget_and_fails_over_budget(file_a, file_b, capsys):
    from scope_profiler.h5reader import read_h5

    assert (
        check_rows(
            read_h5(file_a),
            read_h5(file_b),
            max_regression=60,
        )
        == []
    )
    failures = check_rows(
        read_h5(file_a),
        read_h5(file_b),
        max_regression=10,
    )
    assert [row["name"] for row in failures] == ["solve"]
    assert check_files(file_a, file_b, max_regression=60) == 0
    assert check_files(file_a, file_b, max_regression=10) == 1
    assert "FAIL" in capsys.readouterr().out


def test_check_can_fail_on_new_regions(file_a, file_b):
    assert check_main([str(file_a), str(file_b), "--fail-on-new"]) == 1


def test_dispatch_from_main_cli(file_a, file_b, capsys):
    cli_main(["diff", str(file_a), str(file_b)])
    out = capsys.readouterr().out
    assert "Regions (3)" in out


def test_diff_listed_in_top_level_help(capsys):
    with pytest.raises(SystemExit):
        cli_main(["--help"])
    assert "diff" in capsys.readouterr().out
