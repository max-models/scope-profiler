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
    format_overall_change,
    overall_change,
)
from scope_profiler.diff import main as diff_main

NS = 1_000_000_000


def _write_sample_h5(path, rank_regions, label=None, total=None):
    with h5py.File(path, "w") as h5file:
        meta_grp = h5file.create_group("metadata")
        if label is not None:
            meta_grp.attrs["label"] = label
        if total is not None:
            meta_grp.attrs["start_time_ns"] = 0
            meta_grp.attrs["finalize_time_ns"] = int(total * NS)
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
            },
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
    # |delta|: teardown 2.0, solve 2.0, setup 0.0 - ties keep alpha order.
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


def test_diff_files_uses_summary_reader_for_scalar_metrics(
    tmp_path,
    monkeypatch,
    capsys,
):
    from scope_profiler.h5writer import ProfilingWriter
    from scope_profiler.profile_manager import RankPayload

    paths = []
    for name, duration in (("a", NS), ("b", 2 * NS)):
        path = tmp_path / f"{name}.h5"
        rank_payload = RankPayload(
            regions={
                "solve": (
                    np.asarray([0], dtype=np.int64),
                    np.asarray([duration], dtype=np.int64),
                ),
            },
            likwid={},
            likwid_environment={},
            exclusive_totals={"solve": duration},
        )
        with ProfilingWriter(path) as writer:
            writer.write_rank(0, rank_payload)
        paths.append(path)

    def reject_eager_read(*args, **kwargs):
        raise AssertionError("scalar diff used the eager reader")

    monkeypatch.setattr("scope_profiler.diff.read_profile", reject_eager_read)
    diff_files(*paths, metric="total")
    assert "+100%" in capsys.readouterr().out


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
    lines = [line for line in out.splitlines() if "solve" in line]
    assert lines  # solve row present with the calls metric


def test_cli_threshold_flag(file_a, file_b, capsys):
    diff_main([str(file_a), str(file_b), "--threshold", "10"])
    out = capsys.readouterr().out
    assert "setup" not in out.split("Regions")[1]


def test_cli_top_flag_keeps_the_largest_changes(file_a, file_b, capsys):
    diff_main([str(file_a), str(file_b), "--top", "1"])
    out = capsys.readouterr().out
    assert "Regions (1 of 3)" in out
    table = out.split("Regions (1 of 3)")[1]
    assert "solve" in table
    assert "setup" not in table


def test_cli_top_rejects_zero(file_a, file_b, capsys):
    with pytest.raises(SystemExit):
        diff_main([str(file_a), str(file_b), "--top", "0"])
    assert "must be at least 1" in capsys.readouterr().err


def test_check_top_shortens_the_table_but_checks_every_region(file_a, file_b, capsys):
    assert check_main([str(file_a), str(file_b), "--fail-on-new", "--top", "1"]) == 1
    out = capsys.readouterr().out
    table, violations = out.split("Budget violations:")
    assert "solve" in table
    assert "teardown" not in table.split("Only in b")[0]
    assert violations.strip() == "solve, teardown"


def _timed_file(path, total):
    _write_sample_h5(path, {0: {"solve": ([0], [int(total * NS)])}}, total=total)
    return path


def test_overall_change_of_a_faster_run(tmp_path):
    from scope_profiler.h5reader import read_h5

    a = read_h5(_timed_file(tmp_path / "a.h5", 10))
    b = read_h5(_timed_file(tmp_path / "b.h5", 8))
    change = overall_change(a, b)
    assert change["faster"] is True
    assert change["speedup"] == pytest.approx(1.25)
    assert change["relative_change_pct"] == pytest.approx(-20)
    assert change["absolute_diff_seconds"] == pytest.approx(-2)
    assert format_overall_change(change) == (
        "Overall: b is 1.25x faster than a (-20%, -2 s)"
    )
    assert format_overall_change(overall_change(b, a)) == (
        "Overall: b is 1.25x slower than a (+25%, +2 s)"
    )
    assert format_overall_change(overall_change(a, a)) == "Overall: same total time"


def test_overall_change_without_total_times(file_a, file_b):
    from scope_profiler.h5reader import read_h5

    change = overall_change(read_h5(file_a), read_h5(file_b))
    assert set(change.values()) == {None}
    assert "no total run time" in format_overall_change(change)


def test_overall_change_with_a_zero_time():
    change = {
        "absolute_diff_seconds": 1.0,
        "relative_change_pct": None,
        "speedup": 0.0,
        "faster": False,
    }
    assert format_overall_change(change) == "Overall: b is slower than a (+1 s)"


def test_diff_and_check_print_the_overall_line(tmp_path, capsys):
    a = _timed_file(tmp_path / "a.h5", 10)
    b = _timed_file(tmp_path / "b.h5", 8)
    diff_main([str(a), str(b)])
    assert "Overall: b is 1.25x faster than a" in capsys.readouterr().out
    check_main([str(a), str(b)])
    assert "Overall: b is 1.25x faster than a" in capsys.readouterr().out


def test_diff_json_is_the_compare_profiles_payload(tmp_path, capsys):
    import json

    a = _timed_file(tmp_path / "a.h5", 10)
    b = _timed_file(tmp_path / "b.h5", 8)
    assert diff_main([str(a), str(b), "--json", "--top", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"]["speedup"] == pytest.approx(1.25)
    assert payload["regions"]["returned"] == 1
    assert [row["name"] for row in payload["improvements"]] == ["solve"]


def test_check_json_adds_the_verdict(file_a, file_b, capsys):
    import json

    assert check_main([str(file_a), str(file_b), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["check"] == {
        "passed": False,
        "max_regression_pct": 0.0,
        "fail_on_new": False,
        "violations": ["solve"],
    }
    args = [
        str(file_a),
        str(file_b),
        "--json",
        "--metric",
        "p95",
        "--max-regression",
        "60",
    ]
    assert check_main(args) == 0
    assert json.loads(capsys.readouterr().out)["check"]["passed"] is True


def _runs(tmp_path, prefix, totals):
    return [_timed_file(tmp_path / f"{prefix}{i}.h5", t) for i, t in enumerate(totals)]


def test_repeated_runs_give_means_and_a_noise_band(tmp_path):
    from scope_profiler.h5reader import read_h5

    base = [read_h5(p) for p in _runs(tmp_path, "a", [10.0, 10.2, 9.8])]
    same = [read_h5(p) for p in _runs(tmp_path, "s", [10.1, 9.9, 10.3])]
    fast = [read_h5(p) for p in _runs(tmp_path, "f", [8.0, 8.1, 7.9])]

    (row,) = diff_rows(base, same)
    assert row["a"] == pytest.approx(10.0)
    assert row["b"] == pytest.approx(10.1)
    assert row["noise"] == pytest.approx(2 * np.hypot(0.2, 0.2) / np.sqrt(3))
    assert row["significant"] is False

    (row,) = diff_rows(base, fast)
    assert row["delta"] == pytest.approx(-2.0)
    assert row["significant"] is True

    (row,) = diff_rows(base, fast[0])
    assert row["noise"] is None
    assert row["significant"] is None

    assert overall_change(base, same)["significant"] is False
    assert overall_change(base, fast)["significant"] is True
    assert "beyond the noise of" in format_overall_change(overall_change(base, fast))
    assert "within the noise of" in format_overall_change(overall_change(base, same))


def test_diff_cli_takes_globs_of_repeated_runs(tmp_path, capsys):
    _runs(tmp_path, "a", [10.0, 10.2, 9.8])
    _runs(tmp_path, "s", [10.1, 9.9, 10.3])
    diff_main([str(tmp_path / "a*.h5"), str(tmp_path / "s*.h5")])
    out = capsys.readouterr().out
    assert "and 2 more run(s)" in out
    assert "within the noise" in out
    assert "noise" in out.split("Regions")[1]
    assert "% ~" in out


def test_check_ignores_regressions_within_the_noise(tmp_path, capsys):
    import json

    _runs(tmp_path, "a", [10.0, 10.2, 9.8])
    _runs(tmp_path, "s", [10.1, 9.9, 10.3])
    _runs(tmp_path, "w", [12.0, 12.1, 11.9])
    base, same, slow = (str(tmp_path / f"{p}*.h5") for p in "asw")
    assert check_main([base, same]) == 0
    assert check_main([base, slow]) == 1
    assert check_main([base, same, "--json"]) == 0
    payload = capsys.readouterr().out.split("\n{", 1)[1]
    payload = json.loads("{" + payload)
    assert payload["baseline"]["runs"] == 3
    assert payload["regions"]["items"][0]["significant"] is False
    assert payload["regressions"] == []


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
