"""Tests for ``scope-profiler pproc --summary`` and its LIKWID counter table."""

import h5py
import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.likwid_data import LikwidRegionResult, write_likwid_results
from scope_profiler.post_processing import main as pproc_main
from scope_profiler.summary import likwid_tables, print_likwid_table


def _result(tag, group="CLOCK", nthreads=1, cpus=None, base=1.0):
    """A LIKWID result with predictable, distinguishable numbers."""
    return LikwidRegionResult(
        tag=tag,
        group_id=0,
        group_name=group,
        cpus=list(range(nthreads)) if cpus is None else cpus,
        times=np.full(nthreads, 0.5 * base),
        call_counts=np.full(nthreads, 3, dtype=np.int64),
        event_names=["INSTR_RETIRED_ANY", "CAS_COUNT_RD", "CAS_COUNT_RD"],
        counter_names=["FIXC0", "MBOX0C0", "MBOX1C0"],
        events=np.array([[100.0], [10.0], [20.0]]).repeat(nthreads, axis=1) * base,
        metric_names=["Clock [MHz]", "CPI"],
        metrics=np.array([[2400.0], [0.5]]).repeat(nthreads, axis=1) * base,
        source="full_api",
    )


def _write(path, results_by_rank, with_regions=True):
    """Write a merged-looking file holding LIKWID results (and timings)."""
    with h5py.File(path, "w") as f:
        for rank, results in results_by_rank.items():
            grp = f.create_group(f"rank{rank}")
            if with_regions:
                for result in results:
                    rgrp = grp.create_group(f"regions/{result.tag}")
                    rgrp.create_dataset(
                        "start_times", data=np.array([0], dtype=np.int64)
                    )
                    rgrp.create_dataset(
                        "end_times", data=np.array([10**9], dtype=np.int64)
                    )
            write_likwid_results(grp, results)
    return path


def test_tables_put_regions_in_columns_and_counters_in_rows(tmp_path):
    """A run has few regions and many counters, so counters are the rows."""
    path = _write(tmp_path / "d.h5", {0: [_result("solve"), _result("io", base=2.0)]})

    (table,) = likwid_tables(read_h5(path))
    assert table["rank"] == 0
    assert table["group"] == "CLOCK"
    # io has twice solve's runtime, so it leads: costliest region first, as in
    # the region table (HDF5 would have handed them back alphabetically).
    assert table["columns"] == ["io", "solve"]

    sections = dict(table["sections"])
    # Repeated events stay distinguishable by their counter register.
    event_names = [name for name, _ in sections["Events"]]
    assert event_names == [
        "INSTR_RETIRED_ANY",
        "CAS_COUNT_RD:MBOX0C0",
        "CAS_COUNT_RD:MBOX1C0",
    ]
    # Values line up with the column order: io (base=2) first, solve second.
    assert dict(sections["Events"])["INSTR_RETIRED_ANY"] == [200.0, 100.0]
    assert dict(sections["Metrics"])["CPI"] == [1.0, 0.5]
    assert dict(sections[""])["call count"] == [3, 3]


def test_one_table_per_event_group(tmp_path):
    """Columns are only comparable within a group, so groups get own tables."""
    path = _write(
        tmp_path / "d.h5",
        {0: [_result("solve", group="CLOCK"), _result("io", group="MEM_DP")]},
    )

    tables = likwid_tables(read_h5(path))
    assert [(t["group"], t["columns"]) for t in tables] == [
        ("CLOCK", ["solve"]),
        ("MEM_DP", ["io"]),
    ]


def test_one_table_per_rank(tmp_path):
    """Each rank measured its own counters and gets its own table."""
    path = _write(tmp_path / "d.h5", {0: [_result("solve")], 1: [_result("solve")]})

    tables = likwid_tables(read_h5(path))
    assert [t["rank"] for t in tables] == [0, 1]


def test_multithreaded_region_labels_columns_by_cpu(tmp_path):
    """Per-thread values are shown as such, not folded into a fake aggregate."""
    path = _write(tmp_path / "d.h5", {0: [_result("solve", nthreads=2, cpus=[4, 5])]})

    (table,) = likwid_tables(read_h5(path))
    assert table["columns"] == ["solve@cpu4", "solve@cpu5"]


def test_filters_apply_to_the_counter_table(tmp_path):
    """--include/--exclude/--ranks select LIKWID regions as they do timings."""
    path = _write(
        tmp_path / "d.h5",
        {0: [_result("solve"), _result("io")], 1: [_result("solve")]},
    )
    results = read_h5(path)

    (table,) = likwid_tables(results, include=["solve"], ranks=[0])
    assert table["columns"] == ["solve"]

    (table,) = likwid_tables(results, exclude=["io"], ranks=[0])
    assert table["columns"] == ["solve"]


def test_no_tables_without_likwid_data(tmp_path):
    """Files recorded without LIKWID produce no counter tables at all."""
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("rank0/regions/solve")
        grp.create_dataset("start_times", data=np.array([0], dtype=np.int64))
        grp.create_dataset("end_times", data=np.array([10**9], dtype=np.int64))

    assert likwid_tables(read_h5(path)) == []


def test_counters_render_as_integers_not_exponents(tmp_path, capsys):
    """Raw event counts are large integers and must stay readable."""
    result = _result("solve")
    result.events = np.array([[69617470.0], [10.0], [20.0]])
    path = _write(tmp_path / "d.h5", {0: [result]})

    (table,) = likwid_tables(read_h5(path))
    print_likwid_table(table)
    out = capsys.readouterr().out
    assert "69617470" in out
    assert "6.96175e+07" not in out
    # Derived metrics keep significant digits rather than being truncated.
    assert "2400" in out


def test_pproc_summary_prints_both_tables(tmp_path, capsys):
    """--summary prints the region table and the LIKWID table."""
    path = _write(tmp_path / "d.h5", {0: [_result("solve")]})

    pproc_main([str(path), "--summary"])
    out = capsys.readouterr().out

    assert "region" in out and "solve" in out
    assert "LIKWID counters (rank 0, group CLOCK)" in out
    assert "CAS_COUNT_RD:MBOX0C0" in out
    assert "Clock [MHz]" in out


def test_pproc_summary_without_likwid_prints_only_regions(tmp_path, capsys):
    """No LIKWID data means no empty counter section."""
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("rank0/regions/solve")
        grp.create_dataset("start_times", data=np.array([0], dtype=np.int64))
        grp.create_dataset("end_times", data=np.array([10**9], dtype=np.int64))

    pproc_main([str(path), "--summary"])
    out = capsys.readouterr().out

    assert "solve" in out
    assert "LIKWID" not in out


def test_pproc_summary_alone_renders_no_plots(tmp_path, capsys):
    """A bare --summary is a text job; it must not write plot files."""
    path = _write(tmp_path / "d.h5", {0: [_result("solve")]})
    output = tmp_path / "figures"

    pproc_main([str(path), "--summary"])
    capsys.readouterr()
    assert not output.exists()


def test_pproc_summary_covers_every_file(tmp_path, capsys):
    """Each input file gets its own pair of tables."""
    first = _write(tmp_path / "a.h5", {0: [_result("solve")]})
    second = _write(tmp_path / "b.h5", {0: [_result("solve")]})

    pproc_main([str(first), str(second), "--summary"])
    out = capsys.readouterr().out

    assert out.count("LIKWID counters") == 2
    assert "a.h5" in out and "b.h5" in out


def test_pproc_summary_sort_orders_the_region_table(tmp_path, capsys):
    """--summary-sort reorders regions without touching the counter table."""
    path = _write(tmp_path / "d.h5", {0: [_result("zebra"), _result("alpha")]})

    pproc_main([str(path), "--summary", "--summary-sort", "name"])
    out = capsys.readouterr().out

    region_block = out.split("LIKWID counters")[0]
    assert region_block.index("alpha") < region_block.index("zebra")


def test_pproc_summary_rejects_an_unknown_sort_key(tmp_path):
    """Sort keys are validated by argparse rather than failing later."""
    path = _write(tmp_path / "d.h5", {0: [_result("solve")]})
    with pytest.raises(SystemExit):
        pproc_main([str(path), "--summary", "--summary-sort", "nonsense"])
