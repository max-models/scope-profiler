import marshal
import pstats

import pytest

from scope_profiler import read_h5
from scope_profiler.post_processing import export_main
from scope_profiler.prof_export import build_pstats_dict, export_prof
from scope_profiler.tests.test_post_processing import _write_sample_h5

MS = 1_000_000  # nanoseconds per millisecond, the unit stored in the HDF5 files


def _nested_file_data():
    """One rank whose regions nest: main > (setup, solve > assemble)."""
    return {
        0: {
            "main": ([0], [100 * MS]),
            "setup": ([0], [20 * MS]),
            "solve": ([20 * MS], [90 * MS]),
            "assemble": ([30 * MS], [60 * MS]),
        }
    }


def _calls(*specs):
    """Build the CallArrays ``build_pstats_dict`` expects, from seconds.

    The ``parent`` column of each spec is not passed through -- nesting is
    reconstructed from the intervals, which is the only way the exporter can
    ever receive it. It is kept in the specs as documentation of the shape
    each test is describing.
    """
    from collections import defaultdict

    import numpy as np

    from scope_profiler.call_stack import build_call_arrays
    from scope_profiler.mpi_region import MPIRegion
    from scope_profiler.region import Region

    intervals = defaultdict(list)
    for name, start, end, _parent in specs:
        intervals[name].append((round(start * 1e9), round(end * 1e9)))
    regions = [
        MPIRegion(
            name=name,
            regions={
                0: Region(
                    np.array([s for s, _ in calls], dtype=np.int64),
                    np.array([e for _, e in calls], dtype=np.int64),
                )
            },
        )
        for name, calls in intervals.items()
    ]
    return build_call_arrays(regions, rank=0)


def _stats_of(path):
    with open(path, "rb") as f:
        return marshal.load(f)


def test_build_pstats_dict_nesting_and_self_time():
    calls = _calls(
        ("main", 0.0, 1.0, None),
        ("child", 0.1, 0.4, 0),
        ("child", 0.5, 0.7, 0),
    )

    stats = build_pstats_dict(calls)

    main = stats[("~", 0, "main")]
    child = stats[("~", 0, "child")]

    assert main[:2] == (1, 1)
    assert main[2] == pytest.approx(0.5)  # 1.0 total minus 0.3 + 0.2 of children
    assert main[3] == pytest.approx(1.0)
    assert main[4] == {}

    assert child[:2] == (2, 2)
    assert child[2] == pytest.approx(0.5)
    assert child[3] == pytest.approx(0.5)
    assert child[4] == {
        ("~", 0, "main"): (2, 2, pytest.approx(0.5), pytest.approx(0.5))
    }


def test_build_pstats_dict_counts_recursion_like_cprofile():
    calls = _calls(
        ("recurse", 0.0, 1.0, None),
        ("recurse", 0.2, 0.6, 0),
    )

    entry = build_pstats_dict(calls)[("~", 0, "recurse")]

    # One primitive call, two total, and the shared seconds counted once.
    assert entry[0] == 1
    assert entry[1] == 2
    assert entry[2] == pytest.approx(1.0)
    assert entry[3] == pytest.approx(1.0)


def test_build_pstats_dict_rejects_partial_overlap():
    """Self time can no longer go negative, because the input cannot exist.

    'long' would be reconstructed as a child of 'short' because it starts
    inside it, even though it runs past its end - which used to be clamped
    to keep pstats' tt from going negative. build_call_arrays refuses the
    intervals outright instead.
    """
    from scope_profiler.call_stack import NestingError

    with pytest.raises(NestingError):
        _calls(("short", 0.0, 0.5, None), ("long", 0.1, 2.0, 0))


def test_build_pstats_dict_synthetic_root():
    calls = _calls(("a", 0.0, 1.0, None), ("b", 2.0, 2.5, None))

    stats = build_pstats_dict(calls, root_name="<run>")

    root_key = ("~", 0, "<run>")
    assert stats[root_key][:4] == (1, 1, 0.0, pytest.approx(1.5))
    for name in ("a", "b"):
        assert root_key in stats[("~", 0, name)][4]


def test_export_prof_readable_by_pstats(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())

    written = export_prof(read_h5(h5_file), tmp_path / "profile.prof", verbose=False)

    assert written == [tmp_path / "profile_rank0.prof"]

    stats = pstats.Stats(str(written[0]))
    names = {key[2] for key in stats.stats}
    assert {"main", "setup", "solve", "assemble"} <= names

    # main: 0.1s wall, minus setup (0.02) and solve (0.07); solve minus assemble.
    assert stats.stats[("~", 0, "main")][2] == pytest.approx(0.01)
    assert stats.stats[("~", 0, "solve")][2] == pytest.approx(0.04)
    assert stats.stats[("~", 0, "assemble")][3] == pytest.approx(0.03)
    assert stats.stats[("~", 0, "assemble")][4].keys() == {("~", 0, "solve")}
    assert stats.total_calls == 5  # four regions plus the synthetic root


def test_export_prof_per_rank_and_per_file(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    _write_sample_h5(file_one, {rank: _nested_file_data()[0] for rank in (0, 1)})
    _write_sample_h5(file_two, {rank: _nested_file_data()[0] for rank in (0, 1)})

    runs = [read_h5(file_one), read_h5(file_two)]
    written = export_prof(runs, tmp_path / "profile.prof", ranks=[0, 1], verbose=False)

    assert [path.name for path in written] == [
        "profile_run_one_rank0.prof",
        "profile_run_one_rank1.prof",
        "profile_run_two_rank0.prof",
        "profile_run_two_rank1.prof",
    ]
    for path in written:
        assert pstats.Stats(str(path)).total_calls == 5


def test_export_prof_rejects_unknown_rank(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())

    with pytest.raises(ValueError, match="Invalid rank"):
        export_prof(
            read_h5(h5_file),
            tmp_path / "profile.prof",
            ranks=[3],
            verbose=False,
        )


def test_cli_export_prof_without_plots(tmp_path, capsys):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())
    out_dir = tmp_path / "figures"

    export_main(
        [
            "prof",
            str(h5_file),
            "-o",
            str(out_dir),
        ]
    )

    prof_file = out_dir / "profile_rank0.prof"
    assert prof_file.exists()
    assert not list(out_dir.glob("*.png"))
    assert "snakeviz" in capsys.readouterr().out

    stats = _stats_of(prof_file)
    assert {key[2] for key in stats} >= {"main", "setup", "solve", "assemble"}


def test_cli_export_prof_with_rank_selection(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())
    out_dir = tmp_path / "figures"

    export_main(["prof", str(h5_file), "-o", str(out_dir), "--ranks", "0"])

    assert (out_dir / "profile_rank0.prof").exists()


def test_cli_export_prof_requires_output(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())

    with pytest.raises(SystemExit):
        export_main(["prof", str(h5_file)])


def test_exported_prof_loads_in_snakeviz(tmp_path):
    snakeviz_stats = pytest.importorskip("snakeviz.stats")

    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())
    written = export_prof(read_h5(h5_file), tmp_path / "profile.prof", verbose=False)

    stats = pstats.Stats(str(written[0]))
    assert len(snakeviz_stats.table_rows(stats)) == 5

    # snakeviz drops entries that neither call nor are called; every region has
    # to survive that, otherwise it cannot be drawn.
    tree = snakeviz_stats.json_stats(stats)
    assert set(tree) == {
        "~:0(<profiling_data rank 0>)",
        "~:0(main)",
        "~:0(setup)",
        "~:0(solve)",
        "~:0(assemble)",
    }
    assert set(tree["~:0(solve)"]["children"]) == {"~:0(assemble)"}
