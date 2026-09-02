import marshal
import pstats

import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.call_stack import build_call_arrays
from scope_profiler.post_processing import export_main
from scope_profiler.prof_export import (
    build_pstats_dict,
    export_prof,
    load_prof,
    to_pstats,
)
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

    stats = build_pstats_dict(calls, call_paths=False)

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

    entry = build_pstats_dict(calls, call_paths=False)[("~", 0, "recurse")]

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


def test_build_pstats_dict_call_paths_keep_contexts_separate():
    """The same region below two parents becomes two SnakeViz tree nodes."""
    calls = _calls(
        ("phase_a", 0.0, 1.0, None),
        ("work", 0.1, 0.4, 0),
        ("phase_b", 2.0, 3.0, None),
        ("work", 2.1, 2.7, 2),
    )

    stats = build_pstats_dict(calls, root_name="<run>")
    phase_a = ("~", 0, "phase_a")
    phase_b = ("~", 0, "phase_b")
    work_a = ("~", 0, "phase_a > work")
    work_b = ("~", 0, "phase_b > work")

    assert {key[2] for key in stats} == {
        "<run>",
        "phase_a",
        "phase_a > work",
        "phase_b",
        "phase_b > work",
    }
    assert stats[work_a][:4] == (1, 1, pytest.approx(0.3), pytest.approx(0.3))
    assert stats[work_b][:4] == (1, 1, pytest.approx(0.6), pytest.approx(0.6))
    assert stats[work_a][4] == {phase_a: stats[work_a][:4]}
    assert stats[work_b][4] == {phase_b: stats[work_b][:4]}


def test_build_pstats_dict_uses_captured_source_location():
    from scope_profiler.mpi_region import MPIRegion
    from scope_profiler.region import Region

    calls = build_call_arrays(
        [
            MPIRegion(
                "solve",
                {
                    0: Region(
                        np.array([0], dtype=np.int64),
                        np.array([1_000_000_000], dtype=np.int64),
                        source_file="solver.py",
                        source_lineno=42,
                    )
                },
            )
        ],
        rank=0,
    )

    stats = build_pstats_dict(calls)
    assert ("solver.py", 42, "solve") in stats


def test_export_prof_readable_by_pstats(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())

    written = export_prof(read_h5(h5_file), tmp_path / "profile.prof", verbose=False)

    assert written == [tmp_path / "profile_rank0.prof"]

    stats = pstats.Stats(str(written[0]))
    names = {key[2] for key in stats.stats}
    assert {
        "main",
        "main > setup",
        "main > solve",
        "main > solve > assemble",
    } <= names

    # main: 0.1s wall, minus setup (0.02) and solve (0.07); solve minus assemble.
    assert stats.stats[("~", 0, "main")][2] == pytest.approx(0.01)
    assert stats.stats[("~", 0, "main > solve")][2] == pytest.approx(0.04)
    assert stats.stats[("~", 0, "main > solve > assemble")][3] == pytest.approx(0.03)
    assert stats.stats[("~", 0, "main > solve > assemble")][4].keys() == {
        ("~", 0, "main > solve")
    }
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
    assert {key[2] for key in stats} >= {
        "main",
        "main > setup",
        "main > solve",
        "main > solve > assemble",
    }


def test_build_pstats_of_an_empty_run_is_an_empty_stats():
    """pstats refuses to build a Stats from an empty dict; we hand back one."""
    from scope_profiler.call_stack import build_call_arrays
    from scope_profiler.prof_export import build_pstats

    stats = build_pstats(build_call_arrays([], rank=0))

    assert isinstance(stats, pstats.Stats)
    assert stats.stats == {}
    assert stats.total_calls == 0


def test_scope_profile_dumps_the_same_file_as_write_prof_file(tmp_path):
    """The inherited cProfile.Profile.dump_stats works on our stats dict."""
    from scope_profiler.prof_export import ScopeProfile, write_prof_file

    calls = _calls(("main", 0.0, 1.0, None), ("child", 0.1, 0.4, 0))
    stats_dict = build_pstats_dict(calls, root_name="<run>")

    dumped = tmp_path / "dumped.prof"
    ScopeProfile(stats_dict).dump_stats(str(dumped))
    written = write_prof_file(tmp_path / "written.prof", stats_dict)

    assert _stats_of(dumped) == _stats_of(written)


def test_pstats_reporting_api_works_on_the_built_stats(capsys):
    """sort_stats/print_callers/print_callees, the reason to build a Stats."""
    from scope_profiler.prof_export import build_pstats

    calls = _calls(
        ("main", 0.0, 1.0, None),
        ("setup", 0.0, 0.2, 0),
        ("solve", 0.2, 0.9, 0),
    )

    stats = build_pstats(calls, root_name="<run>")

    # Self time of every region, and one synthetic root spanning the run.
    assert stats.total_tt == pytest.approx(1.0)
    stats.sort_stats("cumulative").print_stats()
    stats.print_callers()
    stats.print_callees()
    out = capsys.readouterr().out
    assert "main > solve" in out
    assert "was called by" in out
    assert "called..." in out


def test_to_pstats_returns_real_pstats_stats(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())

    result = to_pstats(read_h5(h5_file))

    assert set(result) == {("profiling_data", 0)}
    stats = result[("profiling_data", 0)]
    assert isinstance(stats, pstats.Stats)

    names = {key[2] for key in stats.stats}
    assert {
        "main",
        "main > setup",
        "main > solve",
        "main > solve > assemble",
    } <= names

    # The regular pstats.Stats API works without ever touching disk.
    stats.sort_stats("cumulative")
    assert "main" in stats.get_stats_profile().func_profiles


def test_to_pstats_matches_written_prof_file(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())

    in_memory = to_pstats(read_h5(h5_file))[("profiling_data", 0)]
    written = export_prof(read_h5(h5_file), tmp_path / "profile.prof", verbose=False)
    from_disk = load_prof(written[0])

    assert in_memory.stats == from_disk.stats


def test_load_prof_reads_back_a_written_file(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())
    written = export_prof(read_h5(h5_file), tmp_path / "profile.prof", verbose=False)

    stats = load_prof(written[0])

    assert isinstance(stats, pstats.Stats)
    assert {key[2] for key in stats.stats} >= {"main", "main > setup"}


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
        "~:0(main > setup)",
        "~:0(main > solve)",
        "~:0(main > solve > assemble)",
    }
    assert set(tree["~:0(main > solve)"]["children"]) == {
        "~:0(main > solve > assemble)"
    }
