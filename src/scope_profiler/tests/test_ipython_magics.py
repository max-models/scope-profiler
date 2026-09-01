"""Tests for :mod:`scope_profiler.ipython_magics`.

Runs the magics through a real ``InteractiveShell`` (no notebook/kernel
needed) and checks the recorded ``ProfilingResults`` and the printed tables,
the same way ``%%time``/``%%capture`` are tested in IPython itself.

Skipped entirely if the optional ``notebook`` extra (IPython) is not
installed, since it is not a dependency of the base package.
"""

import sys

import pytest

IPython = pytest.importorskip(
    "IPython", reason="the optional 'notebook' extra is not installed"
)

from IPython.core.error import UsageError
from IPython.testing import globalipapp

from scope_profiler.ipython_magics import ScopeMagics


@pytest.fixture
def shell():
    # ``globalipapp`` rebinds its own module-level ``get_ipython`` the first
    # time it runs, so it must be looked up via the module each call --
    # importing the name directly would freeze on the pre-init stub.
    ip = globalipapp.get_ipython()
    ip.register_magics(ScopeMagics)
    magics = ip.magics_manager.registry["ScopeMagics"]
    magics._runs.clear()
    magics._order.clear()
    yield ip


def _magics(shell):
    return shell.magics_manager.registry["ScopeMagics"]


def test_scope_cell_records_and_prints(shell, capsys):
    shell.run_cell_magic("scope", "warmup", "x = sum(range(1000))")
    magics = _magics(shell)
    assert magics._order == ["warmup"]
    results = magics._runs["warmup"]
    assert results.get_regions(include="warmup")
    assert shell.user_ns["x"] == sum(range(1000))
    out = capsys.readouterr().out
    assert "warmup" in out


def test_scope_cell_default_name_and_quiet(shell, capsys):
    shell.run_cell_magic("scope", "-q", "pass")
    magics = _magics(shell)
    assert magics._order == ["cell"]
    assert capsys.readouterr().out == ""


def test_scope_timeit_runs_n_times(shell, capsys):
    shell.run_line_magic("scope_timeit", "-n 5 1 + 1")
    magics = _magics(shell)
    results = magics._runs["timeit"]
    (region,) = results.get_regions(include="timeit")
    assert region.num_calls == 5
    assert "5 runs" in capsys.readouterr().out


def test_scope_last_defaults_to_most_recent(shell, capsys):
    shell.run_cell_magic("scope", "first", "pass")
    shell.run_cell_magic("scope", "second", "pass")
    capsys.readouterr()
    shell.run_line_magic("scope_last", "")
    out = capsys.readouterr().out
    assert "second" in out
    assert "first" not in out


def test_scope_last_named(shell, capsys):
    shell.run_cell_magic("scope", "first", "pass")
    shell.run_cell_magic("scope", "second", "pass")
    capsys.readouterr()
    shell.run_line_magic("scope_last", "first")
    assert "first" in capsys.readouterr().out


def test_scope_last_unknown_name_raises(shell):
    shell.run_cell_magic("scope", "first", "pass")
    with pytest.raises(UsageError):
        shell.run_line_magic("scope_last", "missing")


def test_scope_last_no_runs_raises(shell):
    with pytest.raises(UsageError):
        shell.run_line_magic("scope_last", "")


def test_scope_compare_two_most_recent(shell, capsys):
    shell.run_cell_magic("scope", "baseline", "x = sum(range(1000))")
    shell.run_cell_magic("scope", "candidate", "x = sum(range(2000))")
    capsys.readouterr()
    shell.run_line_magic("scope_compare", "")
    out = capsys.readouterr().out
    assert "baseline" in out
    assert "candidate" in out


def test_scope_compare_named(shell, capsys):
    shell.run_cell_magic("scope", "a", "pass")
    shell.run_cell_magic("scope", "b", "pass")
    shell.run_cell_magic("scope", "c", "pass")
    capsys.readouterr()
    shell.run_line_magic("scope_compare", "a c")
    out = capsys.readouterr().out
    assert "'a'" in out
    assert "'c'" in out


def test_scope_compare_needs_two_runs(shell):
    shell.run_cell_magic("scope", "only", "pass")
    with pytest.raises(UsageError):
        shell.run_line_magic("scope_compare", "")


def test_scope_compare_one_name_raises(shell):
    shell.run_cell_magic("scope", "a", "pass")
    shell.run_cell_magic("scope", "b", "pass")
    with pytest.raises(UsageError):
        shell.run_line_magic("scope_compare", "a")


line_profiler = pytest.importorskip(
    "line_profiler", reason="the optional 'line-profiler' extra is not installed"
)


def test_scope_line_records_and_prints_line_stats(shell, capsys):
    cell = (
        "from scope_profiler import ProfileManager\n"
        "\n"
        '@ProfileManager.profile("compute")\n'
        "def compute(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += i * i\n"
        "    return total\n"
        "\n"
        "compute(100)\n"
    )
    shell.run_cell_magic("scope_line", "demo", cell)
    magics = _magics(shell)
    assert magics._order == ["demo"]
    out = capsys.readouterr().out
    assert "demo" in out
    assert "Function: compute" in out
    assert "Line #" in out


def test_scope_line_quiet_flags(shell, capsys):
    shell.run_cell_magic("scope_line", "-q -Q demo", "pass")
    assert capsys.readouterr().out == ""


def test_scope_export_prof(shell, tmp_path):
    shell.run_cell_magic("scope", "naive", "x = sum(range(1000))")
    out_path = tmp_path / "run.prof"
    shell.run_line_magic("scope_export", str(out_path))
    assert list(tmp_path.glob("run*.prof"))


def test_scope_export_speedscope(shell, tmp_path):
    shell.run_cell_magic("scope", "naive", "x = sum(range(1000))")
    out_path = tmp_path / "run.speedscope.json"
    shell.run_line_magic("scope_export", str(out_path))
    assert out_path.exists()


def test_scope_export_named_run(shell, tmp_path):
    shell.run_cell_magic("scope", "first", "pass")
    shell.run_cell_magic("scope", "second", "pass")
    out_path = tmp_path / "first.prof"
    shell.run_line_magic("scope_export", f"-n first {out_path}")
    assert list(tmp_path.glob("first*.prof"))


def test_scope_reset_clears_everything(shell, capsys):
    shell.run_cell_magic("scope", "a", "pass")
    shell.run_cell_magic("scope", "b", "pass")
    capsys.readouterr()
    shell.run_line_magic("scope_reset", "")
    magics = _magics(shell)
    assert magics._runs == {}
    assert magics._order == []
    with pytest.raises(UsageError):
        shell.run_line_magic("scope_last", "")


def test_scope_reset_one_name(shell):
    shell.run_cell_magic("scope", "a", "pass")
    shell.run_cell_magic("scope", "b", "pass")
    shell.run_line_magic("scope_reset", "a")
    magics = _magics(shell)
    assert magics._order == ["b"]
    with pytest.raises(UsageError):
        shell.run_line_magic("scope_last", "a")


def test_scope_reset_unknown_name_raises(shell):
    shell.run_cell_magic("scope", "a", "pass")
    with pytest.raises(UsageError):
        shell.run_line_magic("scope_reset", "missing")


RECURSIVE_CELL = (
    "def inner(n):\n"
    "    return sum(i * i for i in range(n))\n"
    "\n"
    "def outer(n):\n"
    "    return inner(n) + inner(n)\n"
    "\n"
    "recursive_result = outer(200)\n"
)


def test_scope_recursive_records_every_call(shell, capsys):
    shell.run_cell_magic("scope_recursive", "explore", RECURSIVE_CELL)
    magics = _magics(shell)
    results = magics._runs["explore"]
    names = {region.name for region in results.get_regions()}
    assert "__main__.outer" in names
    assert "__main__.inner" in names
    # inner() is called twice by outer(), and neither is decorated.
    (inner,) = results.get_regions(include=r"__main__\.inner$")
    assert inner.num_calls == 2
    assert "%%scope_recursive 'explore'" in capsys.readouterr().out


def test_scope_recursive_assigns_into_user_namespace(shell):
    shell.run_cell_magic("scope_recursive", "-q", RECURSIVE_CELL)
    assert shell.user_ns["recursive_result"] == 2 * sum(i * i for i in range(200))


def test_scope_recursive_registers_nothing_globally(shell):
    """The magic must not leave a decorated function behind.

    A ``@ProfileManager.profile`` helper would stay registered for the life
    of the kernel, so every later session would rebind it -- and hand it to
    line_profiler in a %%scope_line session.
    """
    from scope_profiler.profile_manager import ProfileManager

    before = sum(len(entries) for entries in ProfileManager._decorators.values())
    for _ in range(3):
        shell.run_cell_magic("scope_recursive", "-q", "pass")
    after = sum(len(entries) for entries in ProfileManager._decorators.values())
    assert after == before


def test_scope_recursive_restores_the_previous_profiler(shell):
    before = sys.getprofile()
    shell.run_cell_magic("scope_recursive", "-q", RECURSIVE_CELL)
    assert sys.getprofile() is before


def test_scope_recursive_does_not_leak_into_scope_line(shell, capsys):
    """Regression: %%scope_line printed a table for this module's own code."""
    shell.run_cell_magic("scope_recursive", "-q", RECURSIVE_CELL)
    capsys.readouterr()
    cell = (
        "from scope_profiler import ProfileManager\n"
        "\n"
        '@ProfileManager.profile("compute")\n'
        "def compute(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += i * i\n"
        "    return total\n"
        "\n"
        "compute(50)\n"
    )
    shell.run_cell_magic("scope_line", "-q", cell)
    out = capsys.readouterr().out
    assert "ipython_magics.py" not in out
    assert "run_cell_recursively" not in out
    # The user's own function is still line-profiled, with its rows filled in.
    assert "Function: compute" in out
    assert "total += i * i" in out


def test_scope_agg_records_counts_without_events(shell, capsys):
    cell = (
        "from scope_profiler import ProfileManager\n"
        "for _ in range(25):\n"
        '    with ProfileManager.profile_region("hot"):\n'
        "        pass\n"
    )
    shell.run_cell_magic("scope_agg", "hotloop", cell)
    magics = _magics(shell)
    (hot,) = magics._runs["hotloop"].get_regions(include="hot$")
    assert hot.num_calls == 25
    assert "%%scope_agg 'hotloop'" in capsys.readouterr().out


def test_scope_load_reads_a_file_written_elsewhere(shell, tmp_path, capsys):
    from scope_profiler import ProfileManager

    path = tmp_path / "cluster_run.h5"
    with ProfileManager.session(file_path=str(path), verbose=False):
        with ProfileManager.profile_region("solve"):
            sum(range(1000))

    shell.run_line_magic("scope_load", str(path))
    magics = _magics(shell)
    assert magics._order == ["cluster_run"]
    assert magics._runs["cluster_run"].get_regions(include="solve")
    assert "cluster_run" in capsys.readouterr().out


def test_scope_load_named_and_comparable(shell, tmp_path):
    from scope_profiler import ProfileManager

    path = tmp_path / "run.h5"
    with ProfileManager.session(file_path=str(path), verbose=False):
        with ProfileManager.profile_region("solve"):
            sum(range(1000))

    shell.run_line_magic("scope_load", f"-q -n baseline {path}")
    shell.run_cell_magic("scope", "candidate", "x = sum(range(1000))")
    shell.run_line_magic("scope_compare", "baseline candidate")


def test_scope_load_missing_file_raises(shell, tmp_path):
    with pytest.raises(FileNotFoundError):
        shell.run_line_magic("scope_load", str(tmp_path / "absent.h5"))


pandas = pytest.importorskip("pandas", reason="the optional 'pproc' extra is missing")


def test_scope_df_returns_region_frame(shell):
    shell.run_cell_magic("scope", "-q run_a", "x = sum(range(1000))")
    frame = shell.run_line_magic("scope_df", "")
    assert list(frame["name"]) == ["scope_profiler.session", "run_a"]
    assert frame["num_calls"].tolist() == [1, 1]


def test_scope_df_events_and_per_rank(shell):
    shell.run_cell_magic("scope", "-q run_a", "x = sum(range(1000))")
    events = shell.run_line_magic("scope_df", "--events")
    assert list(events.columns) == [
        "name",
        "rank",
        "call_index",
        "start",
        "end",
        "duration",
    ]
    per_rank = shell.run_line_magic("scope_df", "--per-rank")
    assert "rank" in per_rank.columns


def test_scope_df_include_filter(shell):
    shell.run_cell_magic("scope", "-q run_a", "x = sum(range(1000))")
    frame = shell.run_line_magic("scope_df", "--include run_a")
    assert list(frame["name"]) == ["run_a"]


def test_scope_df_events_on_aggregation_run_explains_empty(shell, capsys):
    cell = (
        "from scope_profiler import ProfileManager\n"
        'with ProfileManager.profile_region("hot"):\n'
        "    pass\n"
    )
    shell.run_cell_magic("scope_agg", "-q agg", cell)
    capsys.readouterr()
    frame = shell.run_line_magic("scope_df", "--events")
    assert frame.empty
    assert "aggregation mode" in capsys.readouterr().out


def test_scope_df_conflicting_flags_raise(shell):
    shell.run_cell_magic("scope", "-q run_a", "pass")
    with pytest.raises(UsageError):
        shell.run_line_magic("scope_df", "--events --per-rank")


FAILING_CELL = (
    "def work():\n"
    "    return sum(range(1000))\n"
    "\n"
    "work()\n"
    'raise ValueError("kaboom")\n'
)


@pytest.mark.parametrize("magic", ["scope", "scope_agg", "scope_recursive"])
def test_failing_cell_still_records_the_partial_run(shell, magic, capsys):
    """A cell that raises should report the error and keep what it measured."""
    shell.run_cell_magic(magic, "boom", FAILING_CELL)
    magics = _magics(shell)
    assert magics._order == ["boom"]
    out = capsys.readouterr().out
    assert "kaboom" in out
    # The table is still printed, so the partial profile is not lost.
    assert "boom" in out


def test_scope_recursive_restores_profiler_after_failure(shell):
    before = sys.getprofile()
    shell.run_cell_magic("scope_recursive", "-q boom", FAILING_CELL)
    assert sys.getprofile() is before


def test_scope_recursive_failure_traceback_starts_in_the_cell(shell, capsys):
    shell.run_cell_magic("scope_recursive", "-q boom", FAILING_CELL)
    out = capsys.readouterr().out
    # The magic's own frame is sliced off the traceback.
    assert "ipython_magics.py" not in out
    assert "exec(code, namespace)" not in out
    assert 'raise ValueError("kaboom")' in out


@pytest.mark.parametrize(
    "magic, line, cell",
    [
        ("scope", "-q", "x = sum(range(100))"),
        ("scope_agg", "-q", "x = sum(range(100))"),
        ("scope_recursive", "-q", "x = sum(range(100))"),
        ("scope_line", "-q -Q", "x = sum(range(100))"),
    ],
)
def test_recording_magics_write_nothing_to_disk(
    shell, tmp_path, monkeypatch, magic, line, cell
):
    """The magics run with deactivate_file_output=True, by design."""
    monkeypatch.chdir(tmp_path)
    shell.run_cell_magic(magic, line, cell)
    assert list(tmp_path.iterdir()) == []
