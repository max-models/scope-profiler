"""Tests for the top-level ``scope-profiler`` CLI dispatch."""

import pytest

from scope_profiler import ProfileManager, __version__, read_h5
from scope_profiler.__main__ import _COMMANDS
from scope_profiler.__main__ import main as cli_main
from scope_profiler.post_processing import _DEFAULT_PLOTS, _PLOT_CATALOG


def test_version_flag_prints_version_and_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--version"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "scope-profiler" in out
    assert __version__ in out


def test_no_args_prints_help_and_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main([])

    assert exc_info.value.code != 0
    assert "usage: scope-profiler" in capsys.readouterr().out


def test_help_lists_plot_export_and_not_pproc(capsys):
    with pytest.raises(SystemExit):
        cli_main(["--help"])

    out = capsys.readouterr().out
    assert "plot" in out
    assert "export" in out
    assert "report" in out
    assert "pproc" not in out


def test_run_help_lists_line_profile_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["run", "--help"])

    assert exc_info.value.code == 0
    assert "--line-profile" in capsys.readouterr().out


def test_run_help_lists_memory_profile_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["run", "--help"])

    assert exc_info.value.code == 0
    assert "--memory-profile" in capsys.readouterr().out


def test_run_help_lists_low_touch_options(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["run", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--no-recursive" in output
    assert "--no-output" in output
    assert "--label" in output
    assert "--entrypoint" in output
    assert "--include" in output
    assert "--exclude" in output
    assert "--tag" in output


def test_run_help_lists_mpi_call_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["run", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--mpi-calls" in output
    assert "--no-mpi-calls" in output


def test_run_line_profile_flag_is_passed_to_setup(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("print('hello')\n", encoding="utf-8")
    calls = {}

    def fake_setup(**kwargs):
        calls["setup"] = kwargs

    def fake_run_script(
        path,
        script_args=None,
        only_user_code=True,
        recursive=True,
        include_patterns=(),
        exclude_patterns=(),
        entrypoint=None,
    ):
        calls["run_script"] = {
            "path": path,
            "script_args": script_args,
            "only_user_code": only_user_code,
            "recursive": recursive,
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "entrypoint": entrypoint,
        }

    def fake_finalize(verbose=True):
        calls["finalize"] = {"verbose": verbose}

    monkeypatch.setattr("scope_profiler.__main__.ProfileManager.setup", fake_setup)
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.run_script",
        fake_run_script,
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.finalize",
        fake_finalize,
    )

    cli_main(
        [
            "run",
            "--line-profile",
            "--memory-profile",
            "--all",
            "-q",
            str(script),
            "--",
            "arg",
        ],
    )

    assert calls["setup"]["use_line_profiler"] is True
    assert calls["setup"]["use_memray"] is True
    assert calls["setup"]["recursive_profile"] is True
    assert calls["setup"]["profile_mpi_calls"] is None
    assert calls["run_script"] == {
        "path": str(script),
        "script_args": ["arg"],
        "only_user_code": False,
        "recursive": True,
        "include_patterns": [],
        "exclude_patterns": [],
        "entrypoint": None,
    }
    assert calls["finalize"] == {"verbose": False}


def test_run_does_not_enable_mpi_calls_by_default(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("pass\n", encoding="utf-8")
    calls = {}

    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.setup",
        lambda **kwargs: calls.update(setup=kwargs),
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.run_script",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.finalize",
        lambda **kwargs: None,
    )

    cli_main(["run", str(script)])

    # None lets setup() use its False default, or a config file opt in.
    assert calls["setup"]["profile_mpi_calls"] is None


def test_run_can_disable_recursive_tracing_and_file_output(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("pass\n", encoding="utf-8")
    calls = {}

    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.setup",
        lambda **kwargs: calls.update(setup=kwargs),
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.run_script",
        lambda *args, **kwargs: calls.update(run_script=kwargs),
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.finalize",
        lambda **kwargs: calls.update(finalize=kwargs),
    )

    cli_main(["run", "--no-recursive", "--no-output", "--label", "trial", str(script)])

    assert calls["setup"]["recursive_profile"] is False
    assert calls["setup"]["deactivate_file_output"] is True
    assert calls["setup"]["label"] == "trial"
    assert calls["run_script"]["recursive"] is False
    assert calls["finalize"] == {"verbose": True, "return_results": True}


def test_run_can_enable_automatic_mpi_call_profiling(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("pass\n", encoding="utf-8")
    calls = {}

    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.setup",
        lambda **kwargs: calls.update(setup=kwargs),
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.run_script",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.finalize",
        lambda **kwargs: None,
    )

    cli_main(["run", "--mpi-calls", str(script)])

    assert calls["setup"]["profile_mpi_calls"] is True


def test_run_can_disable_automatic_mpi_call_profiling(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("pass\n", encoding="utf-8")
    events = []
    calls = {}

    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.setup",
        lambda **kwargs: calls.update(setup=kwargs),
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.run_script",
        lambda *args, **kwargs: events.append("run"),
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.finalize",
        lambda **kwargs: events.append("finalize"),
    )
    cli_main(["run", "--no-mpi-calls", str(script)])

    assert events == ["run", "finalize"]
    assert calls["setup"]["profile_mpi_calls"] is False


def test_run_activates_regions_without_an_in_source_session(tmp_path):
    script = tmp_path / "instrumented.py"
    output = tmp_path / "profile.h5"
    script.write_text(
        """\
import scope_profiler as sp

@sp.profile("main")
def main():
    for _ in range(3):
        with sp.region("iteration"):
            pass

main()
""",
        encoding="utf-8",
    )

    ProfileManager._reset()
    try:
        cli_main(["run", "-q", "-o", str(output), str(script)])
        results = read_h5(output)

        assert set(results.region_names) == {"main", "iteration"}
        assert results["main"].num_calls == 1
        assert results["iteration"].num_calls == 3
    finally:
        ProfileManager._reset()


def test_run_selects_explicit_regions_by_tag(tmp_path):
    script = tmp_path / "tagged.py"
    output = tmp_path / "tagged.h5"
    script.write_text(
        """\
import scope_profiler as sp

with sp.region("fast", tags=["hot"]):
    pass
with sp.region("slow", tags=["cold"]):
    pass
""",
        encoding="utf-8",
    )

    ProfileManager._reset()
    try:
        cli_main(["run", "-q", "--tag", "hot", "-o", str(output), str(script)])
        results = read_h5(output)
        assert set(results.region_names) == {"fast"}
    finally:
        ProfileManager._reset()


def test_run_toml_config_is_passed_to_setup(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("print('hello')\n", encoding="utf-8")
    config = tmp_path / "profiling.toml"
    config.write_text(
        "[profiling]\nrecursive_profile = false\nfile_path = 'from-config.h5'\n",
        encoding="utf-8",
    )
    calls = {}

    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.setup",
        lambda **kwargs: calls.update(setup=kwargs),
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.run_script",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.finalize",
        lambda **kwargs: None,
    )

    cli_main(["run", "--config", str(config), str(script)])

    assert calls["setup"]["config_path"] == str(config)
    assert calls["setup"]["recursive_profile"] is None
    assert calls["setup"]["file_path"] is None


def test_pproc_is_not_a_command(capsys):
    with pytest.raises(SystemExit):
        cli_main(["pproc", "--help"])

    assert "invalid choice" in capsys.readouterr().err


def test_invalid_choice_does_not_mention_pproc(capsys):
    with pytest.raises(SystemExit):
        cli_main(["bogus-command"])

    assert "pproc" not in capsys.readouterr().err


@pytest.mark.parametrize("command", sorted(_COMMANDS))
def test_top_level_command_help_does_not_crash(command, capsys):
    """Regression test: building each subparser's help text must not raise.

    ``scope-profiler plot <kind> --help`` used to crash (a tuple was passed
    where argparse expected a help string), and nothing exercised ``--help``
    below the top level to catch it.
    """
    with pytest.raises(SystemExit) as exc_info:
        cli_main([command, "--help"])

    assert exc_info.value.code == 0
    assert f"scope-profiler {command}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "plot_kind",
    ["list", "default", "all", "quick", *_PLOT_CATALOG],
)
def test_plot_kind_help_does_not_crash(plot_kind, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["plot", plot_kind, "--help"])

    assert exc_info.value.code == 0
    assert f"scope-profiler plot {plot_kind}" in capsys.readouterr().out


def test_default_plot_preset_is_gantt_and_total_durations():
    assert _DEFAULT_PLOTS == {"gantt", "durations"}


@pytest.mark.parametrize("export_kind", ["prof", "speedscope", "plot-data"])
def test_export_kind_help_does_not_crash(export_kind, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["export", export_kind, "--help"])

    assert exc_info.value.code == 0
    assert f"scope-profiler export {export_kind}" in capsys.readouterr().out
