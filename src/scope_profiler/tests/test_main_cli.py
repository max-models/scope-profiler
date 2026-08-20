"""Tests for the top-level ``scope-profiler`` CLI dispatch."""

import pytest

from scope_profiler import __version__
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
    assert "pproc" not in out


def test_run_help_lists_line_profile_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["run", "--help"])

    assert exc_info.value.code == 0
    assert "--line-profile" in capsys.readouterr().out


def test_run_line_profile_flag_is_passed_to_setup(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("print('hello')\n", encoding="utf-8")
    calls = {}

    def fake_setup(**kwargs):
        calls["setup"] = kwargs

    def fake_run_script(path, script_args=None, only_user_code=True):
        calls["run_script"] = {
            "path": path,
            "script_args": script_args,
            "only_user_code": only_user_code,
        }

    def fake_finalize(verbose=True):
        calls["finalize"] = {"verbose": verbose}

    monkeypatch.setattr("scope_profiler.__main__.ProfileManager.setup", fake_setup)
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.run_script", fake_run_script
    )
    monkeypatch.setattr(
        "scope_profiler.__main__.ProfileManager.finalize", fake_finalize
    )

    cli_main(["run", "--line-profile", "--all", "-q", str(script), "--", "arg"])

    assert calls["setup"]["use_line_profiler"] is True
    assert calls["setup"]["recursive_profile"] is True
    assert calls["run_script"] == {
        "path": str(script),
        "script_args": ["arg"],
        "only_user_code": False,
    }
    assert calls["finalize"] == {"verbose": False}


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
    "plot_kind", ["list", "default", "all", "quick", *_PLOT_CATALOG]
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
