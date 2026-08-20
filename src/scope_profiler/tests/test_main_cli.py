"""Tests for the top-level ``scope-profiler`` CLI dispatch."""

import pytest

from scope_profiler import __version__
from scope_profiler.__main__ import _COMMANDS
from scope_profiler.__main__ import main as cli_main
from scope_profiler.post_processing import _PLOT_CATALOG


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


@pytest.mark.parametrize("export_kind", ["prof", "speedscope", "plot-data"])
def test_export_kind_help_does_not_crash(export_kind, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["export", export_kind, "--help"])

    assert exc_info.value.code == 0
    assert f"scope-profiler export {export_kind}" in capsys.readouterr().out
