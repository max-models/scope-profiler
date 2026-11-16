"""``scope-profiler tags``: list the opt-in tags of the installed packages."""

import json
import sys
from importlib.metadata import EntryPoint

import pytest

import scope_profiler as sp
from scope_profiler import profile_manager, tags_cli
from scope_profiler.__main__ import main as cli_main

_LIBRARY = """
import scope_profiler as sp

TAGS = ("{tag}",)
sp.register_opt_in_tag(*TAGS)


@sp.profile("{tag}.solve", tags=TAGS)
def solve():
    with sp.region("{tag}.inline", tags=TAGS):
        pass


kernel = sp.define_region("{tag}.kernel", tags=TAGS)
"""


@pytest.fixture(autouse=True)
def _fresh_opt_in_tags(monkeypatch):
    """Give every test its own registry; the real one outlives resets."""
    monkeypatch.setattr(profile_manager, "_OPT_IN_TAGS", set())


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Write importable library modules registering a tag named after them."""
    monkeypatch.syspath_prepend(str(tmp_path))
    written = []

    def write(name, source=None):
        text = _LIBRARY.format(tag=name) if source is None else source
        (tmp_path / f"{name}.py").write_text(text, encoding="utf-8")
        written.append(name)
        return name

    yield write
    for name in written:
        sys.modules.pop(name, None)


@pytest.fixture
def declared(monkeypatch):
    """Declare entry points in the ``scope_profiler.opt_in_tags`` group."""
    points = []

    def fake_entry_points(group):
        assert group == tags_cli.ENTRY_POINT_GROUP
        return points

    monkeypatch.setattr(tags_cli, "entry_points", fake_entry_points)

    def declare(name, value):
        points.append(EntryPoint(name, value, tags_cli.ENTRY_POINT_GROUP))

    return declare


def test_opt_in_tags_lists_declared_regions_and_ran_inline_ones(library):
    module = __import__(library("taglib_a"))

    assert sp.opt_in_tags() == {"taglib_a": ("taglib_a.kernel", "taglib_a.solve")}

    sp.setup(output=None, enable_tags=["taglib_a"])
    module.solve()
    sp.finalize(verbose=False)

    assert sp.opt_in_tags()["taglib_a"] == (
        "taglib_a.inline",
        "taglib_a.kernel",
        "taglib_a.solve",
    )


def test_opt_in_tags_ignores_regular_tags():
    sp.register_opt_in_tag("lib")

    @sp.profile("plain", tags=("other",))
    def plain():
        pass

    assert sp.opt_in_tags() == {"lib": ()}


def test_entry_points_are_imported_and_listed(library, declared, capsys):
    declared("taglib_b", library("taglib_b"))

    assert cli_main(["tags"]) == 0

    out = capsys.readouterr().out
    assert "taglib_b (registered by taglib_b)\n" in out
    assert "  taglib_b.kernel\n  taglib_b.solve\n" in out
    assert "--enable-tag" in out


def test_import_flag_and_json_output(library, declared, capsys):
    library("taglib_c")
    sp.register_opt_in_tag("early")

    assert cli_main(["tags", "--import", "taglib_c", "--json"]) == 0

    document = json.loads(capsys.readouterr().out)
    assert document == {
        "tags": [
            {"tag": "early", "module": __name__, "regions": []},
            {
                "tag": "taglib_c",
                "module": "taglib_c",
                "regions": ["taglib_c.kernel", "taglib_c.solve"],
            },
        ],
        "errors": [],
    }


def test_tag_without_regions_says_so(capsys):
    sp.register_opt_in_tag("empty")

    assert tags_cli.main(["--no-entry-points"]) == 0

    out = capsys.readouterr().out
    assert f"empty (registered by {__name__})\n  (no regions known yet)\n" in out


def test_broken_entry_point_is_a_warning(library, declared, capsys):
    declared("broken", library("taglib_broken", "raise RuntimeError('boom')\n"))
    declared("good", library("taglib_d"))

    assert tags_cli.main([]) == 0

    captured = capsys.readouterr()
    assert "warning: could not import entry point broken: RuntimeError: boom" in (
        captured.err
    )
    assert "taglib_d (registered by taglib_d)" in captured.out


def test_failed_import_flag_exits_nonzero(declared, capsys):
    assert tags_cli.main(["--import", "no_such_module_for_tags"]) == 1

    captured = capsys.readouterr()
    assert "could not import no_such_module_for_tags: ModuleNotFoundError" in (
        captured.err
    )
    assert "No opt-in tags registered" in captured.out


def test_no_entry_points_skips_discovery(monkeypatch, capsys):
    def fail(group):
        raise AssertionError("entry points must not be read")

    monkeypatch.setattr(tags_cli, "entry_points", fail)

    assert tags_cli.main(["--no-entry-points", "--json"]) == 0

    assert json.loads(capsys.readouterr().out) == {"tags": [], "errors": []}


def test_real_entry_point_group_is_readable(capsys):
    assert tags_cli.main([]) == 0
    capsys.readouterr()
