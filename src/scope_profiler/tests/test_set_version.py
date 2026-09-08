"""``tools/set_version.py``: the release version, in every file that spells it.

The script is not part of the installed package, so it is imported from the
repository by path. Tests that would rewrite the real files copy the repository
files into ``tmp_path`` and point the script's sites at the copies instead.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY / "tools" / "set_version.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(),
    reason="tools/ is not part of an installed package",
)


@pytest.fixture
def script():
    """The script, imported as a module."""
    spec = importlib.util.spec_from_file_location("set_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandbox(script, tmp_path, monkeypatch):
    """The real files, copied so a rewrite cannot touch the repository."""
    for site in script.VERSION_SITES:
        copy = tmp_path / site.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(site.path.read_text())
        monkeypatch.setattr(site, "path", copy)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## Unreleased\n\n- something\n")
    monkeypatch.setattr(script, "CHANGELOG", changelog)
    return tmp_path


def test_the_repository_is_consistent_right_now(script, capsys):
    """The check the release process runs, run against the real files."""
    assert script.check() == 0


def test_every_site_is_found_in_its_real_file(script):
    """A renamed file or reformatted line must fail loudly, not be skipped."""
    for site in script.VERSION_SITES:
        assert re.match(r"^\d+\.\d+\.\d+$", site.read()), site.name


def test_setting_a_version_rewrites_every_site(script, sandbox):
    assert script.main(["1.2.3"]) == 0

    for site in script.VERSION_SITES:
        assert site.read() == "1.2.3"


def test_setting_the_same_version_again_changes_nothing(script, sandbox):
    script.main(["1.2.3"])
    before = {site.name: site.path.read_text() for site in script.VERSION_SITES}

    assert script.main(["1.2.3"]) == 0

    for site in script.VERSION_SITES:
        assert site.path.read_text() == before[site.name]


def test_only_the_version_on_the_line_is_replaced(script, sandbox):
    """Neighbouring text -- other version-shaped strings -- must survive."""
    pyproject = next(
        site for site in script.VERSION_SITES if site.name == "pyproject.toml"
    )
    before = pyproject.path.read_text()

    script.main(["9.9.9"])

    after = pyproject.path.read_text()
    assert after.count("9.9.9") == 1
    # Everything else in the file is untouched, line count included.
    assert len(after.splitlines()) == len(before.splitlines())
    assert 'requires-python = ">=3.10"' in after


def test_check_reports_disagreement(script, sandbox, capsys):
    script.VERSION_SITES[1].write("0.0.1")

    assert script.check() == 1
    assert "disagree" in capsys.readouterr().err


def test_a_malformed_version_is_rejected(script, sandbox):
    for bad in ("1.2", "v1.2.3", "1.2.3rc1", "latest"):
        with pytest.raises(SystemExit):
            script.main([bad])


def test_a_missing_version_line_is_reported_not_skipped(script, sandbox):
    """Silently skipping a file it could not parse is the one thing it must not do."""
    site = script.VERSION_SITES[0]
    site.path.write_text("[project]\nname = 'scope-profiler'\n")

    with pytest.raises(SystemExit, match="no version found"):
        site.read()


def test_the_changelog_heading_is_only_retitled_when_asked(script, sandbox):
    script.main(["2.0.0"])
    assert "## Unreleased" in script.CHANGELOG.read_text()

    script.main(["2.0.0", "--changelog"])

    text = script.CHANGELOG.read_text()
    assert "## Unreleased" not in text
    assert re.search(r"^## 2\.0\.0 - \d{4}-\d{2}-\d{2}$", text, re.MULTILINE)


def test_retitling_a_changelog_without_an_unreleased_heading_is_harmless(
    script,
    sandbox,
):
    script.CHANGELOG.write_text("# Changelog\n\n## 1.0.0 - 2020-01-01\n")

    assert script.main(["2.0.0", "--changelog"]) == 0
    assert "## 2.0.0" not in script.CHANGELOG.read_text()


def test_check_and_a_version_cannot_be_combined(script, sandbox):
    with pytest.raises(SystemExit):
        script.main(["1.2.3", "--check"])
