"""Tests for ``scope-profiler site``: Astro pages for a catalogue of runs."""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("yaml")
pytest.importorskip("plotly")

from scope_profiler.__main__ import main as cli_main
from scope_profiler.astro_site import (
    MARKER,
    SiteError,
    build_site,
    load_manifest,
    slugify,
)
from scope_profiler.tests.test_post_processing import _write_sample_h5

CONTENT = "src/content/docs/catalogue"
PUBLIC = "public/catalogue"


def _write_profiles(directory, ranks=(1, 2, 4), **metadata):
    """One file per rank count, with calls that nest properly."""
    directory.mkdir(parents=True, exist_ok=True)
    for count in ranks:
        setup = 10_000_000 // count
        solve = 50_000_000 // count
        data = {
            rank: {
                "setup": ([0], [setup]),
                "solve": ([setup + 10], [setup + 10 + solve]),
            }
            for rank in range(count)
        }
        _write_sample_h5(
            directory / f"np{count}.h5",
            data,
            {
                key: value
                for key, value in {
                    "mpi_size": count,
                    "omp_num_threads": 1,
                    "num_nodes": 1,
                    "timestamp": "2026-10-17T10:00:00Z",
                    "scope_profiler_version": "0.7.4",
                    "hostname": "secret-host",
                    "chip_information": "Test CPU",
                    **metadata,
                }.items()
                if value is not None  # None leaves a field out
            },
        )


def _manifest(tmp_path, text, base="/cat"):
    """Write a manifest; ``base=None`` leaves the base path out."""
    path = tmp_path / "catalogue.yaml"
    path.write_text(
        (f"base: {base!r}\n" if base is not None and "base:" not in text else "")
        + text,
        encoding="utf-8",
    )
    return path


BASIC = """
title: Probe catalogue
description: A {tricky} <description> with $math.
cases:
  - title: Poisson solver
    group: Solvers
    description: Strong scaling.
    tags: [cpu, scaling]
    info: {commit: 843a82cd, note: "uses {braces} and <tags>"}
    runs:
      - path: data/np*.h5
  - title: Single run
    runs:
      - {path: data/np2.h5, label: only run}
"""


def _tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_slugify():
    assert slugify("Poisson cube, GPU (256³)") == "poisson-cube-gpu-256"
    assert slugify("---") == ""


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("title: x\n", "'cases' must be a non-empty list"),
        ("cases: []\n", "'cases' must be a non-empty list"),
        ("bogus: 1\ncases: [{title: a, runs: [x]}]\n", "unknown key"),
        ("cases: [{runs: [data/np1.h5]}]\n", "'title' or 'id' is required"),
        ("cases: [{title: a}]\n", "case 'a'.runs must be a non-empty list"),
        (
            "cases: [{title: a, runs: [data/np1.h5], colour: red}]\n",
            "unknown key",
        ),
        (
            "cases: [{title: a, scaling: sideways, runs: [data/np1.h5]}]\n",
            "scaling must be one of",
        ),
        (
            "cases: [{title: a, speedup_x: time, runs: [data/np1.h5]}]\n",
            "speedup_x must be one of",
        ),
        (
            "cases: [{title: a, include: '(', runs: [data/np1.h5]}]\n",
            "not a regular expression",
        ),
        (
            "cases: [{title: a, ranks: zero, runs: [data/np1.h5]}]\n",
            "must be a rank or a list of ranks",
        ),
        (
            "cases: [{title: a, runs: [data/np1.h5]}, {title: A, runs: [data/np1.h5]}]\n",
            "the id is used by another case",
        ),
        ("cases: [{title: a, runs: [data/gone.h5]}]\n", "is not a file"),
        ("cases: [{title: a, runs: [data/none*.h5]}]\n", "no file matches"),
        (
            "cases: [{title: a, runs: [{path: 'data/np*.h5', label: x}]}]\n",
            "cannot be given to a pattern",
        ),
        ("cases: [{title: a, runs: [{label: x}]}]\n", "'path' is required"),
        ("cases: [{title: '!!!', runs: [data/np1.h5]}]\n", "cannot make an id"),
        ("cases: [\n", "not valid YAML"),
    ],
)
def test_invalid_manifests_say_what_is_wrong(tmp_path, text, message):
    _write_profiles(tmp_path / "data")

    with pytest.raises(SiteError, match=re.escape(message)):
        load_manifest(_manifest(tmp_path, text))


def test_missing_manifest_is_a_site_error(tmp_path):
    with pytest.raises(SiteError, match="cannot read"):
        load_manifest(tmp_path / "nothing.yaml")


def test_globs_are_expanded_in_natural_order(tmp_path):
    _write_profiles(tmp_path / "data", ranks=(1, 2, 10))

    manifest = load_manifest(
        _manifest(tmp_path, "cases: [{title: a, runs: ['data/np*.h5']}]\n"),
    )

    assert [run.path.name for run in manifest.cases[0].runs] == [
        "np1.h5",
        "np2.h5",
        "np10.h5",
    ]


def test_defaults_apply_to_every_case_and_cases_override_them(tmp_path):
    _write_profiles(tmp_path / "data")

    manifest = load_manifest(
        _manifest(
            tmp_path,
            """
defaults: {scaling: weak, ranks: [0]}
cases:
  - {title: a, runs: [data/np1.h5]}
  - {title: b, scaling: none, runs: [data/np1.h5]}
""",
        ),
    )

    first, second = manifest.cases
    assert (first.scaling, first.ranks) == ("weak", (0,))
    assert (second.scaling, second.ranks) == ("none", (0,))


def test_build_writes_pages_and_chart_data(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    result = build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    assert (result.cases, result.runs) == (2, 4)
    content = out / CONTENT
    case_dir = content / "solvers" / "poisson-solver"
    for page in ("index.mdx", "np1.mdx", "np2.mdx", "np4.mdx"):
        assert (case_dir / page).is_file()
    assert (content / "single-run" / "only-run.mdx").is_file()
    assert (content / "index.mdx").is_file()
    assert json.loads((out / PUBLIC / "index.json").read_text())["cases"][0]["id"] == (
        "poisson-solver"
    )

    # Every chart a page refers to exists, and is scope-profiler plot data.
    for page in content.rglob("*.mdx"):
        for url in re.findall(r'src="([^"]+)"', page.read_text()):
            assert url.startswith("/catalogue/")
            chart = out / "public" / url.lstrip("/")
            assert json.loads(chart.read_text())["format"] == (
                "scope-profiler-plot-data"
            )


def _data(out, *parts):
    """The JSON data file of a generated page (``parts`` below the catalogue)."""
    return _strict_json((out / CONTENT).joinpath(*parts).read_text())


def _strict_json(text):
    def refuse(constant):
        raise ValueError(f"{constant} is not JSON")

    return json.loads(text, parse_constant=refuse)


def test_chart_data_is_valid_json_when_a_run_lacks_a_region(tmp_path):
    """A region only some runs recorded must not leave NaN in the files."""
    _write_profiles(tmp_path / "data", ranks=(1,))
    setup, solve = 10_000_000, 50_000_000
    _write_sample_h5(
        tmp_path / "data" / "np2.h5",
        {
            rank: {
                "setup": ([0], [setup]),
                "solve": ([setup + 10], [setup + 10 + solve]),
                "exchange": ([setup + solve + 20], [setup + solve + 30]),
            }
            for rank in range(2)
        },
        {"mpi_size": 2, "omp_num_threads": 1, "num_nodes": 1},
    )
    out = tmp_path / "docs"

    build_site(
        load_manifest(
            _manifest(tmp_path, "cases: [{title: a, runs: ['data/np*.h5']}]\n"),
        ),
        out,
    )

    charts = list((out / PUBLIC).rglob("*.plot.json"))
    assert charts
    for chart in charts:
        _strict_json(chart.read_text())  # raises on NaN or Infinity


def test_non_finite_numbers_are_dropped_or_nulled():
    from scope_profiler.astro_site import _finite

    payload = {
        "bars": [
            {"region": "a", "value": 1.0},
            {"region": "b", "value": float("nan")},
        ],
        "ratio": float("inf"),
        "nested": {"rows": [{"x": float("-inf")}, {"x": 2}]},
    }

    # Chart data leaves out the rows that miss a measurement ...
    assert _finite(payload, drop_rows=True) == {
        "bars": [{"region": "a", "value": 1.0}],
        "ratio": None,
        "nested": {"rows": [{"x": 2}]},
    }
    # ... and the page data keeps every row, with null where a number is missing.
    assert _finite(payload) == {
        "bars": [{"region": "a", "value": 1.0}, {"region": "b", "value": None}],
        "ratio": None,
        "nested": {"rows": [{"x": None}, {"x": 2}]},
    }


def test_pages_link_to_each_other_with_the_base_path(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    content = out / CONTENT
    index = (content / "index.mdx").read_text()
    assert "[Poisson solver](/cat/catalogue/solvers/poisson-solver/)" in index
    assert "[Single run](/cat/catalogue/single-run/)" in index
    run = (content / "solvers" / "poisson-solver" / "np2.mdx").read_text()
    assert "[← Poisson solver](/cat/catalogue/solvers/poisson-solver/)" in run
    # The components are imported relative to the page, however deep it is.
    assert (
        'import Summary from "../../../../../components/report/Summary.astro";' in run
    )
    single = (content / "single-run" / "only-run.mdx").read_text()
    assert (
        'import Summary from "../../../../components/report/Summary.astro";' in single
    )
    # The runs table links each run's page; chart data is addressed from the site
    # root, and the component adds the base path.
    case = _data(out, "solvers", "poisson-solver", "index.data.json")
    assert case["links"]["runs"][1] == "/cat/catalogue/solvers/poisson-solver/np2/"
    panels = _data(out, "solvers", "poisson-solver", "np2.data.json")["charts"][
        "panels"
    ]
    assert panels
    assert all(
        p["src"].startswith("/catalogue/solvers/poisson-solver/np2/") for p in panels
    )


def test_the_base_path_is_required_and_checked(tmp_path):
    _write_profiles(tmp_path / "data")
    runs = "cases: [{title: a, runs: [data/np1.h5]}]\n"
    no_base = load_manifest(_manifest(tmp_path, runs, base=None))

    with pytest.raises(SiteError, match="base path is needed"):
        build_site(no_base, tmp_path / "docs")
    assert not (tmp_path / "docs").exists()  # nothing is written without one
    with pytest.raises(SiteError, match="must start with '/'"):
        build_site(no_base, tmp_path / "docs", base="cat")
    with pytest.raises(SiteError, match="must start with '/'"):
        load_manifest(_manifest(tmp_path, runs, base="cat"))


def test_the_base_argument_overrides_the_manifest_and_may_be_empty(tmp_path):
    _write_profiles(tmp_path / "data")
    manifest = load_manifest(_manifest(tmp_path, BASIC, base="/cat"))

    build_site(manifest, tmp_path / "other", base="/other/")
    build_site(manifest, tmp_path / "root", base="")

    other = (tmp_path / "other" / CONTENT / "index.mdx").read_text()
    assert "(/other/catalogue/single-run/)" in other
    root = (tmp_path / "root" / CONTENT / "index.mdx").read_text()
    assert "(/catalogue/single-run/)" in root


def test_case_page_compares_the_runs(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    case = (out / CONTENT / "solvers" / "poisson-solver" / "index.mdx").read_text()
    assert "Tags: `cpu`, `scaling`" in case
    for section in ("RunsTable", "Changes", "CompareTable", "Charts"):
        assert f"<{section} data={{data}} />" in case
    data = _data(out, "solvers", "poisson-solver", "index.data.json")
    report = data["report"]
    assert report["labels"] == ["np1", "np2", "np4"]
    assert [row["label"] for row in report["runs"]["rows"]] == ["np1", "np2", "np4"]
    assert report["runs"]["rows"][0]["change"] == {"kind": "base"}
    assert len(report["candidates"]) == 2
    assert {row["name"] for row in report["table"]["rows"]} >= {"setup", "solve"}
    assert "Speedup" in [panel["title"] for panel in data["charts"]["panels"]]
    # A single run has nothing to compare.
    single = (out / CONTENT / "single-run" / "index.mdx").read_text()
    assert "CompareTable" not in single
    assert "- [only run](/cat/catalogue/single-run/only-run/)" in single


def test_regions_are_compared_per_rank(tmp_path):
    """Adding up ranks would make a run on more ranks look slower."""
    _write_profiles(tmp_path / "data", ranks=(1, 2))
    out = tmp_path / "docs"

    build_site(
        load_manifest(
            _manifest(tmp_path, "cases: [{title: a, runs: ['data/np*.h5']}]\n"),
        ),
        out,
    )

    table = _data(out, "a", "index.data.json")["report"]["table"]
    assert table["per_rank"] is True
    solve = next(row for row in table["rows"] if row["name"] == "solve")
    # 'solve' lasts 50 ms on one rank and 25 ms on each of two.
    assert [cell["values"]["total"] for cell in solve["cells"]] == ["50 ms", "25 ms"]


def test_overview_pages_lead_their_sidebar_group(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    head = 'sidebar:\n  label: "Overview"\n  order: 0\n'
    assert head in (out / CONTENT / "index.mdx").read_text()
    assert head in (out / CONTENT / "single-run" / "index.mdx").read_text()
    run = (out / CONTENT / "single-run" / "only-run.mdx").read_text()
    assert 'label: "only run"' in run
    assert "order: 1" in run


def test_user_text_cannot_break_the_mdx(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    index = (out / CONTENT / "index.mdx").read_text()
    assert r"A \{tricky\} \<description\> with \$math." in index
    case = (out / CONTENT / "solvers" / "poisson-solver" / "index.mdx").read_text()
    assert r"uses \{braces\} and \<tags\>" in case
    # In the frontmatter the text is a quoted YAML string, so it stays literal.
    assert 'description: "A {tricky} <description> with $math."' in index


def test_only_identifying_metadata_is_published(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    everything = "".join(
        path.read_text() for path in (out / CONTENT).rglob("*") if path.is_file()
    )
    assert "secret-host" not in everything
    report = _data(out, "solvers", "poisson-solver", "np2.data.json")["report"]
    assert report["meta"]["host"] == ["Test CPU"]
    assert report["meta"]["file"]["path"] is None
    keys = [key for key, _ in report["metadata"]]
    assert "timestamp" in keys and "hostname" not in keys
    runs = _data(out, "solvers", "poisson-solver", "index.data.json")["report"]["runs"]
    assert all(row["host"] is None for row in runs["rows"])


def test_metadata_all_publishes_everything_the_profile_recorded(tmp_path):
    _write_profiles(tmp_path / "data", ranks=(2,))
    out = tmp_path / "docs"
    manifest = load_manifest(
        _manifest(tmp_path, "cases: [{title: a, metadata: all, runs: [data/np2.h5]}]\n")
    )

    build_site(manifest, out)

    report = _data(out, "a", "np2.data.json")["report"]
    assert ["hostname", "secret-host"] in report["metadata"]
    assert report["meta"]["host"] == ["secret-host", "Test CPU"]


def test_an_unknown_metadata_level_is_refused(tmp_path):
    _write_profiles(tmp_path / "data")

    with pytest.raises(SiteError, match="metadata must be 'minimal' or 'all'"):
        load_manifest(
            _manifest(
                tmp_path, "cases: [{title: a, metadata: some, runs: [data/np1.h5]}]\n"
            )
        )


def test_build_is_deterministic(tmp_path):
    _write_profiles(tmp_path / "data")
    manifest = load_manifest(_manifest(tmp_path, BASIC))

    build_site(manifest, tmp_path / "one")
    build_site(manifest, tmp_path / "two")

    assert _tree(tmp_path / "one") == _tree(tmp_path / "two")


def test_no_charts_writes_pages_without_chart_data(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out, charts=False)

    assert not list((out / PUBLIC).rglob("*.plot.json"))
    for page in (out / CONTENT).rglob("*.mdx"):
        assert "<ScopeChart" not in page.read_text()


def test_single_rank_runs_skip_the_rank_sections(tmp_path):
    _write_profiles(tmp_path / "data", ranks=(1,))
    out = tmp_path / "docs"

    build_site(
        load_manifest(
            _manifest(tmp_path, "cases: [{title: a, runs: [data/np1.h5]}]\n")
        ),
        out,
    )

    data = _data(out, "a", "np1.data.json")
    titles = [panel["title"] for panel in data["charts"]["panels"]]
    assert "Region durations" in titles
    assert "Rank imbalance" not in titles
    assert data["report"]["balance"] is None and data["report"]["matrix"] is None
    run = (out / CONTENT / "a" / "np1.mdx").read_text()
    assert "LoadBalance" not in run and "RankMatrix" not in run
    assert "<RegionTable data={data} />" in run


def test_rebuild_replaces_only_what_a_previous_build_wrote(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"
    manifest = load_manifest(_manifest(tmp_path, BASIC))
    build_site(manifest, out)
    stale = out / CONTENT / "stale.mdx"
    stale.write_text("old")

    build_site(manifest, out)

    assert not stale.exists()
    assert (out / CONTENT / MARKER).is_file()


def test_hand_written_content_is_never_deleted(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"
    mine = out / CONTENT / "mine.mdx"
    mine.parent.mkdir(parents=True)
    mine.write_text("hand written")

    with pytest.raises(SiteError, match="was not written by scope-profiler site"):
        build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    assert mine.read_text() == "hand written"
    # --no-clean writes next to it instead.
    build_site(load_manifest(_manifest(tmp_path, BASIC)), out, clean=False)
    assert mine.read_text() == "hand written"
    assert (out / CONTENT / "index.mdx").is_file()


def test_runs_with_the_same_label_are_refused(tmp_path):
    _write_profiles(tmp_path / "data")
    manifest = load_manifest(
        _manifest(
            tmp_path,
            """
cases:
  - title: a
    runs:
      - {path: data/np1.h5, label: same}
      - {path: data/np2.h5, label: same}
""",
        ),
    )

    with pytest.raises(SiteError, match="two runs are called 'same'"):
        build_site(manifest, tmp_path / "docs")


def test_an_axis_the_runs_did_not_record_is_refused(tmp_path):
    _write_profiles(tmp_path / "data", num_nodes=None)
    manifest = load_manifest(
        _manifest(
            tmp_path,
            "cases: [{title: a, speedup_x: nodes, runs: ['data/np*.h5']}]\n",
        ),
    )

    with pytest.raises(SiteError, match="case 'a'"):
        build_site(manifest, tmp_path / "docs")


def test_an_unreadable_profile_names_its_case_and_file(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "broken.h5").write_text("not hdf5")
    manifest = load_manifest(
        _manifest(tmp_path, "cases: [{title: a, runs: [data/broken.h5]}]\n"),
    )

    with pytest.raises(SiteError, match=r"case 'a': cannot read .*broken\.h5"):
        build_site(manifest, tmp_path / "docs")


def test_site_command_writes_the_project(tmp_path, capsys):
    _write_profiles(tmp_path / "data")
    manifest = _manifest(tmp_path, BASIC)
    out = tmp_path / "docs"

    assert cli_main(["site", str(manifest), "-o", str(out)]) == 0

    assert (out / CONTENT / "index.mdx").is_file()
    assert "2 case(s), 4 run(s)" in capsys.readouterr().out


def test_site_command_takes_the_base_path_from_the_command_line(tmp_path):
    _write_profiles(tmp_path / "data")
    manifest = _manifest(tmp_path, BASIC, base=None)
    out = tmp_path / "docs"

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["site", str(manifest), "-o", str(out)])
    assert exc_info.value.code == 2

    assert cli_main(["site", str(manifest), "-o", str(out), "--base", "/x"]) == 0
    assert "(/x/catalogue/single-run/)" in (out / CONTENT / "index.mdx").read_text()


def test_site_command_check_only_validates(tmp_path, capsys):
    _write_profiles(tmp_path / "data")
    manifest = _manifest(tmp_path, BASIC)
    out = tmp_path / "docs"

    assert cli_main(["site", str(manifest), "-o", str(out), "--check"]) == 0

    assert "ok" in capsys.readouterr().out
    assert not out.exists()


def test_site_command_reports_a_bad_manifest_as_a_usage_error(tmp_path, capsys):
    manifest = _manifest(tmp_path, "cases: []\n")

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["site", str(manifest)])

    assert exc_info.value.code == 2
    assert "scope-profiler site: error:" in capsys.readouterr().err
