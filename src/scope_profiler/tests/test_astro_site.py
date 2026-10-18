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
    assert (content / "solvers" / "poisson-solver.mdx").is_file()
    for page in ("np1.mdx", "np2.mdx", "np4.mdx"):
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
    case = _data(out, "solvers", "poisson-solver.data.json")
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

    case = (out / CONTENT / "solvers" / "poisson-solver.mdx").read_text()
    assert "Tags: `cpu`, `scaling`" in case
    for section in ("RunsTable", "CompareControls", "CompareTable", "Charts"):
        assert f"<{section} data={{data}} />" in case
    data = _data(out, "solvers", "poisson-solver.data.json")
    report = data["report"]
    assert report["labels"] == ["np1", "np2", "np4"]
    assert [row["label"] for row in report["runs"]["rows"]] == ["np1", "np2", "np4"]
    assert report["runs"]["rows"][0]["change"] == {"kind": "base"}
    assert len(report["candidates"]) == 2
    assert {row["name"] for row in report["table"]["rows"]} >= {"setup", "solve"}
    assert "Speedup" in [panel["title"] for panel in data["charts"]["panels"]]
    # A single run has nothing to compare.
    single = (out / CONTENT / "single-run.mdx").read_text()
    assert "CompareTable" not in single and "CompareControls" not in single
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

    table = _data(out, "a.data.json")["report"]["table"]
    assert table["per_rank"] is True
    solve = next(row for row in table["rows"] if row["name"] == "solve")
    # 'solve' lasts 50 ms on one rank and 25 ms on each of two.
    assert [cell["values"]["total"] for cell in solve["cells"]] == ["50 ms", "25 ms"]


def test_the_sidebar_lists_each_case_once_and_not_its_runs(tmp_path):
    """A study can have hundreds of runs: the case page lists them, the sidebar does not."""
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    case = (out / CONTENT / "solvers" / "poisson-solver.mdx").read_text()
    assert 'sidebar:\n  label: "Poisson solver"\n  order: 1\n' in case
    single = (out / CONTENT / "single-run.mdx").read_text()
    assert 'label: "Single run"\n  order: 2\n' in single
    for run in ("np1", "np2", "np4"):
        text = (out / CONTENT / "solvers" / "poisson-solver" / f"{run}.mdx").read_text()
        assert "sidebar:\n  hidden: true\n" in text or "  hidden: true" in text
    assert "hidden" not in case
    # The catalogue's own page leads the catalogue's sidebar group.
    assert (
        'sidebar:\n  label: "Overview"\n  order: 0\n'
        in (out / CONTENT / "index.mdx").read_text()
    )


def test_run_pages_link_to_the_runs_either_side(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    folder = out / CONTENT / "solvers" / "poisson-solver"
    first = (folder / "np1.mdx").read_text()
    middle = (folder / "np2.mdx").read_text()
    last = (folder / "np4.mdx").read_text()
    base = "/cat/catalogue/solvers/poisson-solver"
    assert f"[np2 ›]({base}/np2/)" in first and "‹" not in first
    assert f"[‹ np1]({base}/np1/)" in middle and f"[np4 ›]({base}/np4/)" in middle
    assert f"[‹ np2]({base}/np2/)" in last and "›]" not in last


def test_the_comparison_data_lets_the_browser_pick_any_pair(tmp_path):
    _write_profiles(tmp_path / "data", ranks=(1, 2, 4, 8, 16, 32))
    out = tmp_path / "docs"

    build_site(
        load_manifest(
            _manifest(tmp_path, "cases: [{title: a, runs: ['data/np*.h5']}]\n")
        ),
        out,
    )

    report = _data(out, "a.data.json")["report"]
    dataset = report["dataset"]
    assert [run["label"] for run in dataset["runs"]] == [
        f"np{n}" for n in (1, 2, 4, 8, 16, 32)
    ]
    assert dataset["per_rank"] is True
    assert set(dataset["regions"]) >= {"path:setup", "path:solve"}
    run = dataset["runs"][5]
    assert run["wall"] > 0 and run["order"]
    key = next(iter(run["values"]))
    total, own, _avg, calls = run["values"][key]
    assert total >= own >= 0 and calls >= 1
    # The report's own comparison covers only the first few runs, as the reference the
    # page's computation is checked against.
    assert len(report["candidates"]) == 4
    assert len(report["table"]["labels"]) == 5


def test_user_text_cannot_break_the_mdx(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    index = (out / CONTENT / "index.mdx").read_text()
    assert r"A \{tricky\} \<description\> with \$math." in index
    # What the manifest says about the case is data, which the page draws as text.
    case = (out / CONTENT / "solvers" / "poisson-solver.mdx").read_text()
    assert "{braces}" not in case and "<tags>" not in case
    info = _data(out, "solvers", "poisson-solver.data.json")["info"]
    assert ["note", "uses {braces} and <tags>"] in info
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
    runs = _data(out, "solvers", "poisson-solver.data.json")["report"]["runs"]
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


def test_a_case_page_draws_the_rank_charts_over_its_runs(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(
        load_manifest(
            _manifest(tmp_path, "cases: [{title: a, runs: ['data/np*.h5']}]\n")
        ),
        out,
    )

    panels = _data(out, "a.data.json")["charts"]["panels"]
    titles = [panel["title"] for panel in panels]
    assert "Rank imbalance" in titles and "Rank heatmap" in titles
    # The durations of several runs are grouped; there is nothing to stack.
    durations = next(p for p in panels if p["title"] == "Region durations")
    assert durations["stack"] is None


def test_a_runs_durations_chart_is_written_both_stacked_and_grouped(tmp_path):
    _write_profiles(tmp_path / "data", ranks=(2,))
    out = tmp_path / "docs"

    build_site(
        load_manifest(
            _manifest(tmp_path, "cases: [{title: a, runs: [data/np2.h5]}]\n")
        ),
        out,
    )

    panels = _data(out, "a", "np2.data.json")["charts"]["panels"]
    durations = next(p for p in panels if p["title"] == "Region durations")
    stack = durations["stack"]
    assert stack["stacked"] is True
    assert durations["src"] == stack["stacked_src"]
    segments = {}
    for kind in ("stacked_src", "flat_src"):
        data = _strict_json((out / "public" / stack[kind].lstrip("/")).read_text())
        segments[kind] = {bar.get("segment") for bar in data["bars"]}
    assert "self" in segments["stacked_src"] and segments["flat_src"] == {None}


def test_the_case_info_is_metadata_not_a_table_of_its_own(tmp_path):
    _write_profiles(tmp_path / "data")
    out = tmp_path / "docs"

    build_site(load_manifest(_manifest(tmp_path, BASIC)), out)

    case = (out / CONTENT / "solvers" / "poisson-solver.mdx").read_text()
    assert "## Information" not in case
    assert "## Metadata\n\n<Metadata data={data} rows={data.info} />" in case
    info = _data(out, "solvers", "poisson-solver.data.json")["info"]
    assert info[0] == ["commit", "843a82cd"]
    # Each run's own metadata begins with it.
    run = _data(out, "solvers", "poisson-solver", "np2.data.json")["report"]["metadata"]
    assert run[: len(info)] == info
    assert ["timestamp", "2026-10-17T10:00:00Z"] in run
    # A case with no info has no section for it.
    single = (out / CONTENT / "single-run.mdx").read_text()
    assert "Metadata" not in single


def _attach(tmp_path, runs_yaml):
    _write_profiles(tmp_path / "data")
    (tmp_path / "params").mkdir(exist_ok=True)
    (tmp_path / "params" / "np1.yml").write_text("steps: 10\n")
    (tmp_path / "params" / "np2.yml").write_text("steps: 20\n")
    (tmp_path / "run.py").write_text("print('hello')\n")
    manifest = load_manifest(_manifest(tmp_path, runs_yaml))
    out = tmp_path / "docs"
    build_site(manifest, out)
    return out


def test_files_are_attached_to_runs_and_shown_on_their_pages(tmp_path):
    out = _attach(
        tmp_path,
        """
cases:
  - title: a
    files: [run.py]
    runs:
      - {path: data/np1.h5, files: ["params/{stem}.yml"]}
      - {path: data/np2.h5, files: ["params/{stem}.yml"]}
""",
    )

    first = _data(out, "a", "np1.data.json")["files"]
    second = _data(out, "a", "np2.data.json")["files"]
    # The case's file goes with every run, then the run's own (here by its name).
    assert [f["name"] for f in first] == ["run.py", "np1.yml"]
    assert [f["name"] for f in second] == ["run.py", "np2.yml"]
    parameters = second[1]
    assert parameters["text"] == "steps: 20\n"
    assert parameters["language"] == "yaml" and first[0]["language"] == "python"
    assert parameters["url"] == "/catalogue/a/np2/files/np2.yml"
    # The file itself is there to download.
    assert (
        out / "public" / "catalogue" / "a" / "np2" / "files" / "np2.yml"
    ).read_text() == ("steps: 20\n")
    page = (out / CONTENT / "a" / "np2.mdx").read_text()
    assert "## Files\n\n<Files data={data} />" in page
    counts = _data(out, "a.data.json")["links"]["files"]
    assert counts == [2, 2]


def test_a_run_without_files_has_no_files_section(tmp_path):
    out = _attach(tmp_path, "cases: [{title: a, runs: [data/np1.h5]}]\n")

    assert _data(out, "a", "np1.data.json")["files"] == []
    assert "Files" not in (out / CONTENT / "a" / "np1.mdx").read_text()


def test_a_large_or_binary_file_is_offered_for_download_only(tmp_path):
    from scope_profiler.astro_site import MAX_SHOWN_FILE_BYTES

    _write_profiles(tmp_path / "data")
    (tmp_path / "big.txt").write_text("x" * (MAX_SHOWN_FILE_BYTES + 1))
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)))
    out = tmp_path / "docs"

    build_site(
        load_manifest(
            _manifest(
                tmp_path,
                "cases: [{title: a, runs: [{path: data/np1.h5, files: [big.txt, blob.bin]}]}]\n",
            )
        ),
        out,
    )

    big, blob = _data(out, "a", "np1.data.json")["files"]
    assert big["text"] is None and big["reason"] == "too large to show"
    assert blob["text"] is None and blob["reason"] == "not a text file"
    assert (
        out / "public" / "catalogue" / "a" / "np1" / "files" / "big.txt"
    ).stat().st_size == (MAX_SHOWN_FILE_BYTES + 1)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("files: [missing.py]", "is not a file"),
        ("files: ['nothing*.py']", "no file matches"),
        ("files: 3", "must be a path or a list of paths"),
    ],
)
def test_attached_files_are_checked(tmp_path, entry, message):
    _write_profiles(tmp_path / "data")

    with pytest.raises(SiteError, match=message):
        load_manifest(
            _manifest(
                tmp_path,
                f"cases: [{{title: a, runs: [{{path: data/np1.h5, {entry}}}]}}]\n",
            )
        )


def test_two_attached_files_with_one_name_are_refused(tmp_path):
    _write_profiles(tmp_path / "data")
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / "params.yml").write_text("a: 1\n")
    (tmp_path / "two" / "params.yml").write_text("a: 2\n")
    manifest = load_manifest(
        _manifest(
            tmp_path,
            "cases: [{title: a, runs: [{path: data/np1.h5, files: [one/params.yml, two/params.yml]}]}]\n",
        )
    )

    with pytest.raises(SiteError, match="two attached files are called 'params.yml'"):
        build_site(manifest, tmp_path / "docs")


NIGHTLY = """
cases:
  - title: Nightly solve
    group: Solvers
    executions:
      - {id: night-2, date: "2026-10-02", info: {commit: bbb}, runs: [{path: "data/two/np*.h5"}]}
      - {id: night-1, date: "2026-10-01", info: {commit: aaa}, runs: [{path: "data/one/np*.h5"}]}
"""


def _nightly(tmp_path, text=NIGHTLY):
    _write_profiles(tmp_path / "data" / "one")
    _write_profiles(tmp_path / "data" / "two")
    out = tmp_path / "docs"
    result = build_site(load_manifest(_manifest(tmp_path, text)), out)
    return out, result


def test_executions_of_a_case_share_a_page_and_sidebar_entry(tmp_path):
    out, result = _nightly(tmp_path)

    content = out / CONTENT / "solvers"
    # The latest execution (by date, not by listing) is the case's own page.
    latest = _data(out, "solvers", "nightly-solve.data.json")
    assert [row["id"] for row in latest["executions"]] == ["night-2", "night-1"]
    assert [row["current"] for row in latest["executions"]] == [True, False]
    assert latest["executions"][0]["link"] == "/cat/catalogue/solvers/nightly-solve/"
    assert (content / "nightly-solve" / "night-2" / "np1.mdx").is_file()
    # An earlier execution has a page of its own, out of the sidebar.
    earlier = (content / "nightly-solve" / "night-1.mdx").read_text()
    assert "hidden: true" in earlier
    assert "hidden" not in (content / "nightly-solve.mdx").read_text()
    assert (content / "nightly-solve" / "night-1" / "np4.mdx").is_file()
    assert result.runs == 6
    # The page of an execution says which one it is.
    rows = dict(_data(out, "solvers", "nightly-solve", "night-1.data.json")["info"])
    assert rows["commit"] == "aaa" and rows["execution"] == "night-1"
    # Back from a run of the latest execution goes to the case's page.
    run_page = (content / "nightly-solve" / "night-2" / "np1.mdx").read_text()
    assert "(/cat/catalogue/solvers/nightly-solve/)" in run_page


def test_executions_are_charted_over_time(tmp_path):
    out, _ = _nightly(tmp_path)

    charts = _data(out, "solvers", "nightly-solve.data.json")["charts"]
    trend = charts["panels"][-1]
    assert trend["title"] == "Executions over time"
    plot = json.loads((out / "public" / trend["src"].lstrip("/")).read_text())
    assert plot["plot"] == "trend"
    assert [e["id"] for e in plot["executions"]] == ["night-1", "night-2"]
    assert [s["label"] for s in plot["series"]] == ["np1", "np2", "np4"]
    assert all(len(s["values"]) == 2 for s in plot["series"])


def test_index_counts_the_executions(tmp_path):
    out, _ = _nightly(tmp_path)

    index = json.loads((out / PUBLIC / "index.json").read_text())
    assert index["cases"][0]["executions"] == 2
    assert "Executions" in (out / CONTENT / "index.mdx").read_text()


@pytest.mark.parametrize(
    "text, message",
    [
        (
            "cases:\n  - {title: A, runs: [{path: x.h5}], executions: [{id: a}]}\n",
            "either 'runs' or 'executions'",
        ),
        ("cases:\n  - {title: A, executions: []}\n", "non-empty list"),
        (
            "cases:\n  - {title: A, executions: [{runs: []}]}\n",
            "'id', 'label' or 'date'",
        ),
        (
            "cases:\n  - {title: A, executions: [{id: a, runs: [x], bogus: 1}]}\n",
            "bogus",
        ),
    ],
)
def test_invalid_executions_are_refused(tmp_path, text, message):
    with pytest.raises(SiteError, match=message):
        load_manifest(_manifest(tmp_path, text))


def test_duplicate_execution_ids_are_refused(tmp_path):
    _write_profiles(tmp_path / "data")
    text = (
        "cases:\n  - title: A\n    executions:\n"
        "      - {id: a, runs: [{path: data/np1.h5}]}\n"
        "      - {id: a, runs: [{path: data/np2.h5}]}\n"
    )
    with pytest.raises(SiteError, match="used by another execution"):
        load_manifest(_manifest(tmp_path, text))


def _discovered(tmp_path, meta=True):
    for night, date in (("n1", "2026-10-01"), ("n2", "2026-10-02")):
        _write_profiles(tmp_path / "data" / night)
        if meta:
            (tmp_path / "data" / night / "meta.yml").write_text(
                f"date: {date}\ninfo: {{commit: c-{night}}}\nlabel: Night {night[1]}\n"
            )
    return (
        "cases:\n  - title: Found\n    executions:\n"
        '      path: "data/{execution}/np*.h5"\n'
    )


def test_executions_are_found_by_a_pattern_with_their_meta_file(tmp_path):
    text = _discovered(tmp_path)

    manifest = load_manifest(_manifest(tmp_path, text))

    executions = manifest.cases[0].executions
    assert [(e.id, e.label, e.date, e.info) for e in executions] == [
        ("n1", "Night 1", "2026-10-01", {"commit": "c-n1"}),
        ("n2", "Night 2", "2026-10-02", {"commit": "c-n2"}),
    ]
    assert [len(e.runs) for e in executions] == [3, 3]
    build_site(manifest, tmp_path / "docs")
    assert (tmp_path / "docs" / CONTENT / "found.mdx").is_file()


def test_found_executions_need_no_meta_file(tmp_path):
    text = _discovered(tmp_path, meta=False)

    executions = load_manifest(_manifest(tmp_path, text)).cases[0].executions

    assert [(e.id, e.label, e.date) for e in executions] == [
        ("n1", "n1", None),
        ("n2", "n2", None),
    ]


@pytest.mark.parametrize(
    "path, message",
    [
        ("data/np*.h5", "must contain {execution} once"),
        ("data/{execution}x/np*.h5", "whole directory name"),
        ("nothing/{execution}/np*.h5", "no directory matches"),
    ],
)
def test_a_bad_execution_pattern_is_refused(tmp_path, path, message):
    _discovered(tmp_path)
    text = f"cases:\n  - title: Found\n    executions:\n      path: {path!r}\n"
    with pytest.raises(SiteError, match=re.escape(message)):
        load_manifest(_manifest(tmp_path, text))


def test_a_meta_file_with_unknown_keys_is_refused(tmp_path):
    text = _discovered(tmp_path)
    (tmp_path / "data" / "n1" / "meta.yml").write_text("colour: red\n")
    with pytest.raises(SiteError, match="colour"):
        load_manifest(_manifest(tmp_path, text))


FIGURES = """
cases:
  - title: With extras
    images: [figs/case.png]
    code:
      - {path: src/solver.py, url: "https://example.org/solver.py"}
      - https://example.org/repo
      - {url: "https://example.org/other", label: Other code}
    runs:
      - {path: data/np1.h5, images: ["figs/run-{stem}.svg", figs/doc.pdf]}
      - path: data/np2.h5
"""


def _with_extras(tmp_path):
    _write_profiles(tmp_path / "data")
    (tmp_path / "figs").mkdir()
    for name in ("case.png", "run-np1.svg", "doc.pdf"):
        (tmp_path / "figs" / name).write_bytes(b"x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "solver.py").write_text("print('hi')\n")
    out = tmp_path / "docs"
    build_site(load_manifest(_manifest(tmp_path, FIGURES)), out)
    return out


def test_runs_and_cases_show_images_and_pdfs(tmp_path):
    out = _with_extras(tmp_path)

    run = _data(out, "with-extras", "np1.data.json")
    assert [(f["name"], f["pdf"]) for f in run["images"]] == [
        ("run-np1.svg", False),
        ("doc.pdf", True),
    ]
    for figure in run["images"]:
        assert (out / "public" / figure["url"].lstrip("/")).is_file()
    assert "<Gallery" in (out / CONTENT / "with-extras" / "np1.mdx").read_text()
    # A run without figures has no section.
    assert _data(out, "with-extras", "np2.data.json")["images"] == []
    assert "Gallery" not in (out / CONTENT / "with-extras" / "np2.mdx").read_text()
    case = _data(out, "with-extras.data.json")
    assert [f["name"] for f in case["images"]] == ["case.png"]


def test_a_case_ends_with_its_code_and_links(tmp_path):
    out = _with_extras(tmp_path)

    code = _data(out, "with-extras.data.json")["code"]
    assert [(c["name"], c["link"], c["url"] is not None) for c in code] == [
        ("solver.py", "https://example.org/solver.py", True),
        ("https://example.org/repo", "https://example.org/repo", False),
        ("Other code", "https://example.org/other", False),
    ]
    assert code[0]["text"] == "print('hi')\n"
    page = (out / CONTENT / "with-extras.mdx").read_text()
    assert page.rstrip().endswith("<Files data={data} files={data.code} />")


@pytest.mark.parametrize(
    "entry, message",
    [
        ("images: [notes.txt]", "not an image or PDF"),
        ("code: [{label: x}]", "'path' or 'url' is required"),
        ("code: [{url: ftp://x}]", "must start with http"),
        ("code: [{path: nothing.py}]", "is not a file"),
    ],
)
def test_invalid_figures_and_code_are_refused(tmp_path, entry, message):
    _write_profiles(tmp_path / "data")
    (tmp_path / "notes.txt").write_text("x")
    text = f"cases:\n  - title: A\n    {entry}\n    runs: [data/np1.h5]\n"
    with pytest.raises(SiteError, match=re.escape(message)):
        load_manifest(_manifest(tmp_path, text))


def test_the_trend_follows_a_run_by_its_size_when_labels_drift(tmp_path):
    _write_profiles(tmp_path / "data" / "one")
    _write_profiles(tmp_path / "data" / "two")
    text = (
        "cases:\n  - title: Drift\n    executions:\n"
        "      - id: a\n        date: '2026-10-01'\n        runs:\n"
        + "".join(
            f"          - {{path: data/one/np{n}.h5, label: run{i}}}\n"
            for i, n in enumerate((1, 2, 4), start=1)
        )
        + "      - id: b\n        date: '2026-10-02'\n        runs:\n"
        + "".join(
            f"          - {{path: data/two/np{n}.h5, label: '{n} ranks'}}\n"
            for n in (1, 2, 4)
        )
    )
    out = tmp_path / "docs"
    build_site(load_manifest(_manifest(tmp_path, text)), out)

    panel = _data(out, "drift.data.json")["charts"]["panels"][-1]
    plot = json.loads((out / "public" / panel["src"].lstrip("/")).read_text())
    # One line per size, named after the newest execution, with a point in each.
    assert [s["label"] for s in plot["series"]] == ["1 ranks", "2 ranks", "4 ranks"]
    assert all(None not in s["values"] for s in plot["series"])
