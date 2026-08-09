"""The optional run label: setup(label=...) and how post-processing uses it."""

import json
from time import sleep

import pytest

from scope_profiler import ProfileManager, read_h5
from scope_profiler.plotting_scripts import collect_region_statistics


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


def _profile(out, label=None):
    ProfileManager.setup(file_path=str(out), label=label)
    with ProfileManager.profile_region("solve"):
        sleep(0.0001)
    ProfileManager.finalize(verbose=False)
    return read_h5(str(out))


def test_label_round_trips_through_the_file(tmp_path):
    """The label is metadata, so it survives into every post-processing step."""
    reader = _profile(tmp_path / "run.h5", label="128 ranks")

    assert reader.label == "128 ranks"
    assert reader.metadata["label"] == "128 ranks"
    assert reader.display_label == "128 ranks"


def test_without_a_label_the_file_stem_is_used(tmp_path):
    """The previous behaviour, unchanged for runs that set no label."""
    reader = _profile(tmp_path / "run.h5")

    assert reader.label is None
    assert "label" not in reader.metadata
    assert reader.display_label == "run"


def test_in_memory_results_carry_the_label(tmp_path):
    """finalize(return_results=True) reads it from the same metadata."""
    ProfileManager.setup(
        file_path=str(tmp_path / "run.h5"), flush_to_disk=False, label="in memory"
    )
    with ProfileManager.profile_region("solve"):
        sleep(0.0001)

    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results.label == "in memory"
    assert results.display_label == "in memory"


def test_label_names_the_run_in_charts_and_statistics(tmp_path):
    """Chart legends and the JSON statistics default to the label."""
    labelled = _profile(tmp_path / "a.h5", label="128 ranks")
    plain = _profile(tmp_path / "b.h5")

    payload = collect_region_statistics([labelled, plain])

    assert [f["label"] for f in payload["files"]] == ["128 ranks", "b"]


def test_label_leads_the_summary_heading(tmp_path, capsys):
    """Printed by finalize(), print_summary() and `scope-profiler pproc`."""
    reader = _profile(tmp_path / "run.h5", label="128 ranks")

    reader.print_summary()

    heading = capsys.readouterr().out.splitlines()[0]
    assert heading.startswith("128 ranks - ")
    assert "run.h5" in heading, "the path still identifies the file on disk"


def test_pproc_label_overrides_the_stored_one(tmp_path, capsys):
    """`scope-profiler pproc --label` renames runs for one report."""
    from scope_profiler.post_processing import main

    labelled = tmp_path / "a.h5"
    plain = tmp_path / "b.h5"
    _profile(labelled, label="stored")
    _profile(plain)
    output_dir = tmp_path / "figures"

    main(
        [
            str(labelled),
            str(plain),
            "-o",
            str(output_dir),
            "--summary",
            "--label",
            "128 ranks",
            "--label",
            "256 ranks",
            "--export-prof",
        ]
    )

    out = capsys.readouterr().out
    assert "128 ranks - " in out and "256 ranks - " in out
    assert "stored - " not in out

    payload = json.loads(
        (output_dir / "region_statistics.json").read_text(encoding="utf-8")
    )
    assert [f["label"] for f in payload["files"]] == ["128 ranks", "256 ranks"]

    # The label goes into exported filenames too, with the spaces made safe.
    assert (output_dir / "profile_128_ranks_rank0.prof").exists()

    # The file itself keeps what the run recorded.
    assert read_h5(str(labelled)).label == "stored"


def test_pproc_label_count_must_match_the_files(tmp_path):
    """Silently pairing them off by position would mislabel a whole report."""
    from scope_profiler.post_processing import main

    _profile(tmp_path / "a.h5")
    _profile(tmp_path / "b.h5")

    with pytest.raises(SystemExit):
        main(
            [
                str(tmp_path / "a.h5"),
                str(tmp_path / "b.h5"),
                "--summary",
                "--label",
                "only-one",
            ]
        )


def test_empty_label_is_treated_as_no_label(tmp_path):
    """So that `label=os.environ.get(...)` cannot produce a blank heading."""
    reader = _profile(tmp_path / "run.h5", label="")

    assert reader.label is None
    assert reader.display_label == "run"
