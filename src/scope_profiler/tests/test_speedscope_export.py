import json

import pytest

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.post_processing import main
from scope_profiler.speedscope_export import (
    build_speedscope_document,
    export_speedscope,
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
    """Build the call list the document builder expects, in seconds."""
    return [
        {"name": name, "start": start, "end": end, "parent": parent}
        for name, start, end, parent in specs
    ]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _replay(profile, frames):
    """Replay an evented profile, returning (frame name, start, end) per call.

    Raises if the events are not a balanced, correctly ordered stack machine —
    which is what speedscope requires and refuses to import without.
    """
    stack = []
    closed = []
    last_at = float("-inf")
    for event in profile["events"]:
        assert event["at"] >= last_at, "events must be ordered by timestamp"
        last_at = event["at"]
        if event["type"] == "O":
            stack.append((frames[event["frame"]]["name"], event["at"]))
        else:
            assert stack, "close event with nothing open"
            name, start = stack.pop()
            assert (
                frames[event["frame"]]["name"] == name
            ), "closed a frame that was not on top of the stack"
            closed.append((name, start, event["at"]))
    assert not stack, "profile ended with frames still open"
    return closed


def test_document_events_replay_as_a_call_stack():
    calls = _calls(
        ("main", 10.0, 11.0, None),
        ("setup", 10.0, 10.2, 0),
        ("solve", 10.2, 10.9, 0),
        ("assemble", 10.3, 10.6, 2),
    )

    document = build_speedscope_document([("rank 0", calls)], name="run")
    (profile,) = document["profiles"]

    # Timestamps are rebased on the first call, so the profile starts at zero.
    assert profile["startValue"] == pytest.approx(0.0)
    assert profile["endValue"] == pytest.approx(1.0)
    assert profile["unit"] == "seconds"
    assert profile["type"] == "evented"

    replayed = _replay(profile, document["shared"]["frames"])
    assert replayed == [
        ("setup", pytest.approx(0.0), pytest.approx(0.2)),
        ("assemble", pytest.approx(0.3), pytest.approx(0.6)),
        ("solve", pytest.approx(0.2), pytest.approx(0.9)),
        ("main", pytest.approx(0.0), pytest.approx(1.0)),
    ]


def test_document_clips_partial_overlap():
    # "long" is reconstructed as a child of "short" because it starts inside
    # it, even though it runs past its end: an evented profile cannot express
    # that, so the child is clipped to its parent.
    calls = _calls(("short", 0.0, 0.5, None), ("long", 0.1, 2.0, 0))

    document = build_speedscope_document([("rank 0", calls)], name="run")
    replayed = _replay(document["profiles"][0], document["shared"]["frames"])

    assert replayed == [
        ("long", pytest.approx(0.1), pytest.approx(0.5)),
        ("short", pytest.approx(0.0), pytest.approx(0.5)),
    ]


def test_document_handles_recursion():
    calls = _calls(("recurse", 0.0, 1.0, None), ("recurse", 0.2, 0.6, 0))

    document = build_speedscope_document([("rank 0", calls)], name="run")

    # The frame table is keyed by name, so recursion reuses one frame.
    assert document["shared"]["frames"] == [{"name": "recurse"}]
    assert len(_replay(document["profiles"][0], document["shared"]["frames"])) == 2


def test_export_writes_one_profile_per_rank(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, {rank: _nested_file_data()[0] for rank in (0, 1)})

    written = export_speedscope(
        ProfilingH5Reader(h5_file),
        tmp_path / "profile.speedscope.json",
        ranks=[0, 1],
        verbose=False,
    )

    assert written == [tmp_path / "profile.speedscope.json"]

    document = _load(written[0])
    assert document["$schema"] == "https://www.speedscope.app/file-format-schema.json"
    assert document["exporter"].startswith("scope-profiler@")
    assert document["activeProfileIndex"] == 0
    assert [profile["name"] for profile in document["profiles"]] == ["rank 0", "rank 1"]
    assert {frame["name"] for frame in document["shared"]["frames"]} == {
        "main",
        "setup",
        "solve",
        "assemble",
    }

    for profile in document["profiles"]:
        replayed = _replay(profile, document["shared"]["frames"])
        assert len(replayed) == 4
        main_call = next(call for call in replayed if call[0] == "main")
        assert main_call[2] - main_call[1] == pytest.approx(0.1)


def test_export_defaults_to_rank_zero_and_splits_per_file(tmp_path):
    file_one = tmp_path / "run_one.h5"
    file_two = tmp_path / "run_two.h5"
    _write_sample_h5(file_one, {rank: _nested_file_data()[0] for rank in (0, 1)})
    _write_sample_h5(file_two, {rank: _nested_file_data()[0] for rank in (0, 1)})

    readers = [ProfilingH5Reader(file_one), ProfilingH5Reader(file_two)]
    written = export_speedscope(
        readers, tmp_path / "profile.speedscope.json", verbose=False
    )

    assert [path.name for path in written] == [
        "profile_run_one.speedscope.json",
        "profile_run_two.speedscope.json",
    ]
    for path in written:
        document = _load(path)
        assert [profile["name"] for profile in document["profiles"]] == ["rank 0"]


def test_export_rejects_unknown_rank(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())

    with pytest.raises(ValueError, match="Invalid rank"):
        export_speedscope(
            ProfilingH5Reader(h5_file),
            tmp_path / "profile.speedscope.json",
            ranks=[3],
            verbose=False,
        )


def test_cli_export_speedscope_without_plots(tmp_path, capsys):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())
    out_dir = tmp_path / "figures"

    main(
        [
            str(h5_file),
            "-o",
            str(out_dir),
            "--export-speedscope",
            "--skip-plot-images",
        ]
    )

    speedscope_file = out_dir / "profile.speedscope.json"
    assert speedscope_file.exists()
    assert not list(out_dir.glob("*.png"))
    assert "speedscope.app" in capsys.readouterr().out

    document = _load(speedscope_file)
    assert {frame["name"] for frame in document["shared"]["frames"]} == {
        "main",
        "setup",
        "solve",
        "assemble",
    }


def test_cli_export_speedscope_alongside_plots(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())
    out_dir = tmp_path / "figures"

    main([str(h5_file), "-o", str(out_dir), "--export-speedscope", "--ranks", "0"])

    assert (out_dir / "profile.speedscope.json").exists()
    assert (out_dir / "flame_plot.png").exists()


def test_cli_export_speedscope_requires_output(tmp_path):
    h5_file = tmp_path / "profiling_data.h5"
    _write_sample_h5(h5_file, _nested_file_data())

    with pytest.raises(SystemExit):
        main([str(h5_file), "--export-speedscope"])
