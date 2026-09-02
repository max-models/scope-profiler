"""Tests for the Chrome Trace / Perfetto exporter."""

import json

from scope_profiler import read_h5
from scope_profiler.chrome_trace_export import (
    build_chrome_trace_document,
    export_chrome_trace,
)
from scope_profiler.post_processing import export_main
from scope_profiler.tests.test_post_processing import _write_sample_h5


def _profile(tmp_path):
    path = tmp_path / "run.h5"
    _write_sample_h5(
        path,
        {
            0: {
                "main": ([0], [100 * 1_000_000]),
                "solve": ([20 * 1_000_000], [80 * 1_000_000]),
            }
        },
    )
    return read_h5(path)


def test_build_document_emits_process_thread_and_duration_events(tmp_path):
    document = build_chrome_trace_document(_profile(tmp_path), label="demo")

    assert document["metadata"]["name"] == "demo"
    events = document["traceEvents"]
    assert [event["ph"] for event in events[:2]] == ["M", "M"]
    durations = [event for event in events if event["ph"] == "X"]
    assert [(event["name"], event["ts"], event["dur"]) for event in durations] == [
        ("main", 0.0, 100_000.0),
        ("solve", 20_000.0, 60_000.0),
    ]
    assert {event["pid"] for event in events} == {0}
    assert {event["tid"] for event in durations} == {0}


def test_export_chrome_trace_writes_json(tmp_path):
    output = tmp_path / "out" / "profile.trace.json"
    written = export_chrome_trace(_profile(tmp_path), output, verbose=False)

    assert written == [output]
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["traceEvents"]
    assert document["traceEvents"][-1]["name"] == "solve"


def test_cli_export_chrome_trace(tmp_path, capsys):
    profile = tmp_path / "run.h5"
    _write_sample_h5(profile, {0: {"work": ([0], [1_000_000])}})

    export_main(["chrome-trace", str(profile), "-o", str(tmp_path / "out")])

    output = tmp_path / "out" / "profile.trace.json"
    assert output.exists()
    assert "ui.perfetto.dev" in capsys.readouterr().out
