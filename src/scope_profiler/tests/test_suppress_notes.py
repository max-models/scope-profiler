import subprocess
import sys

import scope_profiler as sp


def _session_output(capsys, tmp_path, **kwargs):
    with sp.session(file_path=str(tmp_path / "p.h5"), **kwargs):
        with sp.region("work"):
            pass
    return capsys.readouterr().out


def test_session_prints_notes_by_default(capsys, tmp_path):
    assert "Durations are in seconds" in _session_output(capsys, tmp_path)


def test_session_suppress_notes(capsys, tmp_path):
    out = _session_output(capsys, tmp_path, suppress_notes=True)
    assert "work" in out
    assert "Durations are in seconds" not in out


def test_run_suppress_notes(tmp_path):
    script = tmp_path / "s.py"
    script.write_text(
        "import scope_profiler as sp\nwith sp.region('work'):\n    pass\n"
    )
    cmd = [sys.executable, "-m", "scope_profiler", "run", "-o", str(tmp_path / "o.h5")]
    plain = subprocess.run(
        cmd + [str(script)], capture_output=True, text=True, check=False
    )
    quiet = subprocess.run(
        cmd + ["--suppress-notes", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "Durations are in seconds" in plain.stdout
    assert "work" in quiet.stdout
    assert "Durations are in seconds" not in quiet.stdout
