import subprocess
import sys


def test_run_has_session_root(tmp_path):
    script = tmp_path / "s.py"
    script.write_text(
        "import scope_profiler as sp\nwith sp.region('work'):\n    pass\n",
    )
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "scope_profiler",
            "run",
            "-o",
            str(tmp_path / "o.h5"),
            str(script),
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "scope_profiler.session" in out
    assert "% session" in out
