import numpy as np
import pytest

from scope_profiler.__main__ import main
from scope_profiler.h5writer import ProfilingWriter
from scope_profiler.profile_manager import RankPayload


def test_line_profile_cli_prints_persisted_records(tmp_path, capsys):
    path = tmp_path / "profile.h5"
    record = {
        "region": "solve",
        "filename": "app.py",
        "function": "solve",
        "first_lineno": 10,
        "line_numbers": np.asarray([11, 12]),
        "hits": np.asarray([1, 5]),
        "times": np.asarray([10.0, 25.0]),
        "unit": 1e-9,
    }
    payload = RankPayload(
        regions={"solve": (np.asarray([0]), np.asarray([1]))},
        likwid={},
        likwid_environment={},
        line_profile=[record],
    )
    with ProfilingWriter(path) as writer:
        writer.write_rank(0, payload)

    assert main(["line-profile", str(path), "--function", "solve"]) == 0
    output = capsys.readouterr().out
    assert "Rank 0 | solve | solve (app.py:10)" in output
    assert "│ 11" in output
    assert "1e-08" in output
    assert "% time" in output
    assert "28.57" in output


def test_line_profile_is_listed_in_top_level_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "line-profile" in capsys.readouterr().out
