"""Regression checks for the nested summary table's public output."""

import re
from io import StringIO

import numpy as np
import pytest

from scope_profiler import MPIRegion, ProfilingResults, Region
from scope_profiler.summary import print_region_table, region_rows


@pytest.fixture
def nested_results():
    def region(name, starts, ends):
        return MPIRegion(name, {0: Region(np.array(starts), np.array(ends))})

    return ProfilingResults(
        {
            "scope_profiler.session": region(
                "scope_profiler.session", [0], [10_000_000_000]
            ),
            "work": region("work", [1_000_000_000], [9_000_000_000]),
            "leaf": region(
                "leaf", [2_000_000_000, 5_000_000_000], [3_000_000_000, 7_000_000_000]
            ),
        }
    )


def test_summary_frame_precision_and_own_rows(nested_results):
    stream = StringIO()
    print_region_table(
        region_rows(nested_results), stream=stream, suppress_notes=True, total_time=10
    )
    output = stream.getvalue()
    lines = output.rstrip().splitlines()
    assert lines[0].strip().startswith("╭")
    assert lines[-1].strip().startswith("╰")
    assert len({len(line) for line in lines}) == 1
    header = lines[1]
    assert (
        header.index("region") < header.index("% session") < header.index("total [s]")
    )
    assert header.count("│") == 2
    assert "avg" not in header and "% parent" not in header
    assert "TOTAL" not in output and "(1x)" not in output
    assert "leaf (2x)" in output
    data = lines[3:-1]
    assert len(data) == 5
    expected = [
        ("100.00%", "10.000000"),
        ("20.00%", "2.000000"),
        ("80.00%", "8.000000"),
        ("50.00%", "5.000000"),
        ("30.00%", "3.000000"),
    ]
    for line, (percent, duration) in zip(data, expected):
        assert re.findall(r"\d+\.\d+%?", line) == [percent, duration]
    assert "(own)" in data[1] and "(own)" in data[3]
    # Numeric indentation follows depth, but contains no tree glyphs.
    for column, values in (
        ("% session", [item[0] for item in expected]),
        ("total [s]", [item[1] for item in expected]),
    ):
        start = header.index(column)
        for line, value, indent in zip(data, values, (0, 3, 3, 5, 5)):
            assert line[start : line.index(value)] == " " * indent


@pytest.mark.parametrize("filters", [{"include": "work|leaf"}, {"exclude": "leaf"}])
def test_summary_filter_preserves_exclusive_time(nested_results, filters):
    rows = region_rows(nested_results, **filters)
    work = next(row for row in rows if row["name"] == "work")
    assert work["exclusive"] == pytest.approx(5)


def test_explicit_average_column_includes_own_average(nested_results):
    stream = StringIO()
    print_region_table(
        region_rows(nested_results),
        stream=stream,
        suppress_notes=True,
        columns=["region", "avg"],
    )
    own_lines = [line for line in stream.getvalue().splitlines() if "(own)" in line]
    assert "2.000e+00" in own_lines[0]
    assert "5.000e+00" in own_lines[1]
