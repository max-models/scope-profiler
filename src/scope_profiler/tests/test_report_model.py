"""The models behind the HTML report and the catalogue pages: what a report says, as data."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scope_profiler.report_data import _rank_balance
from scope_profiler.report_model import (
    change_model,
    comparison_page_model,
    deviation,
    hotspots_model,
    line_profile_model,
    matrix_model,
    report_rows,
    run_page_model,
    summary_model,
)
from scope_profiler.tests.test_html_report import (
    _comparison_runs,
    _line_profile_results,
    _nested_results,
    _ranked_results,
)


def _text(parts) -> str:
    """The plain text of a model's parts, a region shown as its name."""
    return "".join(
        part if isinstance(part, str) else part.get("region", part.get("text"))
        for part in parts
    )


def _summary(results):
    from scope_profiler.report_data import _hotspot_base, _hotspot_entries

    rows = report_rows(results, None, None, None, "start")
    entries = _hotspot_entries(rows)
    return summary_model(
        results,
        rows,
        entries,
        _hotspot_base(rows, entries),
        _rank_balance(results, rows, None),
    )


def _as_json(model):
    def default(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return json.loads(json.dumps(model, default=default))


def test_summary_names_the_largest_hotspot_with_a_region_part():
    summary = _summary(_nested_results())

    assert summary["empty"] is False
    labels = [kpi["label"] for kpi in summary["kpis"]]
    assert labels[:3] == ["Wall time", "Ranks", "Regions"]
    # The loop's own time, outside the calls it makes, is the largest leaf of the tree.
    top = next(kpi for kpi in summary["kpis"] if kpi["label"] == "Top hotspot")
    assert top["value"] == [{"region": "loop"}]
    first = summary["findings"][0]
    assert first["tone"] == "info"
    assert {"region": "loop"} in first["parts"]
    text = _text(first["parts"])
    assert text.startswith("Time spent directly in loop")
    assert "is the largest hotspot" in text


def test_summary_flags_an_unbalanced_region_and_links_to_the_section():
    summary = _summary(_ranked_results([2, 30, 5, 8]))

    imbalance = next(kpi for kpi in summary["kpis"] if kpi["label"] == "Load imbalance")
    assert imbalance["tone"] == "warn"
    warning = next(f for f in summary["findings"] if f["tone"] == "warn")
    assert {"anchor": "balance", "text": "load balance"} in warning["parts"]
    assert "is unevenly distributed across ranks" in _text(warning["parts"])


def test_summary_of_nothing_is_empty():
    from scope_profiler.results import ProfilingResults

    empty = ProfilingResults({})

    assert summary_model(empty, [], [], 0.0, None) == {
        "empty": True,
        "kpis": [],
        "findings": [],
    }


def test_hotspots_need_two_entries_to_rank():
    assert hotspots_model([], 1.0) == {"items": []}
    one = [
        {
            "name": "a",
            "context": [],
            "kind": "leaf",
            "time": 1.0,
            "calls": 1,
            "imbalance": None,
        }
    ]
    assert hotspots_model(one, 1.0) == {"items": []}


def test_hotspots_carry_share_details_and_a_flag_for_slow_ranks():
    entries = [
        {
            "name": "a",
            "context": ["outer"],
            "kind": "leaf",
            "time": 0.6,
            "calls": 3,
            "imbalance": 40.0,
        },
        {
            "name": "b",
            "context": [],
            "kind": "own",
            "time": 0.2,
            "calls": 1,
            "imbalance": None,
        },
    ]

    items = hotspots_model(entries, 1.0)["items"]

    assert [item["rank"] for item in items] == [1, 2]
    assert items[0]["share"] == pytest.approx(60.0)
    assert items[0]["details"] == ["600 ms", "3 calls", "200 ms/call"]
    assert items[0]["flag"] == "slowest rank +40%"
    assert items[0]["title"] == "outer › a"
    assert items[1]["flag"] is None


@pytest.mark.parametrize(
    ("base", "value", "kind", "tone"),
    [
        (None, None, "none", None),
        (None, 2.0, "new", None),
        (2.0, None, "gone", None),
        (0.0, 3.0, "new", None),
        (10.0, 5.0, "delta", "faster"),
        (10.0, 20.0, "delta", "slower"),
        (10.0, 10.1, "delta", "same"),
    ],
)
def test_change_model_classifies_a_change(base, value, kind, tone):
    change = change_model(base, value, "total")

    assert change["kind"] == kind
    assert change.get("tone") == tone


def test_change_model_reports_size_percent_and_speed():
    change = change_model(10.0, 5.0, "total")

    assert change["pct"] == pytest.approx(-50.0)
    assert change["label"] == "−50.0%"
    assert change["title"] == "−5 s · 2.00× faster"
    assert change["size"] == pytest.approx(5.0)
    # Call counts are neither faster nor slower.
    calls = change_model(100, 150, "calls")
    assert calls["tone"] == "neutral" and calls["title"] == "+50 calls"


def test_deviation_shades_only_what_is_beyond_the_noise():
    assert deviation(1.0, 1.0) is None
    assert deviation(1.01, 1.0) is None
    assert deviation(1.5, 1.0) == {"sign": 1, "alpha": 0.55}
    assert deviation(0.5, 1.0) == {"sign": -1, "alpha": 0.55}
    assert deviation(1.0, 0.0) is None


def test_matrix_model_shades_cells_against_the_regions_mean():
    results = _ranked_results([2, 30, 5, 8])
    rows = report_rows(results, None, None, None, "start")

    matrix = matrix_model(_rank_balance(results, rows, None))

    assert matrix["ranks"] == [0, 1, 2, 3]
    work = next(row for row in matrix["rows"] if row["name"] == "work")
    slowest = work["cells"][1]
    assert slowest["shade"]["sign"] == 1
    assert slowest["label"].startswith("+")
    assert slowest["title"].startswith("rank 1: ")
    assert matrix["totals"]["cells"][0]["rank"] == 0
    # Ranks that wait for each other spend the same time in regions overall, so none
    # of them is called the slowest.
    assert matrix["slowest_rank"] is None
    assert "ranks waiting for each other" in matrix["overall"]


def test_matrix_model_names_the_rank_that_is_slowest_overall():
    values = np.array([1.0, 1.0, 3.0])
    balance = {
        "ranks": [0, 1, 2],
        "own": True,
        "regions": [
            {"name": "a", "values": values, "mean": float(values.mean())},
        ],
        "totals": values,
    }

    matrix = matrix_model(balance)

    assert matrix["slowest_rank"] == 2
    assert matrix["overall"].startswith("Rank 2, in red in the header")


def test_line_profile_model_lists_every_line_and_marks_the_hottest(tmp_path):
    results = _line_profile_results(
        tmp_path,
        [
            (
                "alpha",
                "def alpha():\n    a = 1\n    # note\n    return a\n",
                [(2, 5.0), (4, 700.0)],
            )
        ],
    )
    from scope_profiler.report_data import _line_profile_functions

    entry = _line_profile_functions(results, None)[0]

    model = line_profile_model(entry)

    assert model["has_source"] is True
    assert [row["number"] for row in model["rows"]] == [1, 2, 3, 4]
    assert [row["recorded"] for row in model["rows"]] == [False, True, False, True]
    assert [row["hot"] for row in model["rows"] if row["recorded"]] == [False, True]
    assert model["rows"][3]["percent"] == pytest.approx(100 * 700 / 705)
    assert model["rows"][3]["seconds_text"].endswith("µs")
    assert 'class="tk-kw"' in model["rows"][0]["source"]


def test_run_page_model_has_every_section_and_is_json_serializable():
    results = _ranked_results([2, 30, 5, 8], extra_regions=2)

    model = _as_json(run_page_model(results))

    assert model["kind"] == "run"
    assert model["summary"]["kpis"] and model["hotspots"]["items"]
    assert model["balance"]["rows"] and model["matrix"]["rows"]
    assert model["line_profiles"] is None and model["hardware"] == []
    # Every region's first row has an id the links point at.
    ids = [g["id"] for g in model["regions"]["groups"] if g["id"]]
    assert sorted(ids) == sorted(model["region_ids"].values())
    assert set(model["region_ids"]) == {g["name"] for g in model["regions"]["groups"]}


def test_run_page_model_can_hold_back_host_and_metadata_fields():
    results = _ranked_results([1, 1])
    results._metadata = {
        "hostname": "secret",
        "chip_information": "cpu",
        "timestamp": "2026-01-01T00:00:00",
    }

    model = run_page_model(
        results, host_keys=("chip_information",), metadata_keys=("timestamp",)
    )

    assert model["meta"]["host"] == ["cpu"]
    assert model["metadata"] == [["timestamp", "2026-01-01T00:00:00"]]


def test_comparison_page_model_compares_against_the_baseline(tmp_path):
    baseline, candidate = _comparison_runs(tmp_path)

    model = _as_json(comparison_page_model([baseline, candidate]))

    assert model["kind"] == "comparison"
    assert model["labels"] == ["base", "cand"]
    (item,) = model["candidates"]
    assert item["label"] == "cand" and item["base_label"] == "base"
    assert {change["name"] for change in item["improvements"]} >= {"old", "kernel"}
    assert [change["name"] for change in item["regressions"]][:1] == ["io"]
    names = {row["name"]: row for row in model["table"]["rows"]}
    # A region only the candidate has is new; one only the baseline has is gone.
    assert names["fresh"]["cells"][1]["changes"]["total"]["kind"] == "new"
    assert names["old"]["cells"][1]["changes"]["total"]["kind"] == "gone"
    assert names["kernel"]["cells"][1]["changes"]["total"]["tone"] == "faster"


def test_a_comparison_leaves_the_rank_charts_out_unless_asked():
    from scope_profiler.report_data import collect_chart_payloads

    runs = [_ranked_results([2, 8, 5, 9], file_path=f"run{n}.h5") for n in (1, 2)]

    default, _ = collect_chart_payloads(runs, None, None, None, comparison=True)
    asked, _ = collect_chart_payloads(
        runs, None, None, None, comparison=True, rank_charts=True
    )

    assert "Rank imbalance" not in [chart[0] for chart in default]
    titles = [chart[0] for chart in asked]
    assert "Rank imbalance" in titles and "Rank heatmap" in titles
    heatmap = next(chart for chart in asked if chart[0] == "Rank heatmap")
    # Every point says which run it is of, so a page can show only the runs it compares.
    assert {point["file"] for point in heatmap[1]["points"]} == {"run1", "run2"}


def test_a_single_runs_durations_chart_can_also_be_drawn_the_other_way(tmp_path):
    from scope_profiler.report_data import collect_chart_payloads
    from scope_profiler.report_model import chart_panels

    results = _nested_results()

    plain, _ = collect_chart_payloads([results], None, None, None)
    toggled, _ = collect_chart_payloads([results], None, None, None, stack_toggle=True)

    durations = next(chart for chart in toggled if chart[0] == "Region durations")
    assert durations[2]["stacked"] is True
    segments = {bar.get("segment") for bar in durations[1]["bars"]}
    flat = {bar.get("segment") for bar in durations[2]["alternate"]["bars"]}
    assert "self" in segments and flat == {None}
    # Without the option the chart has no alternative, and the panel says so.
    assert "alternate" not in next(c for c in plain if c[0] == "Region durations")[2]
    panel = next(p for p in chart_panels(toggled) if p["title"] == "Region durations")
    assert panel["stack"]["stacked"] is True and panel["stack"]["alternate"]
    assert "alternate" not in panel["options"]
    assert (
        next(p for p in chart_panels(plain) if p["title"] == "Region durations")[
            "stack"
        ]
        is None
    )


def test_several_runs_durations_have_no_stack_alternative():
    """The browser's chart builder cannot stack the segments of several runs."""
    from scope_profiler.report_data import collect_chart_payloads

    runs = [_ranked_results([2, 8, 5, 9], file_path=f"run{n}.h5") for n in (1, 2)]

    charts, _ = collect_chart_payloads(
        runs, None, None, None, comparison=True, stack_toggle=True
    )

    durations = next(chart for chart in charts if chart[0] == "Region durations")
    assert "alternate" not in durations[2]
