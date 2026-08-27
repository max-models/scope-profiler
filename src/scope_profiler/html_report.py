"""Self-contained HTML reports for profiling results.

The report deliberately has no plotting dependency: it is useful on a remote
machine immediately after a run, and can be opened locally in any browser.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

from scope_profiler.h5reader import read_h5
from scope_profiler.inspection import _json_safe
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import normalize_region_table_columns, region_rows

_STYLE = """
body { color: #1f2937; font: 15px/1.45 system-ui, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; }
h1, h2, h3 { color: #111827; } section { margin: 2rem 0; }
.chart { min-height: 360px; margin: 1rem 0 2rem; }
.facts { display: flex; flex-wrap: wrap; gap: .75rem; }
.fact { background: #f3f4f6; border-radius: .4rem; padding: .5rem .75rem; }
.overview { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: .5rem;
            padding: .25rem 1.25rem; }
.overview li { margin: .5rem 0; }
.overview .flag { color: #b45309; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; }
th, td { border-bottom: 1px solid #d1d5db; padding: .45rem .6rem; text-align: right; }
th { background: #f9fafb; position: sticky; top: 0; } th:first-child, td:first-child { text-align: left; }
details { margin: .75rem 0; } summary { cursor: pointer; font-weight: 600; }
.muted { color: #6b7280; } code { overflow-wrap: anywhere; }
.region-row { cursor: pointer; }
.region-row:hover { background: #f3f4f6; }
.region-row td:first-child { display: flex; align-items: center; gap: .4rem; }
.toggle-icon { display: inline-block; width: .9em; color: #6b7280; }
.bar-cell { position: relative; }
.bar { position: absolute; left: 0; top: .15rem; bottom: .15rem; background: #bfdbfe;
       border-radius: .2rem; z-index: 0; }
.bar-cell span { position: relative; z-index: 1; }
.region-detail td { background: #f9fafb; text-align: left; padding: .75rem 1.25rem; }
.region-detail pre { background: #111827; color: #e5e7eb; padding: .6rem .75rem;
                      border-radius: .4rem; overflow-x: auto; margin: .35rem 0 .9rem; }
.rank-table { width: auto; min-width: 40%; }
.rank-table th, .rank-table td { padding: .3rem .6rem; }
.tag { display: inline-block; background: #e0e7ff; color: #3730a3; border-radius: .3rem;
       padding: .1rem .5rem; margin: 0 .3rem .3rem 0; font-size: .85em; }
"""

_SCRIPT = """
document.querySelectorAll(".region-row").forEach(function (row) {
  row.addEventListener("click", function () {
    var detail = row.nextElementSibling;
    if (!detail || !detail.classList.contains("region-detail")) return;
    var opening = detail.hidden;
    detail.hidden = !opening;
    var icon = row.querySelector(".toggle-icon");
    if (icon) icon.textContent = opening ? "\\u25be" : "\\u25b8";
  });
});
"""


def _text(value) -> str:
    """Convert a possibly numpy-backed value to safe HTML text."""
    value = _json_safe(value)
    if isinstance(value, (list, dict)):
        import json

        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def _seconds(value) -> str:
    return "-" if value is None else f"{value:.6g} s"


_IMBALANCE_FLAG_PCT = 15.0
_HOT_CALL_THRESHOLD = 1000
_HOT_CALL_AVG_SECONDS = 1e-5


def _overview_html(results, rows) -> str:
    """A few sentences summarizing what stands out in this run's regions."""
    timed = [row for row in rows if row["total"] is not None]
    if not timed:
        return '<p class="muted">No timed regions to summarize.</p>'

    points = [
        f"Profiled <strong>{_text(len(rows))}</strong> region(s) across "
        f"<strong>{_text(results.num_ranks)}</strong> rank(s), spanning "
        f"{_seconds(results.time_span)} (setup to finalize: {_seconds(results.total_time)})."
    ]

    total_sum = sum(row["total"] for row in timed)
    hottest = max(timed, key=lambda row: row["total"])
    pct = 100.0 * hottest["total"] / total_sum if total_sum else 0.0
    points.append(
        f"<code>{_text(hottest['name'])}</code> dominates the recorded time: "
        f"{_seconds(hottest['total'])} over {_text(hottest['calls'])} call(s), "
        f"{pct:.1f}% of the summed region time."
    )

    if results.num_ranks > 1:
        imbalanced = [row for row in timed if row["imbalance"] and row["total"] > 0]
        if imbalanced:
            worst = max(imbalanced, key=lambda row: row["imbalance"])
            if worst["imbalance"] >= _IMBALANCE_FLAG_PCT:
                points.append(
                    '<span class="flag">⚠</span> <code>'
                    f"{_text(worst['name'])}</code> is unevenly distributed across "
                    f"ranks: the slowest rank spends {worst['imbalance']:.0f}% more "
                    "time than the per-rank average, which may be worth "
                    "investigating for load balancing."
                )

    chatty = [
        row
        for row in timed
        if row["calls"] >= _HOT_CALL_THRESHOLD
        and row["avg"] is not None
        and row["avg"] < _HOT_CALL_AVG_SECONDS
    ]
    if chatty:
        worst = max(chatty, key=lambda row: row["calls"])
        points.append(
            f"<code>{_text(worst['name'])}</code> was called "
            f"{_text(worst['calls'])} times at ~{worst['avg'] * 1e6:.1f} µs on "
            "average; frequent short calls like this can make timer overhead "
            "itself measurable."
        )

    untimed = len(rows) - len(timed)
    if untimed:
        points.append(
            f"{_text(untimed)} region(s) recorded no calls on the selected ranks."
        )

    return "<ul>" + "".join(f"<li>{point}</li>" for point in points) + "</ul>"


def _metadata_table(metadata: dict) -> str:
    if not metadata:
        return '<p class="muted">No metadata recorded.</p>'
    entries = "".join(
        f"<tr><th>{_text(key)}</th><td><code>{_text(value)}</code></td></tr>"
        for key, value in sorted(metadata.items())
    )
    return f"<table><tbody>{entries}</tbody></table>"


def _rank_breakdown_html(region, ranks) -> str:
    """Per-rank calls/total/avg/min/max table for one region's detail row."""
    selected = sorted(
        region.ranks if ranks is None else [r for r in ranks if r in region.regions]
    )
    if not selected:
        return ""
    rows = []
    for rank in selected:
        data = region.regions[rank]
        rows.append(
            "<tr>"
            f"<td>{rank}</td><td>{_text(data.num_calls)}</td>"
            f"<td>{_text(f'{data.total_duration:.6g}')}</td>"
            f"<td>{_text(f'{data.average_duration:.6g}') if data.num_calls else '-'}</td>"
            f"<td>{_text(f'{data.min_duration:.6g}') if data.num_calls else '-'}</td>"
            f"<td>{_text(f'{data.max_duration:.6g}') if data.num_calls else '-'}</td>"
            "</tr>"
        )
    return (
        "<table class='rank-table'><thead><tr>"
        "<th>rank</th><th>calls</th><th>total [s]</th><th>avg [s]</th>"
        "<th>min [s]</th><th>max [s]</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _region_detail_html(region, ranks) -> str:
    """Expandable detail for one region: call site, tags and per-rank stats."""
    parts = []
    if region.tags:
        parts.append(
            "".join(f'<span class="tag">{_text(tag)}</span>' for tag in region.tags)
        )
    if region.has_source:
        parts.append(
            f"<p class='muted'>{_text(region.source_file)}:{_text(region.source_lineno)}</p>"
        )
        if region.source_text:
            parts.append(f"<pre><code>{_text(region.source_text.rstrip())}</code></pre>")
    if region.has_gpu_timing:
        parts.append(
            f"<p>GPU total: {_seconds(region.gpu_total_duration)}, "
            f"GPU average: {_seconds(region.gpu_average_duration)}</p>"
        )
    breakdown = _rank_breakdown_html(region, ranks)
    if breakdown:
        parts.append("<p class='muted'>Per rank</p>" + breakdown)
    if not parts:
        parts.append("<p class='muted'>No additional detail captured.</p>")
    return "".join(parts)


def _region_table(results, rows, ranks, columns) -> str:
    selected_columns = normalize_region_table_columns(columns)
    headers = "".join(f"<th>{_text(header)}</th>" for _, header in selected_columns)
    keys = [key for key, _ in selected_columns]
    max_total = max((row["total"] or 0.0 for row in rows), default=0.0)

    def cell(row, key) -> str:
        value = row["num_ranks"] if key == "ranks" else row[key]
        if key == "name":
            return f'<span class="toggle-icon">▸</span><span>{_text(value)}</span>'
        text = _text(f"{value:.6g}") if isinstance(value, float) else _text(value)
        if key == "total" and max_total:
            width = 100.0 * (value or 0.0) / max_total
            return f'<span class="bar" style="width:{width:.4g}%"></span><span>{text}</span>'
        return text

    body_rows = []
    for row in rows:
        region = results.get_region(row["name"])
        cells = "".join(
            f'<td class="bar-cell">{cell(row, key)}</td>'
            if key == "total"
            else f"<td>{cell(row, key)}</td>"
            for key in keys
        )
        body_rows.append(f'<tr class="region-row">{cells}</tr>')
        body_rows.append(
            '<tr class="region-detail" hidden>'
            f'<td colspan="{len(keys)}">{_region_detail_html(region, ranks)}</td>'
            "</tr>"
        )
    body = "".join(body_rows)
    if not rows:
        body = f'<tr><td colspan="{len(keys)}">No regions recorded.</td></tr>'
    return (
        "<table><thead><tr>" + headers + "</tr></thead><tbody>" + body + "</tbody></table>"
    )


def _chart_sections(runs, include, exclude, ranks) -> str:
    """Render Plotly charts inline, or explain why the optional extra is absent."""
    try:
        from scope_profiler.plotting_scripts import plot_durations, plot_gantt
    except ImportError:
        return (
            '<section><h2>Charts</h2><p class="muted">Charts require '
            "<code>scope-profiler[pproc]</code>; the statistics and metadata "
            "above remain available without it.</p></section>"
        )

    charts: list[tuple[str, object]] = []
    failures: list[str] = []
    for run in runs:
        try:
            charts.append(
                (
                    f"Timeline: {run.display_label}",
                    plot_gantt(
                        run,
                        include=include,
                        exclude=exclude,
                        ranks=ranks,
                        show=False,
                        verbose=False,
                        backend="plotly",
                        return_fig=True,
                    ),
                )
            )
        except (ImportError, ValueError) as exc:
            failures.append(f"Timeline for {run.display_label}: {exc}")

    try:
        charts.append(
            (
                "Region durations",
                plot_durations(
                    runs,
                    include=include,
                    exclude=exclude,
                    ranks=ranks,
                    sort_by="total",
                    stack_children=True,
                    show=False,
                    verbose=False,
                    backend="plotly",
                    return_fig=True,
                ),
            )
        )
    except (ImportError, ValueError) as exc:
        failures.append(f"Region durations: {exc}")

    fragments = []
    include_plotlyjs = True
    for title, figure in charts:
        if figure is None:
            continue
        fragments.append(
            f'<h3>{_text(title)}</h3><div class="chart">'
            f"{figure.to_html(full_html=False, include_plotlyjs=include_plotlyjs)}</div>"
        )
        include_plotlyjs = False
    if failures:
        fragments.append(
            '<p class="muted">Unavailable chart(s): '
            + _text("; ".join(failures))
            + "</p>"
        )
    if not fragments:
        fragments.append('<p class="muted">No charts could be rendered.</p>')
    return "<section><h2>Charts</h2>" + "".join(fragments) + "</section>"


def create_html_report(
    profiling_data: (
        ProfilingResults | str | Path | Sequence[ProfilingResults | str | Path]
    ),
    filepath: str | Path,
    *,
    include=None,
    exclude=None,
    ranks: list[int] | None = None,
    sort: str = "total",
    columns=None,
    include_charts: bool = True,
) -> Path:
    """Write a standalone HTML summary for one or more profiling results."""
    if isinstance(profiling_data, (ProfilingResults, str, Path)):
        profiling_data = [profiling_data]
    runs = [
        item if isinstance(item, ProfilingResults) else read_h5(item)
        for item in profiling_data
    ]
    if not runs:
        raise ValueError("At least one profiling result is required.")

    sections = []
    for results in runs:
        rows = region_rows(
            results, include=include, exclude=exclude, ranks=ranks, sort=sort
        )
        facts = [
            ("File", str(Path(results.file_path).resolve())),
            ("Ranks", results.num_ranks),
            ("Regions", len(rows)),
            ("Profiled window", _seconds(results.time_span)),
            ("Setup to finalize", _seconds(results.total_time)),
        ]
        facts_html = "".join(
            f'<div class="fact"><strong>{_text(name)}:</strong> {_text(value)}</div>'
            for name, value in facts
        )
        sections.append(
            f'<section><h2>{_text(results.display_label)}</h2><div class="facts">{facts_html}</div>'
            f'<div class="overview">{_overview_html(results, rows)}</div>'
            f"<h3>Region statistics</h3>{_region_table(results, rows, ranks, columns)}"
            f"<details><summary>Metadata</summary>{_metadata_table(results.metadata)}</details></section>"
        )

    charts = _chart_sections(runs, include, exclude, ranks) if include_charts else ""
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>scope-profiler report</title><style>"
        + _STYLE
        + "</style></head><body><h1>scope-profiler report</h1>"
        + "".join(sections)
        + charts
        + "<script>"
        + _SCRIPT
        + "</script></body></html>\n"
    )
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
