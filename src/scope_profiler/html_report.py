"""Self-contained HTML reports for profiling results.

The report deliberately has no plotting dependency: it is useful on a remote
machine immediately after a run, and can be opened locally in any browser.
"""

from __future__ import annotations

import html
import linecache
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.h5reader import read_h5
from scope_profiler.inspection import _json_safe
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import (
    _region_durations,
    normalize_region_table_columns,
    region_rows,
)

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
th[data-key] { cursor: pointer; user-select: none; }
th[data-key]:hover { color: #2563eb; }
th[data-key]::after { content: ""; display: inline-block; width: .6em; }
th[data-sort-dir="asc"]::after { content: "\\25b4"; }
th[data-sort-dir="desc"]::after { content: "\\25be"; }
.spark { display: block; }
.call-tree, .call-tree ul { list-style: none; margin: 0; padding-left: 1.1rem; }
.call-tree { padding-left: 0; }
.call-tree > li { margin: .2rem 0; }
.call-tree summary { font-weight: normal; }
.call-tree .recursive { color: #6b7280; font-style: italic; }

@media print {
  body { max-width: 100%; }
  .region-row { cursor: default; }
  tr.region-detail[hidden] { display: table-row !important; }
  details:not([open]) > *:not(summary) { display: block !important; }
  th { position: static; }
  table, .chart, .call-tree { break-inside: avoid; }
}
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

document.querySelectorAll("table.region-stats").forEach(function (table) {
  var headerRow = table.tHead.rows[0];
  Array.prototype.forEach.call(headerRow.cells, function (th) {
    var key = th.dataset.key;
    if (!key) return;
    th.addEventListener("click", function () {
      var ascending = th.dataset.sortDir !== "asc";
      Array.prototype.forEach.call(headerRow.cells, function (cell) {
        delete cell.dataset.sortDir;
      });
      th.dataset.sortDir = ascending ? "asc" : "desc";
      var bodies = Array.prototype.slice.call(table.tBodies);
      bodies.sort(function (a, b) {
        var av = a.dataset[key];
        var bv = b.dataset[key];
        var an = parseFloat(av);
        var bn = parseFloat(bv);
        var cmp =
          !isNaN(an) && !isNaN(bn) ? an - bn : String(av).localeCompare(String(bv));
        return ascending ? cmp : -cmp;
      });
      bodies.forEach(function (tbody) {
        table.appendChild(tbody);
      });
    });
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
            parts.append(
                f"<pre><code>{_text(region.source_text.rstrip())}</code></pre>"
            )
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


def _sparkline_svg(durations, width: int = 90, height: int = 22) -> str:
    """Tiny inline SVG trend line of a region's call durations, in call order."""
    values = np.asarray(durations, dtype=float)
    if values.size < 2:
        return ""
    if values.size > 200:
        values = values[np.linspace(0, values.size - 1, 200).astype(int)]
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    xs = np.linspace(0, width, values.size)
    ys = (
        np.full(values.size, height / 2.0)
        if span == 0
        else height - 2 - (values - lo) / span * (height - 4)
    )
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    fill_points = f"0,{height} {points} {width},{height}"
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" preserveAspectRatio="none">'
        f'<polygon points="{fill_points}" fill="#bfdbfe" opacity="0.5"></polygon>'
        f'<polyline points="{points}" fill="none" stroke="#2563eb" '
        'stroke-width="1.4"></polyline></svg>'
    )


def _region_table(results, rows, ranks, columns) -> str:
    selected_columns = normalize_region_table_columns(columns)
    headers = "".join(
        f'<th data-key="{key}">{_text(header)}</th>' for key, header in selected_columns
    )
    headers += "<th>trend</th>"
    keys = [key for key, _ in selected_columns]
    max_total = max((row["total"] or 0.0 for row in rows), default=0.0)
    session_total = next(
        (row["total"] for row in rows if row["name"] == "scope_profiler.session"),
        None,
    )

    def cell(row, key) -> str:
        if key == "ranks":
            value = row["num_ranks"]
        elif key == "percent":
            value = (
                100.0 * row["total"] / session_total
                if row["total"] is not None and session_total
                else None
            )
        else:
            value = row[key]
        if key == "name":
            return f'<span class="toggle-icon">▸</span><span>{_text(value)}</span>'
        text = _text(f"{value:.6g}") if isinstance(value, float) else _text(value)
        if key == "total" and max_total:
            width = 100.0 * (value or 0.0) / max_total
            return f'<span class="bar" style="width:{width:.4g}%"></span><span>{text}</span>'
        return text

    def sort_value(row, key) -> str:
        if key == "ranks":
            value = row["num_ranks"]
        elif key == "percent":
            value = (
                100.0 * row["total"] / session_total
                if row["total"] is not None and session_total
                else None
            )
        else:
            value = row[key]
        return "" if value is None else str(value)

    body_groups = []
    for row in rows:
        region = results.get_region(row["name"])
        cells = "".join(
            (
                f'<td class="bar-cell">{cell(row, key)}</td>'
                if key == "total"
                else f"<td>{cell(row, key)}</td>"
            )
            for key in keys
        )
        cells += f"<td>{_sparkline_svg(_region_durations(region, ranks))}</td>"
        data_attrs = " ".join(
            f'data-{key}="{_text(sort_value(row, key))}"' for key in keys
        )
        body_groups.append(
            f"<tbody {data_attrs}>"
            f'<tr class="region-row">{cells}</tr>'
            '<tr class="region-detail" hidden>'
            f'<td colspan="{len(keys) + 1}">{_region_detail_html(region, ranks)}</td>'
            "</tr></tbody>"
        )
    body = "".join(body_groups)
    if not rows:
        body = f'<tbody><tr><td colspan="{len(keys) + 1}">No regions recorded.</td></tr></tbody>'
    return (
        '<table class="region-stats"><thead><tr>'
        + headers
        + "</tr></thead>"
        + body
        + "</table>"
    )


def _name_call_tree(nodes):
    """Collapse per-call ``call_graph()`` nodes into a name-level tree.

    Returns ``(roots, children_of)``: ``roots`` are region names ever called
    with no parent, in first-seen order; ``children_of`` maps a region name
    to the distinct child names it was ever seen calling, also in first-seen
    order. Distinct call instances of the same name collapse onto one node,
    since the tree describes region structure, not individual calls.
    """
    name_of_call = {node["call_id"]: node["name"] for node in nodes}
    roots: list[str] = []
    seen_roots: set[str] = set()
    children_of: dict[str, list[str]] = {}
    seen_edges: set[tuple[str, str]] = set()
    for node in nodes:
        parent_id = node["parent_id"]
        name = node["name"]
        if parent_id is None:
            if name not in seen_roots:
                seen_roots.add(name)
                roots.append(name)
            continue
        parent_name = name_of_call.get(parent_id)
        if parent_name is None:
            continue
        edge = (parent_name, name)
        if edge not in seen_edges:
            seen_edges.add(edge)
            children_of.setdefault(parent_name, []).append(name)
    return roots, children_of


def _render_call_tree_node(name, children_of, stats, path) -> str:
    row = stats.get(name)
    calls = row["calls"] if row else 0
    total = row["total"] if row else None
    label = f"<code>{_text(name)}</code> — {_text(calls)} call(s), {_seconds(total)}"
    if name in path:
        return f'<li>{label} <span class="recursive">(recursive)</span></li>'
    children = children_of.get(name, [])
    if not children:
        return f"<li>{label}</li>"
    inner = "".join(
        _render_call_tree_node(child, children_of, stats, path | {name})
        for child in children
    )
    return f"<li><details><summary>{label}</summary><ul>{inner}</ul></details></li>"


def _call_tree_html(results, rows, include, exclude, ranks) -> str:
    """Nested view of which region calls which, reconstructed from timestamps."""
    rank = ranks[0] if ranks else 0
    if rank >= results.num_ranks:
        return '<p class="muted">Call tree unavailable: no data for the selected ranks.</p>'
    try:
        nodes = results.call_graph(rank=rank, include=include, exclude=exclude)
    except (ValueError, KeyError) as exc:
        return f'<p class="muted">Call tree unavailable: {_text(exc)}</p>'
    if not nodes:
        return '<p class="muted">No calls recorded on this rank.</p>'

    roots, children_of = _name_call_tree(nodes)
    stats = {row["name"]: row for row in rows}
    roots.sort(key=lambda name: -(stats.get(name, {}).get("total") or 0.0))
    body = "".join(
        _render_call_tree_node(name, children_of, stats, frozenset()) for name in roots
    )
    return (
        f'<p class="muted">Reconstructed from call timestamps on rank {rank}.</p>'
        f'<ul class="call-tree">{body}</ul>'
    )


def _line_profile_html(results, ranks) -> str:
    """Per-line timings from ``line_profiler``, one table per profiled function."""
    available = results.line_profile
    selected_ranks = sorted(
        available if ranks is None else [rank for rank in ranks if rank in available]
    )
    sections = []
    for rank in selected_ranks:
        for record in available.get(rank, []):
            unit = record["unit"]
            total_time = float(np.sum(record["times"])) * unit
            table_rows = []
            for line, hits, elapsed in zip(
                record["line_numbers"], record["hits"], record["times"]
            ):
                seconds = float(elapsed) * unit
                per_hit = seconds / int(hits) if hits else 0.0
                percent = 100.0 * seconds / total_time if total_time else 0.0
                source = linecache.getline(record["filename"], int(line)).rstrip("\n")
                table_rows.append(
                    "<tr>"
                    f"<td>{_text(int(line))}</td><td>{_text(int(hits))}</td>"
                    f"<td>{_text(f'{seconds:.6g}')}</td>"
                    f"<td>{_text(f'{per_hit:.6g}')}</td>"
                    f"<td>{_text(f'{percent:.2f}')}</td>"
                    f"<td><code>{_text(source)}</code></td>"
                    "</tr>"
                )
            sections.append(
                f"<h4>Rank {rank} · {_text(record['region'])} · "
                f"{_text(record['function'])} ({_text(record['filename'])}:"
                f"{_text(record['first_lineno'])})</h4>"
                "<table class='rank-table'><thead><tr><th>line</th><th>hits</th>"
                "<th>time [s]</th><th>per hit [s]</th><th>% time</th><th>source</th>"
                "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"
            )
    if not sections:
        return '<p class="muted">No line-profile records for the selected ranks.</p>'
    return "".join(sections)


def _chart_sections(runs, include, exclude, ranks) -> str:
    """Render Plotly charts inline, or explain why the optional extra is absent."""
    try:
        from scope_profiler.plotting_scripts import (
            plot_durations,
            plot_flame,
            plot_flame_graph,
            plot_gantt,
            plot_rank_heatmap,
        )
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
                    f"Timeline: {run.display_label} (rank 0)",
                    plot_gantt(
                        run,
                        include=include,
                        exclude=exclude,
                        ranks=[0],
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

    try:
        charts.append(
            (
                "Rank heatmap",
                plot_rank_heatmap(
                    runs,
                    include=include,
                    exclude=exclude,
                    ranks=ranks,
                    exclusive=True,
                    show=False,
                    verbose=False,
                    backend="plotly",
                    return_fig=True,
                ),
            )
        )
    except (ImportError, ValueError) as exc:
        failures.append(f"Rank heatmap: {exc}")

    for run in runs:
        try:
            charts.append(
                (
                    f"Flame chart: {run.display_label}",
                    plot_flame(
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
            failures.append(f"Flame for {run.display_label}: {exc}")

        try:
            charts.append(
                (
                    f"Flame graph: {run.display_label}",
                    plot_flame_graph(
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
            failures.append(f"Flame graph for {run.display_label}: {exc}")

    fragments = []
    include_plotlyjs = True
    for title, figure in charts:
        if figure is None:
            continue
        explanation = ""
        if title == "Rank heatmap":
            explanation = (
                '<p class="muted">This heatmap uses exclusive timings. Exclusive '
                "duration is the time spent in a region itself, excluding time "
                "spent in nested child regions; this prevents the enclosing "
                "session region from dominating the heatmap.</p>"
            )
        fragments.append(
            f'<h3>{_text(title)}</h3>{explanation}<div class="chart">'
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
        line_profile_html = (
            f"<details><summary>Line profile</summary>"
            f"{_line_profile_html(results, ranks)}</details>"
            if any(results.line_profile.values())
            else ""
        )
        sections.append(
            f'<section><h2>{_text(results.display_label)}</h2><div class="facts">{facts_html}</div>'
            f'<div class="overview">{_overview_html(results, rows)}</div>'
            f"<h3>Region statistics</h3>{_region_table(results, rows, ranks, columns)}"
            f"<details><summary>Call tree</summary>"
            f"{_call_tree_html(results, rows, include, exclude, ranks)}</details>"
            f"{line_profile_html}"
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
