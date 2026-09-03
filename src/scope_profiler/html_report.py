"""Self-contained HTML reports for profiling results.

The report deliberately has no plotting dependency: it is useful on a remote
machine immediately after a run, and can be opened locally in any browser.
"""

from __future__ import annotations

import html
import json
import linecache
import tempfile
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np

from scope_profiler.inspection import _json_safe
from scope_profiler.profile_io import read_profile
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import (
    _format_counter,
    _region_durations,
    likwid_tables,
    normalize_region_table_columns,
    perf_event_tables,
    region_rows,
)

_STYLE = """
body { color: #1f2937; font: 15px/1.45 system-ui, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; }
h1, h2, h3 { color: #111827; } section { margin: 2rem 0; }
.chart { min-height: 360px; margin: 1rem 0 2rem; }
.chart-duration { min-height: 680px; }
.chart-error { color: #b91c1c; padding: 1rem; }
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
.region-row.region-selected { background: #fef3c7; box-shadow: inset 4px 0 #d97706; }
.region-row.region-selected:hover { background: #fde68a; }
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
.filter-bar { display: flex; align-items: center; gap: .6rem; margin: 1rem 0 1.5rem; }
.filter-bar label { font-weight: 600; }
.region-filter { flex: 1; max-width: 34rem; font: inherit; padding: .4rem .6rem;
                 border: 1px solid #d1d5db; border-radius: .4rem; }
.region-filter:focus { border-color: #2563eb; outline: 2px solid #bfdbfe; }
.filter-count { color: #6b7280; font-size: .9em; white-space: nowrap; }
.selection-status { color: #92400e; font-size: .9em; white-space: nowrap; }
.clear-selection { background: transparent; border: 0; color: #2563eb; cursor: pointer;
                   font: inherit; padding: .2rem; text-decoration: underline; }
.clear-selection[hidden] { display: none; }
.empty-state { color: #6b7280; text-align: center; font-style: italic; }
.toc { background: #f9fafb; border: 1px solid #d1d5db; border-radius: .5rem;
       padding: .75rem 1rem; margin: 1rem 0 1.5rem; }
.toc strong { margin-right: .75rem; }
.toc a { display: inline-block; margin: .2rem .75rem .2rem 0; }
.overview a { color: inherit; }
.chart-controls { display: flex; gap: .5rem; margin: .75rem 0; }
.chart-controls button { background: #fff; border: 1px solid #9ca3af; border-radius: .35rem;
                         color: #374151; cursor: pointer; font: inherit; padding: .35rem .65rem; }
.chart-controls button:hover { background: #f3f4f6; }
.chart-panel { border: 1px solid #e5e7eb; border-radius: .5rem; padding: .25rem 1rem; }
.chart-heading { color: #111827; font-size: 1.17em; font-weight: 700; }
.table-scroll { overflow-x: auto; }
.back-to-top { text-align: right; }
.call-tree, .call-tree ul { list-style: none; margin: 0; padding-left: 1.1rem; }
.call-tree { padding-left: 0; }
.call-tree > li { margin: .2rem 0; }
.call-tree summary { font-weight: normal; }
.call-tree .recursive { color: #6b7280; font-style: italic; }

@media print {
  body { max-width: 100%; }
  .filter-bar, .chart-controls, .back-to-top, .toc { display: none; }
  .region-row { cursor: default; }
  tr.region-detail[hidden] { display: table-row !important; }
  details:not([open]) > *:not(summary) { display: block !important; }
  th { position: static; }
  table, .chart, .call-tree { break-inside: avoid; }
}
"""

_SCRIPT = """
(function () {
  var listeners = [];
  var selectedRegion = null;
  var status = document.getElementById("region-selection");
  var clear = document.getElementById("clear-region-selection");

  function select(region, run, shouldScroll) {
    selectedRegion = region || null;
    var target = null;
    document.querySelectorAll("tbody[data-region]").forEach(function (body) {
      var match = selectedRegion !== null && body.dataset.region === selectedRegion;
      var row = body.querySelector(".region-row");
      if (row) row.classList.toggle("region-selected", match);
      if (match && !target && (!run || body.dataset.run === run)) target = row;
    });
    if (status) status.textContent = selectedRegion ? "Highlighted: " + selectedRegion : "";
    if (clear) clear.hidden = !selectedRegion;
    listeners.forEach(function (listener) {
      try { listener(selectedRegion); } catch (error) { /* keep other views responsive */ }
    });
    if (target && shouldScroll !== false) {
      var detail = target.nextElementSibling;
      if (detail && detail.classList.contains("region-detail")) {
        detail.hidden = false;
        var icon = target.querySelector(".toggle-icon");
        if (icon) icon.textContent = "\\u25be";
      }
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  window.scopeProfilerSelectRegion = select;
  window.scopeProfilerOnRegionSelect = function (listener) {
    listeners.push(listener);
    listener(selectedRegion);
  };
  if (clear) clear.addEventListener("click", function () { select(null); });

  document.querySelectorAll(".region-row").forEach(function (row) {
    row.addEventListener("click", function () {
      var detail = row.nextElementSibling;
      if (!detail || !detail.classList.contains("region-detail")) return;
      var opening = detail.hidden;
      detail.hidden = !opening;
      var icon = row.querySelector(".toggle-icon");
      if (icon) icon.textContent = opening ? "\\u25be" : "\\u25b8";
      var body = row.closest("tbody[data-region]");
      if (body) select(body.dataset.region, body.dataset.run, false);
    });
  });
})();

// Region filtering, in the same comma-separated syntax the profiling-data
// site uses: each term is a case-insensitive substring of the region name, "^"
// anchors a term to the start, and an empty box shows every region. The charts
// are drawn by a deferred module, which registers through the hook below --
// classic scripts run first, so the hook is in place by the time it does.
(function () {
  var input = document.getElementById("region-filter");
  var count = document.getElementById("region-filter-count");
  if (!input) return;
  var listeners = [];

  function terms() {
    return input.value
      .split(",")
      .map(function (term) { return term.trim().toLowerCase(); })
      .filter(Boolean);
  }

  function matches(name, active) {
    var lowered = String(name == null ? "" : name).toLowerCase();
    return active.some(function (term) {
      return term.charAt(0) === "^"
        ? lowered.indexOf(term.slice(1)) === 0
        : lowered.indexOf(term) !== -1;
    });
  }

  function apply() {
    var active = terms();
    var shown = 0;
    var total = 0;
    document.querySelectorAll("table.region-stats").forEach(function (table) {
      var visible = 0;
      var filterable = 0;
      Array.prototype.forEach.call(table.tBodies, function (tbody) {
        var region = tbody.dataset.region;
        if (region === undefined) return;
        filterable += 1;
        var match = !active.length || matches(region, active);
        tbody.hidden = !match;
        if (match) visible += 1;
      });
      var empty = table.querySelector("tbody.region-empty");
      if (empty) empty.hidden = !filterable || visible > 0;
      shown += visible;
      total += filterable;
    });
    if (count) {
      count.textContent = !active.length || !total
        ? ""
        : shown + " of " + total + " region" + (total === 1 ? "" : "s");
    }
    listeners.forEach(function (listener) {
      try { listener(active.slice()); } catch (error) { /* one chart must not stop the rest */ }
    });
  }

  // Charts register here to redraw themselves when the filter changes.
  window.scopeProfilerOnRegionFilter = function (listener) {
    listeners.push(listener);
    listener(terms());
  };

  var timer = null;
  input.addEventListener("input", function () {
    window.clearTimeout(timer);
    timer = window.setTimeout(apply, 150);
  });
  apply();
})();

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

document.querySelectorAll("[data-chart-action]").forEach(function (button) {
  button.addEventListener("click", function () {
    var open = button.dataset.chartAction === "expand";
    document.querySelectorAll("details.chart-panel").forEach(function (panel) {
      panel.open = open;
    });
    if (open && window.Plotly) {
      document.querySelectorAll("details.chart-panel .chart").forEach(function (chart) {
        if (chart.data) window.Plotly.Plots.resize(chart);
      });
    }
  });
});

document.querySelectorAll("details.chart-panel").forEach(function (panel) {
  panel.addEventListener("toggle", function () {
    var chart = panel.querySelector(".chart");
    if (panel.open && chart && chart.data && window.Plotly) {
      window.Plotly.Plots.resize(chart);
    }
  });
});
"""


def _plotlyjs_version() -> str:
    """The plotly.js version the installed plotly would have inlined.

    Pinned rather than "latest" so a report keeps rendering the way it did
    when it was written, and so the CDN and inline modes agree.
    """
    try:
        from plotly.offline._plotlyjs_version import __plotlyjs_version__

        return str(__plotlyjs_version__)
    except ImportError:  # pragma: no cover - plotly is checked by the caller
        return "3.7.0"


_FILTER_BAR = (
    '<div class="filter-bar">'
    '<label for="region-filter">Filter regions</label>'
    '<input id="region-filter" class="region-filter" type="search" autocomplete="off"'
    ' placeholder="e.g. solve, ^prop:"'
    ' title="Comma-separated, case-insensitive substring match.'
    ' Prefix a term with ^ to anchor it to the start of the region name."'
    ' aria-label="Filter regions">'
    '<span class="filter-count" id="region-filter-count" aria-live="polite"></span>'
    '<span class="selection-status" id="region-selection" aria-live="polite"></span>'
    '<button class="clear-selection" id="clear-region-selection" type="button" hidden>'
    "Clear highlight</button>"
    "</div>"
)


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


def _overview_html(results, rows, region_ids=None) -> str:
    """A few sentences summarizing what stands out in this run's regions."""
    region_ids = {} if region_ids is None else region_ids

    def region_link(name: str) -> str:
        label = f"<code>{_text(name)}</code>"
        target = region_ids.get(name)
        return f'<a href="#{_text(target)}">{label}</a>' if target else label

    timed = [row for row in rows if row["total"] is not None]
    if not timed:
        return '<p class="muted">No timed regions to summarize.</p>'

    points = [
        (
            f"Profiled <strong>{_text(len(rows))}</strong> region(s) across "
            f"<strong>{_text(results.num_ranks)}</strong> rank(s), spanning "
            f"{_seconds(results.time_span)} (setup to finalize: {_seconds(results.total_time)})."
        ),
    ]

    # Exclusive time, not inclusive: an enclosing region's total is mostly its
    # children's, so ranking by it just names whatever sits nearest the top of
    # the call tree. Exclusive time sums to the time actually attributed to
    # regions, so the percentage is a share of a whole rather than of a total
    # that counts nested time once per level.
    #
    # A region reached under two different parents gets one display row per
    # path, all carrying the same figures, so rank over one row per name.
    by_name = {}
    for row in timed:
        by_name.setdefault(row["name"], row)
    unique = list(by_name.values())

    def own_time(row):
        # Legacy profiles whose call tree cannot be rebuilt have no exclusive
        # figure; inclusive is the only thing left to rank them by.
        return row["total"] if row["exclusive"] is None else row["exclusive"]

    exclusive_sum = sum(own_time(row) for row in unique)
    hottest = max(unique, key=own_time)
    pct = 100.0 * own_time(hottest) / exclusive_sum if exclusive_sum else 0.0
    points.append(
        f"{region_link(hottest['name'])} dominates the recorded time: "
        f"{_seconds(own_time(hottest))} in the region itself, excluding nested "
        f"regions, over {_text(hottest['calls'])} call(s) -- "
        f"{pct:.1f}% of the time attributed to regions.",
    )

    # Naming the largest inclusive total too, when it is a different region,
    # answers the obvious next question: why is the region at the top of the
    # table not the one called out above?
    widest = max(unique, key=lambda row: row["total"])
    if widest["name"] != hottest["name"]:
        points.append(
            f"{region_link(widest['name'])} has the largest total, "
            f"{_seconds(widest['total'])}, but "
            f"{_seconds(widest['total'] - own_time(widest))} of that is spent "
            "in the regions nested inside it.",
        )

    if results.num_ranks > 1:
        imbalanced = [row for row in timed if row["imbalance"] and row["total"] > 0]
        if imbalanced:
            worst = max(imbalanced, key=lambda row: row["imbalance"])
            if worst["imbalance"] >= _IMBALANCE_FLAG_PCT:
                points.append(
                    '<span class="flag">⚠</span> '
                    f"{region_link(worst['name'])} is unevenly distributed across "
                    f"ranks: the slowest rank spends {worst['imbalance']:.0f}% more "
                    "time than the per-rank average, which may be worth "
                    "investigating for load balancing.",
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
            f"{region_link(worst['name'])} was called "
            f"{_text(worst['calls'])} times at ~{worst['avg'] * 1e6:.1f} µs on "
            "average; frequent short calls like this can make timer overhead "
            "itself measurable.",
        )

    untimed = len(rows) - len(timed)
    if untimed:
        points.append(
            f"{_text(untimed)} region(s) recorded no calls on the selected ranks.",
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
        region.ranks if ranks is None else [r for r in ranks if r in region.regions],
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
            "</tr>",
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
            "".join(f'<span class="tag">{_text(tag)}</span>' for tag in region.tags),
        )
    if region.has_source:
        parts.append(
            f"<p class='muted'>{_text(region.source_file)}:{_text(region.source_lineno)}</p>",
        )
        if region.source_text:
            parts.append(
                f"<pre><code>{_text(region.source_text.rstrip())}</code></pre>",
            )
    if region.has_gpu_timing:
        parts.append(
            f"<p>GPU total: {_seconds(region.gpu_total_duration)}, "
            f"GPU average: {_seconds(region.gpu_average_duration)}</p>",
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


def _region_table(results, rows, ranks, columns, region_ids=None) -> str:
    region_ids = {} if region_ids is None else region_ids
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
        # The sort keys follow the chosen columns, so the filter gets a hook of
        # its own rather than depending on "name" being one of them.
        region_attr = _text(row["name"])
        run_attr = _text(results.display_label)
        row_id = region_ids.get(row["name"])
        id_attr = f' id="{_text(row_id)}"' if row_id else ""
        body_groups.append(
            f'<tbody data-region="{region_attr}" data-run="{run_attr}" {data_attrs}>'
            f'<tr class="region-row"{id_attr}>{cells}</tr>'
            '<tr class="region-detail" hidden>'
            f'<td colspan="{len(keys) + 1}">{_region_detail_html(region, ranks)}</td>'
            "</tr></tbody>",
        )
    body = "".join(body_groups)
    if rows:
        body += (
            '<tbody class="region-empty" hidden><tr>'
            f'<td colspan="{len(keys) + 1}" class="empty-state">'
            "No regions match the filter.</td></tr></tbody>"
        )
    else:
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
        available if ranks is None else [rank for rank in ranks if rank in available],
    )
    sections = []
    for rank in selected_ranks:
        for record in available.get(rank, []):
            unit = record["unit"]
            total_time = float(np.sum(record["times"])) * unit
            table_rows = []
            for line, hits, elapsed in zip(
                record["line_numbers"],
                record["hits"],
                record["times"],
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
                    "</tr>",
                )
            sections.append(
                f"<h4>Rank {rank} · {_text(record['region'])} · "
                f"{_text(record['function'])} ({_text(record['filename'])}:"
                f"{_text(record['first_lineno'])})</h4>"
                "<table class='rank-table'><thead><tr><th>line</th><th>hits</th>"
                "<th>time [s]</th><th>per hit [s]</th><th>% time</th><th>source</th>"
                "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>",
            )
    if not sections:
        return '<p class="muted">No line-profile records for the selected ranks.</p>'
    return "".join(sections)


def _counter_table(headers, rows) -> str:
    """Render a horizontally scrollable hardware-counter table."""
    heading = "".join(f"<th>{_text(value)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{_text(value) if index == 0 else _text(_format_counter(value))}</td>"
            for index, value in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-scroll"><table class="rank-table"><thead><tr>'
        + heading
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _hardware_sections(runs, include, exclude, ranks) -> str:
    """Render counter tables only for runs that recorded hardware metrics."""
    fragments = []
    for run in runs:
        for table in likwid_tables(run, include=include, exclude=exclude, ranks=ranks):
            sections = []
            for heading, rows in table["sections"]:
                if not rows:
                    continue
                section_heading = f"<h4>{_text(heading)}</h4>" if heading else ""
                sections.append(
                    section_heading
                    + _counter_table(
                        ("counter", *table["columns"]),
                        ((name, *values) for name, values in rows),
                    ),
                )
            fragments.append(
                '<details class="counter-panel">'
                f"<summary>LIKWID: {_text(run.display_label)}, rank "
                f"{_text(table['rank'])}, group {_text(table['group'])}</summary>"
                + "".join(sections)
                + "</details>",
            )

        for table in perf_event_tables(
            run,
            include=include,
            exclude=exclude,
            ranks=ranks,
        ):
            fragments.append(
                '<details class="counter-panel">'
                f"<summary>Linux perf events: {_text(run.display_label)}, "
                f"rank {_text(table['rank'])}</summary>"
                + _counter_table(("region", "calls", *table["events"]), table["rows"])
                + "</details>",
            )

    if not fragments:
        return ""
    return (
        '<section id="hardware-counters"><h2>Hardware counters</h2>'
        '<p class="muted">Counters are shown only for runs and ranks that recorded '
        "them. LIKWID groups include their raw events and derived metrics; Linux "
        "perf-event values are totals across each region's calls.</p>"
        + "".join(fragments)
        + '<p class="back-to-top"><a href="#top">Back to top</a></p></section>'
    )


def _chart_description(title: str, payload: dict) -> str:
    """Explain how to read one chart in the report."""
    if title.startswith("Timeline:"):
        text = (
            "Each bar is one recorded region call on rank 0. Its position and "
            "width show when the call started and how long it ran; colors "
            "identify regions."
        )
    elif title == "Region durations":
        if payload.get("options", {}).get("stack_children"):
            text = (
                "Each bar shows a region's total recorded duration. The stacked "
                "segments divide that time between the region itself and its "
                "direct child regions."
            )
        else:
            text = (
                "Grouped bars compare each region's total recorded duration "
                "across the profiled runs."
            )
    elif title == "Rank heatmap":
        text = (
            "This heatmap uses exclusive timings. Exclusive duration is the time "
            "spent in a region itself, excluding time spent in nested child "
            "regions; this prevents the enclosing session region from dominating "
            "the heatmap."
        )
    elif title == "Duration over time":
        text = (
            "Each line follows a region's mean call duration over elapsed run time. "
            "The shaded range spans the fastest to slowest selected rank, so widening "
            "bands reveal changing rank imbalance."
        )
    elif title == "Rank imbalance":
        text = (
            "Each line compares a region's total duration by rank; its dashed line "
            "marks the mean across ranks. Points far from the mean identify stragglers."
        )
    elif title.startswith("Call graph:"):
        text = (
            "Nodes are regions and links show caller-to-callee relationships on the "
            "selected rank. Repeated invocations are combined into one node per region."
        )
    elif title.startswith("LIKWID:"):
        text = (
            "Bars compare the selected LIKWID metric across regions and ranks. "
            "The hardware-counter tables above retain every recorded event and metric."
        )
    elif title.startswith("Flame chart:"):
        text = (
            "Each frame is one recorded call on the selected ranks. Frame nesting "
            "shows parent-child relationships, and width represents inclusive "
            "duration."
        )
    elif title.startswith("Flame graph:"):
        text = (
            "Repeated calls with the same call path are combined. Frame nesting "
            "shows the aggregated call hierarchy, and width represents total "
            "inclusive duration."
        )
    else:
        return ""
    return f'<p class="muted">{text}</p>'


def _chart_sections(runs, include, exclude, ranks, charts_cdn: bool = False) -> str:
    """Build embedded chart payloads for the bundled browser renderer."""
    try:
        from plotly.offline import get_plotlyjs

        from scope_profiler.plotting_scripts import (
            available_likwid_metrics,
            plot_callgraph,
            plot_duration_timeseries,
            plot_durations,
            plot_flame,
            plot_flame_graph,
            plot_gantt,
            plot_imbalance,
            plot_likwid,
            plot_rank_heatmap,
        )
    except ImportError:
        return (
            '<section id="charts"><h2>Charts</h2><p class="muted">Charts require '
            "<code>scope-profiler[pproc]</code>; the statistics and metadata "
            "above remain available without it.</p></section>"
        )

    charts: list[tuple[str, dict]] = []
    failures: list[str] = []

    def collect(title, plotter, path: Path, *args, **kwargs) -> None:
        try:
            plotter(
                *args,
                data_filepath=path,
                data_format="json",
                show=False,
                verbose=False,
                # Plot functions prepare their export payload before handing
                # the Canvas to a renderer. This private sentinel reaches the
                # no-output path and avoids materializing an unused Python
                # figure; the browser package owns rendering for reports.
                backend="data-only",
                **kwargs,
            )
        except (ImportError, ValueError) as exc:
            failures.append(f"{title}: {exc}")
            return
        charts.append((title, json.loads(path.read_text(encoding="utf-8"))))

    with tempfile.TemporaryDirectory(prefix="scope-profiler-report-") as directory:
        payload_dir = Path(directory)
        for index, run in enumerate(runs):
            collect(
                f"Timeline: {run.display_label} (rank 0)",
                plot_gantt,
                payload_dir / f"gantt-{index}.json",
                run,
                include=include,
                exclude=exclude,
                ranks=[0],
            )

        collect(
            "Region durations",
            plot_durations,
            payload_dir / "durations.json",
            runs,
            include=include,
            exclude=exclude,
            ranks=ranks,
            sort_by="total",
            # The browser builder compares runs as grouped bars, or decomposes
            # one run into stacked child segments. Combining both encodings in
            # one Cartesian axis would merge equal segment names across runs.
            stack_children=len(runs) == 1,
        )
        collect(
            "Duration over time",
            plot_duration_timeseries,
            payload_dir / "duration-timeseries.json",
            runs,
            include=include,
            exclude=exclude,
            ranks=ranks,
        )
        selected_rank_counts = [
            len(
                [
                    rank
                    for rank in (range(run.num_ranks) if ranks is None else ranks)
                    if 0 <= rank < run.num_ranks
                ],
            )
            for run in runs
        ]
        if any(count > 1 for count in selected_rank_counts):
            collect(
                "Rank imbalance",
                plot_imbalance,
                payload_dir / "rank-imbalance.json",
                runs,
                metric="total",
                include=include,
                exclude=exclude,
                ranks=ranks,
            )
        collect(
            "Rank heatmap",
            plot_rank_heatmap,
            payload_dir / "rank-heatmap.json",
            runs,
            include=include,
            exclude=exclude,
            ranks=ranks,
            exclusive=True,
        )

        for index, run in enumerate(runs):
            selected_ranks = [
                rank
                for rank in (range(run.num_ranks) if ranks is None else ranks)
                if 0 <= rank < run.num_ranks
            ]
            if selected_ranks:
                callgraph_rank = selected_ranks[0]
                collect(
                    f"Call graph: {run.display_label} (rank {callgraph_rank})",
                    plot_callgraph,
                    payload_dir / f"callgraph-{index}.json",
                    run,
                    rank=callgraph_rank,
                    include=include,
                    exclude=exclude,
                    compact=True,
                )
            collect(
                f"Flame chart: {run.display_label}",
                plot_flame,
                payload_dir / f"flame-chart-{index}.json",
                run,
                include=include,
                exclude=exclude,
                ranks=ranks,
            )
            collect(
                f"Flame graph: {run.display_label}",
                plot_flame_graph,
                payload_dir / f"flame-graph-{index}.json",
                run,
                include=include,
                exclude=exclude,
                ranks=ranks,
            )

        likwid_metrics = available_likwid_metrics(runs)
        if likwid_metrics:
            metric = likwid_metrics[0]
            collect(
                f"LIKWID: {metric}",
                plot_likwid,
                payload_dir / "likwid.json",
                runs,
                metric=metric,
                include=include,
                exclude=exclude,
                ranks=ranks,
            )

    fragments = []
    chart_documents = []
    for index, (title, payload) in enumerate(charts):
        chart_id = f"scope-profiler-chart-{index}"
        is_duration_chart = payload.get("plot") == "durations"
        chart_class = "chart chart-duration" if is_duration_chart else "chart"
        explanation = _chart_description(title, payload)
        fragments.append(
            '<details class="chart-panel" open>'
            '<summary><span class="chart-heading" role="heading" aria-level="3">'
            f"{_text(title)}</span></summary>{explanation}"
            f'<div class="{chart_class}" id="{chart_id}"></div></details>',
        )
        chart_documents.append(
            {
                "id": chart_id,
                "payload": payload,
                "options": {"layout": {"height": 680}} if is_duration_chart else {},
            },
        )

    if failures:
        fragments.append(
            '<p class="muted">Unavailable chart(s): '
            + _text("; ".join(failures))
            + "</p>",
        )
    if not charts:
        fragments.append('<p class="muted">No charts could be rendered.</p>')
        return (
            '<section id="charts"><h2>Charts</h2>' + "".join(fragments) + "</section>"
        )

    # Escape '<' so profile labels such as '</script>' cannot terminate the
    # inline module. The bundled builders and the payloads make the document
    # durable; whether the Plotly runtime travels with it is the caller's
    # choice, since inlining it costs ~4.7 MB in every report.
    documents_json = json.dumps(chart_documents, ensure_ascii=False).replace(
        "<",
        "\\u003c",
    )
    plotly_builders = (
        files("scope_profiler._assets")
        .joinpath("scope-profiler-plotly-0.2.0.js")
        .read_text(encoding="utf-8")
    )
    if charts_cdn:
        # The exact version this plotly would have inlined, so a report served
        # from the CDN draws with the same runtime as one carrying it.
        runtime = (
            f'<script src="https://cdn.plot.ly/plotly-{_text(_plotlyjs_version())}'
            '.min.js" crossorigin="anonymous"></script>'
        )
    else:
        runtime = "<script>" + get_plotlyjs() + "</script>"
    interactions = r"""
const payloadRegions = (payload) => {
  const rows = [
    ...(payload.intervals ?? []), ...(payload.bars ?? []),
    ...(payload.points ?? []), ...(payload.calls ?? []),
    ...(payload.regions ?? []),
  ];
  return new Set(rows.map((row) => row.region ?? row.name).filter(Boolean));
};

const traceMatchesRegion = (trace, region) => {
  const name = String(trace.name ?? "");
  return name === region || name.endsWith(` / ${region}`) ||
    name === `${region} mean` || name.endsWith(` / ${region} mean`);
};

const highlightFigure = (chart, figure, region) => {
  if (!region || !payloadRegions(chart.payload).has(region)) return figure;
  const kind = chart.payload.plot;
  for (const trace of figure.data) {
    if (trace.type === "sankey") {
      const labels = trace.node?.label ?? [];
      const original = Array.isArray(trace.node?.color) ? trace.node.color : [];
      trace.node.color = labels.map((label, index) =>
        label === region ? (original[index] ?? "#d97706") : "rgba(156,163,175,0.22)");
      const sources = trace.link?.source ?? [], targets = trace.link?.target ?? [];
      trace.link.color = sources.map((source, index) =>
        labels[source] === region || labels[targets[index]] === region
          ? "rgba(217,119,6,0.72)" : "rgba(156,163,175,0.12)");
    } else if (trace.type === "icicle") {
      const labels = trace.labels ?? [];
      const original = Array.isArray(trace.marker?.colors) ? trace.marker.colors : [];
      trace.marker.colors = labels.map((label, index) =>
        label === region ? (original[index] ?? "#d97706") : "rgba(156,163,175,0.22)");
    } else if (trace.type === "heatmap") {
      figure.layout.shapes = [...(figure.layout.shapes ?? []), {
        type: "rect", xref: "x", yref: "paper", x0: region, x1: region,
        x0shift: -0.5, x1shift: 0.5, y0: 0, y1: 1,
        fillcolor: "rgba(245,158,11,0.16)", line: { color: "#d97706", width: 3 },
      }];
    } else if (trace.type === "bar" && trace.orientation === "h") {
      trace.opacity = traceMatchesRegion(trace, region) ? 1 : 0.16;
    } else if (trace.type === "bar") {
      trace.marker.opacity = (trace.x ?? []).map((value) => value === region ? 1 : 0.16);
      trace.marker.line = { ...(trace.marker.line ?? {}),
        color: (trace.x ?? []).map((value) => value === region ? "#92400e" : "rgba(0,0,0,0.12)"),
        width: (trace.x ?? []).map((value) => value === region ? 2 : 0.5) };
    } else if (trace.type === "scatter") {
      trace.opacity = traceMatchesRegion(trace, region) ? 1 : 0.14;
    }
  }
  return figure;
};

const regionFromPoint = (chart, point) => {
  const regions = payloadRegions(chart.payload);
  const candidates = [point.label, point.x, point.y, point.data?.name,
    point.source?.label, point.target?.label];
  for (const candidate of candidates) {
    if (regions.has(candidate)) return candidate;
  }
  const traceName = String(point.data?.name ?? "");
  for (const region of regions) {
    if (traceName === `${region} mean` || traceName.endsWith(` / ${region}`) ||
        traceName.endsWith(` / ${region} mean`)) return region;
  }
  return null;
};

const runFromPoint = (chart, point, region) => {
  if (Array.isArray(point.customdata) && typeof point.customdata[0] === "string") {
    return point.customdata[0];
  }
  const name = String(point.data?.name ?? "");
  return region && name.endsWith(` / ${region}`) ? name.slice(0, -region.length - 3) : null;
};

let activeTerms = [];
let selectedRegion = null;
const draw = (chart) => {
  const target = document.getElementById(chart.id);
  const options = activeTerms.length
    ? { ...chart.options, filterRegion: (region) => activeTerms.some((term) =>
        term.startsWith('^')
          ? String(region).toLowerCase().startsWith(term.slice(1))
          : String(region).toLowerCase().includes(term)) }
    : chart.options;
  try {
    const figure = highlightFigure(chart, buildFigure(chart.payload, options), selectedRegion);
    target.classList.remove('chart-error');
    const rendered = globalThis.Plotly.react(target, figure.data, figure.layout,
      { responsive: true, displaylogo: false });
    if (!target.dataset.regionClickBound) {
      target.dataset.regionClickBound = "true";
      Promise.resolve(rendered).then(() => target.on("plotly_click", (event) => {
        const point = event.points?.[0];
        if (!point) return;
        const region = regionFromPoint(chart, point);
        if (region && typeof globalThis.scopeProfilerSelectRegion === "function") {
          globalThis.scopeProfilerSelectRegion(region, runFromPoint(chart, point, region));
        }
      }));
    }
    return rendered;
  } catch (error) {
    target.classList.add('chart-error');
    target.textContent = `Could not render chart: ${error.message}`;
  }
};

const redraw = () => { for (const chart of scopeProfilerCharts) draw(chart); };
if (typeof globalThis.scopeProfilerOnRegionFilter === "function") {
  globalThis.scopeProfilerOnRegionFilter((terms) => { activeTerms = terms; redraw(); });
}
if (typeof globalThis.scopeProfilerOnRegionSelect === "function") {
  globalThis.scopeProfilerOnRegionSelect((region) => { selectedRegion = region; redraw(); });
}
if (typeof globalThis.scopeProfilerOnRegionFilter !== "function" &&
    typeof globalThis.scopeProfilerOnRegionSelect !== "function") redraw();
"""
    script = (
        runtime
        + '<script type="module">'
        + plotly_builders
        + "\nconst scopeProfilerCharts = "
        + documents_json
        + ";\n"
        # Redraw on every filter change rather than only once: the region
        # filter is handed to the builders, which decide what a filtered chart
        # means for their own payload. Plotly.react diffs against what is
        # already drawn, so typing does not tear each chart down and rebuild
        # it. The hook is installed by the report's classic script, which runs
        # before this deferred module.
        # A blocked or offline CDN leaves Plotly undefined. Say that once,
        # rather than letting every chart report its own confusing TypeError.
        + "if (!globalThis.Plotly) {\n"
        + "  for (const chart of scopeProfilerCharts) {\n"
        + "    const target = document.getElementById(chart.id);\n"
        + "    target.classList.add('chart-error');\n"
        + "    target.textContent = 'Charts need Plotly, which this report "
        + "loads from https://cdn.plot.ly and could not reach. Rebuild "
        + "without --charts-cdn to embed it.';\n"
        + "  }\n"
        + "} else {\n"
        + interactions
        + "}\n</script>"
    )
    controls = (
        '<div class="chart-controls" aria-label="Chart display controls">'
        '<button type="button" data-chart-action="expand">Expand all charts</button>'
        '<button type="button" data-chart-action="collapse">Collapse all charts</button>'
        "</div>"
    )
    return (
        '<section id="charts"><h2>Charts</h2>'
        + controls
        + "".join(fragments)
        + script
        + '<p class="back-to-top"><a href="#top">Back to top</a></p></section>'
    )


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
    charts_cdn: bool = False,
    include_charts: bool = True,
) -> Path:
    """Write a standalone HTML summary for one or more profiling results."""
    if isinstance(profiling_data, (ProfilingResults, str, Path)):
        profiling_data = [profiling_data]
    runs = [
        item if isinstance(item, ProfilingResults) else read_profile(item)
        for item in profiling_data
    ]
    if not runs:
        raise ValueError("At least one profiling result is required.")

    sections = []
    run_links = []
    for run_index, results in enumerate(runs):
        rows = region_rows(
            results,
            include=include,
            exclude=exclude,
            ranks=ranks,
            sort=sort,
            # Populates each row's "exclusive" time. The table's own % column
            # is computed from the inclusive total either way; the overview
            # needs exclusive time to name a hot spot rather than a parent.
            percentage_mode="exclusive",
        )
        section_id = f"run-{run_index}"
        region_ids = {
            row["name"]: f"{section_id}-region-{row_index}"
            for row_index, row in enumerate(rows)
        }
        run_links.append((section_id, results.display_label))
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
            f'<section id="{section_id}"><h2>{_text(results.display_label)}</h2>'
            f'<div class="facts">{facts_html}</div>'
            f'<div class="overview">{_overview_html(results, rows, region_ids)}</div>'
            f"<h3>Region statistics</h3>"
            f"{_region_table(results, rows, ranks, columns, region_ids)}"
            f"<details><summary>Call tree</summary>"
            f"{_call_tree_html(results, rows, include, exclude, ranks)}</details>"
            f"{line_profile_html}"
            f"<details><summary>Metadata</summary>{_metadata_table(results.metadata)}</details>"
            f'<p class="back-to-top"><a href="#top">Back to top</a></p></section>',
        )

    hardware = _hardware_sections(runs, include, exclude, ranks)
    charts = (
        _chart_sections(runs, include, exclude, ranks, charts_cdn=charts_cdn)
        if include_charts
        else ""
    )
    navigation_links = [
        f'<a href="#{_text(section_id)}">{_text(label)}</a>'
        for section_id, label in run_links
    ]
    if hardware:
        navigation_links.append('<a href="#hardware-counters">Hardware counters</a>')
    if charts:
        navigation_links.append('<a href="#charts">Charts</a>')
    navigation = (
        '<nav class="toc" aria-label="Report contents"><strong>Contents</strong>'
        + "".join(navigation_links)
        + "</nav>"
    )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>scope-profiler report</title><style>"
        + _STYLE
        + '</style></head><body><h1 id="top">scope-profiler report</h1>'
        + navigation
        + _FILTER_BAR
        + "".join(sections)
        + hardware
        + charts
        + "<script>"
        + _SCRIPT
        + "</script></body></html>\n"
    )
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
