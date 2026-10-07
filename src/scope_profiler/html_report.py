"""Self-contained HTML reports for profiling results.

The report deliberately has no plotting dependency: it is useful on a remote
machine immediately after a run, and can be opened locally in any browser.
"""

from __future__ import annotations

import html
import json
import linecache
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

import numpy as np

from scope_profiler.inspection import _json_safe
from scope_profiler.profile_io import read_profile
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import (
    _format_counter,
    _region_durations,
    _session_total,
    format_region_table,
    gpu_timing_warnings,
    likwid_tables,
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
.run-meta { color: #4b5563; margin: -.5rem 0 1rem; }
.run-meta span + span::before { color: #9ca3af; content: "·"; margin: 0 .5rem; }
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
.region-row td:first-child, .own-row td:first-child { display: flex; align-items: center;
                                                     gap: .4rem; }
.region-row:focus-visible { outline: 2px solid #2563eb; outline-offset: -2px; }
.tree-toggle { background: none; border: 0; color: #6b7280; cursor: pointer; display: inline-block; font: inherit;
               padding: 0; width: .9em; }
.tree-toggle:hover { color: #2563eb; }
.region-stats:not(.flat):not(.filtering) > tbody.tree-hidden { display: none; }
.region-stats:not(.flat):not(.filtering) > tbody.tree-collapsed .own-row { display: none; }
.region-stats.flat .tree-toggle, .region-stats.filtering .tree-toggle { visibility: hidden; }
.table-tools { display: flex; align-items: baseline; gap: .5rem; margin: .5rem 0 -.25rem; }
.table-tools button, .chart-controls button { background: #fff; border: 1px solid #9ca3af;
  border-radius: .35rem; color: #374151; cursor: pointer; font: inherit; font-size: .9em;
  padding: .2rem .6rem; }
.table-tools button:hover, .chart-controls button:hover { background: #f3f4f6; }
.hotspots { margin: 1.25rem 0; max-width: 56rem; }
.hotspots ol { display: grid; gap: .15rem; list-style: none; margin: .5rem 0; padding: 0; }
.hotspot { align-items: center; background: none; border: 0; border-radius: .3rem;
           cursor: pointer; display: grid; font: inherit; gap: .75rem;
           grid-template-columns: minmax(10rem, 18rem) 1fr 11rem; padding: .2rem .4rem;
           text-align: left; width: 100%; }
.hotspot:hover { background: #f3f4f6; }
.hotspot-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hotspot-track { background: #f3f4f6; border-radius: .2rem; height: .7rem; }
.hotspot-fill { background: #60a5fa; border-radius: .2rem; display: block; height: 100%; }
.hotspot-value { color: #4b5563; font-variant-numeric: tabular-nums; text-align: right; }
.region-stats { border: 1px solid #d1d5db; border-collapse: separate; border-radius: .5rem;
                border-spacing: 0; width: auto; min-width: 50%; }
.region-stats > thead > tr > th, .region-stats > tbody > tr > td {
  border-bottom: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .9em; padding: .2rem .75rem; text-align: left; white-space: pre; }
.region-stats > thead > tr > th { border-bottom: 1px solid #d1d5db; }
.region-stats > thead > tr > th:first-child { border-top-left-radius: .5rem; }
.region-stats > thead > tr > th:last-child { border-top-right-radius: .5rem; }
.region-stats > tbody > tr.region-detail > td { font-family: inherit; font-size: inherit;
                                                white-space: normal; }
.region-stats .indent { color: #9ca3af; }
.region-stats.flat .indent { display: none; }
.own-row td { color: #6b7280; }
.table-note { font-size: .9em; margin-top: -.25rem; }
.bar-cell { position: relative; }
.bar { position: absolute; left: .75rem; bottom: .1rem; height: 3px; background: #93c5fd;
       border-radius: 2px; z-index: 0; max-width: calc(100% - 1.5rem); }
.bar-cell > span:not(.bar) { position: relative; z-index: 1; }
.own-row .bar { background: #c7d2fe; }
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
.chart-panel { border: 1px solid #e5e7eb; border-radius: .5rem; padding: .25rem 1rem; }
.chart-heading { color: #111827; font-size: 1.17em; font-weight: 700; }
.table-scroll { overflow-x: auto; }
.back-to-top { text-align: right; }

@media print {
  body { max-width: 100%; }
  .filter-bar, .chart-controls, .table-tools, .back-to-top, .toc { display: none; }
  tbody.tree-hidden { display: table-row-group !important; }
  tbody.tree-collapsed .own-row { display: table-row !important; }
  .region-row { cursor: default; }
  tr.region-detail[hidden] { display: table-row !important; }
  details:not([open]) > *:not(summary) { display: block !important; }
  th { position: static; }
  .chart { break-inside: avoid; }
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
      var body = target.parentNode;
      if (window.scopeProfilerRevealRegion) window.scopeProfilerRevealRegion(body);
      var detail = body.querySelector(".region-detail");
      if (detail) {
        detail.hidden = false;
        target.setAttribute("aria-expanded", "true");
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
    function toggleDetail() {
      var detail = row.parentNode.querySelector(".region-detail");
      if (!detail) return;
      detail.hidden = !detail.hidden;
      row.setAttribute("aria-expanded", String(!detail.hidden));
      var body = row.closest("tbody[data-region]");
      if (body) select(body.dataset.region, body.dataset.run, false);
    }
    row.addEventListener("click", toggleDetail);
    row.addEventListener("keydown", function (event) {
      if (event.target !== row || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault();
      toggleDetail();
    });
  });

  document.querySelectorAll(".hotspot").forEach(function (button) {
    button.addEventListener("click", function () {
      select(button.dataset.region, button.dataset.run);
    });
  });
})();

// Collapsible call tree. Each region's tbody carries its call path, so a
// collapsed row hides every tbody whose path extends its own. Filtering and
// column sorting both show rows outside the tree, and switch collapsing off.
(function () {
  function bodies(table) {
    return Array.prototype.filter.call(table.tBodies, function (body) {
      return body.dataset.path !== undefined;
    });
  }

  function refresh(table) {
    var prefixes = bodies(table)
      .filter(function (body) { return body.classList.contains("tree-collapsed"); })
      .map(function (body) { return body.dataset.path + " > "; });
    bodies(table).forEach(function (body) {
      var path = body.dataset.path;
      body.classList.toggle("tree-hidden", prefixes.some(function (prefix) {
        return path.indexOf(prefix) === 0;
      }));
    });
  }

  function setCollapsed(body, collapsed) {
    var button = body.querySelector(".tree-toggle");
    if (!button) return;
    body.classList.toggle("tree-collapsed", collapsed);
    button.setAttribute("aria-expanded", String(!collapsed));
    button.textContent = collapsed ? "\u25b8" : "\u25be";
  }

  // Expand every collapsed ancestor, so a selected region can be shown.
  window.scopeProfilerRevealRegion = function (body) {
    var table = body.closest("table.region-stats");
    var path = body.dataset.path;
    if (!table || path === undefined) return;
    bodies(table).forEach(function (other) {
      if (path.indexOf(other.dataset.path + " > ") === 0) setCollapsed(other, false);
    });
    refresh(table);
  };

  document.querySelectorAll(".tree-toggle").forEach(function (button) {
    button.addEventListener("click", function (event) {
      event.stopPropagation();  // the row's own click opens its detail
      var body = button.closest("tbody");
      setCollapsed(body, !body.classList.contains("tree-collapsed"));
      refresh(body.closest("table"));
    });
  });

  document.querySelectorAll("[data-tree-action]").forEach(function (button) {
    button.addEventListener("click", function () {
      var table = document.getElementById(button.dataset.table);
      if (!table) return;
      var collapse = button.dataset.treeAction === "collapse";
      // "Collapse all" keeps the top level open: a lone root row says nothing.
      bodies(table).forEach(function (body) {
        setCollapsed(body, collapse && Number(body.dataset.depth) > 0);
      });
      refresh(table);
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
      table.classList.toggle("filtering", active.length > 0);
      shown += visible;
      total += filterable;
    });
    document.querySelectorAll(".hotspots").forEach(function (block) {
      var shown = 0;
      block.querySelectorAll("li[data-region]").forEach(function (item) {
        item.hidden = active.length > 0 && !matches(item.dataset.region, active);
        if (!item.hidden) shown += 1;
      });
      block.hidden = shown === 0;
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
      // Sorted rows no longer follow the call tree, so drop its indentation.
      table.classList.add("flat");
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
_UNCOVERED_FLAG_PCT = 25.0
_HOTSPOT_COUNT = 8
_SESSION = "scope_profiler.session"
_MIN_TIMESERIES_CALLS = 3


def _pooled_by_name(rows) -> dict[str, dict]:
    """Own time, total and calls per region name, over all of its call paths.

    Own time rather than inclusive: an enclosing region's total is mostly its
    children's, so ranking by it just names whatever sits nearest the top of
    the call tree. Own times sum to the time actually attributed to regions.
    """
    pooled: dict[str, dict] = {}
    for row in rows:
        if row["total"] is None:
            continue
        if row["name"] == _SESSION and "call_path" not in row:
            # Without a call tree, the session's time outside every region
            # is unknown: its fallback "exclusive" time is its whole total.
            continue
        entry = pooled.setdefault(
            row["name"], {"name": row["name"], "own": 0.0, "total": 0.0, "calls": 0}
        )
        # Legacy profiles whose call tree cannot be rebuilt have no exclusive
        # figure; inclusive is the only thing left to rank them by.
        entry["own"] += row["total"] if row["exclusive"] is None else row["exclusive"]
        entry["total"] += row["total"]
        entry["calls"] += row["calls"]
    return pooled


def _overview_html(results, rows, region_ids=None) -> str:
    """A few sentences on what stands out in this run's regions."""
    region_ids = {} if region_ids is None else region_ids

    def region_link(name: str) -> str:
        label = f"<code>{_text(name)}</code>"
        target = region_ids.get(name)
        return f'<a href="#{_text(target)}">{label}</a>' if target else label

    timed = [row for row in rows if row["total"] is not None]
    if not timed:
        return '<p class="muted">No timed regions to summarize.</p>'

    pooled = _pooled_by_name(timed)
    # The session root's own time is the time outside every region: worth a
    # note of its own, but not a hot spot or a "largest total" to explain.
    session = pooled.pop(_SESSION, None)
    points = []
    if pooled:
        own_sum = sum(entry["own"] for entry in pooled.values())
        hottest = max(pooled.values(), key=lambda entry: entry["own"])
        pct = 100.0 * hottest["own"] / own_sum if own_sum else 0.0
        points.append(
            f"{region_link(hottest['name'])} dominates the recorded time: "
            f"{_seconds(hottest['own'])} in the region itself, excluding nested "
            f"regions, over {_text(hottest['calls'])} call(s) -- "
            f"{pct:.1f}% of the time attributed to regions.",
        )
        # Naming the largest inclusive total too, when it is a different
        # region, answers the obvious next question: why is the region at the
        # top of the table not the one called out above?
        widest = max(pooled.values(), key=lambda entry: entry["total"])
        if widest["name"] != hottest["name"]:
            points.append(
                f"{region_link(widest['name'])} has the largest total, "
                f"{_seconds(widest['total'])}, but "
                f"{_seconds(widest['total'] - widest['own'])} of that is spent "
                "in the regions nested inside it.",
            )
    if session and pooled and session["total"]:
        uncovered = 100.0 * session["own"] / session["total"]
        if uncovered >= _UNCOVERED_FLAG_PCT:
            points.append(
                f"{uncovered:.0f}% of the session ({_seconds(session['own'])}) "
                "is outside every region; adding regions there would show where "
                "that time goes.",
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
    if not points:
        return ""
    return "<ul>" + "".join(f"<li>{point}</li>" for point in points) + "</ul>"


def _hotspots_html(results, rows) -> str:
    """The regions with the most own time, as a short ranked bar list."""
    pooled = _pooled_by_name(rows)
    session = pooled.get(_SESSION)
    ranked = sorted(pooled.values(), key=lambda entry: -entry["own"])
    ranked = [entry for entry in ranked if entry["own"] > 0][:_HOTSPOT_COUNT]
    # A single region has nothing to be ranked against.
    if len(ranked) < 2:
        return ""
    base = session["total"] if session else sum(e["own"] for e in pooled.values())
    widest = ranked[0]["own"]
    run = _text(results.display_label)
    items = []
    for entry in ranked:
        name = _text(entry["name"])
        # The session root's own time is whatever no other region covers.
        label = "<em>outside any region</em>" if entry["name"] == _SESSION else name
        share = f" · {100.0 * entry['own'] / base:.1f}%" if base else ""
        items.append(
            f'<li data-region="{name}"><button class="hotspot" type="button"'
            f' data-region="{name}" data-run="{run}" title="{name}">'
            f'<span class="hotspot-name">{label}</span>'
            '<span class="hotspot-track"><span class="hotspot-fill"'
            f' style="width:{100.0 * entry["own"] / widest:.4g}%"></span></span>'
            f'<span class="hotspot-value">{entry["own"]:.6f} s{share}</span>'
            "</button></li>"
        )
    return (
        '<div class="hotspots"><h3>Hot spots</h3>'
        '<p class="muted">Time spent in each region itself, excluding nested '
        "regions, over all of its call paths. Click one to find it in the table."
        "</p><ol>" + "".join(items) + "</ol></div>"
    )


def _plural(count, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _format_timestamp(value) -> str:
    """An ISO timestamp as minutes in UTC, or as recorded when unparseable."""
    text = str(value)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return text
    if moment.tzinfo is None:
        return moment.strftime("%Y-%m-%d %H:%M")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _run_meta_html(results, region_count: int) -> str:
    """One line on where, when and for how long a run was profiled."""
    metadata = results.metadata or {}
    parts = []
    if results.file_path and str(results.file_path) != ".":
        path = Path(results.file_path)
        parts.append(
            f'<span title="{_text(path.resolve())}"><code>{_text(path.name)}</code></span>'
        )
    if metadata.get("timestamp"):
        parts.append(f"<span>{_text(_format_timestamp(metadata['timestamp']))}</span>")
    parts.extend(
        f"<span>{_text(metadata[key])}</span>"
        for key in ("hostname", "chip_information")
        if metadata.get(key)
    )
    parts.append(f"<span>{_plural(results.num_ranks, 'rank')}</span>")
    parts.append(f"<span>{_plural(region_count, 'region')}</span>")
    if results.time_span is not None:
        parts.append(
            f'<span title="setup to finalize: {_text(_seconds(results.total_time))}">'
            f"{_text(_seconds(results.time_span))} profiled</span>"
        )
    return f'<p class="run-meta">{"".join(parts)}</p>'


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


# The tree glyphs and column indentation that format_region_table() puts in
# front of a cell's value. Kept in their own span so that re-sorting the table
# by a column, which breaks the hierarchy, can hide them.
_CELL_INDENT = re.compile(r"^[ │└─]*")


def _indented_cell(text: str) -> str:
    indent = _CELL_INDENT.match(text).group()  # type: ignore[union-attr]
    value = _text(text[len(indent) :])
    return f'<span class="indent">{_text(indent)}</span>{value}' if indent else value


def _region_table(
    results, rows, ranks, columns, region_ids=None, table_id="regions"
) -> str:
    """Region statistics laid out like the terminal summary table.

    Cell text comes from :func:`~scope_profiler.summary.format_region_table`,
    so the report shows the same tree, call counts and ``(own)`` rows. Each
    region is one ``<tbody>``, holding its row, its ``(own)`` row when it has
    children, and its expandable detail row, so sorting, filtering and
    collapsing the tree move them together.
    """
    region_ids = {} if region_ids is None else region_ids
    selected_columns, display_rows = format_region_table(rows, columns)
    headers = "".join(
        f'<th data-key="{key}">{_text(header)}</th>' for key, header in selected_columns
    )
    headers += "<th>trend</th>"
    keys = [key for key, _ in selected_columns]
    max_total = max((row["total"] or 0.0 for row in rows), default=0.0)
    session_total = _session_total(rows)

    # (own) rows follow exactly the regions that have children in the table.
    parents = {row.get("call_path") for row, _, is_own in display_rows if is_own}

    def cells(display, bar_value, toggle='<span class="tree-toggle"></span>') -> str:
        rendered = []
        for key in keys:
            text = _indented_cell(display[key])
            if key == "name":
                rendered.append(f"<td>{toggle}<span>{text}</span></td>")
            elif key == "total" and max_total and display[key]:
                width = 100.0 * (bar_value or 0.0) / max_total
                rendered.append(
                    f'<td class="bar-cell"><span class="bar" style="width:{width:.4g}%">'
                    f"</span><span>{text}</span></td>"
                )
            else:
                rendered.append(f"<td>{text}</td>")
        return "".join(rendered)

    def sort_value(row, key) -> str:
        if key == "ranks":
            value = row["num_ranks"]
        elif key == "percent":
            value = (
                100.0 * row["coverage"] / session_total
                if row.get("coverage") is not None and session_total
                else None
            )
        elif key == "parent_percent":
            value = (
                100.0 * row["coverage"] / row["parent_coverage"]
                if row.get("coverage") is not None and row.get("parent_coverage")
                else None
            )
        else:
            value = row[key]
        return "" if value is None else str(value)

    groups: list[list[str]] = []
    linked: set[str] = set()
    for row, display, is_own in display_rows:
        if is_own:
            # Directly after its parent region's row, ahead of the detail row.
            groups[-1].insert(
                -1,
                f'<tr class="own-row">{cells(display, row["exclusive"])}<td></td></tr>',
            )
            continue
        region = results.get_region(row["name"])
        data_attrs = " ".join(
            f'data-{key}="{_text(sort_value(row, key))}"' for key in keys
        )
        # The sort keys follow the chosen columns, so the filter gets a hook of
        # its own rather than depending on "name" being one of them.
        region_attr = _text(row["name"])
        run_attr = _text(results.display_label)
        # A region reached along several call paths has one row per path;
        # links from the overview target the first of them.
        row_id = None if row["name"] in linked else region_ids.get(row["name"])
        linked.add(row["name"])
        id_attr = f' id="{_text(row_id)}"' if row_id else ""
        # Rows without a reconstructable call path stay outside the tree.
        tree_attrs = (
            f' data-path="{_text(row["call_path"])}" data-depth="{row["depth"]}"'
            if "call_path" in row
            else ""
        )
        if row.get("call_path") in parents:
            toggle = (
                '<button class="tree-toggle" type="button" aria-expanded="true"'
                ' title="Collapse or expand the nested regions">▾</button>'
            )
            row_cells = cells(display, row["total"], toggle)
        else:
            row_cells = cells(display, row["total"])
        head = (
            f'<tbody data-region="{region_attr}" data-run="{run_attr}"'
            f"{tree_attrs} {data_attrs}>"
            f'<tr class="region-row" tabindex="0" aria-expanded="false"{id_attr}>'
            f"{row_cells}"
            f"<td>{_sparkline_svg(_region_durations(region, ranks))}</td></tr>"
        )
        detail = (
            '<tr class="region-detail" hidden>'
            f'<td colspan="{len(keys) + 1}">{_region_detail_html(region, ranks)}</td>'
            "</tr></tbody>"
        )
        # (own) rows are inserted between the two as they come.
        groups.append([head, detail])
    body = "".join("".join(group) for group in groups)
    if rows:
        body += (
            '<tbody class="region-empty" hidden><tr>'
            f'<td colspan="{len(keys) + 1}" class="empty-state">'
            "No regions match the filter.</td></tr></tbody>"
        )
    else:
        body = f'<tbody><tr><td colspan="{len(keys) + 1}">No regions recorded.</td></tr></tbody>'
    notes = ["Durations are in seconds.", *gpu_timing_warnings(rows)]
    if parents:
        notes.append("(own) rows show a region's time excluding its children.")
    if session_total is not None and "percent" in keys:
        notes.append(
            "% session uses wall-clock coverage; overlapping recursive calls count once."
        )
    if rows:
        notes.append("Click a row for its call site and per-rank breakdown.")
    tools = (
        '<div class="table-tools">'
        f'<button type="button" data-tree-action="collapse" data-table="{table_id}">'
        "Collapse all</button>"
        f'<button type="button" data-tree-action="expand" data-table="{table_id}">'
        "Expand all</button></div>"
        if parents
        else ""
    )
    # No scrolling wrapper: an overflow container would stop the sticky
    # header from following the page on a long table.
    return (
        tools
        + f'<table class="region-stats" id="{table_id}"><thead><tr>'
        + headers
        + "</tr></thead>"
        + body
        + "</table>"
        + f'<p class="muted table-note">{_text(" ".join(notes))}</p>'
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
        text = (
            "Grouped bars compare each region's total recorded duration "
            "across the profiled runs."
        )
    elif title.startswith("Change:"):
        text = (
            "Percent change in each region's total duration, candidate over "
            "baseline. Bars above zero got slower; a region measured in only "
            "one run leaves a gap rather than reading as a 100% change."
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
            "Each line follows the mean call duration of a region called at least "
            f"{_MIN_TIMESERIES_CALLS} times over elapsed run time. "
            "The shaded range spans the fastest to slowest selected rank, so widening "
            "bands reveal changing rank imbalance."
        )
    elif title == "Rank imbalance":
        text = (
            "Each line compares a region's total duration by rank; its dashed line "
            "marks the mean across ranks. Points far from the mean identify stragglers."
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
            collect_region_statistics,
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

    charts: list[tuple[str, dict, dict]] = []
    failures: list[str] = []

    def collect(
        title,
        plotter,
        path: Path,
        *args,
        chart_options: dict | None = None,
        **kwargs,
    ) -> None:
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
        charts.append(
            (title, json.loads(path.read_text(encoding="utf-8")), chart_options or {}),
        )

    selected_ranks = [
        [
            rank
            for rank in (range(run.num_ranks) if ranks is None else ranks)
            if 0 <= rank < run.num_ranks
        ]
        for run in runs
    ]
    # Following a region over time needs a few calls to follow; a region
    # entered once or twice would only add a stray point to the chart.
    repeated = sorted(
        {
            region.name
            for run in runs
            for region in run.get_regions(include=include, exclude=exclude)
            if _region_durations(region, ranks).size >= _MIN_TIMESERIES_CALLS
        }
    )

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
                # Every row is already labelled with its region.
                chart_options={"layout": {"showlegend": False}},
            )

        if len(runs) == 2:
            # A baseline/candidate pair wants "what changed?", which reads
            # better as one signed bar per region than as two bars a viewer
            # has to subtract by eye.
            statistics = collect_region_statistics(
                runs,
                ranks=ranks,
                include=include,
                exclude=exclude,
            )
            if statistics["files"]:
                baseline, candidate = (file["label"] for file in statistics["files"])
                charts.append(
                    (
                        f"Change: {candidate} vs {baseline}",
                        {"plot": "region_statistics", **statistics},
                        {"comparison": "percent"},
                    ),
                )
        if len(runs) > 1:
            # For a single run the region table and hot spots already rank
            # every region; the bars earn their space comparing runs.
            collect(
                "Region durations",
                plot_durations,
                payload_dir / "durations.json",
                runs,
                include=include,
                exclude=exclude,
                ranks=ranks,
                sort_by="total",
                stack_children=False,
            )

        for index, run in enumerate(runs):
            collect(
                f"Flame graph: {run.display_label}",
                plot_flame_graph,
                payload_dir / f"flame-graph-{index}.json",
                run,
                include=include,
                exclude=exclude,
                ranks=ranks,
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

        if repeated:
            collect(
                "Duration over time",
                plot_duration_timeseries,
                payload_dir / "duration-timeseries.json",
                runs,
                include=[f"{re.escape(name)}$" for name in repeated],
                exclude=exclude,
                ranks=ranks,
            )
        # Both rank views are empty or trivial with a single rank.
        if any(len(selected) > 1 for selected in selected_ranks):
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

    # Open only the views that orient a reader; the rest wait, collapsed, for
    # a reader looking for them, rather than all competing for attention.
    opened = {
        next(index for index, chart in enumerate(charts) if chart[0].startswith(kind))
        for kind in ("Timeline:", "Change:", "Flame graph:")
        if any(chart[0].startswith(kind) for chart in charts)
    }
    fragments = []
    chart_documents = []
    for index, (title, payload, chart_options) in enumerate(charts):
        chart_id = f"scope-profiler-chart-{index}"
        is_duration_chart = payload.get("plot") == "durations"
        chart_class = "chart chart-duration" if is_duration_chart else "chart"
        explanation = _chart_description(title, payload)
        fragments.append(
            f'<details class="chart-panel"{" open" if index in opened else ""}>'
            '<summary><span class="chart-heading" role="heading" aria-level="3">'
            f"{_text(title)}</span></summary>{explanation}"
            f'<div class="{chart_class}" id="{chart_id}"></div></details>',
        )
        chart_documents.append(
            {
                "id": chart_id,
                "payload": payload,
                "options": (
                    {**chart_options, "layout": {"height": 680}}
                    if is_duration_chart
                    else chart_options
                ),
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
    if (trace.type === "icicle") {
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
  const identity = point.customdata?.identity;
  if (identity && regions.has(identity.region)) return identity.region;
  const candidates = [point.label, point.x, point.y, point.data?.name];
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
  if (point.customdata?.identity?.file != null) return point.customdata.identity.file;
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
  // buildFigure dispatches region_statistics to the ranked summary; only
  // buildComparisonFigure draws the signed per-region change.
  const build = options.comparison ? buildComparisonFigure : buildFigure;
  try {
    const figure = highlightFigure(chart, build(chart.payload, options), selectedRegion);
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
    sort: str = "start",
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
        overview = _overview_html(results, rows, region_ids)
        line_profile_html = (
            f"<details><summary>Line profile</summary>"
            f"{_line_profile_html(results, ranks)}</details>"
            if any(results.line_profile.values())
            else ""
        )
        sections.append(
            f'<section id="{section_id}"><h2>{_text(results.display_label)}</h2>'
            # A region on several call paths has a row per path.
            f"{_run_meta_html(results, len({row['name'] for row in rows}))}"
            + (f'<div class="overview">{overview}</div>' if overview else "")
            + _hotspots_html(results, rows)
            + "<h3>Region statistics</h3>"
            + _region_table(
                results,
                rows,
                ranks,
                columns,
                region_ids,
                table_id=f"{section_id}-regions",
            )
            + f"{line_profile_html}"
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
