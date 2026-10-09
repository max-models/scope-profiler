"""Self-contained HTML reports for profiling results.

The report deliberately has no plotting dependency: it is useful on a remote
machine immediately after a run, and can be opened locally in any browser.

One run gets a full report: a summary, its hotspots, the region table,
load balance across ranks, line profiles, hardware counters and charts.
Several runs get a comparison report instead - what changed between them --
which links to a full report built for each run alongside it.
"""

from __future__ import annotations

import builtins
import functools
import html
import io
import json
import keyword
import linecache
import os
import re
import sys
import tempfile
import tokenize
from collections import Counter
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

_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

_STYLE = """
body { color: #1f2937; font: 15px/1.45 system-ui, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; background: #fff; }
h1, h2, h3 { color: #111827; } section { margin: 2rem 0; }
h2 { border-bottom: 1px solid #e5e7eb; padding-bottom: .3rem; }
.report-header h1 { margin: .1rem 0 .3rem; }
.chart { min-height: 360px; margin: 1rem 0 2rem; }
.chart-duration { min-height: 680px; }
.chart-error { color: #b91c1c; padding: 1rem; }
.run-meta { color: #4b5563; margin: 0 0 1rem; }
.run-meta span + span::before { color: #9ca3af; content: "·"; margin: 0 .5rem; }
.kpis { display: flex; flex-wrap: wrap; gap: .5rem 2.5rem; margin: 1rem 0 1.25rem; }
.kpi { display: flex; flex-direction: column; min-width: 0; }
.kpi-label { color: #6b7280; font-size: .85em; }
.kpi-value { color: #111827; font-size: 1.15em; font-variant-numeric: tabular-nums;
             font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kpi-value a { color: inherit; text-decoration: none; }
.kpi-value a:hover { color: #2563eb; }
.kpi-detail { color: #6b7280; font-size: .85em; }
.kpi.warn .kpi-value { color: #b45309; }
.kpi.good .kpi-value { color: #15803d; }
.kpi.bad .kpi-value { color: #b91c1c; }
.findings { margin: .5rem 0 1rem; padding-left: 1.25rem; }
.findings li { margin: .2rem 0; }
.findings a { color: inherit; }
.flag { color: #b45309; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; }
th, td { border-bottom: 1px solid #d1d5db; padding: .45rem .6rem; text-align: right; }
th { background: #f9fafb; position: sticky; top: var(--filter-bar-height, 0px); z-index: 2; } th:first-child, td:first-child { text-align: left; }
details { margin: .75rem 0; } summary { cursor: pointer; font-weight: 600; }
.muted { color: #6b7280; } code { overflow-wrap: anywhere; }
.region-row, .select-row { cursor: pointer; }
.region-row:hover, .select-row:hover { background: #f3f4f6; }
.region-row.region-selected, .select-row.region-selected { background: #fef3c7;
  box-shadow: inset 4px 0 #d97706; }
.region-row.region-selected:hover, .select-row.region-selected:hover { background: #fde68a; }
.region-row td:first-child, .own-row td:first-child { display: flex; align-items: center;
                                                     gap: .4rem; }
.region-row:focus-visible { outline: 2px solid #2563eb; outline-offset: -2px; }
.tree-toggle { background: none; border: 0; color: #6b7280; cursor: pointer; display: inline-block; font: inherit;
               padding: 0; width: .9em; }
.tree-toggle:hover { color: #2563eb; }
.region-stats:not(.flat):not(.filtering) > tbody.tree-hidden { display: none; }
.region-stats:not(.flat):not(.filtering) > tbody.tree-collapsed .own-row { display: none; }
.region-stats.flat .tree-toggle, .region-stats.filtering .tree-toggle { visibility: hidden; }
.table-tools { display: flex; flex-wrap: wrap; align-items: baseline; gap: .5rem;
               margin: .5rem 0 -.25rem; }
.table-tools .tools-label { color: #6b7280; font-size: .9em; margin-left: .5rem; }
.table-tools .tools-label:first-child { margin-left: 0; }
.table-tools button, .chart-controls button, .chart-tools button { background: #fff;
  border: 1px solid #9ca3af; border-radius: .35rem; color: #374151; cursor: pointer;
  font: inherit; font-size: .9em; padding: .2rem .6rem; }
.table-tools button:hover, .chart-controls button:hover, .chart-tools button:hover {
  background: #f3f4f6; }
.table-tools button[aria-pressed="true"], .chart-tools button[aria-pressed="true"] {
  background: #1f2937; border-color: #1f2937; color: #fff; }
.hotspots { margin: 1.25rem 0; }
.hotspots ol { display: grid; gap: .2rem; list-style: none; margin: .5rem 0; padding: 0; }
.hotspot { align-items: center; background: none; border: 0; border-radius: .4rem;
              color: inherit; cursor: pointer; display: grid; font: inherit; gap: .2rem .9rem;
              grid-template-columns: 1.6rem minmax(12rem, 20rem) 1fr 4rem; padding: .3rem .5rem;
              text-align: left; width: 100%; }
.hotspot:hover { background: #f3f4f6; }
.hs-rank { color: #9ca3af; font-variant-numeric: tabular-nums; text-align: right; }
.hs-name { display: flex; flex-direction: column; min-width: 0; }
.hs-label, .hs-path { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hs-bar { display: flex; flex-direction: column; gap: .2rem; min-width: 0; }
.hs-path { color: #6b7280; font-size: .82em; }
.hs-tag { color: #6b7280; font-size: .85em; font-weight: 400; margin-left: .3rem; }
.hs-track { background: #f3f4f6; height: .5rem; overflow: hidden; }
.hs-fill { background: #60a5fa; display: block; height: 100%; }
.hs-fill.own { background: #818cf8; } .hs-fill.outside { background: #9ca3af; }
.hs-share { font-variant-numeric: tabular-nums; font-weight: 600; text-align: right; }
.hs-detail { color: #6b7280; font-size: .82em; font-variant-numeric: tabular-nums;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.region-stats { border: 1px solid #d1d5db; border-collapse: separate; border-radius: .5rem;
                border-spacing: 0; width: auto; min-width: 50%; }
.region-stats > thead > tr > th, .region-stats > tbody > tr > td {
  border-bottom: 0; font-family: MONO;
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
.filter-bar { align-items: center; background: #fff;
  border-bottom: 1px solid #e5e7eb; display: flex; flex-wrap: wrap; gap: .6rem;
  margin: -2rem 0 1rem; padding: .6rem 0; position: sticky; top: 0; z-index: 20; }
.region-limits { align-items: center; display: flex; flex-wrap: wrap; gap: .4rem; }
.region-limits label { font-weight: 600; margin-left: .4rem; }
.region-limits input[type="range"] { accent-color: #1f2937; width: 9rem; }
.region-limits output { color: #374151; font-size: .9em; min-width: 8.5rem;
  font-variant-numeric: tabular-nums; }
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
.toc { margin: .25rem 0 1.25rem; }
.toc strong { margin-right: .75rem; }
.toc a { display: inline-block; margin: .2rem .75rem .2rem 0; }
.chart-controls { display: flex; gap: .5rem; margin: .75rem 0; }
.chart-panel { border: 1px solid #e5e7eb; border-radius: .5rem; padding: .25rem 1rem; }
.chart-heading { color: #111827; font-size: 1.17em; font-weight: 700; }
.chart-tools { align-items: center; display: flex; flex-wrap: wrap; gap: .4rem;
  justify-content: flex-end; margin: .25rem 0 -.5rem; }
.chart-tools .chart-open { margin-left: .6rem; }
.table-scroll { overflow-x: auto; }
/* A scrolling wrapper is its headers' sticky frame: the page bar's offset
   would push them down over the first row. */
.table-scroll th { top: 0; }
.back-to-top { text-align: right; }
.meta-table th { background: none; position: static; vertical-align: top; white-space: nowrap;
                 width: 1%; }
.meta-table td { text-align: left; }
.meta-table td code { white-space: pre-wrap; word-break: break-all; }
.lp-functions { border: 1px solid #e5e7eb; border-radius: .5rem; overflow: hidden; }
.lp-function { border-bottom: 1px solid #e5e7eb; margin: 0; }
.lp-function:last-child { border-bottom: 0; }
.lp-functions:not(.show-all) .lp-extra { display: none; }
.lp-function > summary { align-items: center; display: grid; font-weight: 400; gap: .75rem;
  grid-template-columns: minmax(8rem, 16rem) minmax(5rem, 10rem) minmax(6rem, 1fr) 6rem 4.5rem 2.5rem;
  padding: .3rem .8rem; }
.lp-function > summary:hover { background: #f9fafb; }
.lp-function[open] > summary { background: #f9fafb; border-bottom: 1px solid #e5e7eb; }
.lp-func { font-family: MONO; font-weight: 650; overflow: hidden; text-overflow: ellipsis;
           white-space: nowrap; }
.lp-region, .lp-loc { font-size: .85em; overflow: hidden; text-overflow: ellipsis;
                      white-space: nowrap; }
.lp-region { color: #3730a3; } .lp-loc { color: #6b7280; }
.lp-bar { background: #f3f4f6; height: .5rem; overflow: hidden; }
.lp-bar > span { background: #f87171; display: block; height: 100%; }
.lp-total, .lp-share { font-variant-numeric: tabular-nums; text-align: right; }
.lp-share { color: #6b7280; }
.lp-block { margin: .25rem 0 1rem; }
.lp-head { align-items: baseline; display: flex; gap: .75rem; margin: .2rem 0 .35rem; }
.region-detail .lp-scroll { background: #fff; border: 1px solid #e5e7eb; border-radius: .4rem;
                            max-width: 62rem; }
.lp-scroll { max-height: 32rem; overflow: auto; padding-bottom: .6rem; }
.lp-table { font-family: MONO; font-size: .85em; margin: 0; width: 100%; }
.lp-table th { background: #f9fafb; font-weight: 600; position: sticky; top: 0; z-index: 1; }
.lp-table th, .lp-table td { border-bottom: 0; padding: .08rem .7rem; text-align: right;
                             white-space: nowrap; }
.lp-table th.lp-src, .lp-table td.lp-src { text-align: left; white-space: pre; width: 100%; }
.lp-table td.lp-lineno { color: #9ca3af; }
.lp-table td.lp-pct { background: linear-gradient(to right, #fecaca var(--pct), transparent var(--pct)); }
.lp-table tr.lp-idle td { color: #9ca3af; }
.lp-table tr.lp-idle td.lp-src { color: #1f2937; }
.lp-table tr.lp-hot td { background-color: #fef2f2; }
.lp-table tr.lp-hot td.lp-pct { background-color: #fef2f2; font-weight: 650; }
.lp-table tbody tr:hover td { background-color: #f3f4f6; }
.lp-src .tk-kw { color: #cf222e; } .lp-src .tk-const, .lp-src .tk-num { color: #0550ae; }
.lp-src .tk-str { color: #0a3069; } .lp-src .tk-com { color: #6e7781; font-style: italic; }
.lp-src .tk-fn { color: #8250df; } .lp-src .tk-dec { color: #8250df; }
.lp-src .tk-bi { color: #953800; } .lp-src .tk-self { color: #953800; font-style: italic; }
.balance-table td, .balance-table th { white-space: nowrap; }
.balance-table td:first-child { max-width: 18rem; overflow: hidden; text-overflow: ellipsis; }
.balance-table tbody tr.extra { display: none; }
.balance-table.show-all tbody tr.extra { display: table-row; }
.spread { background: #f3f4f6; border-radius: .2rem; display: block; height: .6rem;
          min-width: 9rem; position: relative; }
.spread-range { background: #fca5a5; border-radius: .2rem; height: 100%; position: absolute; top: 0; }
.spread-mean { background: #111827; height: 140%; position: absolute; top: -20%; width: 2px; }
.rank-matrix { font-size: .85em; width: auto; }
.rank-matrix th, .rank-matrix td { padding: .2rem .45rem; white-space: nowrap; }
.rank-matrix td.heat { font-variant-numeric: tabular-nums; min-width: 2.6rem; text-align: center; }
.rank-matrix.compact td.heat { color: transparent; min-width: .9rem; padding: .2rem; }
.rank-matrix th.slowest { color: #b91c1c; }
.rank-matrix tfoot td { border-top: 2px solid #d1d5db; font-weight: 600; }
.legend { align-items: center; color: #6b7280; display: flex; font-size: .85em; gap: .5rem; }
.legend-scale { background: linear-gradient(to right, rgba(37,99,235,.55), #fff, rgba(220,38,38,.55));
                border: 1px solid #e5e7eb; border-radius: .2rem; display: inline-block;
                height: .7rem; width: 8rem; }
.runs-table td, .runs-table th { text-align: left; white-space: nowrap; }
.runs-table td.run-file { max-width: 16rem; overflow: hidden; text-overflow: ellipsis; }
.compare-candidate { margin: 1rem 0 2rem; }
.changes { display: grid; gap: 1.5rem; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); }
.changes h4 { margin: .5rem 0; }
.change-list { display: grid; gap: .15rem; list-style: none; margin: 0; padding: 0; }
.change-item { align-items: center; display: grid; gap: .1rem .6rem;
               grid-template-columns: minmax(8rem, 1fr) 6rem 5.5rem; padding: .2rem .3rem; }
.change-item .hs-name { font-size: .95em; }
.change-track { background: #f3f4f6; height: .5rem; overflow: hidden; }
.change-fill { display: block; height: 100%; }
.change-fill.faster { background: #22c55e; } .change-fill.slower { background: #ef4444; }
.change-value { font-size: .9em; font-variant-numeric: tabular-nums; text-align: right; }
.compare-table { font-size: .92em; min-width: 60%; width: auto; }
.compare-table th { white-space: nowrap; }
.compare-table td { font-variant-numeric: tabular-nums; white-space: nowrap; }
.compare-table td:first-child { max-width: 24rem; overflow: hidden; text-overflow: ellipsis; }
.compare-table .cmp-name { display: inline-block; }
.compare-table.flat .cmp-name { padding-left: 0 !important; }
.compare-table [data-m] { display: none; }
.compare-table.metric-total [data-m="total"], .compare-table.metric-own [data-m="own"],
.compare-table.metric-avg [data-m="avg"], .compare-table.metric-calls [data-m="calls"] {
  display: inline-flex; }
.delta { justify-content: flex-end; }
.delta-text { font-variant-numeric: tabular-nums; }
.faster .delta-text, .delta-text.faster { color: #15803d; }
.slower .delta-text, .delta-text.slower { color: #b91c1c; }
.same .delta-text, .neutral .delta-text { color: #6b7280; }
.badge { color: #6b7280; font-size: .9em; font-style: italic; }
.region-toast { align-items: center; background: #111827; border-radius: .5rem; bottom: 1.25rem;
                box-shadow: 0 6px 20px rgba(0,0,0,.25); color: #f9fafb; display: flex; gap: .75rem;
                padding: .55rem .6rem .55rem 1rem; position: fixed; right: 1.25rem; z-index: 10; }
.region-toast[hidden] { display: none; }
.region-toast button { background: none; border: 1px solid #6b7280; border-radius: .35rem;
                       color: inherit; cursor: pointer; font: inherit; font-size: .9em;
                       padding: .15rem .55rem; }
.region-toast button:hover { background: #374151; }
.region-toast .toast-close { border: 0; font-size: 1.1em; }

/* Theme: a flat grey page with white boxes - the header, each section and
   each chart - outlined rather than shadowed. */
:root { --page: #f4f5f7; --surface: #fff; --line: #e3e6ea; --text: #1f2937;
        --muted: #6b7280; --accent: #2563eb; --radius: .6rem; }
body { background: var(--page); color: var(--text); margin-top: 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.filter-bar { background: var(--surface); border: 1px solid var(--line); border-top: 0;
  border-radius: 0 0 var(--radius) var(--radius); margin: 0 0 1.25rem;
  padding: .6rem 1rem; }
.report-header { background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1.1rem 1.5rem .3rem; }
.report-header h1 { font-size: 1.6em; letter-spacing: -.01em; }
.run-meta span + span::before { content: ""; margin: 0; }
.run-meta { display: flex; flex-wrap: wrap; gap: .4rem; }
.run-meta span { background: var(--page); border: 1px solid var(--line);
  border-radius: 999px; font-size: .85em; padding: .1rem .6rem; }
.toc { align-items: center; display: flex; flex-wrap: wrap; gap: .35rem; margin: .9rem 0; }
.toc strong { color: var(--muted); font-size: .85em; font-weight: 600; margin-right: .3rem;
  text-transform: uppercase; letter-spacing: .04em; }
.toc a { background: var(--surface); border: 1px solid var(--line); border-radius: .4rem;
  color: var(--text); font-size: .9em; margin: 0; padding: .2rem .65rem; }
.toc a:hover { border-color: var(--accent); color: var(--accent); text-decoration: none; }
section { background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); margin: 1.25rem 0; padding: .4rem 1.5rem 1rem; }
h2 { border-bottom: 0; font-size: 1.3em; margin: .9rem 0 .6rem; padding-bottom: 0; }
section h3 { border-top: 1px solid var(--line); margin: 1.75rem -1.5rem .75rem;
  padding: 1.1rem 1.5rem 0; }
.kpis { gap: .6rem; }
.kpi { background: var(--page); border: 1px solid var(--line); border-radius: .5rem;
  flex: 1 1 9rem; padding: .55rem .8rem; }
.kpi-label { font-size: .75em; letter-spacing: .04em; text-transform: uppercase; }
.kpi-value { font-size: 1.25em; }
th { background: #f8f9fb; color: #374151; font-weight: 600; }
th, td { border-bottom-color: var(--line); }
.region-stats, .chart-panel, .lp-functions { border-color: var(--line); }
.region-stats > thead > tr > th { border-bottom-color: var(--line); }
.chart-panel { background: var(--surface); margin: .75rem 0; padding: .35rem 1.1rem; }
.chart-panel > summary { padding: .4rem 0; }
.chart-panel[open] > summary { border-bottom: 1px solid var(--line); margin-bottom: .5rem; }
.table-tools button, .chart-controls button, .chart-tools button, .region-toast button {
  border-radius: .4rem; }
.table-tools button, .chart-controls button, .chart-tools button {
  border-color: #d1d5db; color: #374151; }
.table-tools button[aria-pressed="true"], .chart-tools button[aria-pressed="true"] {
  background: var(--accent); border-color: var(--accent); }
.region-filter { background: var(--page); border-color: var(--line); }
.region-filter:focus { background: var(--surface); }
.region-limits input[type="range"] { accent-color: var(--accent); }
.hs-fill, .bar { background: #93b4f5; }
.back-to-top { font-size: .85em; margin: .5rem 0 0; }

@media print {
  body, section, .report-header { background: #fff; border: 0; padding: 0; }
  body { max-width: 100%; }
  .filter-bar, .chart-controls, .chart-tools, .table-tools, .back-to-top, .toc,
  .region-toast { display: none; }
  tbody.tree-hidden { display: table-row-group !important; }
  tbody.tree-collapsed .own-row { display: table-row !important; }
  .region-row { cursor: default; }
  tr.region-detail[hidden] { display: table-row !important; }
  details:not([open]) > *:not(summary) { display: block !important; }
  th { position: static; }
  .chart { break-inside: avoid; }
  .balance-table tbody tr.extra { display: table-row; }
  .lp-scroll { max-height: none; overflow: visible; } .lp-table th { position: static; }
  .lp-functions .lp-extra { display: block; }
}
""".replace("MONO", _MONO)

_SCRIPT = """
// Region selection, shared by every view. A selection highlights the region
// everywhere; how it moves the page depends on where it came from:
//   "scroll": jump to the region's table row (the hotspot list, whose
//             whole purpose is finding a region in the table);
//   "toast":  stay put and offer the jump instead (chart clicks - yanking
//             the page away from the chart being explored is disorienting);
//   "none":   nothing (a click on the table row itself).
(function () {
  var listeners = [];
  var selectedRegion = null;
  var status = document.getElementById("region-selection");
  var clear = document.getElementById("clear-region-selection");
  var toast = document.getElementById("region-toast");
  var toastTimer = null;
  var SELECTABLE = "tbody[data-region] > .region-row, tr.select-row[data-region]";

  function owner(row) {
    return row.classList.contains("region-row") ? row.parentNode : row;
  }

  // A function longer than its capped box opens scrolled to its hottest line.
  function scrollToHotLines(detail) {
    detail.querySelectorAll(".lp-scroll").forEach(function (box) {
      var hot = box.querySelector("tr.lp-hot");
      if (!hot || box.dataset.scrolled) return;
      box.dataset.scrolled = "1";
      var head = box.querySelector("thead");
      var offset = hot.offsetTop - (head ? head.offsetHeight : 0);
      if (offset + hot.offsetHeight > box.clientHeight) {
        box.scrollTop = offset - (box.clientHeight - hot.offsetHeight) / 3;
      }
    });
  }

  function reveal(target) {
    var body = owner(target);
    if (target.classList.contains("region-row")) {
      if (window.scopeProfilerRevealRegion) window.scopeProfilerRevealRegion(body);
      var detail = body.querySelector(".region-detail");
      if (detail) {
        detail.hidden = false;
        target.setAttribute("aria-expanded", "true");
        scrollToHotLines(detail);
      }
    }
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function hideToast() {
    if (toast) toast.hidden = true;
    window.clearTimeout(toastTimer);
  }

  function showToast(region, target) {
    if (!toast) return;
    toast.querySelector(".toast-text").textContent = "Highlighted " + region;
    var jump = toast.querySelector(".toast-jump");
    jump.hidden = !target;
    jump.onclick = function () { hideToast(); if (target) reveal(target); };
    toast.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(hideToast, 6000);
  }

  function select(region, run, mode) {
    mode = mode || "scroll";
    selectedRegion = region || null;
    var target = null;
    document.querySelectorAll(SELECTABLE).forEach(function (row) {
      var data = owner(row).dataset;
      var match = selectedRegion !== null && data.region === selectedRegion;
      row.classList.toggle("region-selected", match);
      if (match && !target && (!run || data.run === run)) target = row;
    });
    if (status) status.textContent = selectedRegion ? "Highlighted: " + selectedRegion : "";
    if (clear) clear.hidden = !selectedRegion;
    listeners.forEach(function (listener) {
      try { listener(selectedRegion); } catch (error) { /* keep other views responsive */ }
    });
    if (!selectedRegion) { hideToast(); return; }
    if (mode === "scroll" && target) reveal(target);
    else if (mode === "toast") showToast(selectedRegion, target);
  }

  window.scopeProfilerSelectRegion = select;
  window.scopeProfilerOnRegionSelect = function (listener) {
    listeners.push(listener);
    listener(selectedRegion);
  };
  if (clear) clear.addEventListener("click", function () { select(null); });
  if (toast) toast.querySelector(".toast-close").addEventListener("click", hideToast);

  document.querySelectorAll(".region-row").forEach(function (row) {
    function toggleDetail() {
      var detail = row.parentNode.querySelector(".region-detail");
      if (!detail) return;
      detail.hidden = !detail.hidden;
      row.setAttribute("aria-expanded", String(!detail.hidden));
      if (!detail.hidden) scrollToHotLines(detail);
      var body = row.closest("tbody[data-region]");
      if (body) select(body.dataset.region, body.dataset.run, "none");
    }
    row.addEventListener("click", toggleDetail);
    row.addEventListener("keydown", function (event) {
      if (event.target !== row || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault();
      toggleDetail();
    });
  });

  document.querySelectorAll("tr.select-row").forEach(function (row) {
    row.addEventListener("click", function () {
      var region = row.dataset.region;
      select(region === selectedRegion ? null : region, row.dataset.run, "none");
    });
  });

  document.querySelectorAll(".hotspot").forEach(function (button) {
    button.addEventListener("click", function () {
      select(button.dataset.region, button.dataset.run, "scroll");
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
    button.textContent = collapsed ? "▸" : "▾";
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
  // Table headers stick just below the bar, whose height grows when it wraps.
  var bar = input && input.closest(".filter-bar");
  if (bar && window.ResizeObserver) {
    new ResizeObserver(function () {
      document.documentElement.style.setProperty(
        "--filter-bar-height", bar.offsetHeight + "px");
    }).observe(bar);
  }
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

  // The Regions/Depth sliders' region set, handed over by the chart module
  // (scopeProfilerSetRegionLimit below); null while they limit nothing.
  var limit = null;

  function allowed(region, active) {
    return (!active.length || matches(region, active)) &&
      (!limit || limit.has(region));
  }

  function filterTables(active) {
    var limiting = active.length > 0 || limit !== null;
    var shown = 0;
    var total = 0;
    document.querySelectorAll("table.region-stats").forEach(function (table) {
      var visible = 0;
      var filterable = 0;
      Array.prototype.forEach.call(table.tBodies, function (tbody) {
        var region = tbody.dataset.region;
        if (region === undefined) return;
        filterable += 1;
        var match = allowed(region, active);
        tbody.hidden = !match;
        if (match) visible += 1;
      });
      var empty = table.querySelector("tbody.region-empty");
      if (empty) empty.hidden = !filterable || visible > 0;
      table.classList.toggle("filtering", limiting);
      shown += visible;
      total += filterable;
    });
    // Every other per-region element: hotspots, load balance, the
    // comparison table.
    document.querySelectorAll("[data-filter-region]").forEach(function (item) {
      item.hidden = limiting && !allowed(item.dataset.filterRegion, active);
    });
    document.querySelectorAll(".hotspots").forEach(function (block) {
      block.hidden = !block.querySelector("li[data-filter-region]:not([hidden])");
    });
    if (count) {
      count.textContent = !limiting || !total
        ? ""
        : shown + " of " + total + " region" + (total === 1 ? "" : "s");
    }
  }

  function apply() {
    var active = terms();
    filterTables(active);
    listeners.forEach(function (listener) {
      try { listener(active.slice()); } catch (error) { /* one chart must not stop the rest */ }
    });
  }

  // The chart module passes the sliders' region set here, so the tables keep
  // to the same regions as the charts.
  window.scopeProfilerSetRegionLimit = function (regions) {
    limit = regions;
    filterTables(terms());
  };

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

// Comparison table: which metric the cells show, and whether rows follow the
// call tree or the size of the change.
document.querySelectorAll("table.compare-table").forEach(function (table) {
  var metric = "total";
  var order = "tree";
  var tools = document.querySelectorAll('[data-compare-table="' + table.id + '"]');

  function press(attribute, value) {
    tools.forEach(function (button) {
      if (button.hasAttribute(attribute)) {
        button.setAttribute("aria-pressed", String(button.getAttribute(attribute) === value));
      }
    });
  }

  function arrange() {
    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    var key = order === "tree" ? "order" : "change" + metric.charAt(0).toUpperCase() + metric.slice(1);
    rows.sort(function (a, b) {
      var av = parseFloat(a.dataset[key]);
      var bv = parseFloat(b.dataset[key]);
      av = isNaN(av) ? -1 : av;
      bv = isNaN(bv) ? -1 : bv;
      return order === "tree" ? av - bv : bv - av;
    });
    rows.forEach(function (row) { body.appendChild(row); });
    table.classList.toggle("flat", order !== "tree");
  }

  tools.forEach(function (button) {
    button.addEventListener("click", function () {
      if (button.dataset.compareMetric) {
        table.classList.remove("metric-" + metric);
        metric = button.dataset.compareMetric;
        table.classList.add("metric-" + metric);
        press("data-compare-metric", metric);
      } else {
        order = button.dataset.compareSort;
        press("data-compare-sort", order);
      }
      arrange();
    });
  });
});

document.querySelectorAll("[data-show-all]").forEach(function (button) {
  button.addEventListener("click", function () {
    var table = document.getElementById(button.dataset.showAll);
    if (!table) return;
    var all = table.classList.toggle("show-all");
    button.textContent = all ? button.dataset.lessLabel : button.dataset.moreLabel;
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


def _filter_bar(limits: str = "") -> str:
    """The bar that stays at the top of the page: the region filter, the
    charts' Regions and Depth sliders (``limits``) when there are any, and the
    highlight status."""
    return (
        '<div class="filter-bar">'
        '<label for="region-filter">Filter regions</label>'
        '<input id="region-filter" class="region-filter" type="search"'
        ' autocomplete="off" placeholder="e.g. solve, ^prop:"'
        ' title="Comma-separated, case-insensitive substring match.'
        ' Prefix a term with ^ to anchor it to the start of the region name."'
        ' aria-label="Filter regions">'
        '<span class="filter-count" id="region-filter-count" aria-live="polite">'
        "</span>" + limits + '<span class="selection-status" id="region-selection"'
        ' aria-live="polite"></span>'
        '<button class="clear-selection" id="clear-region-selection" type="button"'
        " hidden>Clear highlight</button>"
        "</div>"
    )


_REGION_TOAST = (
    '<div class="region-toast" id="region-toast" role="status" hidden>'
    '<span class="toast-text"></span>'
    '<button class="toast-jump" type="button">Show in table</button>'
    '<button class="toast-close" type="button" aria-label="Dismiss">×</button>'
    "</div>"
)


def _text(value) -> str:
    """Convert a possibly numpy-backed value to safe HTML text."""
    value = _json_safe(value)
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def _seconds(value) -> str:
    return "-" if value is None else f"{value:.6g} s"


def _duration_text(seconds) -> str:
    """A duration in the unit that keeps it short: 51.2 ms, not 0.0512 s."""
    if seconds is None:
        return "-"
    magnitude = abs(seconds)
    if magnitude == 0:
        return "0 s"
    if magnitude < 1e-6:
        return f"{seconds * 1e9:.3g} ns"
    if magnitude < 1e-3:
        return f"{seconds * 1e6:.3g} µs"
    if magnitude < 1:
        return f"{seconds * 1e3:.3g} ms"
    if magnitude < 1000:
        return f"{seconds:.4g} s"
    return f"{seconds:,.0f} s"


def _signed_pct(value: float) -> str:
    """A percent change with a real minus sign, so columns of them line up."""
    return f"{value:+.1f}%".replace("-", "−")


_IMBALANCE_FLAG_PCT = 15.0
_HOT_CALL_THRESHOLD = 1000
_HOT_CALL_AVG_SECONDS = 1e-5
_UNCOVERED_FLAG_PCT = 25.0
_HOTSPOT_COUNT = 8
_SESSION = "scope_profiler.session"
_SPEEDUP_REGIONS = 8
# Past this many regions, the Regions slider opens on only the ones with the
# most time: more lanes and bars than this cannot be read anyway, and the
# browser spends seconds drawing them.
_TIMELINE_REGIONS = 500
# Which scaling chart(s) a comparison of run sizes shows. Nothing in a profile
# records the problem size, so the report cannot tell a strong-scaling study
# (fixed total problem) from a weak-scaling one (fixed problem per rank); the
# caller says which, and by default both are drawn.
SCALING_MODES = ("strong", "weak", "both")
_BALANCE_ROWS = 12
_MATRIX_REGIONS = 20
_MATRIX_RANKS = 64
_MATRIX_TEXT_RANKS = 24
_CHANGE_COUNT = 5
# Line profiles: functions listed before "Show all", the share of a function's
# time that makes a line worth showing, and the fewest lines worth folding.
_LP_FUNCTIONS = 10
# Changes smaller than this read as noise rather than as faster or slower.
_SAME_PCT = 2.0
# Own-time changes below this share of the baseline are not listed as changes.
_CHANGE_FLOOR_PCT = 0.5


def _plural(count, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _region_linker(region_ids):
    """Render a region name as a link to its row in the region table."""

    def region_link(name: str) -> str:
        label = f"<code>{_text(name)}</code>"
        target = region_ids.get(name)
        return f'<a href="#{_text(target)}">{label}</a>' if target else label

    return region_link


def _hotspot_entries(rows) -> list[dict]:
    """Where the time goes: the leaves of the call tree, largest first.

    A region's own time - its time outside every nested region - is a leaf
    of the tree the region table draws: a leaf region's own time is all of
    its time, and a parent's is its ``(own)`` row. Ranking leaves rather
    than inclusive totals names the code that costs the time instead of
    whatever encloses it, and the entries add up to the session.
    """
    timed = [row for row in rows if row["total"] is not None]
    paths = {row["call_path"] for row in timed if "call_path" in row}
    parents = {path.rsplit(" > ", 1)[0] for path in paths if " > " in path}
    entries = []
    for row in timed:
        if "call_path" in row:
            path = row["call_path"]
            if row["name"] == _SESSION:
                # The session root's own time is the time outside every region.
                kind = "outside"
            else:
                kind = "own" if path in parents else "leaf"
            context = [part for part in path.split(" > ")[:-1] if part != _SESSION]
        elif row["name"] == _SESSION:
            # Without a call tree, the session's time outside every region
            # is unknown: its fallback "exclusive" time is its whole total.
            continue
        else:
            kind, context = "leaf", []
        # Legacy profiles whose call tree cannot be rebuilt have no exclusive
        # figure; inclusive is the only thing left to rank them by.
        own = row["total"] if row["exclusive"] is None else row["exclusive"]
        if own <= 0:
            continue
        entries.append(
            {
                "name": row["name"],
                "context": context,
                "kind": kind,
                "time": own,
                "calls": row["calls"],
                # A parent's imbalance is that of its total, not its own time.
                "imbalance": row["imbalance"] if kind == "leaf" else None,
            }
        )
    entries.sort(key=lambda entry: -entry["time"])
    return entries


def _hotspot_base(rows, entries) -> float:
    """What a hotspot's share is a share of: the session, when recorded."""
    session = _session_total(rows)
    if session and any("call_path" in row for row in rows):
        return session
    return sum(entry["time"] for entry in entries)


def _hotspot_label(entry) -> str:
    if entry["kind"] == "outside":
        return '<span class="hs-label"><em>outside any region</em></span>'
    label = f"<strong>{_text(entry['name'])}</strong>"
    if entry["kind"] == "own":
        label += '<span class="hs-tag" title="Time in this region outside its nested regions">(own)</span>'
    return f'<span class="hs-label">{label}</span>'


def _hotspot_context(entry) -> str:
    if not entry["context"]:
        return ""
    trail = " › ".join(_text(part) for part in entry["context"])
    return f'<span class="hs-path">in {trail}</span>'


def _hotspots_html(results, entries, base) -> str:
    """The largest leaves of the call tree, as a ranked bar list."""
    # A single entry has nothing to be ranked against.
    if len(entries) < 2:
        return ""
    run = _text(results.display_label)
    items = []
    for position, entry in enumerate(entries[:_HOTSPOT_COUNT], start=1):
        name = _text(entry["name"])
        share = 100.0 * entry["time"] / base if base else 0.0
        details = [_duration_text(entry["time"])]
        if entry["kind"] != "outside":
            details.append(_plural(entry["calls"], "call"))
            if entry["calls"]:
                details.append(f"{_duration_text(entry['time'] / entry['calls'])}/call")
        if entry["imbalance"] is not None and entry["imbalance"] >= _IMBALANCE_FLAG_PCT:
            details.append(
                f'<span class="flag">⚠ slowest rank +{entry["imbalance"]:.0f}%</span>'
            )
        title = " › ".join([*entry["context"], entry["name"]])
        items.append(
            f'<li data-region="{name}" data-filter-region="{name}">'
            f'<button class="hotspot" type="button" data-region="{name}"'
            f' data-run="{run}" title="{_text(title)}">'
            f'<span class="hs-rank">{position}</span>'
            f'<span class="hs-name">{_hotspot_label(entry)}'
            f"{_hotspot_context(entry)}</span>"
            '<span class="hs-bar">'
            f'<span class="hs-track"><span class="hs-fill {entry["kind"]}"'
            f' style="width:{min(share, 100.0):.4g}%"></span></span>'
            f'<span class="hs-detail">{" · ".join(details)}</span></span>'
            f'<span class="hs-share">{share:.1f}%</span>'
            "</button></li>"
        )
    return (
        '<div class="hotspots"><h3>Hotspots</h3>'
        '<p class="muted">The leaves of the call tree with the most time: regions '
        "without nested regions, and the <em>own</em> time parents spend outside "
        "theirs. Shares are of the session. Click one to find it in the table.</p>"
        "<ol>" + "".join(items) + "</ol></div>"
    )


def _kpi(label: str, value: str, detail: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    detail_html = f'<span class="kpi-detail">{detail}</span>' if detail else ""
    return (
        f'<div class="kpi{tone_class}"><span class="kpi-label">{label}</span>'
        f'<span class="kpi-value">{value}</span>{detail_html}</div>'
    )


def _summary_html(results, rows, entries, base, balance, region_ids, section_id):
    """Headline numbers and a few sentences on what stands out in one run."""
    region_link = _region_linker(region_ids)
    timed = [row for row in rows if row["total"] is not None]
    if not timed:
        return '<p class="muted">No timed regions to summarize.</p>'

    kpis = []
    if results.time_span:
        kpis.append(
            _kpi(
                "Wall time",
                _text(_duration_text(results.time_span)),
                "setup to finalize",
            )
        )
    kpis.append(_kpi("Ranks", _text(results.num_ranks)))
    kpis.append(_kpi("Regions", _text(len({row["name"] for row in rows}))))
    findings: list[tuple[str, str]] = []

    regions_only = [entry for entry in entries if entry["kind"] != "outside"]
    outside = next((entry for entry in entries if entry["kind"] == "outside"), None)
    # A lone session root has no regions for the outside time to be outside of.
    if outside is not None and base and regions_only:
        uncovered = 100.0 * outside["time"] / base
        kpis.append(
            _kpi(
                "In regions",
                f"{100.0 - uncovered:.0f}%",
                "of the session",
                "warn" if uncovered >= _UNCOVERED_FLAG_PCT else "",
            )
        )
        if uncovered >= _UNCOVERED_FLAG_PCT:
            findings.append(
                (
                    "warn",
                    (
                        f"{uncovered:.0f}% of the session ({_duration_text(outside['time'])}) "
                        "is outside every region; adding regions there would show "
                        "where that time goes."
                    ),
                )
            )

    if regions_only:
        top = regions_only[0]
        share = f"{100.0 * top['time'] / base:.1f}%" if base else ""
        kpis.append(
            _kpi(
                "Top hotspot",
                region_link(top["name"]),
                f"{share} of the session" if share else "",
            )
        )
        where = f" (in {_text(' › '.join(top['context']))})" if top["context"] else ""
        subject = (
            f"Time spent directly in {region_link(top['name'])}{where}, outside "
            "its nested regions,"
            if top["kind"] == "own"
            else f"{region_link(top['name'])}{where}"
        )
        findings.insert(
            0,
            (
                "info",
                f"{subject} is the largest hotspot: {_duration_text(top['time'])} over "
                f"{_plural(top['calls'], 'call')}"
                + (f", {share} of the session." if share else "."),
            ),
        )
        if len(regions_only) >= 3 and base:
            top_three = sum(entry["time"] for entry in regions_only[:3])
            findings.insert(
                1,
                (
                    "info",
                    (
                        f"The three largest hotspots account for "
                        f"{100.0 * top_three / base:.0f}% of the session."
                    ),
                ),
            )

    if balance is not None:
        worst = balance["regions"][0]
        tone = "warn" if worst["imbalance"] >= _IMBALANCE_FLAG_PCT else ""
        kpis.append(
            _kpi(
                "Load imbalance",
                f"+{worst['imbalance']:.0f}%",
                f"{_text(worst['name'])}, slowest on rank {worst['slowest']}",
                tone,
            )
        )
        if tone:
            findings.append(
                (
                    "warn",
                    (
                        f"{region_link(worst['name'])} is unevenly distributed across "
                        f"ranks: rank {worst['slowest']} spends {worst['imbalance']:.0f}% "
                        f"more time in it than the per-rank average "
                        f"({_duration_text(worst['excess'])} more). See "
                        f'<a href="#{section_id}-balance">load balance</a>.'
                    ),
                )
            )
        counts = balance["slowest_counts"]
        if counts:
            rank, slowest_in = counts.most_common(1)[0]
            if slowest_in >= 2 and slowest_in * 2 >= len(balance["regions"]):
                findings.append(
                    (
                        "info",
                        (
                            f"Rank {rank} is the slowest rank in {slowest_in} of "
                            f"{len(balance['regions'])} regions."
                        ),
                    )
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
        findings.append(
            (
                "warn",
                (
                    f"{region_link(worst['name'])} was called "
                    f"{_text(worst['calls'])} times at ~{worst['avg'] * 1e6:.1f} µs on "
                    "average; frequent short calls like this can make timer overhead "
                    "itself measurable."
                ),
            )
        )

    untimed = len(rows) - len(timed)
    if untimed:
        findings.append(
            (
                "info",
                f"{_text(untimed)} region(s) recorded no calls on the selected ranks.",
            )
        )
    findings_html = (
        '<ul class="findings">'
        + "".join(f'<li class="{tone}">{text}</li>' for tone, text in findings)
        + "</ul>"
        if findings
        else ""
    )
    return f'<div class="kpis">{"".join(kpis)}</div>{findings_html}'


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
            f'<span title="{_text(path)}"><code>{_text(path.name)}</code></span>'
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
    return f'<table class="meta-table"><tbody>{entries}</tbody></table>'


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


def _region_detail_html(results, region, ranks, functions=(), table_id="lp") -> str:
    """Expandable detail for one region: call site, tags, ranks and line profile."""
    parts = []
    if region.tags:
        parts.append(
            "".join(f'<span class="tag">{_text(tag)}</span>' for tag in region.tags),
        )
    if region.has_source:
        parts.append(
            f"<p class='muted'>{_text(region.source_file)}:{_text(region.source_lineno)}</p>",
        )
        # A run with line profiles shows the code, with its timings, there;
        # the captured snippet would only repeat it.
        if region.source_text and not any(results.line_profile.values()):
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
    if functions:
        parts.append(
            "<p class='muted'>Line profile</p>"
            + "".join(
                '<div class="lp-block"><p class="lp-head">'
                f'<span class="lp-func">{_text(entry["function"])}</span>'
                f"{_lp_location_html(results, entry)}"
                f'<span class="lp-total">{_text(_duration_text(entry["total"]))}</span></p>'
                + _line_profile_table(entry, f"{table_id}-{index}")
                + "</div>"
                for index, entry in enumerate(functions)
            )
        )
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
    results,
    rows,
    ranks,
    columns,
    region_ids=None,
    table_id="regions",
    line_profiles=None,
) -> str:
    """Region statistics laid out like the terminal summary table.

    Cell text comes from :func:`~scope_profiler.summary.format_region_table`,
    so the report shows the same tree, call counts and ``(own)`` rows. Each
    region is one ``<tbody>``, holding its row, its ``(own)`` row when it has
    children, and its expandable detail row, so sorting, filtering and
    collapsing the tree move them together.
    """
    region_ids = {} if region_ids is None else region_ids
    line_profiles = {} if line_profiles is None else line_profiles
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
        functions = line_profiles.get(row["name"], ())
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
            f'<td colspan="{len(keys) + 1}">'
            # A region on several call paths has a detail row per path, so
            # the ids of its line tables carry the row's position.
            + _region_detail_html(
                results, region, ranks, functions, f"{table_id}-{len(groups)}-lp"
            )
            + "</td></tr></tbody>"
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
        notes.append(
            "Click a row for its call site, line profile and per-rank breakdown."
            if line_profiles
            else "Click a row for its call site and per-rank breakdown."
        )
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


# Line profiles of scope-profiler's own frames - the session's __enter__,
# for one - say nothing about the code being profiled.
_PACKAGE_DIR = Path(__file__).resolve().parent


def _is_own_frame(filename) -> bool:
    try:
        return Path(filename).resolve().is_relative_to(_PACKAGE_DIR)
    except (OSError, ValueError):
        return False


def _line_profile_functions(results, ranks) -> list[dict]:
    """Line timings per profiled function, summed over the selected ranks."""
    available = results.line_profile
    selected_ranks = sorted(
        available if ranks is None else [rank for rank in ranks if rank in available],
    )
    functions: dict[tuple, dict] = {}
    for rank in selected_ranks:
        for record in available.get(rank, []):
            if _is_own_frame(record["filename"]):
                continue
            key = (
                record["region"],
                record["filename"],
                record["function"],
                int(record["first_lineno"]),
            )
            entry = functions.setdefault(
                key, {"lines": {}, "ranks": set(), "stored": {}}
            )
            entry["ranks"].add(rank)
            if record.get("source") and not entry["stored"]:
                first = int(record["source_first_lineno"])
                entry["stored"] = dict(
                    enumerate(str(record["source"]).split("\n"), start=first)
                )
            unit = float(record["unit"])
            for line, hits, elapsed in zip(
                record["line_numbers"], record["hits"], record["times"]
            ):
                stats = entry["lines"].setdefault(int(line), [0, 0.0])
                stats[0] += int(hits)
                stats[1] += float(elapsed) * unit
    result = []
    for (region, filename, function, first_lineno), entry in functions.items():
        result.append(
            {
                "region": region,
                "filename": filename,
                "function": function,
                "first_lineno": first_lineno,
                "lines": entry["lines"],
                # The source stored with the timings; files recorded before
                # it was stored are read from the source file instead.
                "stored": entry["stored"],
                "ranks": sorted(entry["ranks"]),
                "total": sum(seconds for _, seconds in entry["lines"].values()),
            }
        )
    result.sort(key=lambda entry: -entry["total"])
    return result


def _dedent_width(lines) -> int:
    """The indentation every non-blank line shares, to strip from all of them."""
    widths = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    return min(widths, default=0)


_BUILTIN_NAMES = frozenset(dir(builtins))
# f-strings (3.12+) and t-strings (3.14+) arrive as several tokens.
_STRING_TOKENS = frozenset(
    {tokenize.STRING}
    | {
        getattr(tokenize, name)
        for name in (
            "FSTRING_START",
            "FSTRING_MIDDLE",
            "FSTRING_END",
            "TSTRING_START",
            "TSTRING_MIDDLE",
            "TSTRING_END",
        )
        if hasattr(tokenize, name)
    }
)
_LAYOUT_TOKENS = frozenset(
    {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}
)


def _highlight_python(lines: list[str]) -> list[str]:
    """Python source lines as HTML, with each token in a classed span.

    The standard library's tokenizer rather than a highlighting package: the
    report takes no dependency for it, and line profiles are always Python.
    The lines are a slice of a file, so they may end inside a statement or a
    string; whatever tokenizes before that is highlighted, the rest is plain.
    """
    marks: list[list[tuple[int, int, str]]] = [[] for _ in lines]
    previous = None
    decorator = False
    try:
        tokens = tokenize.generate_tokens(io.StringIO("\n".join(lines) + "\n").readline)
        for token in tokens:
            kind = None
            if token.type == tokenize.COMMENT:
                kind = "tk-com"
            elif token.type in _STRING_TOKENS:
                kind = "tk-str"
            elif token.type == tokenize.NUMBER:
                kind = "tk-num"
            elif token.type == tokenize.OP:
                if token.string == "@" and (
                    previous is None or previous.type == tokenize.NEWLINE
                ):
                    decorator = True
                elif token.string != ".":
                    decorator = False
                if decorator:
                    kind = "tk-dec"
            elif token.type == tokenize.NAME:
                after_dot = previous is not None and previous.string == "."
                if decorator:
                    kind = "tk-dec"
                elif previous is not None and previous.string in ("def", "class"):
                    kind = "tk-fn"
                elif token.string in ("True", "False", "None"):
                    kind = "tk-const"
                elif keyword.iskeyword(token.string):
                    kind = "tk-kw"
                elif token.string in ("self", "cls"):
                    kind = "tk-self"
                elif token.string in _BUILTIN_NAMES and not after_dot:
                    kind = "tk-bi"
            if token.type not in _LAYOUT_TOKENS or token.type == tokenize.NEWLINE:
                previous = token
            if kind is None:
                continue
            (first_row, first_col), (last_row, last_col) = token.start, token.end
            # A triple-quoted string spans lines: mark its part of each.
            for row in range(first_row, min(last_row, len(lines)) + 1):
                start = first_col if row == first_row else 0
                end = last_col if row == last_row else len(lines[row - 1])
                if end > start:
                    marks[row - 1].append((start, end, kind))
    except (tokenize.TokenError, SyntaxError):
        pass
    rendered = []
    for line, line_marks in zip(lines, marks):
        parts = []
        cursor = 0
        for start, end, kind in sorted(line_marks):
            if start < cursor:
                continue
            parts.append(html.escape(line[cursor:start]))
            parts.append(f'<span class="{kind}">{html.escape(line[start:end])}</span>')
            cursor = end
        parts.append(html.escape(line[cursor:]))
        rendered.append("".join(parts))
    return rendered


@functools.lru_cache(maxsize=256)
def _source_path(filename: str) -> str:
    """Where to read a recorded source file from, here.

    Profiles store paths relative to the working directory or to the
    ``sys.path`` entry the file was imported from, so a relative path is
    looked up against both, in that order.
    """
    if os.path.isabs(filename) or os.path.isfile(filename):
        return filename
    for root in sys.path:
        candidate = os.path.join(root or os.curdir, filename)
        if os.path.isfile(candidate):
            return candidate
    return filename


def _line_profile_table(entry, table_id) -> str:
    """One function's source, each line with the time spent on it."""
    recorded = entry["lines"]
    total = entry["total"]
    first = min(entry["first_lineno"], min(recorded))
    last = max(recorded)
    stored = entry.get("stored") or {}
    path = _source_path(entry["filename"])
    sources = {
        number: (
            stored[number]
            if number in stored
            else linecache.getline(path, number).rstrip("\n")
        ).expandtabs()
        for number in range(first, last + 1)
    }
    # With the source at hand, show the function as it reads, unrecorded
    # lines (comments, blank lines, the def) included; without it, only
    # the lines that have timings.
    numbers = (
        list(range(first, last + 1)) if any(sources.values()) else sorted(recorded)
    )
    strip = _dedent_width(sources[number] for number in numbers)
    highlighted = dict(
        zip(numbers, _highlight_python([sources[number][strip:] for number in numbers]))
    )
    hottest = max(recorded, key=lambda number: recorded[number][1])
    body = []
    for number in numbers:
        source = highlighted[number]
        if number not in recorded:
            body.append(
                f'<tr class="lp-idle"><td class="lp-lineno">{number}</td>'
                f'<td></td><td></td><td></td><td></td><td class="lp-src">{source}</td></tr>'
            )
            continue
        hits, seconds = recorded[number]
        percent = 100.0 * seconds / total if total else 0.0
        per_hit = seconds / hits if hits else 0.0
        row_class = ' class="lp-hot"' if number == hottest and total else ""
        body.append(
            f'<tr{row_class}><td class="lp-lineno">{number}</td>'
            f"<td>{hits:,}</td><td>{_text(_duration_text(seconds))}</td>"
            f"<td>{_text(_duration_text(per_hit))}</td>"
            f'<td class="lp-pct" style="--pct:{min(percent, 100.0):.3g}%">{percent:.2f}%</td>'
            f'<td class="lp-src">{source}</td></tr>'
        )
    return (
        f'<div class="lp-scroll"><table class="lp-table" id="{table_id}"><thead><tr>'
        "<th>line</th><th>hits</th><th>time</th><th>per hit</th><th>% time</th>"
        '<th class="lp-src">source</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div>"
        + (
            ""
            if any(sources.values())
            else '<p class="muted table-note">No source: this profile does not store '
            f"it, and <code>{_text(entry['filename'])}</code> could not be read from "
            "here. Profiles recorded with the current scope-profiler carry the "
            "source with them.</p>"
        )
    )


def _lp_location_html(results, entry) -> str:
    """Where a profiled function is, and on how many ranks it ran."""
    location = f"{Path(entry['filename']).name}:{entry['first_lineno']}"
    ranks_note = (
        f" · {_plural(len(entry['ranks']), 'rank')}"
        if len(entry["ranks"]) > 1
        else f" · rank {entry['ranks'][0]}" if results.num_ranks > 1 else ""
    )
    return (
        f'<span class="lp-loc" title="{_text(entry["filename"])}">'
        f"{_text(location)}{_text(ranks_note)}</span>"
    )


def _line_profile_html(results, functions, section_id="run-0") -> str:
    """Line profiles of regions the table does not show: one row per function.

    Regions in the table carry their line profile in their detail row; this
    lists the rest - regions the include/exclude patterns left out, say.
    Every function starts collapsed, and only the largest few are listed
    until asked for.
    """
    grand_total = sum(entry["total"] for entry in functions)
    list_id = f"{section_id}-lp-functions"
    rows = []
    for index, entry in enumerate(functions):
        share = 100.0 * entry["total"] / grand_total if grand_total else 0.0
        extra = " lp-extra" if index >= _LP_FUNCTIONS else ""
        rows.append(
            f'<details class="lp-function{extra}">'
            f'<summary><span class="lp-func">{_text(entry["function"])}</span>'
            f'<span class="lp-region">{_text(entry["region"])}</span>'
            f"{_lp_location_html(results, entry)}"
            f'<span class="lp-bar"><span style="width:{share:.3g}%"></span></span>'
            f'<span class="lp-total">{_text(_duration_text(entry["total"]))}</span>'
            f'<span class="lp-share">{share:.0f}%</span>'
            "</summary>"
            + _line_profile_table(entry, f"{section_id}-lp-{index}")
            + "</details>"
        )
    more = len(functions) - _LP_FUNCTIONS
    toggle = (
        f'<div class="table-tools"><button type="button" data-show-all="{list_id}"'
        f' data-more-label="Show all {len(functions)} functions"'
        f' data-less-label="Show the top {_LP_FUNCTIONS}">'
        f"Show all {len(functions)} functions</button></div>"
        if more > 0
        else ""
    )
    return (
        '<p class="muted">Profiled functions whose region is not in the table '
        "above, largest first; a region in the table shows its line profile when "
        "its row is clicked. Times are summed over the selected ranks.</p>"
        f'<div class="lp-functions" id="{list_id}">' + "".join(rows) + "</div>" + toggle
    )


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


def _top_label(value: int, regions: int) -> str:
    """The Regions slider's readout; mirrored by the report script's topLabel."""
    return f"all {regions}" if value >= regions else f"top {value} of {regions}"


def _depth_label(value: int, max_depth: int) -> str:
    """The Depth slider's readout; mirrored by the report script's depthLabel."""
    if value >= max_depth:
        return "all levels"
    if value == 0:
        return "top level only"
    return f"{value} level{'s' if value > 1 else ''} of children"


def _region_index(runs, include, exclude, ranks) -> dict[str, list]:
    """``{region: [seconds, depth]}`` for the Regions and Depth sliders.

    ``seconds`` is the region's largest total over the runs, summed over the
    selected ranks, so a region costly in any run ranks high. ``depth`` is the
    shallowest level its calls reach on the first rank of any run - 0 for a
    top-level call - or None when that rank's calls do not nest.
    """
    from scope_profiler.call_stack import NestingError, build_call_arrays

    index: dict[str, list] = {}
    for run in runs:
        for region in run.get_regions(include=include, exclude=exclude):
            seconds = sum(
                rank_region.total_duration
                for rank, rank_region in region.regions.items()
                if ranks is None or rank in ranks
            )
            entry = index.setdefault(region.name, [0.0, None])
            entry[0] = max(entry[0], seconds)
        rank = 0 if ranks is None else min(ranks, default=0)
        try:
            arrays = build_call_arrays(run.get_regions(), rank)
        except NestingError:
            continue
        shallowest: dict[str, int] = {}
        for row, depth in zip(arrays.region_index.tolist(), arrays.depth.tolist()):
            name = arrays.names[row]
            shallowest[name] = min(depth, shallowest.get(name, depth))
        for name, depth in shallowest.items():
            if name in index:
                known = index[name][1]
                index[name][1] = depth if known is None else min(known, depth)
    return index


def _region_limits(controls: dict) -> str:
    """The Regions and Depth sliders for the filter bar, each only when it can
    change what is drawn: two regions or more, and calls nested at least one
    level deep. A slider at its maximum means all."""

    def slider(kind, label, maximum, value, minimum, text, hint):
        control = f"region-{kind}"
        return (
            f'<label for="{control}" title="{hint}">{label}</label>'
            f'<input type="range" class="region-{kind}" id="{control}"'
            f' title="{hint}"'
            f' min="{minimum}" max="{maximum}" step="1" value="{value}">'
            f'<output for="{control}">{text}</output>'
        )

    html = ""
    regions = controls["regions"]
    if regions > 1:
        top = controls["top"] or regions
        html += slider(
            "top",
            "Regions",
            regions,
            top,
            1,
            _top_label(top, regions),
            "Draw only the regions with the most time, in every chart",
        )
    max_depth = controls["max_depth"]
    if max_depth >= 1:
        html += slider(
            "depth",
            "Depth",
            max_depth,
            max_depth,
            0,
            _depth_label(max_depth, max_depth),
            "Hide regions, and timeline calls, nested deeper than this",
        )
    return f'<span class="region-limits">{html}</span>' if html else ""


def _chart_description(title: str, payload: dict, comparison: bool = False) -> str:
    """Explain how to read one chart in the report."""
    if title.startswith("Timeline:"):
        text = (
            "Each bar is one recorded region call on rank 0. Its position and "
            "width show when the call started and how long it ran; colors "
            "identify regions."
        )
    elif title == "Region durations":
        what = (
            "mean call duration"
            if payload.get("metrics") == ["avg"]
            else "total recorded duration"
        )
        if payload.get("options", {}).get("stack_children"):
            text = (
                f"Each bar shows a region's {what}, summed over the selected "
                "ranks. The stacked segments divide that time between the "
                "region itself (self) and the regions it calls directly."
            )
        elif not comparison:
            text = (
                f"Each bar shows a region's {what}, summed over the selected "
                "ranks, longest first."
            )
        else:
            text = (
                f"Grouped bars compare each region's {what} across the "
                "profiled runs."
            )
    elif title.startswith("Change:"):
        text = (
            "Percent change in each region's total duration, candidate over "
            "baseline. Bars above zero got slower; a region measured in only "
            "one run leaves a gap rather than reading as a 100% change."
        )
    elif title == "Speedup":
        # Marked so the x-axis buttons can rename it along with the chart.
        field = (
            "<span data-axis-label>"
            + _text(payload.get("options", {}).get("x_label", "ranks"))
            + "</span>"
        )
        text = (
            "Each line is one region's speedup: its mean call duration on the run "
            f"with the fewest {field} divided by its mean call duration on "
            "each run. The dashed line is ideal scaling; a region below it gains "
            f"less than its extra {field} would allow. The "
            f"{_SPEEDUP_REGIONS} regions with the most time on that first run are "
            "shown. Valid for a strong-scaling study: the same total problem size "
            "on every run."
        )
    elif title == "Weak scaling":
        # Marked so the x-axis buttons can rename it along with the chart.
        field = (
            "<span data-axis-label>"
            + _text(payload.get("options", {}).get("x_label", "ranks"))
            + "</span>"
        )
        text = (
            "Each line is one region's weak-scaling efficiency: its mean call "
            f"duration on the run with the fewest {field} divided by its "
            "mean call duration on each run. The dashed line at 1 is ideal: the "
            "same time per call however large the run; a region below it loses "
            "time to costs that grow with the run, such as communication or load "
            f"imbalance. The {_SPEEDUP_REGIONS} regions with the most time on that "
            "first run are shown. Valid for a weak-scaling study: the problem "
            "grows with the run, so every run has the same problem size per "
            "rank or core."
        )
    elif title == "Rank heatmap":
        text = (
            "This heatmap uses exclusive timings. Exclusive duration is the time "
            "spent in a region itself, excluding time spent in nested child "
            "regions; this prevents the enclosing session region from dominating "
            "the heatmap."
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
    else:
        return ""
    return f'<p class="muted">{text}</p>'


def _rank_balance(results, rows, ranks) -> dict | None:
    """Per-rank time in each region, for the load-balance views.

    Own time rather than inclusive: a parent's inclusive total repeats every
    child's imbalance, so ranking it would mostly name the session. Own times
    are disjoint, so they also add up to each rank's time inside regions.
    Without a call tree there is no own time, and totals are used instead.
    """
    selected = sorted(
        rank
        for rank in (range(results.num_ranks) if ranks is None else ranks)
        if 0 <= rank < results.num_ranks
    )
    if len(selected) < 2:
        return None
    timed = [
        row for row in rows if row["total"] is not None and row["name"] != _SESSION
    ]
    own = bool(timed) and all("call_path" in row for row in timed)
    regions = []
    for name in dict.fromkeys(row["name"] for row in timed):
        region = results.get_region(name)
        values = np.asarray(
            [
                (
                    (
                        region.regions[rank].total_exclusive_duration
                        if own
                        else region.regions[rank].total_duration
                    )
                    if rank in region.regions
                    else 0.0
                )
                for rank in selected
            ],
            dtype=float,
        )
        mean = float(values.mean())
        if mean <= 0:
            continue
        peak = int(values.argmax())
        regions.append(
            {
                "name": name,
                "values": values,
                "mean": mean,
                "min": float(values.min()),
                "max": float(values.max()),
                "slowest": selected[peak],
                "imbalance": (float(values.max()) / mean - 1.0) * 100.0,
                "excess": float(values.max()) - mean,
            }
        )
    if not regions:
        return None
    # Largest excess first: the time balancing a region could save.
    regions.sort(key=lambda entry: -entry["excess"])
    return {
        "ranks": selected,
        "regions": regions,
        "own": own,
        "totals": np.sum([entry["values"] for entry in regions], axis=0),
        "slowest_counts": Counter(
            entry["slowest"] for entry in regions if entry["excess"] > 0
        ),
    }


def _deviation_style(value: float, mean: float) -> str:
    """Background for a matrix cell: blue below the region's mean, red above."""
    if mean <= 0:
        return ""
    deviation = (value - mean) / mean
    if abs(deviation) < _SAME_PCT / 100.0:
        return ""
    strength = min(abs(deviation) / 0.5, 1.0)
    color = "220,38,38" if deviation > 0 else "37,99,235"
    return f' style="background:rgba({color},{0.08 + 0.47 * strength:.2f})"'


def _load_balance_html(results, balance, region_ids, section_id) -> str:
    """How evenly each region's time is spread over the ranks."""
    run = _text(results.display_label)
    metric = "own time" if balance["own"] else "total time"
    scale = max(entry["max"] for entry in balance["regions"]) or 1.0
    table_id = f"{section_id}-balance-table"
    body = []
    for index, entry in enumerate(balance["regions"]):
        name = _text(entry["name"])
        low = 100.0 * entry["min"] / scale
        high = 100.0 * entry["max"] / scale
        flag = ' class="flag"' if entry["imbalance"] >= _IMBALANCE_FLAG_PCT else ""
        extra = " extra" if index >= _BALANCE_ROWS else ""
        body.append(
            f'<tr class="select-row{extra}" data-region="{name}" data-run="{run}"'
            f' data-filter-region="{name}">'
            f'<td title="{name}">{name}</td>'
            f"<td>{_text(_duration_text(entry['mean']))}</td>"
            f"<td>{_text(_duration_text(entry['min']))}</td>"
            f"<td>{_text(_duration_text(entry['max']))}</td>"
            f"<td>{entry['slowest']}</td>"
            f"<td{flag}>+{entry['imbalance']:.0f}%</td>"
            f"<td>{_text(_duration_text(entry['excess']))}</td>"
            '<td><span class="spread">'
            f'<span class="spread-range" style="left:{low:.3g}%;width:{max(high - low, 0.5):.3g}%"></span>'
            f'<span class="spread-mean" style="left:{100.0 * entry["mean"] / scale:.3g}%"></span>'
            "</span></td></tr>"
        )
    more = len(balance["regions"]) - _BALANCE_ROWS
    toggle = (
        f'<div class="table-tools"><button type="button" data-show-all="{table_id}"'
        f' data-more-label="Show all {len(balance["regions"])} regions"'
        f' data-less-label="Show the top {_BALANCE_ROWS}">'
        f"Show all {len(balance['regions'])} regions</button></div>"
        if more > 0
        else ""
    )
    table = (
        f'<div class="table-scroll"><table class="balance-table" id="{table_id}"><thead><tr>'
        "<th>region</th><th>mean</th><th>min</th><th>max</th><th>slowest rank</th>"
        '<th title="slowest rank over the per-rank mean">imbalance</th>'
        '<th title="slowest rank minus the per-rank mean">excess</th>'
        "<th>min · mean · max</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
        + toggle
    )
    return (
        f'<div id="{section_id}-balance"><h3>Load balance</h3>'
        f'<p class="muted">Each region\'s {metric} per rank'
        + (
            ", excluding nested regions, so a parent does not repeat its "
            "children's imbalance"
            if balance["own"]
            else ""
        )
        + ". <em>Excess</em> is how much longer the slowest rank spends than the "
        "average rank: the most balancing that region could save. Regions are "
        "ordered by it.</p>"
        + table
        + _rank_matrix_html(results, balance, section_id)
        + "</div>"
    )


def _rank_matrix_html(results, balance, section_id) -> str:
    """Every rank side by side, region by region, against the region's mean."""
    run = _text(results.display_label)
    ranks = balance["ranks"][:_MATRIX_RANKS]
    columns = len(ranks)
    by_size = sorted(balance["regions"], key=lambda entry: -entry["mean"])
    shown = by_size[:_MATRIX_REGIONS]
    totals = balance["totals"][:columns]
    totals_mean = float(np.mean(totals))
    # Ranks that wait for each other end up with the same time in regions;
    # the largest of near-equal totals is noise, not a slowest rank.
    slowest_rank = (
        ranks[int(np.argmax(totals))]
        if totals_mean
        and (float(np.max(totals)) - totals_mean) / totals_mean >= _SAME_PCT / 100.0
        else None
    )
    compact = columns > _MATRIX_TEXT_RANKS

    def cells(values, mean) -> str:
        rendered = []
        for rank, value in zip(ranks, values[:columns]):
            deviation = 100.0 * (value - mean) / mean if mean else 0.0
            text = _signed_pct(deviation) if abs(deviation) >= 0.05 else "0%"
            rendered.append(
                f'<td class="heat"{_deviation_style(value, mean)}'
                f' title="rank {rank}: {_text(_duration_text(value))} ({text} vs mean)">'
                f"{text}</td>"
            )
        return "".join(rendered)

    header = "".join(
        (
            f'<th class="slowest" title="slowest rank overall">{rank}</th>'
            if rank == slowest_rank
            else f"<th>{rank}</th>"
        )
        for rank in ranks
    )
    body = "".join(
        f'<tr class="select-row" data-region="{_text(entry["name"])}" data-run="{run}"'
        f' data-filter-region="{_text(entry["name"])}">'
        f'<td title="{_text(entry["name"])}">{_text(entry["name"])}</td>'
        f"<td>{_text(_duration_text(entry['mean']))}</td>"
        + cells(entry["values"], entry["mean"])
        + "</tr>"
        for entry in shown
    )
    footer = (
        "<tfoot><tr><td>all regions</td>"
        f"<td>{_text(_duration_text(totals_mean))}</td>"
        + cells(totals, totals_mean)
        + "</tr></tfoot>"
    )
    notes = []
    if len(by_size) > len(shown):
        notes.append(f"The {len(shown)} regions with the most time per rank are shown.")
    if len(balance["ranks"]) > columns:
        notes.append(
            f"The first {columns} of {len(balance['ranks'])} ranks are shown; the "
            "rank heatmap chart shows all of them."
        )
    overall = (
        f"Rank {slowest_rank}, in red in the header, spends the most time in "
        "regions overall."
        if slowest_rank is not None
        else "Every rank spends about the same time in regions overall, so the "
        "differences above are ranks waiting for each other."
    )
    return (
        f'<h3 id="{section_id}-ranks">Rank comparison</h3>'
        '<p class="muted">Each cell compares one rank\'s time in a region with the '
        f"region's mean over the ranks. {overall} Hover a cell for its time.</p>"
        '<p class="legend"><span>faster than the mean</span><span class="legend-scale">'
        "</span><span>slower</span></p>"
        f'<div class="table-scroll"><table class="rank-matrix{" compact" if compact else ""}">'
        "<thead><tr><th>region</th><th>mean</th>"
        + header
        + "</tr></thead><tbody>"
        + body
        + "</tbody>"
        + footer
        + "</table></div>"
        + (f'<p class="muted table-note">{" ".join(notes)}</p>' if notes else "")
    )


def _threads(run) -> int | None:
    """A run's OpenMP thread count, when its metadata recorded one."""
    try:
        return int(run.metadata["omp_num_threads"])
    except (KeyError, TypeError, ValueError):
        return None


def _has_call_timestamps(run) -> bool:
    """Whether a run recorded its calls, rather than only summary statistics.

    Aggregated and summary-only profiles keep per-region totals but no call
    timestamps, so the nesting a stacked durations bar needs cannot be
    reconstructed from them.
    """
    return all(
        rank_region.stored_summary is None
        for region in run.get_regions()
        for rank_region in region.regions.values()
    )


def _scaling_field(runs) -> str | None:
    """The parallelism compared runs differ in, for a speedup chart.

    MPI ranks whenever they change, even when threads change too (the cores
    axis is then a button away); threads when only they change; None when the
    runs share both, as two versions of the same code do.
    """
    if len({run.num_ranks for run in runs}) > 1:
        return "num_ranks"
    threads = [_threads(run) for run in runs]
    threads_vary = None not in threads and len(set(threads)) > 1
    return "omp_num_threads" if threads_vary else None


#: ``speedup_x`` choices for a comparison report's speedup chart, each naming
#: the run property it puts on the x-axis. ``"auto"`` picks for itself (see
#: :func:`_scaling_field`).
SPEEDUP_X_FIELDS = {
    "ranks": "num_ranks",
    "nodes": "num_nodes",
    "threads": "omp_num_threads",
    "cores": "total_cores",
}
SPEEDUP_X_CHOICES = ("auto", *SPEEDUP_X_FIELDS)


def _scaling_value(run, field: str) -> int | None:
    """A run's value of a scaling field, or None when the run did not record it."""
    if field == "num_ranks":
        return run.num_ranks
    if field == "num_nodes":
        return run.num_nodes
    threads = _threads(run)
    if threads is None:
        return None
    return threads if field == "omp_num_threads" else run.num_ranks * threads


class SpeedupAxisError(ValueError):
    """The requested speedup x-axis is unknown, or some run did not record it."""


def _check_speedup_x(runs, speedup_x: str) -> str | None:
    """The scaling field ``speedup_x`` names, checked against every run.

    ``"auto"`` resolves to :func:`_scaling_field`'s choice. A named field
    every run has to have recorded: a speedup chart silently missing some
    runs, or silently drawn over another axis, would misread the study.
    """
    if speedup_x not in SPEEDUP_X_CHOICES:
        raise SpeedupAxisError(
            f"speedup_x must be one of {', '.join(SPEEDUP_X_CHOICES)}, "
            f"got {speedup_x!r}",
        )
    if speedup_x == "auto":
        return _scaling_field(runs)
    field = SPEEDUP_X_FIELDS[speedup_x]
    missing = [run.display_label for run in runs if _scaling_value(run, field) is None]
    if missing:
        raise SpeedupAxisError(
            f"the {speedup_x} axis needs {field!r} in every run's metadata, "
            f"but {', '.join(missing)} did not record it"
            + (
                " (files written by older scope-profiler versions lack the node "
                "count of a multi-rank run)"
                if field == "num_nodes"
                else ""
            )
            + "; choose another axis, or 'auto'",
        )
    return field


def _speedup_axes(runs, field: str) -> list[str]:
    """The scaling fields a speedup chart over ``field`` can switch to.

    Every field all runs recorded and differ in, in :data:`SPEEDUP_X_FIELDS`
    order; ``field`` itself is always one. A field whose values match another
    one's run for run - cores when every run has one thread - would draw the
    same chart again and is left out.
    """
    order = list(SPEEDUP_X_FIELDS.values())
    kept: dict[str, tuple] = {}
    for candidate in [field, *(other for other in order if other != field)]:
        values = tuple(_scaling_value(run, candidate) for run in runs)
        if candidate != field and (
            None in values or len(set(values)) < 2 or values in kept.values()
        ):
            continue
        kept[candidate] = values
    return [candidate for candidate in order if candidate in kept]


def _comparison_entries(per_run_rows) -> list[dict]:
    """Align every run's region rows by call path, in call-tree order.

    The baseline's tree sets the order. A path only a later run has is placed
    after the last row below its parent, so the merged rows still read as one
    tree; rows without a call path match by name.
    """
    entries: list[dict] = []
    index: dict[str, int] = {}

    def key_of(row) -> str:
        return (
            f"path:{row['call_path']}" if "call_path" in row else f"name:{row['name']}"
        )

    def insert_at(row) -> int:
        if "call_path" not in row or " > " not in row["call_path"]:
            return len(entries)
        parent = "path:" + row["call_path"].rsplit(" > ", 1)[0]
        if parent not in index:
            return len(entries)
        position = index[parent] + 1
        prefix = parent + " > "
        while position < len(entries) and entries[position]["key"].startswith(prefix):
            position += 1
        return position

    for run_index, rows in enumerate(per_run_rows):
        for row in rows:
            key = key_of(row)
            if key not in index:
                position = insert_at(row)
                entries.insert(
                    position,
                    {
                        "key": key,
                        "name": row["name"],
                        "depth": row.get("depth", 0),
                        "context": (
                            [
                                part
                                for part in row["call_path"].split(" > ")[:-1]
                                if part != _SESSION
                            ]
                            if "call_path" in row
                            else []
                        ),
                        "values": [None] * len(per_run_rows),
                    },
                )
                index = {entry["key"]: i for i, entry in enumerate(entries)}
            entries[index[key]]["values"][run_index] = row
    return entries


def _metric_value(row, metric):
    if row is None or row["total"] is None:
        return None
    if metric == "own":
        return row["total"] if row["exclusive"] is None else row["exclusive"]
    if metric == "calls":
        return row["calls"]
    return row[metric]


def _metric_text(value, metric) -> str:
    if value is None:
        return "-"
    return f"{value:,}" if metric == "calls" else _duration_text(value)


_COMPARE_METRICS = (
    ("total", "total"),
    ("own", "own"),
    ("avg", "avg/call"),
    ("calls", "calls"),
)


def _change_html(base, value, metric) -> tuple[str, float | None]:
    """A change cell's content, and the size of the change for sorting."""
    if base is None and value is None:
        return "-", None
    if base is None:
        return '<span class="badge new">new</span>', abs(value)
    if value is None:
        return '<span class="badge gone">gone</span>', abs(base)
    delta = value - base
    if not base:
        return '<span class="badge new">new</span>', abs(delta)
    pct = 100.0 * delta / base
    if metric == "calls":
        tone = "same" if delta == 0 else "neutral down" if delta < 0 else "neutral"
        title = f"{delta:+,} calls"
    else:
        tone = "same" if abs(pct) < _SAME_PCT else "faster" if delta < 0 else "slower"
        speed = (
            ""
            if not value
            else (
                f" · {base / value:.2f}× faster"
                if delta < 0
                else f" · {value / base:.2f}× slower"
            )
        )
        title = f"{'+' if delta >= 0 else '−'}{_duration_text(abs(delta))}{speed}"
    content = (
        f'<span class="delta {tone}" title="{_text(title)}">'
        f'<span class="delta-text">{_signed_pct(pct)}</span></span>'
    )
    return content, abs(delta)


def _per_rank_row(row) -> dict:
    """A row's times per rank: its totals over the ranks that ran it.

    Summed over ranks, a region's time is what it costs; compared across
    runs of different sizes, that grows with the rank count even as the run
    gets faster. Per rank, it follows what a reader waits for.
    """
    count = row.get("num_ranks") or 1
    return {
        **row,
        "total": None if row["total"] is None else row["total"] / count,
        "exclusive": None if row["exclusive"] is None else row["exclusive"] / count,
    }


def _comparison_table_html(runs, entries, per_rank=False) -> str:
    """Every region's duration in every run, against the baseline."""
    labels = [_text(run.display_label) for run in runs]
    header = f"<th>region</th><th>{labels[0]}</th>" + "".join(
        f"<th>{label}</th><th>vs {labels[0]}</th>" for label in labels[1:]
    )
    body = []
    for order, entry in enumerate(entries):
        name = _text(entry["name"])
        cells = []
        changes = {}
        for run_index, row in enumerate(entry["values"]):
            values = []
            deltas = []
            for metric, _ in _COMPARE_METRICS:
                value = _metric_value(row, metric)
                values.append(
                    f'<span data-m="{metric}">{_text(_metric_text(value, metric))}</span>'
                )
                if run_index:
                    base = _metric_value(entry["values"][0], metric)
                    content, size = _change_html(base, value, metric)
                    deltas.append(f'<span data-m="{metric}">{content}</span>')
                    # Sorting by change follows the last run against the baseline.
                    changes[metric] = size
            cells.append(f"<td>{''.join(values)}</td>")
            if run_index:
                cells.append(f"<td>{''.join(deltas)}</td>")
        change_attrs = "".join(
            f' data-change-{metric}="{"" if size is None else f"{size:.9g}"}"'
            for metric, size in changes.items()
        )
        title = " › ".join([*entry["context"], entry["name"]])
        body.append(
            f'<tr class="select-row" data-region="{name}" data-filter-region="{name}"'
            f' data-order="{order}"{change_attrs}>'
            f'<td title="{_text(title)}"><span class="cmp-name"'
            f' style="padding-left:{1.1 * entry["depth"]:.3g}em">{name}</span></td>'
            + "".join(cells)
            + "</tr>"
        )
    metric_buttons = "".join(
        f'<button type="button" data-compare-table="compare-table"'
        f' data-compare-metric="{metric}" aria-pressed="{str(metric == "total").lower()}">'
        f"{label}</button>"
        for metric, label in _COMPARE_METRICS
    )
    tools = (
        '<div class="table-tools"><span class="tools-label">Show</span>'
        + metric_buttons
        + '<span class="tools-label">Order</span>'
        '<button type="button" data-compare-table="compare-table" data-compare-sort="tree"'
        ' aria-pressed="true">call tree</button>'
        '<button type="button" data-compare-table="compare-table" data-compare-sort="change"'
        ' aria-pressed="false">largest change</button></div>'
    )
    return (
        tools
        + '<div class="table-scroll"><table class="compare-table metric-total" id="compare-table">'
        + f"<thead><tr>{header}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
        '<p class="muted table-note">'
        + (
            "The runs have different rank counts, so total and own times are per "
            "rank: each region's time divided by the ranks that ran it. "
            if per_rank
            else ""
        )
        + "Regions are matched by call path. <em>own</em> is "
        "the time outside nested regions, so own-time changes add up to the change in "
        "the session. Changes under 2% are greyed out; hover a change for its size in "
        "seconds. Click a row to highlight the region in the charts.</p>"
    )


def _change_items(entries, run_index, direction) -> list[dict]:
    """The largest own-time changes of one run against the baseline.

    Only changes that matter on both scales: at least 2% of the region's own
    baseline time, so timer jitter on a large region does not count, and at
    least 0.5% of the baseline's time in regions, so a tiny region doubling
    does not headline the comparison.
    """
    owns = [
        (
            _metric_value(entry["values"][0], "own") or 0.0,
            _metric_value(entry["values"][run_index], "own") or 0.0,
        )
        for entry in entries
    ]
    floor = _CHANGE_FLOOR_PCT / 100.0 * sum(base for base, _ in owns)
    changes = []
    for entry, (base, value) in zip(entries, owns):
        delta = value - base
        if (
            delta * direction > 0
            and abs(delta) >= floor
            and (not base or 100.0 * abs(delta) / base >= _SAME_PCT)
        ):
            changes.append({**entry, "base": base, "value": value, "delta": delta})
    changes.sort(key=lambda change: -abs(change["delta"]))
    return changes[:_CHANGE_COUNT]


def _change_list_html(changes, scale, tone) -> str:
    if not changes:
        return '<p class="muted">None.</p>'
    items = []
    for change in changes:
        name = _text(change["name"])
        label = (
            "<em>outside any region</em>"
            if change["name"] == _SESSION
            else f"<strong>{name}</strong>"
        )
        context = (
            f'<span class="hs-path">in {" › ".join(_text(p) for p in change["context"])}</span>'
            if change["context"]
            else ""
        )
        pct = (
            f" ({_signed_pct(100.0 * change['delta'] / change['base'])})"
            if change["base"]
            else " (new)"
        )
        width = 100.0 * abs(change["delta"]) / scale if scale else 0.0
        sign = "+" if change["delta"] > 0 else "−"
        items.append(
            f'<li class="change-item" data-filter-region="{name}">'
            f'<span class="hs-name">{label}{context}</span>'
            f'<span class="change-track"><span class="change-fill {tone}"'
            f' style="width:{width:.3g}%"></span></span>'
            f'<span class="change-value delta-text {tone}">{sign}'
            f"{_text(_duration_text(abs(change['delta'])))}"
            f'<span class="muted">{_text(pct)}</span></span></li>'
        )
    return '<ol class="change-list">' + "".join(items) + "</ol>"


def _wall_kpi(base_run, run) -> str:
    before, after = base_run.time_span, run.time_span
    if not before or not after:
        return _kpi("Wall time", _text(_duration_text(after)))
    pct = 100.0 * (after - before) / before
    tone = "" if abs(pct) < _SAME_PCT else "good" if pct < 0 else "bad"
    speed = (
        f"{before / after:.2f}× faster"
        if after < before
        else f"{after / before:.2f}× slower"
    )
    return _kpi(
        "Wall time",
        f"{_text(_duration_text(before))} → {_text(_duration_text(after))}",
        f"{_signed_pct(pct)} · {speed}",
        tone,
    )


def _candidate_html(runs, entries, run_index) -> str:
    """One run against the baseline: headline numbers and what moved."""
    base_run, run = runs[0], runs[run_index]
    improvements = _change_items(entries, run_index, -1)
    regressions = _change_items(entries, run_index, 1)
    scale = max(
        (abs(change["delta"]) for change in improvements + regressions), default=0.0
    )
    kpis = [_wall_kpi(base_run, run)]
    for title, changes in (
        ("Largest improvement", improvements),
        ("Largest regression", regressions),
    ):
        if changes:
            change = changes[0]
            sign = "+" if change["delta"] > 0 else "−"
            kpis.append(
                _kpi(
                    title,
                    f"<code>{_text(change['name'])}</code>",
                    f"{sign}{_text(_duration_text(abs(change['delta'])))} own time",
                )
            )
        else:
            kpis.append(_kpi(title, "none"))
    return (
        f'<div class="compare-candidate"><h3>{_text(run.display_label)} vs '
        f"{_text(base_run.display_label)}</h3>"
        f'<div class="kpis">{"".join(kpis)}</div>'
        '<div class="changes"><div><h4>Faster</h4>'
        + _change_list_html(improvements, scale, "faster")
        + "</div><div><h4>Slower</h4>"
        + _change_list_html(regressions, scale, "slower")
        + "</div></div>"
        '<p class="muted table-note">Changes in own time - time outside nested '
        "regions - so each change is counted once, where it happened.</p></div>"
    )


def _runs_table_html(runs, links) -> str:
    """One line per compared run, with a link to its full report."""
    base_span = runs[0].time_span
    threads = [_threads(run) for run in runs]
    # Threads are worth a column once they differ, or are more than one.
    show_threads = len(set(threads)) > 1 or any((count or 1) > 1 for count in threads)
    body = []
    for index, run in enumerate(runs):
        metadata = run.metadata or {}
        path = (
            Path(run.file_path) if run.file_path and str(run.file_path) != "." else None
        )
        file_cell = (
            f'<td class="run-file" title="{_text(path)}"><code>{_text(path.name)}</code></td>'
            if path
            else "<td>-</td>"
        )
        if index == 0 or not base_span or not run.time_span:
            change = '<span class="badge base">baseline</span>' if index == 0 else "-"
        else:
            change, _ = _change_html(base_span, run.time_span, "total")
        link = (
            f'<a href="{_text(links[index])}">full report</a>'
            if links
            else '<span class="muted">-</span>'
        )
        body.append(
            "<tr>"
            f"<td><strong>{_text(run.display_label)}</strong></td>"
            + file_cell
            + f"<td>{_text(_format_timestamp(metadata['timestamp'])) if metadata.get('timestamp') else '-'}</td>"
            f"<td>{_text(metadata.get('hostname') or '-')}</td>"
            f"<td>{_text(run.num_ranks)}</td>"
            + (
                f"<td>{_text(threads[index] if threads[index] is not None else '-')}</td>"
                if show_threads
                else ""
            )
            + f"<td>{_text(_duration_text(run.time_span))}</td>"
            f"<td>{change}</td><td>{link}</td></tr>"
        )
    note = (
        "Each run's full report is linked on the right."
        if links
        else "Build a full report for one run with "
        "<code>scope-profiler report RUN.h5 -o RUN.html</code>."
    )
    return (
        '<div class="table-scroll"><table class="runs-table"><thead><tr>'
        "<th>run</th><th>file</th><th>recorded</th><th>host</th><th>ranks</th>"
        + ("<th>threads</th>" if show_threads else "")
        + "<th>wall time</th><th>vs baseline</th><th>report</th></tr></thead><tbody>"
        + "".join(body)
        + f'</tbody></table></div><p class="muted table-note">{note}</p>'
    )


def _chart_sections(
    runs,
    include,
    exclude,
    ranks,
    charts_cdn: bool = False,
    comparison: bool = False,
    scaling_field: str | None = None,
    scaling: str = "both",
) -> tuple[str, str]:
    """Build embedded chart payloads for the bundled browser renderer.

    Returns the charts section and the Regions/Depth sliders for the filter
    bar ("" when there is no timeline to drive them).

    A comparison report keeps only the charts that compare runs; each run's
    timeline and rank charts belong to its own report. ``scaling`` picks the
    chart(s) for runs of different sizes: "strong" the speedup, "weak" the
    weak-scaling efficiency, "both" the two of them.
    """
    try:
        from plotly.offline import get_plotlyjs

        from scope_profiler.plotting_scripts import (
            available_likwid_metrics,
            collect_region_statistics,
            plot_durations,
            plot_gantt,
            plot_imbalance,
            plot_likwid,
            plot_rank_heatmap,
            plot_speedup,
            plot_weak_scaling_efficiency,
        )
    except ImportError:
        return (
            (
                '<section id="charts"><h2>Charts</h2><p class="muted">Charts '
                "require <code>scope-profiler[pproc]</code>; the statistics and "
                "metadata above remain available without it.</p></section>"
            ),
            "",
        )

    charts: list[tuple[str, dict, dict]] = []
    failures: list[str] = []

    def payload_of(title, plotter, path: Path, *args, **kwargs) -> dict | None:
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
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def collect(
        title,
        plotter,
        path: Path,
        *args,
        chart_options: dict | None = None,
        **kwargs,
    ) -> None:
        payload = payload_of(title, plotter, path, *args, **kwargs)
        if payload is not None:
            charts.append((title, payload, chart_options or {}))

    selected_ranks = [
        [
            rank
            for rank in (range(run.num_ranks) if ranks is None else ranks)
            if 0 <= rank < run.num_ranks
        ]
        for run in runs
    ]
    # The x-axis of a comparison's speedup chart (see _check_speedup_x).
    scaling_field = scaling_field if comparison else None

    with tempfile.TemporaryDirectory(prefix="scope-profiler-report-") as directory:
        payload_dir = Path(directory)
        for index, run in enumerate([] if comparison else runs):
            title = (
                # Naming the rank only says something with more than one.
                f"Timeline: {run.display_label}"
                + (" (rank 0)" if run.num_ranks > 1 else "")
            )
            payload = payload_of(
                title,
                plot_gantt,
                payload_dir / f"gantt-{index}.json",
                run,
                include=include,
                exclude=exclude,
                ranks=[0],
            )
            if payload is None:
                continue
            # Every row is already labelled with its region.
            charts.append((title, payload, {"layout": {"showlegend": False}}))

        scaling_values = (
            [_scaling_value(run, scaling_field) for run in runs]
            if scaling_field is not None
            else []
        )
        if scaling_field is not None and len(set(scaling_values)) < 2:
            # Only a named axis gets here; "auto" picks one the runs differ in.
            failures.append(
                f"Speedup: every run has the same {scaling_field} "
                f"({scaling_values[0]}), so there is no scaling to show",
            )
        elif scaling_field is not None:
            baseline_run = min(runs, key=lambda run: _scaling_value(run, scaling_field))
            # A line per region gets unreadable fast: the regions with the
            # most time on the smallest run, which is where scaling matters.
            # The same regions on every axis, so switching axes keeps the lines.
            leading = sorted(
                baseline_run.get_regions(include=include, exclude=exclude),
                key=lambda region: -region.total_duration,
            )[:_SPEEDUP_REGIONS]
            # One payload per axis the runs differ in, for each scaling chart;
            # the report's buttons switch between them without recomputing
            # anything in the browser.
            scaling_charts = (
                ("Speedup", plot_speedup, "speedup", ("strong", "both")),
                (
                    "Weak scaling",
                    plot_weak_scaling_efficiency,
                    "weak-scaling",
                    ("weak", "both"),
                ),
            )
            axes = _speedup_axes(runs, scaling_field)
            for title, plotter, file_stem, modes in scaling_charts:
                if scaling not in modes:
                    continue
                variants = []
                for field in axes:
                    payload = payload_of(
                        title if field == scaling_field else f"{title} over {field}",
                        plotter,
                        payload_dir / f"{file_stem}-{field}.json",
                        runs,
                        x_field=field,
                        include=[f"{re.escape(region.name)}$" for region in leading],
                        exclude=exclude,
                        ranks=ranks,
                    )
                    if payload is not None:
                        variants.append(
                            {
                                "field": field,
                                "label": payload["options"]["x_label"],
                                "payload": payload,
                            },
                        )
                default = next(
                    (item for item in variants if item["field"] == scaling_field),
                    None,
                )
                if default is not None:
                    charts.append(
                        (
                            title,
                            default["payload"],
                            {"variants": variants} if len(variants) > 1 else {},
                        ),
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
        collect(
            "Region durations",
            plot_durations,
            payload_dir / "durations.json",
            runs,
            include=include,
            exclude=exclude,
            ranks=ranks,
            # Totals summed over ranks grow with the rank count; across
            # run sizes, the mean call reads like the speedup chart.
            metric="total" if len({run.num_ranks for run in runs}) == 1 else "avg",
            sort_by="total",
            # One run's bars split into the region's own time and its direct
            # children, which the ranked table cannot show at a glance. Runs
            # compare as grouped bars instead: stacking both ways on one axis
            # would merge equal segment names across runs. The split needs
            # call timestamps, which aggregated profiles do not store.
            stack_children=len(runs) == 1 and _has_call_timestamps(runs[0]),
        )
        if not comparison and charts and charts[-1][0] == "Region durations":
            # Where the time went is the first question about one run, so its
            # durations lead, above the timeline.
            charts.insert(0, charts.pop())

        # Both rank views are empty or trivial with a single rank.
        if not comparison and any(len(selected) > 1 for selected in selected_ranks):
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
    # With both scaling charts, the speedup leads and the weak-scaling chart
    # waits; asked for on its own, the weak-scaling chart leads instead.
    lead = "Weak scaling" if scaling == "weak" else "Speedup"
    kinds = (
        (lead, "Change:", "Region durations")
        if comparison
        else ("Region durations", "Timeline:")
    )
    opened = {
        next(index for index, chart in enumerate(charts) if chart[0].startswith(kind))
        for kind in kinds
        if any(chart[0].startswith(kind) for chart in charts)
    }
    if not opened and charts:
        # An aggregated profile has no timeline; never leave every panel shut.
        opened = {0}
    fragments = []
    chart_documents = []
    for index, (title, payload, chart_options) in enumerate(charts):
        chart_id = f"scope-profiler-chart-{index}"
        is_duration_chart = payload.get("plot") == "durations"
        chart_class = "chart chart-duration" if is_duration_chart else "chart"
        explanation = _chart_description(title, payload, comparison)
        chart_options = dict(chart_options)
        variants = chart_options.pop("variants", None)
        axis_buttons = ""
        if variants:
            axis_buttons = (
                f'<span class="tools-label" id="{chart_id}-axis">x-axis</span>'
                + "".join(
                    f'<button type="button" class="chart-axis" data-chart="{chart_id}"'
                    f' data-axis="{position}" aria-describedby="{chart_id}-axis"'
                    f' aria-pressed="{str(variant["payload"] is payload).lower()}">'
                    f'{_text(variant["label"])}</button>'
                    for position, variant in enumerate(variants)
                )
            )
        fragments.append(
            f'<details class="chart-panel"{" open" if index in opened else ""}>'
            '<summary><span class="chart-heading" role="heading" aria-level="3">'
            f"{_text(title)}</span></summary>{explanation}"
            f'<div class="chart-tools">{axis_buttons}'
            '<button type="button"'
            f' class="chart-open" data-chart="{chart_id}"'
            ' title="Open this chart on its own page">'
            "Open in new tab ↗</button></div>"
            f'<div class="{chart_class}" id="{chart_id}"></div></details>',
        )
        chart_documents.append(
            {
                "id": chart_id,
                "title": title,
                "payload": payload,
                "options": (
                    {**chart_options, "layout": {"height": 680}}
                    if is_duration_chart
                    else chart_options
                ),
                **(
                    {
                        "variants": [
                            {"label": variant["label"], "payload": variant["payload"]}
                            for variant in variants
                        ]
                    }
                    if variants
                    else {}
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
            '<section id="charts"><h2>Charts</h2>' + "".join(fragments) + "</section>",
            "",
        )

    # Escape '<' so profile labels such as '</script>' cannot terminate the
    # inline module. The bundled builders and the payloads make the document
    # durable; whether the Plotly runtime travels with it is the caller's
    # choice, since inlining it costs ~4.7 MB in every report.
    documents_json = json.dumps(chart_documents, ensure_ascii=False).replace(
        "<",
        "\\u003c",
    )
    region_index = _region_index(runs, include, exclude, ranks)
    index_json = json.dumps(region_index, ensure_ascii=False).replace("<", "\\u003c")
    num_regions = len(region_index)
    limit_controls = {
        "regions": num_regions,
        "max_depth": max(
            (depth for _, depth in region_index.values() if depth is not None),
            default=0,
        ),
        "top": _TIMELINE_REGIONS if num_regions > _TIMELINE_REGIONS else None,
    }
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
            '.min.js" crossorigin="anonymous" id="scope-profiler-plotly-runtime">'
            "</script>"
        )
    else:
        # The id lets "Open in new tab" carry the runtime to the new page.
        runtime = (
            '<script id="scope-profiler-plotly-runtime">' + get_plotlyjs() + "</script>"
        )
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
const termFilter = (region) => activeTerms.some((term) =>
  term.startsWith('^')
    ? String(region).toLowerCase().startsWith(term.slice(1))
    : String(region).toLowerCase().includes(term));
// The filter bar's Regions and Depth sliders limit every chart to the same
// regions: those no deeper than maxDepth, then the topN of them by time (each
// region's [seconds, depth] in scopeProfilerRegionIndex). The timeline also
// hides the deeper calls of the regions it keeps.
const regionLimits = {};
let limitedRegions = null;
const updateLimitedRegions = () => {
  limitedRegions = Object.keys(regionLimits).length ? limitedSet() : null;
  globalThis.scopeProfilerSetRegionLimit?.(limitedRegions);
};
const limitedSet = () => {
  const { topN, maxDepth } = regionLimits;
  let names = Object.keys(scopeProfilerRegionIndex).filter((name) => {
    const depth = scopeProfilerRegionIndex[name][1];
    return (!activeTerms.length || termFilter(name)) &&
      (maxDepth == null || depth == null || depth <= maxDepth);
  });
  if (topN != null) {
    names = names
      .sort((a, b) => scopeProfilerRegionIndex[b][0] - scopeProfilerRegionIndex[a][0])
      .slice(0, topN);
  }
  return new Set(names);
};
const chartFilter = (region) =>
  (!activeTerms.length || termFilter(region)) &&
  (!limitedRegions || limitedRegions.has(region));
const draw = (chart) => {
  const target = document.getElementById(chart.id);
  // A chart in a collapsed panel waits until the panel opens: a timeline of
  // hundreds of lanes is seconds of drawing that nobody may look at.
  const panel = target.closest('details');
  if (panel && !panel.open) {
    chart.stale = true;
    return;
  }
  chart.stale = false;
  let options = activeTerms.length || limitedRegions
    ? { ...chart.options, filterRegion: chartFilter }
    : chart.options;
  if (chart.payload.plot === "gantt" && regionLimits.maxDepth != null) {
    options = { ...options, maxDepth: regionLimits.maxDepth };
  }
  // buildFigure dispatches region_statistics to the ranked summary; only
  // buildComparisonFigure draws the signed per-region change.
  const build = options.comparison ? buildComparisonFigure : buildFigure;
  try {
    const figure = highlightFigure(chart, build(chart.payload, options), selectedRegion);
    chart.figure = figure;
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
          // "toast": highlight in place, and offer the jump to the table
          // rather than scrolling away from the chart being explored.
          globalThis.scopeProfilerSelectRegion(
            region, runFromPoint(chart, point, region), "toast");
        }
      }));
    }
    return rendered;
  } catch (error) {
    target.classList.add('chart-error');
    target.textContent = `Could not render chart: ${error.message}`;
  }
};

const redraw = () => {
  updateLimitedRegions();
  for (const chart of scopeProfilerCharts) draw(chart);
};
// Both hooks call their listener as soon as it registers, which drew every
// chart twice on load; changes arriving together are drawn once.
let redrawQueued = false;
const scheduleRedraw = () => {
  if (redrawQueued) return;
  redrawQueued = true;
  queueMicrotask(() => { redrawQueued = false; redraw(); });
};
for (const chart of scopeProfilerCharts) {
  const panel = document.getElementById(chart.id).closest('details');
  panel?.addEventListener('toggle', () => { if (panel.open && chart.stale) draw(chart); });
}

// "Open in new tab": the chart as drawn - filter and highlight included --
// on a page of its own, sized to the window. The page carries the Plotly
// runtime this report loaded, so it works offline exactly when this does.
const escapeHtml = (value) => String(value).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const openInNewTab = (chart) => {
  if (!chart.figure) return;
  const runtime = document.getElementById("scope-profiler-plotly-runtime");
  const loader = runtime?.src
    ? `<script crossorigin="anonymous" src="${escapeHtml(runtime.src)}"><\/script>`
    : `<script>${runtime?.textContent ?? ""}<\/script>`;
  const figure = JSON.stringify({ data: chart.figure.data, layout: chart.figure.layout })
    .replace(/</g, "\\u003c");
  const page = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(chart.title)}</title><style>html, body { margin: 0; background: #fff; }
#chart { width: 100vw; }</style></head><body><div id="chart"></div>${loader}<script>
const figure = ${figure};
const target = document.getElementById("chart");
const layout = { ...figure.layout, autosize: true, title: figure.layout.title ?? { text: ${JSON.stringify(escapeHtml(chart.title))} } };
delete layout.width;
// A chart taller than the window keeps its height and scrolls; any other fills it.
if (!(layout.height > window.innerHeight)) { delete layout.height; target.style.height = "100vh"; }
Plotly.newPlot(target, figure.data, layout, { responsive: true, displaylogo: false });
<\/script></body></html>`;
  const url = URL.createObjectURL(new Blob([page], { type: "text/html" }));
  window.open(url, "_blank");
  // The new tab has read the page by then; free the copy.
  setTimeout(() => URL.revokeObjectURL(url), 60000);
};
for (const button of document.querySelectorAll(".chart-open")) {
  const chart = scopeProfilerCharts.find((item) => item.id === button.dataset.chart);
  if (chart) button.addEventListener("click", () => openInNewTab(chart));
}
// At its maximum a slider means all. The readout follows the thumb, while
// the charts, which can take most of a second to draw, wait for it to rest.
// The labels mirror _top_label and _depth_label.
const topLabel = (value, max) => value >= max ? `all ${max}` : `top ${value} of ${max}`;
const depthLabel = (value, max) =>
  value >= max ? "all levels" : value === 0 ? "top level only"
    : `${value} level${value > 1 ? "s" : ""} of children`;
for (const slider of document.querySelectorAll(".region-top, .region-depth")) {
  const isTop = slider.classList.contains("region-top");
  const key = isTop ? "topN" : "maxDepth";
  const readout = slider.nextElementSibling;
  const read = () => {
    const value = Number(slider.value);
    if (value >= Number(slider.max)) delete regionLimits[key];
    else regionLimits[key] = value;
  };
  // The page opens on the slider's own value: past 500 regions, the top 500.
  // The index sets the limits before the first draw.
  read();
  let timer = null;
  const apply = () => {
    window.clearTimeout(timer);
    const before = regionLimits[key];
    read();
    if (regionLimits[key] !== before) scheduleRedraw();
  };
  slider.addEventListener("input", () => {
    const value = Number(slider.value);
    const max = Number(slider.max);
    readout.textContent = isTop ? topLabel(value, max) : depthLabel(value, max);
    window.clearTimeout(timer);
    timer = window.setTimeout(apply, 150);
  });
  slider.addEventListener("change", apply);
}
// The scaling charts' x-axis buttons: each swaps in the payload computed for
// that axis, and the description's axis name with it.
for (const button of document.querySelectorAll(".chart-axis")) {
  const chart = scopeProfilerCharts.find((item) => item.id === button.dataset.chart);
  const variant = chart?.variants?.[Number(button.dataset.axis)];
  if (!variant) continue;
  button.addEventListener("click", () => {
    chart.payload = variant.payload;
    const panel = button.closest(".chart-panel");
    for (const other of panel.querySelectorAll(".chart-axis")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
    for (const label of panel.querySelectorAll("[data-axis-label]")) {
      label.textContent = variant.label;
    }
    draw(chart);
  });
}
if (typeof globalThis.scopeProfilerOnRegionFilter === "function") {
  globalThis.scopeProfilerOnRegionFilter((terms) => { activeTerms = terms; scheduleRedraw(); });
}
if (typeof globalThis.scopeProfilerOnRegionSelect === "function") {
  globalThis.scopeProfilerOnRegionSelect((region) => { selectedRegion = region; scheduleRedraw(); });
}
if (typeof globalThis.scopeProfilerOnRegionFilter !== "function" &&
    typeof globalThis.scopeProfilerOnRegionSelect !== "function") scheduleRedraw();
"""
    script = (
        runtime
        + '<script type="module">'
        + plotly_builders
        + "\nconst scopeProfilerCharts = "
        + documents_json
        + ";\nconst scopeProfilerRegionIndex = "
        + index_json
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
        + '<p class="back-to-top"><a href="#top">Back to top</a></p></section>',
        _region_limits(limit_controls),
    )


def _report_rows(results, include, exclude, ranks, sort):
    return region_rows(
        results,
        include=include,
        exclude=exclude,
        ranks=ranks,
        sort=sort,
        # Populates each row's "exclusive" time. The table's own % column
        # is computed from the inclusive total either way; the hotspots
        # need exclusive time to name the leaves rather than their parents.
        percentage_mode="exclusive",
    )


def _navigation(links) -> str:
    return (
        '<nav class="toc" aria-label="Report contents"><strong>Contents</strong>'
        + "".join(
            f'<a href="#{_text(target)}">{_text(label)}</a>' for target, label in links
        )
        + "</nav>"
    )


_BACK_TO_TOP = '<p class="back-to-top"><a href="#top">Back to top</a></p>'


def _single_run_body(results, include, exclude, ranks, sort, columns, charts, limits):
    """Header, navigation and sections of the report for one run."""
    rows = _report_rows(results, include, exclude, ranks, sort)
    section_id = "run-0"
    region_ids = {
        row["name"]: f"{section_id}-region-{row_index}"
        for row_index, row in enumerate(rows)
    }
    entries = _hotspot_entries(rows)
    base = _hotspot_base(rows, entries)
    balance = _rank_balance(results, rows, ranks)
    # A region on several call paths has a row per path.
    region_count = len({row["name"] for row in rows})
    links = [(f"{section_id}-summary", "Summary")]
    parts = [
        f'<div id="{section_id}-summary">'
        + _summary_html(results, rows, entries, base, balance, region_ids, section_id)
        + "</div>"
    ]
    hotspots = _hotspots_html(results, entries, base)
    if hotspots:
        links.append((f"{section_id}-hotspots", "Hotspots"))
        parts.append(f'<div id="{section_id}-hotspots">{hotspots}</div>')
    # Each region's line profile lives in its table row's detail.
    tabled = {row["name"] for row in rows}
    line_profiles: dict[str, list[dict]] = {}
    unmatched = []
    for entry in _line_profile_functions(results, ranks):
        if entry["region"] in tabled:
            line_profiles.setdefault(entry["region"], []).append(entry)
        else:
            unmatched.append(entry)
    links.append((f"{section_id}-table", "Region statistics"))
    parts.append(
        f'<h3 id="{section_id}-table">Region statistics</h3>'
        + _region_table(
            results,
            rows,
            ranks,
            columns,
            region_ids,
            table_id=f"{section_id}-regions",
            line_profiles=line_profiles,
        )
    )
    if balance is not None:
        links.append((f"{section_id}-balance", "Load balance"))
        parts.append(_load_balance_html(results, balance, region_ids, section_id))
    if unmatched:
        links.append((f"{section_id}-lines", "Line profile"))
        parts.append(
            f'<h3 id="{section_id}-lines">Line profile</h3>'
            + _line_profile_html(results, unmatched, section_id)
        )
    parts.append(
        f'<details id="{section_id}-metadata"><summary>Metadata</summary>'
        f"{_metadata_table(results.metadata)}</details>"
    )
    links.append((f"{section_id}-metadata", "Metadata"))
    sections = (
        f'<section id="{section_id}">' + "".join(parts) + _BACK_TO_TOP + "</section>"
    )

    hardware = _hardware_sections([results], include, exclude, ranks)
    if hardware:
        links.append(("hardware-counters", "Hardware counters"))
    if charts:
        links.append(("charts", "Charts"))
    header = (
        '<header class="report-header">'
        f'<h1 id="top">{_text(results.display_label)}</h1>'
        f"{_run_meta_html(results, region_count)}</header>"
    )
    return (
        # The bar leads the page, above the title, and stays there on scroll.
        _filter_bar(limits)
        + header
        + _navigation(links)
        + sections
        + hardware
        + charts
    )


def _report_file_name(output_path: Path, index: int, label: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label)).strip("-.") or "run"
    return output_path.with_name(f"{output_path.stem}-{index}-{slug}.html")


def _comparison_body(runs, include, exclude, ranks, sort, charts, links, limits):
    """Header, navigation and sections of the report comparing runs."""
    per_run_rows = [_report_rows(run, include, exclude, ranks, sort) for run in runs]
    per_rank = len({run.num_ranks for run in runs}) > 1
    if per_rank:
        per_run_rows = [[_per_rank_row(row) for row in rows] for rows in per_run_rows]
    entries = _comparison_entries(per_run_rows)
    labels = [run.display_label for run in runs]
    title = " vs ".join(labels) if len(runs) <= 3 else f"{len(runs)} runs"
    navigation = [
        ("compare-runs", "Runs"),
        ("compare-changes", "What changed"),
        ("compare-durations", "Durations"),
    ]
    if charts:
        navigation.append(("charts", "Charts"))
    header = (
        '<header class="report-header">'
        f'<h1 id="top">{_text(title)}</h1>'
        f'<p class="run-meta"><span>{_plural(len(runs), "run")}</span>'
        f"<span>baseline: {_text(labels[0])}</span></p></header>"
    )
    sections = (
        '<section id="compare-runs"><h2>Runs</h2>'
        + _runs_table_html(runs, links)
        + "</section>"
        '<section id="compare-changes"><h2>What changed</h2>'
        + "".join(
            _candidate_html(runs, entries, index) for index in range(1, len(runs))
        )
        + "</section>"
        '<section id="compare-durations"><h2>Durations</h2>'
        + _comparison_table_html(runs, entries, per_rank)
        + _BACK_TO_TOP
        + "</section>"
    )
    return _filter_bar(limits) + header + _navigation(navigation) + sections + charts


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
    individual_reports: bool = True,
    scaling: str = "both",
    speedup_x: str = "auto",
) -> Path:
    """Write a standalone HTML report for one or more profiling results.

    One run gets a full report. Several runs get a comparison report, the
    first run being the baseline; with ``individual_reports`` (the default)
    a full report for each run is also written next to it, as
    ``<stem>-<index>-<label>.html``, and linked from the comparison.

    When the compared runs differ in ranks or threads, ``scaling`` says what
    kind of study they are, which the profiles cannot tell: ``"strong"`` (the
    same total problem size on every run) charts each region's speedup,
    ``"weak"`` (the same problem size per rank or core) its weak-scaling
    efficiency, and ``"both"`` (the default) charts both.

    ``speedup_x`` sets the x-axis of a comparison's scaling charts: ``"ranks"``
    (MPI ranks), ``"nodes"``, ``"threads"`` (OpenMP threads) or ``"cores"``
    (ranks times threads). ``"auto"``, the default, uses MPI ranks when they
    change, threads when only they change, and draws no scaling chart when the
    runs differ in neither. A named axis must be recorded by every run, or this raises
    ``ValueError``. Either way, each chart has a button for every other axis
    the runs differ in.
    """
    if scaling not in SCALING_MODES:
        raise ValueError(
            f"scaling must be one of {', '.join(map(repr, SCALING_MODES))}, "
            f"not {scaling!r}.",
        )
    if isinstance(profiling_data, (ProfilingResults, str, Path)):
        profiling_data = [profiling_data]
    runs = [
        item if isinstance(item, ProfilingResults) else read_profile(item)
        for item in profiling_data
    ]
    if not runs:
        raise ValueError("At least one profiling result is required.")
    comparison = len(runs) > 1
    # Checked before anything is written, and whether or not charts are drawn,
    # so a run that cannot have the requested axis fails the same way always.
    scaling_field = _check_speedup_x(runs, speedup_x) if comparison else None
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    charts, limits = (
        _chart_sections(
            runs,
            include,
            exclude,
            ranks,
            charts_cdn=charts_cdn,
            comparison=comparison,
            scaling=scaling,
            scaling_field=scaling_field,
        )
        if include_charts
        else ("", "")
    )
    if comparison:
        links = []
        if individual_reports:
            for index, run in enumerate(runs):
                path = _report_file_name(output_path, index, run.display_label)
                create_html_report(
                    run,
                    path,
                    include=include,
                    exclude=exclude,
                    ranks=ranks,
                    sort=sort,
                    columns=columns,
                    charts_cdn=charts_cdn,
                    include_charts=include_charts,
                )
                links.append(path.name)
        body = _comparison_body(
            runs, include, exclude, ranks, sort, charts, links, limits
        )
        title = "scope-profiler comparison"
    else:
        body = _single_run_body(
            runs[0], include, exclude, ranks, sort, columns, charts, limits
        )
        title = f"{runs[0].display_label} · scope-profiler report"
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_text(title)}</title><style>"
        + _STYLE
        + "</style></head><body>"
        + body
        + _REGION_TOAST
        + "<script>"
        + _SCRIPT
        + "</script></body></html>\n"
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path
