// Every builder, driven by JSON the exporter really wrote.
//
// The hand-written payloads in index.test.js pin down behaviour case by case,
// but they are all single-run, which is exactly how four builders came to drop
// the `file` column without a test noticing. These fixtures come from
// `scope-profiler export plot-data` over two runs of different sizes; see
// fixtures/generate_fixtures.py to regenerate them.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { buildFigure, PLOT_BUILDERS, inferPlotKind } from "../src/index.js";

const directory = join(import.meta.dirname, "fixtures");
const fixtures = readdirSync(directory).filter((name) => name.endsWith(".json"));
const load = (name) => JSON.parse(readFileSync(join(directory, name), "utf8"));

test("the fixtures cover every plot kind the exporter can write", () => {
  assert.ok(fixtures.length >= 14, `only ${fixtures.length} fixtures found`);
  const kinds = new Set(fixtures.map((name) => load(name).plot));
  for (const kind of kinds) assert.ok(PLOT_BUILDERS[kind], `no builder for ${kind}`);
  assert.ok(kinds.has("region_statistics"), "every export writes region_statistics");
});

for (const name of fixtures) {
  test(`buildFigure renders ${name} without naming a builder`, () => {
    const payload = load(name);
    assert.equal(payload.format, "scope-profiler-plot-data");
    const figure = buildFigure(payload);
    assert.ok(figure.data.length > 0, "no traces");
    assert.ok(figure.layout, "no layout");
    // A hole in a data array is a silent misalignment: Plotly draws the trace
    // and simply omits the point, so assert on it rather than on a render.
    for (const trace of figure.data) {
      for (const key of ["x", "y", "z", "values", "labels", "parents"]) {
        const column = trace[key];
        if (!Array.isArray(column)) continue;
        assert.ok(
          !column.flat().some((value) => value === undefined),
          `${name}: undefined in trace.${key}`,
        );
      }
    }
  });

  test(`${name} keeps its kind without the envelope`, () => {
    const { format, format_version, plot, ...bare } = load(name);
    const guessed = inferPlotKind(bare);
    // flame_chart and flame_graph share a payload shape and a builder.
    const expected = plot === "flame_graph" ? "flame" : plot;
    assert.equal(guessed, expected, `${name}: inferred ${guessed}, wrote ${plot}`);
  });
}

// The bugs these fixtures were written for: a builder that ignores the `file`
// column merges two runs into one series, losing rows without saying so.
const plotted = (figure, keys = ["x"]) =>
  figure.data.reduce(
    (total, trace) => total + (keys.some((key) => Array.isArray(trace[key])) ? trace[keys.find((key) => Array.isArray(trace[key]))].length : 0),
    0,
  );

test("the rank heatmap gives every run its own lane", () => {
  const payload = load("rank_heatmap_data.json");
  const figure = buildFigure(payload);
  const cells = figure.data[0].z.flat().filter((value) => value !== null).length;
  assert.equal(cells, payload.points.length, "cells lost to a rank-only key");
  assert.equal(
    new Set(figure.data[0].y).size,
    new Set(payload.points.map((point) => `${point.file}/${point.rank}`)).size,
  );
  assert.ok(figure.data[0].y.every((lane) => lane.includes("run_")), "lanes drop the run");
});

test("imbalance draws a series per run, not one series across runs", () => {
  const payload = load("imbalance_data.json");
  const figure = buildFigure(payload);
  const lines = figure.data.filter((trace) => trace.mode === "lines+markers");
  assert.equal(plotted({ data: lines }), payload.points.length);
  for (const trace of lines) {
    assert.equal(new Set(trace.x).size, trace.x.length, `${trace.name} revisits a rank`);
  }
});

test("the histogram and time series keep every row of both runs", () => {
  const histogram = load("histogram_data.json");
  assert.equal(plotted(buildFigure(histogram)), histogram.bins.length);
  const series = load("duration_timeseries_data.json");
  assert.equal(plotted(buildFigure(series)), series.points.length);
});

test("the gantt gives every run, rank and region its own lane", () => {
  const payload = load("gantt_data.json");
  const figure = buildFigure(payload);
  assert.equal(plotted(figure), payload.intervals.length);
  // One lane per rank would stack a nested profile onto a single row, where
  // the outermost region hides everything inside it.
  assert.equal(
    new Set(figure.layout.yaxis.categoryarray).size,
    new Set(payload.intervals.map((row) => `${row.file}/${row.rank}/${row.region}`)).size,
  );
  for (const lane of figure.layout.yaxis.categoryarray) assert.match(lane, /\(rank \d+\)$/);
});
