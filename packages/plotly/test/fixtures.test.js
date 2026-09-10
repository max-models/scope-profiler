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
const fixtures = readdirSync(directory).filter((name) =>
  name.endsWith(".json"),
);
const load = (name) => JSON.parse(readFileSync(join(directory, name), "utf8"));

// `plot flame_chart` and `plot flame_graph` stamp their documents "flame" and
// "flame_graph"; `flame_chart` is a builder alias the exporter never writes,
// so it is the one kind that cannot have a fixture of its own.
const ALIAS_KINDS = new Set(["flame_chart"]);

test("every builder has a fixture the exporter really wrote", () => {
  const kinds = new Set(fixtures.map((name) => load(name).plot));
  for (const kind of kinds)
    assert.ok(PLOT_BUILDERS[kind], `no builder for ${kind}`);
  // A count was the floor here before, and it passed while two of eighteen
  // builders had no fixture at all. Enumerate the builders instead.
  const uncovered = Object.keys(PLOT_BUILDERS).filter(
    (kind) => !kinds.has(kind) && !ALIAS_KINDS.has(kind),
  );
  assert.deepEqual(uncovered, [], `builders with no fixture: ${uncovered}`);
  assert.ok(
    kinds.has("region_statistics"),
    "every export writes region_statistics",
  );
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
    // Two kinds cannot be told apart by shape alone, and do not need to be:
    // flame_chart and flame_graph share a payload shape and a builder, and
    // both efficiency curves store their y column as `efficiency`. Inference
    // is only a fallback for files written before the envelope existed, and
    // weak_scaling_efficiency has never been written without one.
    const shared = {
      flame_graph: "flame",
      weak_scaling_efficiency: "scaling_efficiency",
    };
    const expected = shared[plot] ?? plot;
    assert.equal(
      guessed,
      expected,
      `${name}: inferred ${guessed}, wrote ${plot}`,
    );
  });
}

// The bugs these fixtures were written for: a builder that ignores the `file`
// column merges two runs into one series, losing rows without saying so.
const plotted = (figure, keys = ["x"]) =>
  figure.data.reduce(
    (total, trace) =>
      total +
      (keys.some((key) => Array.isArray(trace[key]))
        ? trace[keys.find((key) => Array.isArray(trace[key]))].length
        : 0),
    0,
  );

test("the rank heatmap gives every run its own lane", () => {
  const payload = load("rank_heatmap_data.json");
  const figure = buildFigure(payload);
  const cells = figure.data[0].z
    .flat()
    .filter((value) => value !== null).length;
  assert.equal(cells, payload.points.length, "cells lost to a rank-only key");
  assert.equal(
    new Set(figure.data[0].y).size,
    new Set(payload.points.map((point) => `${point.file}/${point.rank}`)).size,
  );
  assert.ok(
    figure.data[0].y.every((lane) => lane.includes("run_")),
    "lanes drop the run",
  );
});

test("imbalance draws a series per run, not one series across runs", () => {
  const payload = load("imbalance_data.json");
  const figure = buildFigure(payload);
  const lines = figure.data.filter((trace) => trace.mode === "lines+markers");
  assert.equal(plotted({ data: lines }), payload.points.length);
  for (const trace of lines) {
    assert.equal(
      new Set(trace.x).size,
      trace.x.length,
      `${trace.name} revisits a rank`,
    );
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
    new Set(figure.layout.yaxis.ticktext).size,
    new Set(
      payload.intervals.map((row) => `${row.file}/${row.rank}/${row.region}`),
    ).size,
  );
  for (const lane of figure.layout.yaxis.ticktext)
    assert.match(lane, /\(rank \d+\)$/);
});
