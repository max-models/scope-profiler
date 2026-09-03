import test from "node:test";
import assert from "node:assert/strict";
import { buildCallgraphFigure, buildDensityFigure, buildDurationTimeseriesFigure, buildDurationsFigure, buildFigure, buildFlameFigure, buildGanttFigure, buildHistogramFigure, buildImbalanceFigure, buildLikwidFigure, buildRankHeatmapFigure, buildRegionSummaryFigure, buildScalingEfficiencyFigure, buildSpeedupFigure, buildWeakScalingFigure, inferPlotKind } from "../src/index.js";

test("gantt gives each region and rank a lane, and honours supplied colors", () => {
  const figure = buildGanttFigure({ colors: { solve: "#123456" }, intervals: [
    { file: "one", rank: 0, region: "solve", start_seconds: 0, end_seconds: 2 },
    { file: "one", rank: 1, region: "solve", start_seconds: 0, end_seconds: 1 },
  ] });
  assert.deepEqual(figure.data[0].y, ["solve (rank 0)", "solve (rank 1)"]);
  assert.equal(figure.data[0].marker.color, "#123456");
});

test("gantt keeps a nested profile legible instead of stacking it on one row", () => {
  // Every region of a rank on one lane means the session bar covers the rest.
  const payload = { intervals: [
    { file: "one", rank: 0, region: "session", start_seconds: 0, end_seconds: 10 },
    { file: "one", rank: 0, region: "setup", start_seconds: 1, end_seconds: 3 },
    { file: "one", rank: 0, region: "solve", start_seconds: 4, end_seconds: 9 },
  ] };
  const figure = buildGanttFigure(payload);
  // A categorical y axis counts up from the bottom, so leaving it un-reversed
  // puts the first (enclosing) region on the bottom lane -- the order
  // `scope-profiler plot gantt` draws.
  assert.deepEqual(figure.layout.yaxis.categoryarray, ["session (rank 0)", "setup (rank 0)", "solve (rank 0)"]);
  assert.equal(figure.layout.yaxis.autorange, undefined);
  assert.equal(new Set(figure.data.flatMap((trace) => trace.y)).size, 3);
  // The opt-out keeps the compact one-row-per-rank view.
  const compact = buildGanttFigure(payload, { laneBy: "rank" });
  assert.deepEqual(compact.layout.yaxis.categoryarray, ["one / rank 0"]);
  assert.equal(compact.layout.yaxis.autorange, "reversed");
});

test("gantt names a lane by its run only when the payload holds several", () => {
  const figure = buildGanttFigure({ intervals: [
    { file: "one", rank: 0, region: "solve", start_seconds: 0, end_seconds: 2 },
    { file: "two", rank: 0, region: "solve", start_seconds: 0, end_seconds: 1 },
  ] });
  assert.deepEqual(figure.data[0].y, ["one / solve (rank 0)", "two / solve (rank 0)"]);
});

test("flame uses parent_call_id rather than matching region names", () => {
  const figure = buildFlameFigure({ calls: [
    { file: "one", rank: 0, call_id: 1, parent_call_id: null, region: "step", start_seconds: 0, end_seconds: 4 },
    { file: "one", rank: 0, call_id: 2, parent_call_id: 1, region: "step", start_seconds: 1, end_seconds: 2 },
  ] });
  assert.deepEqual(figure.data[0].parents, ["", "scope-profiler-root", "one:0:1"]);
});

test("speedup honours a non-rank x field", () => {
  const figure = buildSpeedupFigure({ options: { x_field: "total_cores", baseline: 4 }, points: [
    { region: "solve", total_cores: 4, speedup: 1 }, { region: "solve", total_cores: 8, speedup: 1.8 },
  ] });
  assert.deepEqual(figure.data[0].x, [4, 8]);
  assert.deepEqual(figure.data[1].y, [1, 2]);
});

test("durations preserves stacked child segments", () => {
  const figure = buildDurationsFigure({ metrics: ["total"], colors: { own: "#111111" }, bars: [
    { file: "one", region: "solve", metric: "total", segment: "own", value_seconds: 2 },
    { file: "one", region: "solve", metric: "total", segment: "child", value_seconds: 3 },
  ] });
  assert.equal(figure.layout.barmode, "stack");
  assert.deepEqual(figure.data.map((trace) => trace.name), ["own", "child"]);
  assert.equal(figure.data[0].marker.color, "#111111");
});

test("additional exported data types build Plotly traces", () => {
  assert.equal(buildDurationTimeseriesFigure({ points: [{ region: "solve", time_seconds: 1, mean_duration_seconds: 2, min_duration_seconds: 1, max_duration_seconds: 3 }] }).data[0].type, "scatter");
  assert.equal(buildHistogramFigure({ bins: [{ region: "solve", bin_low_seconds: 0, bin_center_seconds: 1, bin_high_seconds: 2, count: 4 }] }).data[0].type, "bar");
  assert.equal(buildRankHeatmapFigure({ points: [{ rank: 0, region: "solve", total_duration_seconds: 2 }] }).data[0].type, "heatmap");
  assert.equal(buildImbalanceFigure({ metric: "total", points: [{ rank: 0, region: "solve", value_seconds: 2, mean_over_ranks_seconds: 2 }] }).data[0].type, "scatter");
});

test("flame honours filterRegion, re-parenting survivors onto the nearest kept ancestor", () => {
  const payload = { calls: [
    { file: "one", rank: 0, call_id: 1, parent_call_id: null, region: "step", start_seconds: 0, end_seconds: 4, inclusive_duration_seconds: 4 },
    { file: "one", rank: 0, call_id: 2, parent_call_id: 1, region: "noise", start_seconds: 1, end_seconds: 3, inclusive_duration_seconds: 2 },
    { file: "one", rank: 0, call_id: 3, parent_call_id: 2, region: "solve", start_seconds: 1, end_seconds: 2, inclusive_duration_seconds: 1 },
  ] };
  const figure = buildFlameFigure(payload, { filterRegion: (region) => region !== "noise" });
  const trace = figure.data[0];
  assert.deepEqual(trace.labels, ["All calls", "step", "solve"]);
  assert.deepEqual(trace.parents, ["", "scope-profiler-root", "one:0:1"]);
  assert.equal(trace.values[0], 4);
});

test("flame re-parents onto the root when every ancestor is filtered out", () => {
  const figure = buildFlameFigure({ calls: [
    { file: "one", rank: 0, call_id: 1, parent_call_id: null, region: "noise", start_seconds: 0, end_seconds: 4, inclusive_duration_seconds: 4 },
    { file: "one", rank: 0, call_id: 2, parent_call_id: 1, region: "solve", start_seconds: 1, end_seconds: 3, inclusive_duration_seconds: 2 },
  ] }, { filterRegion: (region) => region === "solve" });
  assert.deepEqual(figure.data[0].parents, ["", "scope-profiler-root"]);
  assert.equal(figure.data[0].values[0], 2);
});

test("buildFigure dispatches on the document's own plot kind", () => {
  const figure = buildFigure({ format: "scope-profiler-plot-data", format_version: 1, plot: "gantt", intervals: [
    { file: "one", rank: 0, region: "solve", start_seconds: 0, end_seconds: 2 },
  ] });
  assert.equal(figure.layout.barmode, "overlay");
});

test("buildFigure infers the kind of a payload written before the envelope", () => {
  assert.equal(inferPlotKind({ points: [{ region: "solve", rank: 0, value_seconds: 1, mean_over_ranks_seconds: 1 }] }), "imbalance");
  assert.equal(inferPlotKind({ points: [{ region: "solve", rank: 0, total_duration_seconds: 1 }] }), "rank_heatmap");
  assert.equal(inferPlotKind({ points: [{ file: "one", region: "solve", bin_start_seconds: 0, bin_end_seconds: 1, occupied_seconds: 0.5 }] }), "density");
  assert.equal(inferPlotKind({ bars: [{ series: "run", region: "solve", value: 2 }] }), "likwid");
  assert.equal(inferPlotKind({ calls: [{ call_id: 0, parent_id: null, name: "solve", depth: 0 }] }), "callgraph");
  assert.equal(inferPlotKind({ nothing: true }), undefined);
  assert.equal(buildFigure({ bins: [{ region: "solve", bin_low_seconds: 0, bin_center_seconds: 1, bin_high_seconds: 2, count: 4 }] }).data[0].type, "bar");
});

test("buildFigure refuses a foreign document or a newer format version", () => {
  assert.throws(() => buildFigure({ format: "something-else", intervals: [] }), /scope-profiler-plot-data/);
  assert.throws(() => buildFigure({ format: "scope-profiler-plot-data", format_version: 2, plot: "gantt", intervals: [] }), /upgrade/);
  assert.throws(() => buildFigure({ points: [] }), /options\.plot/);
});

test("the scaling builders read their own y column and ideal line", () => {
  const points = [
    { region: "solve", num_ranks: 2, normalized_runtime: 1, efficiency: 1 },
    { region: "solve", num_ranks: 4, normalized_runtime: 1.4, efficiency: 0.7 },
  ];
  const weak = buildWeakScalingFigure({ options: { baseline: 2 }, points });
  assert.deepEqual(weak.data[0].y, [1, 1.4]);
  assert.deepEqual(weak.data[1].y, [1, 1]);
  assert.equal(weak.layout.yaxis.title, "Normalized runtime");
  const efficiency = buildScalingEfficiencyFigure({ options: { baseline: 2 }, points });
  assert.deepEqual(efficiency.data[0].y, [1, 0.7]);
  assert.equal(efficiency.layout.yaxis.title, "Scaling efficiency");
  // Dispatch alone must pick the right column, with no explicit builder.
  assert.deepEqual(buildFigure({ plot: "weak_scaling", options: { baseline: 2 }, points }).data[0].y, [1, 1.4]);
});

test("density reports occupancy as a fraction of each bin, or raw seconds", () => {
  const payload = { points: [
    { file: "one", region: "solve", bin_start_seconds: 0, bin_end_seconds: 2, occupied_seconds: 1 },
    { file: "one", region: "solve", bin_start_seconds: 2, bin_end_seconds: 4, occupied_seconds: 2 },
  ] };
  const figure = buildDensityFigure(payload);
  assert.deepEqual(figure.data[0].y, ["one / solve"]);
  assert.deepEqual(figure.data[0].x, [1, 3]);
  assert.deepEqual(figure.data[0].z, [[0.5, 1]]);
  assert.equal(figure.data[0].zmax, 1);
  assert.deepEqual(buildDensityFigure(payload, { valueKey: "occupied_seconds" }).data[0].z, [[1, 2]]);
});

test("region summary ranks by the pooled metric and keeps the head of the list", () => {
  const payload = { files: [
    { label: "run", region_statistics: {
      small: { count: 1, total_duration_seconds: 1 },
      big: { count: 4, total_duration_seconds: 9 },
      middle: { count: 2, total_duration_seconds: 5 },
    } },
  ] };
  const figure = buildRegionSummaryFigure(payload, { topN: 2 });
  assert.deepEqual(figure.data[0].y, ["big", "middle"]);
  assert.deepEqual(figure.data[0].x, [9, 5]);
  assert.deepEqual(figure.data[0].customdata, [4, 2]);
  assert.deepEqual(buildRegionSummaryFigure(payload, { metric: "count" }).data[0].y, ["big", "middle", "small"]);
  assert.deepEqual(buildRegionSummaryFigure(payload, { filterRegion: (region) => region !== "big" }).data[0].y, ["middle", "small"]);
});

test("callgraph builds a sankey from either export shape and drops cycles", () => {
  const compact = buildCallgraphFigure({
    regions: [{ name: "step", depth: 0, total_duration: 4 }, { name: "solve", depth: 1, total_duration: 3 }],
    edges: [{ parent: "step", child: "solve" }, { parent: "solve", child: "solve" }],
  });
  assert.equal(compact.data[0].type, "sankey");
  assert.deepEqual(compact.data[0].node.label, ["step", "solve"]);
  assert.deepEqual(compact.data[0].link.source, [0]);
  assert.deepEqual(compact.data[0].link.value, [3]);
  const full = buildCallgraphFigure({ calls: [
    { call_id: 0, parent_id: null, name: "step", depth: 0 },
    { call_id: 1, parent_id: 0, name: "solve", depth: 1 },
    { call_id: 2, parent_id: 0, name: "solve", depth: 1 },
  ] });
  assert.deepEqual(full.data[0].link.value, [2]);
  assert.match(buildCallgraphFigure({ regions: [{ name: "step", depth: 0 }], edges: [] }).layout.annotations[0].text, /No data/);
  assert.throws(() => buildCallgraphFigure({ points: [] }), /regions or calls/);
});

test("likwid groups one hardware-counter metric by series", () => {
  const figure = buildLikwidFigure({ metric: "DP MFLOP/s", bars: [
    { series: "rank 0", region: "solve", value: 900 },
    { series: "rank 1", region: "solve", value: 850 },
  ] }, { logScale: true });
  assert.deepEqual(figure.data.map((trace) => trace.name), ["rank 0", "rank 1"]);
  assert.deepEqual(figure.data[0].x, ["solve"]);
  assert.equal(figure.layout.yaxis.title, "DP MFLOP/s");
  assert.equal(figure.layout.yaxis.type, "log");
});
