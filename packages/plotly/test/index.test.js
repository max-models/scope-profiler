import test from "node:test";
import assert from "node:assert/strict";
import { buildDurationsFigure, buildFlameFigure, buildGanttFigure, buildSpeedupFigure } from "../src/index.js";

test("gantt keeps separate file/rank lanes and supplied colors", () => {
  const figure = buildGanttFigure({ colors: { solve: "#123456" }, intervals: [
    { file: "one", rank: 0, region: "solve", start_seconds: 0, end_seconds: 2 },
    { file: "one", rank: 1, region: "solve", start_seconds: 0, end_seconds: 1 },
  ] });
  assert.deepEqual(figure.data[0].y, ["one / rank 0", "one / rank 1"]);
  assert.equal(figure.data[0].marker.color, "#123456");
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
