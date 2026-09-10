import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { buildRankHeatmapFigure } from "../src/index.js";
const points = Array.from({ length: 8192 }, (_, index) => ({
  region: `region-${index % 128}`,
  rank: Math.floor(index / 128),
  total_duration_seconds: (index % 17) + 1,
}));
const samples = [];
for (let iteration = 0; iteration < 4; iteration++) {
  const start = performance.now();
  const figure = buildRankHeatmapFigure({ points });
  samples.push(performance.now() - start);
  assert.equal(figure.data[0].z.flat().length, points.length);
  assert.equal(figure.data[0].customdata[0][0].identity.rank, 0);
}
console.log(
  JSON.stringify({
    samples_ms: samples,
    memory: process.memoryUsage(),
    rows: points.length,
  }),
);
