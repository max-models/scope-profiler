import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import * as api from "../src/index.js";

const interval = {
  region: "solve",
  start_seconds: 0,
  end_seconds: 2,
  rank: 0,
  file: "run",
  call_id: 1,
};
const call = { ...interval, parent_call_id: null };
const point = {
  region: "solve",
  time_seconds: 1,
  mean_duration_seconds: 2,
  min_duration_seconds: 1,
  max_duration_seconds: 4,
};

test("HTML reports consume point identities before legacy label heuristics", () => {
  const source = readFileSync(
    new URL("../../../src/scope_profiler/html_report.py", import.meta.url),
    "utf8",
  );
  const selection = source.slice(
    source.indexOf("const regionFromPoint ="),
    source.indexOf("let activeTerms ="),
  );
  const helpers = new Function(
    "payloadRegions",
    `${selection}; return { regionFromPoint, runFromPoint };`,
  )(() => new Set(["solve"]));
  const point = {
    customdata: { identity: { region: "solve", file: "run / custom" } },
  };
  assert.equal(helpers.regionFromPoint({ payload: {} }, point), "solve");
  assert.equal(helpers.runFromPoint({}, point, "solve"), "run / custom");
  assert.equal(
    helpers.regionFromPoint({ payload: {} }, { x: "solve" }),
    "solve",
  );
  assert.equal(
    helpers.runFromPoint({}, { customdata: ["legacy"] }, "solve"),
    "legacy",
  );
});

test("record errors identify the array, row and invalid field", () => {
  assert.throws(
    () => api.buildCallgraphFigure({ calls: [{ name: "solve", depth: 0 }] }),
    /call_id/,
  );
  assert.throws(
    () => api.buildCallgraphFigure({ regions: [], edges: "bad" }),
    /edges must/,
  );
  assert.throws(
    () => api.buildCallgraphFigure({ regions: [], edges: [null] }),
    /edges\[0\]/,
  );
  assert.throws(
    () => api.buildFlameFigure({ calls: [{ ...call, parent_call_id: {} }] }),
    /parent_call_id/,
  );
  assert.throws(
    () =>
      api.buildDurationTimeseriesFigure({
        points: [{ ...point, min_duration_seconds: 3 }],
      }),
    /bounds/,
  );
  assert.throws(
    () => api.buildRooflineFigure({ points: [], roofline: "bad" }),
    /roofline must/,
  );
  assert.throws(
    () =>
      api.buildRooflineFigure({
        points: [],
        roofline: [
          { arithmetic_intensity_flops_per_byte: 1, performance_gflops: 0 },
        ],
      }),
    /positive/,
  );
  for (const [row, error] of [
    [null, /intervals\[0\].record/],
    [[], /record/],
    [{ ...interval, region: 42 }, /region/],
    [{ ...interval, start_seconds: "0" }, /start_seconds/],
    [{ ...interval, end_seconds: -1 }, /end_seconds/],
    [{ ...interval, rank: "zero" }, /rank/],
    [{ ...interval, cost: Infinity }, /cost/],
  ])
    assert.throws(() => api.buildGanttFigure({ intervals: [row] }), error);
  assert.throws(() => api.buildGanttFigure({}), /intervals array/);
  assert.throws(() => api.buildFigure([]), /object/);
  assert.throws(() => api.buildFigure(null), /object/);
  assert.throws(
    () => api.buildFigure({ plot: "constructor" }),
    /No figure builder/,
  );
  assert.throws(
    () => api.buildFigure({ plot: "gantt", intervals: [null] }),
    /record/,
  );
  assert.throws(
    () => api.buildRegionSummaryFigure({ files: [{}] }),
    /region_statistics/,
  );
  assert.throws(
    () =>
      api.buildRegionSummaryFigure({
        files: [{ region_statistics: { solve: null } }],
      }),
    /statistics/,
  );
  assert.throws(
    () =>
      api.buildRegionSummaryFigure({
        files: [{ region_statistics: { solve: { count: "one" } } }],
      }),
    /count/,
  );
  assert.throws(
    () =>
      api.buildRooflineFigure({
        points: [
          {
            region: "solve",
            arithmetic_intensity_flops_per_byte: 0,
            performance_gflops: 1,
          },
        ],
      }),
    /positive/,
  );
  assert.throws(
    () => api.buildRankHeatmapFigure({ points: [{ region: "solve" }] }),
    /total_duration_seconds/,
  );
});

test("format versions must be positive integers, with a distinct upgrade error", () => {
  for (const version of [null, "1", -1, 0, 1.5, NaN, Infinity]) {
    assert.throws(
      () =>
        api.validatePlotData({
          plot: "gantt",
          intervals: [],
          format_version: version,
        }),
      /positive integer/,
    );
  }
  assert.throws(
    () =>
      api.validatePlotData({ plot: "gantt", intervals: [], format_version: 2 }),
    /newer/,
  );
  assert.equal(
    api.validatePlotData({ plot: "gantt", intervals: [], format_version: 1 }),
    "gantt",
  );
});

test("flame rejects duplicate IDs and cycles including filtered ancestors", () => {
  assert.throws(
    () => api.buildFlameFigure({ calls: [call, call] }),
    /duplicate call ID/,
  );
  const calls = [
    { ...call, call_id: 1, parent_call_id: 2 },
    { ...call, call_id: 2, parent_call_id: 1 },
  ];
  for (const filterRegion of [undefined, () => false])
    assert.throws(
      () => api.buildFlameFigure({ calls }, { filterRegion }),
      /cycle/,
    );
  assert.throws(
    () => api.buildFlameFigure({ calls: [{ ...call, call_id: null }] }),
    /call_id/,
  );
  assert.throws(
    () =>
      api.buildFlameFigure({
        calls: [{ region: "x", call_id: 1, start_seconds: 0 }],
      }),
    /end_seconds/,
  );
  assert.throws(
    () =>
      api.buildFlameFigure({
        calls: [{ ...call, inclusive_duration_seconds: -1 }],
      }),
    /nonnegative/,
  );
  const separate = api.buildFlameFigure({
    calls: [call, { ...call, file: "other" }],
  });
  assert.equal(new Set(separate.data[0].ids).size, 3);
});

test("collapsed recursive region graphs stay acyclic and report omissions", () => {
  const figure = api.buildCallgraphFigure({
    calls: [
      { call_id: 0, parent_id: null, name: "a", depth: 0 },
      { call_id: 1, parent_id: 0, name: "b", depth: 1 },
      { call_id: 2, parent_id: 1, name: "a", depth: 2 },
    ],
  });
  assert.deepEqual(figure.data[0].link.source, [0]);
  assert.equal(figure.diagnostics[0].code, "cycle-edge-omitted");
  const cyclicCalls = [{ call_id: 0, parent_id: 0, name: "a", depth: 0 }];
  assert.throws(
    () => api.buildCallgraphFigure({ calls: cyclicCalls }),
    /cycle/,
  );
});

test("compact callgraph uses measured edges, preserves zero and exposes unavailable weights", () => {
  const regions = [
    { name: "a", depth: 0 },
    { name: "b", depth: 0 },
    { name: "c", depth: 1, total_duration: 100 },
  ];
  const edges = [
    { parent: "a", child: "c", total_duration: 2 },
    { parent: "b", child: "c", value: 0 },
  ];
  assert.deepEqual(
    api.buildCallgraphFigure({ regions, edges }).data[0].link.value,
    [2, 0],
  );
  const missing = api.buildCallgraphFigure({
    regions,
    edges: edges.map(({ parent, child }) => ({ parent, child })),
  });
  assert.equal(missing.data[0].link.value.length, 0);
  assert.equal(missing.diagnostics.length, 2);
  assert.throws(
    () =>
      api.buildCallgraphFigure({
        regions,
        edges: [{ parent: "a", child: "c", value: -1 }],
      }),
    /nonnegative/,
  );
  const zero = api.buildCallgraphFigure({
    regions,
    edges: [{ parent: "a", child: "c", value: 0 }],
  });
  assert.deepEqual(zero.data[0].link.value, [0]);
});

test("duplicate chart cells cannot silently overwrite measurements", () => {
  for (const [builder, payload] of [
    [
      api.buildDurationsFigure,
      { bars: [{ region: "x", metric: "total", value_seconds: 1 }] },
    ],
    [api.buildLikwidFigure, { bars: [{ region: "x", series: "s", value: 1 }] }],
    [
      api.buildRankHeatmapFigure,
      { points: [{ region: "x", rank: 0, total_duration_seconds: 1 }] },
    ],
    [
      api.buildDensityFigure,
      {
        points: [
          {
            region: "x",
            bin_start_seconds: 0,
            bin_end_seconds: 1,
            occupied_seconds: 1,
          },
        ],
      },
    ],
  ]) {
    const key = Object.keys(payload)[0];
    payload[key].push(payload[key][0]);
    assert.throws(() => builder(payload), /duplicate cell/);
  }
});

test("unequal density grids retain exact boundaries and gaps", () => {
  const points = [
    {
      region: "x",
      bin_start_seconds: 0,
      bin_end_seconds: 1,
      occupied_seconds: 0.5,
    },
    {
      region: "x",
      bin_start_seconds: 2,
      bin_end_seconds: 4,
      occupied_seconds: 1,
    },
  ];
  const figure = api.buildDensityFigure({ points });
  assert.deepEqual(figure.data[0].x, [0, 1, 2, 4]);
  assert.deepEqual(figure.data[0].z, [[0.5, null, 0.5]]);
  const equalWidthGap = api.buildDensityFigure({
    points: [points[0], { ...points[1], bin_end_seconds: 3 }],
  });
  assert.deepEqual(equalWidthGap.data[0].x, [0, 1, 2, 3]);
  assert.deepEqual(equalWidthGap.data[0].z, [[0.5, null, 1]]);
  const raw = api.buildDensityFigure(
    { points },
    { valueKey: "occupied_seconds" },
  );
  assert.deepEqual(raw.data[0].z, [[0.5, null, 1]]);
  assert.throws(
    () =>
      api.buildDensityFigure({
        points: [points[0], { ...points[1], bin_start_seconds: 0.5 }],
      }),
    /overlapping/,
  );
});

test("nested layout overrides preserve axes and themes without mutating options", () => {
  const options = {
    theme: "dark",
    layout: {
      xaxis: { range: [0, 10] },
      font: { size: 20 },
      margin: { l: 20 },
      annotations: [{ text: "note" }],
    },
  };
  const original = structuredClone(options);
  const figure = api.buildGanttFigure({ intervals: [interval] }, options);
  assert.equal(figure.layout.xaxis.title.text, "Time (s)");
  assert.equal(figure.layout.xaxis.gridcolor, api.resolveTheme("dark").grid);
  assert.equal(figure.layout.font.color, api.resolveTheme("dark").text);
  assert.equal(figure.layout.margin.r, 24);
  assert.deepEqual(figure.layout.annotations, options.layout.annotations);
  assert.deepEqual(options, original);
});

test("colors remain stable through filtering and explicit registry overrides", () => {
  const payload = {
    intervals: [interval, { ...interval, region: "assemble" }],
  };
  const color = api.buildGanttFigure(payload).data[1].marker.color;
  assert.equal(
    api.buildGanttFigure(payload, {
      filterRegion: (name) => name === "assemble",
    }).data[0].marker.color,
    color,
  );
  const colors = api.createColorRegistry(["assemble"], { assemble: "red" });
  assert.equal(
    api.buildGanttFigure(payload, { colors }).data[1].marker.color,
    "red",
  );
  assert.ok(Object.isFrozen(colors));
  assert.equal(
    typeof api.buildGanttFigure({
      intervals: [{ ...interval, region: "constructor" }],
    }).data[0].marker.color,
    "string",
  );
});

test("theme snapshots isolate applications and exported presets", () => {
  const defaults = {
    layout: { xaxis: { range: [0, 3] }, annotations: [{ text: "original" }] },
  };
  const isolated = api.createFigureBuilder(defaults);
  defaults.layout.xaxis.range[1] = 999;
  defaults.layout.annotations[0].text = "changed";
  assert.deepEqual(
    isolated({ intervals: [interval] }).layout.xaxis.range,
    [0, 3],
  );
  assert.equal(
    isolated({ intervals: [interval] }).layout.annotations[0].text,
    "original",
  );
  api.resolveTheme("dark").neutral = "red";
  assert.notEqual(api.resolveTheme("dark").neutral, "red");
  const theme = { text: "purple", neutral: "pink" };
  api.setTheme(theme);
  theme.text = "orange";
  const build = api.createFigureBuilder({
    layout: { xaxis: { range: [0, 3] } },
  });
  api.setTheme("light");
  assert.equal(build({ intervals: [interval] }).layout.font.color, "purple");
  assert.equal(
    api.buildFlameFigure({ calls: [call] }, { theme: { neutral: "pink" } })
      .data[0].marker.colors[0],
    "pink",
  );
  assert.deepEqual(api.createFigureBuilder()({ intervals: [] }).data, []);
  api.setTheme();
});

test("time-series variability follows sorted rows and shares legend groups", () => {
  const payload = { points: [{ ...point, time_seconds: 2 }, point] };
  const error = api.buildDurationTimeseriesFigure(payload, {
    variability: "error",
  }).data[0];
  assert.deepEqual(error.error_y.array, [2, 2]);
  assert.deepEqual(error.error_y.arrayminus, [1, 1]);
  const band = api.buildDurationTimeseriesFigure(payload, {
    variability: "band",
  }).data;
  assert.deepEqual(
    band.map((trace) => trace.y),
    [
      [1, 1],
      [4, 4],
      [2, 2],
    ],
  );
  assert.equal(band[1].fill, "tonexty");
  assert.equal(new Set(band.map((trace) => trace.legendgroup)).size, 1);
  const missing = {
    points: [{ region: "x", time_seconds: 0, mean_duration_seconds: 1 }],
  };
  assert.deepEqual(
    api.buildDurationTimeseriesFigure(missing, { variability: "error" }).data[0]
      .error_y.array,
    [null],
  );
  assert.deepEqual(
    api.buildDurationTimeseriesFigure(missing, { variability: "band" }).data[0]
      .y,
    [null],
  );
});

test("comparison deltas expose missing and zero baselines and rank regressions", () => {
  const payload = {
    files: [
      {
        label: "before",
        region_statistics: {
          a: { count: 2 },
          b: { count: 0 },
          c: { count: 4 },
        },
      },
      {
        label: "after",
        region_statistics: {
          a: { count: 4 },
          b: { count: 1 },
          d: { count: 1 },
        },
      },
    ],
  };
  const absolute = api.buildComparisonFigure(payload, {
    metric: "count",
    comparison: "absolute",
  });
  assert.deepEqual(absolute.data[0].y, [2, 1, null, null]);
  const percent = api.buildComparisonFigure(payload, {
    metric: "count",
    comparison: "percent",
    sortBy: "name",
  });
  assert.deepEqual(percent.data[0].y, [100, null, null, null]);
  assert.equal(percent.diagnostics.length, 3);
  assert.throws(
    () =>
      api.buildComparisonFigure(payload, {
        files: [0],
        comparison: "absolute",
      }),
    /two runs/,
  );
  assert.throws(
    () => api.buildComparisonFigure(payload, { comparison: "bad" }),
    /comparison must/,
  );
  assert.throws(
    () =>
      api.buildComparisonFigure(payload, {
        metric: "bad",
        comparison: "absolute",
      }),
    /Unknown/,
  );
});

test("selection identities preserve raw names and linked mean traces", () => {
  const figure = api.buildGanttFigure({ intervals: [interval] });
  assert.deepEqual(
    api.getPointIdentity({ customdata: figure.data[0].customdata[0] }),
    { region: "solve", file: "run", rank: 0, call_id: 1 },
  );
  assert.equal(api.getPointIdentity(null), null);
  const imbalance = api.buildImbalanceFigure({
    points: [
      { region: "x", rank: 0, value_seconds: 1, mean_over_ranks_seconds: 1 },
    ],
  });
  assert.equal(imbalance.data[0].legendgroup, imbalance.data[1].legendgroup);
});

test("render, update and dispose forward arguments and preserve receiver", () => {
  const figure = api.buildGanttFigure(
    { intervals: [] },
    { layout: { uirevision: "session" } },
  );
  const plotly = {
    newPlot(...args) {
      assert.equal(this, plotly);
      return args;
    },
    react(...args) {
      assert.equal(this, plotly);
      return args;
    },
    purge(element) {
      assert.equal(this, plotly);
      return element;
    },
  };
  assert.equal(api.renderFigure(plotly, "chart", figure)[0], "chart");
  assert.equal(
    api.updateFigure(plotly, "chart", figure)[2].uirevision,
    "session",
  );
  assert.equal(api.disposeFigure(plotly, "chart"), "chart");
  assert.throws(() => api.disposeFigure({}, "chart"), /purge/);
  assert.throws(() => api.renderFigure(null, "chart", figure), /newPlot/);
  assert.throws(() => api.updateFigure(null, "chart", figure), /react/);
});

test("scaling custom fields work and invalid baselines fail clearly", () => {
  const points = [{ region: "x", machine: "a", result: 2 }];
  assert.deepEqual(
    api.buildSpeedupFigure({ points }, { xField: "machine", yField: "result" })
      .data[0].y,
    [2],
  );
  assert.throws(
    () => api.buildSpeedupFigure({ points }, { yField: "result" }),
    /num_ranks/,
  );
  assert.throws(
    () =>
      api.buildSpeedupFigure(
        { points },
        { xField: "machine", yField: "missing" },
      ),
    /missing/,
  );
  assert.throws(
    () =>
      api.buildSpeedupFigure({
        points: [{ region: "x", num_ranks: 0, speedup: 1 }],
      }),
    /baseline/,
  );
});
