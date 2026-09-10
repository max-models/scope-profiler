import test from "node:test";
import assert from "node:assert/strict";
import {
  buildCallgraphFigure,
  buildDensityFigure,
  buildDurationTimeseriesFigure,
  buildDurationsFigure,
  buildFigure,
  buildFlameFigure,
  buildGanttFigure,
  buildHistogramFigure,
  buildImbalanceFigure,
  buildLikwidFigure,
  buildRooflineFigure,
  buildRankHeatmapFigure,
  buildRegionSummaryFigure,
  buildComparisonFigure,
  buildScalingEfficiencyFigure,
  buildSpeedupFigure,
  buildWeakScalingEfficiencyFigure,
  buildWeakScalingFigure,
  inferPlotKind,
  renderFigure,
  updateFigure,
  SUMMARY_METRICS,
  resolveTheme,
  setTheme,
  validatePlotData,
} from "../src/index.js";

test("gantt gives each region and rank a lane, and honours supplied colors", () => {
  const figure = buildGanttFigure({
    colors: { solve: "#123456" },
    intervals: [
      {
        file: "one",
        rank: 0,
        region: "solve",
        start_seconds: 0,
        end_seconds: 2,
      },
      {
        file: "one",
        rank: 1,
        region: "solve",
        start_seconds: 0,
        end_seconds: 1,
      },
    ],
  });
  // Bars carry a lane index; the axis carries the names.
  assert.deepEqual(figure.data[0].y, [0, 1]);
  assert.deepEqual(figure.layout.yaxis.ticktext, [
    "solve (rank 0)",
    "solve (rank 1)",
  ]);
  assert.deepEqual(figure.layout.yaxis.tickvals, [0, 1]);
  assert.equal(figure.data[0].marker.color, "#123456");
});

test("gantt keeps a nested profile legible instead of stacking it on one row", () => {
  // Every region of a rank on one lane means the session bar covers the rest.
  const payload = {
    intervals: [
      {
        file: "one",
        rank: 0,
        region: "session",
        start_seconds: 0,
        end_seconds: 10,
      },
      {
        file: "one",
        rank: 0,
        region: "setup",
        start_seconds: 1,
        end_seconds: 3,
      },
      {
        file: "one",
        rank: 0,
        region: "solve",
        start_seconds: 4,
        end_seconds: 9,
      },
    ],
  };
  const figure = buildGanttFigure(payload);
  // The y axis counts up from the bottom, so an ascending range puts the
  // first (enclosing) region on the bottom lane -- the order
  // `scope-profiler plot gantt` draws.
  assert.deepEqual(figure.layout.yaxis.ticktext, [
    "session (rank 0)",
    "setup (rank 0)",
    "solve (rank 0)",
  ]);
  assert.deepEqual(figure.layout.yaxis.range, [-0.5, 2.5]);
  assert.equal(new Set(figure.data.flatMap((trace) => trace.y)).size, 3);
  // The opt-out keeps the compact one-row-per-rank view, rank 0 on top.
  const compact = buildGanttFigure(payload, { laneBy: "rank" });
  assert.deepEqual(compact.layout.yaxis.ticktext, ["one / rank 0"]);
  assert.deepEqual(compact.layout.yaxis.range, [0.5, -0.5]);
});

test("gantt names a lane by its run only when the payload holds several", () => {
  const figure = buildGanttFigure({
    intervals: [
      {
        file: "one",
        rank: 0,
        region: "solve",
        start_seconds: 0,
        end_seconds: 2,
      },
      {
        file: "two",
        rank: 0,
        region: "solve",
        start_seconds: 0,
        end_seconds: 1,
      },
    ],
  });
  assert.deepEqual(figure.layout.yaxis.ticktext, [
    "one / solve (rank 0)",
    "two / solve (rank 0)",
  ]);
});

test("flame uses parent_call_id rather than matching region names", () => {
  const figure = buildFlameFigure({
    calls: [
      {
        file: "one",
        rank: 0,
        call_id: 1,
        parent_call_id: null,
        region: "step",
        start_seconds: 0,
        end_seconds: 4,
      },
      {
        file: "one",
        rank: 0,
        call_id: 2,
        parent_call_id: 1,
        region: "step",
        start_seconds: 1,
        end_seconds: 2,
      },
    ],
  });
  assert.deepEqual(figure.data[0].parents, [
    "",
    "scope-profiler-root",
    "one:0:1",
  ]);
});

test("speedup honours a non-rank x field", () => {
  const figure = buildSpeedupFigure({
    options: { x_field: "total_cores", baseline: 4 },
    points: [
      { region: "solve", total_cores: 4, speedup: 1 },
      { region: "solve", total_cores: 8, speedup: 1.8 },
    ],
  });
  assert.deepEqual(figure.data[0].x, [4, 8]);
  assert.deepEqual(figure.data[1].y, [1, 2]);
});

test("durations preserves stacked child segments", () => {
  const figure = buildDurationsFigure({
    metrics: ["total"],
    colors: { own: "#111111" },
    bars: [
      {
        file: "one",
        region: "solve",
        metric: "total",
        segment: "own",
        value_seconds: 2,
      },
      {
        file: "one",
        region: "solve",
        metric: "total",
        segment: "child",
        value_seconds: 3,
      },
    ],
  });
  assert.equal(figure.layout.barmode, "stack");
  assert.deepEqual(
    figure.data.map((trace) => trace.name),
    ["own", "child"],
  );
  assert.equal(figure.data[0].marker.color, "#111111");
});

test("additional exported data types build Plotly traces", () => {
  assert.equal(
    buildDurationTimeseriesFigure({
      points: [
        {
          region: "solve",
          time_seconds: 1,
          mean_duration_seconds: 2,
          min_duration_seconds: 1,
          max_duration_seconds: 3,
        },
      ],
    }).data[0].type,
    "scatter",
  );
  assert.equal(
    buildHistogramFigure({
      bins: [
        {
          region: "solve",
          bin_low_seconds: 0,
          bin_center_seconds: 1,
          bin_high_seconds: 2,
          count: 4,
        },
      ],
    }).data[0].type,
    "bar",
  );
  assert.equal(
    buildRankHeatmapFigure({
      points: [{ rank: 0, region: "solve", total_duration_seconds: 2 }],
    }).data[0].type,
    "heatmap",
  );
  assert.equal(
    buildImbalanceFigure({
      metric: "total",
      points: [
        {
          rank: 0,
          region: "solve",
          value_seconds: 2,
          mean_over_ranks_seconds: 2,
        },
      ],
    }).data[0].type,
    "scatter",
  );
});

test("flame honours filterRegion, re-parenting survivors onto the nearest kept ancestor", () => {
  const payload = {
    calls: [
      {
        file: "one",
        rank: 0,
        call_id: 1,
        parent_call_id: null,
        region: "step",
        start_seconds: 0,
        end_seconds: 4,
        inclusive_duration_seconds: 4,
      },
      {
        file: "one",
        rank: 0,
        call_id: 2,
        parent_call_id: 1,
        region: "noise",
        start_seconds: 1,
        end_seconds: 3,
        inclusive_duration_seconds: 2,
      },
      {
        file: "one",
        rank: 0,
        call_id: 3,
        parent_call_id: 2,
        region: "solve",
        start_seconds: 1,
        end_seconds: 2,
        inclusive_duration_seconds: 1,
      },
    ],
  };
  const figure = buildFlameFigure(payload, {
    filterRegion: (region) => region !== "noise",
  });
  const trace = figure.data[0];
  assert.deepEqual(trace.labels, ["All calls", "step", "solve"]);
  assert.deepEqual(trace.parents, ["", "scope-profiler-root", "one:0:1"]);
  assert.equal(trace.values[0], 4);
});

test("flame re-parents onto the root when every ancestor is filtered out", () => {
  const figure = buildFlameFigure(
    {
      calls: [
        {
          file: "one",
          rank: 0,
          call_id: 1,
          parent_call_id: null,
          region: "noise",
          start_seconds: 0,
          end_seconds: 4,
          inclusive_duration_seconds: 4,
        },
        {
          file: "one",
          rank: 0,
          call_id: 2,
          parent_call_id: 1,
          region: "solve",
          start_seconds: 1,
          end_seconds: 3,
          inclusive_duration_seconds: 2,
        },
      ],
    },
    { filterRegion: (region) => region === "solve" },
  );
  assert.deepEqual(figure.data[0].parents, ["", "scope-profiler-root"]);
  assert.equal(figure.data[0].values[0], 2);
});

test("buildFigure dispatches on the document's own plot kind", () => {
  const figure = buildFigure({
    format: "scope-profiler-plot-data",
    format_version: 1,
    plot: "gantt",
    intervals: [
      {
        file: "one",
        rank: 0,
        region: "solve",
        start_seconds: 0,
        end_seconds: 2,
      },
    ],
  });
  assert.equal(figure.layout.barmode, "overlay");
});

test("buildFigure infers the kind of a payload written before the envelope", () => {
  assert.equal(
    inferPlotKind({
      points: [
        {
          region: "solve",
          rank: 0,
          value_seconds: 1,
          mean_over_ranks_seconds: 1,
        },
      ],
    }),
    "imbalance",
  );
  assert.equal(
    inferPlotKind({
      points: [{ region: "solve", rank: 0, total_duration_seconds: 1 }],
    }),
    "rank_heatmap",
  );
  assert.equal(
    inferPlotKind({
      points: [
        {
          file: "one",
          region: "solve",
          bin_start_seconds: 0,
          bin_end_seconds: 1,
          occupied_seconds: 0.5,
        },
      ],
    }),
    "density",
  );
  assert.equal(
    inferPlotKind({ bars: [{ series: "run", region: "solve", value: 2 }] }),
    "likwid",
  );
  assert.equal(
    inferPlotKind({
      calls: [{ call_id: 0, parent_id: null, name: "solve", depth: 0 }],
    }),
    "callgraph",
  );
  assert.equal(inferPlotKind({ nothing: true }), undefined);
  assert.equal(
    buildFigure({
      bins: [
        {
          region: "solve",
          bin_low_seconds: 0,
          bin_center_seconds: 1,
          bin_high_seconds: 2,
          count: 4,
        },
      ],
    }).data[0].type,
    "bar",
  );
});

test("buildFigure refuses a foreign document or a newer format version", () => {
  assert.throws(
    () => buildFigure({ format: "something-else", intervals: [] }),
    /scope-profiler-plot-data/,
  );
  assert.throws(
    () =>
      buildFigure({
        format: "scope-profiler-plot-data",
        format_version: 2,
        plot: "gantt",
        intervals: [],
      }),
    /upgrade/,
  );
  assert.throws(() => buildFigure({ points: [] }), /options\.plot/);
});

test("validatePlotData names the required data for each selected builder", () => {
  assert.equal(validatePlotData({ plot: "roofline", points: [] }), "roofline");
  assert.equal(
    validatePlotData({ plot: "callgraph", regions: [], edges: [] }),
    "callgraph",
  );
  assert.throws(
    () => validatePlotData({ plot: "roofline" }),
    /requires a points array/,
  );
  assert.throws(() => buildFigure({ plot: "likwid" }), /requires a bars array/);
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
  const efficiency = buildScalingEfficiencyFigure({
    options: { baseline: 2 },
    points,
  });
  assert.deepEqual(efficiency.data[0].y, [1, 0.7]);
  assert.equal(efficiency.layout.yaxis.title, "Scaling efficiency");
  // Dispatch alone must pick the right column, with no explicit builder.
  assert.deepEqual(
    buildFigure({ plot: "weak_scaling", options: { baseline: 2 }, points })
      .data[0].y,
    [1, 1.4],
  );
});

test("density reports occupancy as a fraction of each bin, or raw seconds", () => {
  const payload = {
    points: [
      {
        file: "one",
        region: "solve",
        bin_start_seconds: 0,
        bin_end_seconds: 2,
        occupied_seconds: 1,
      },
      {
        file: "one",
        region: "solve",
        bin_start_seconds: 2,
        bin_end_seconds: 4,
        occupied_seconds: 2,
      },
    ],
  };
  const figure = buildDensityFigure(payload);
  assert.deepEqual(figure.data[0].y, ["one / solve"]);
  assert.deepEqual(figure.data[0].x, [1, 3]);
  assert.deepEqual(figure.data[0].z, [[0.5, 1]]);
  assert.equal(figure.data[0].zmax, 1);
  assert.deepEqual(
    buildDensityFigure(payload, { valueKey: "occupied_seconds" }).data[0].z,
    [[1, 2]],
  );
});

test("region summary ranks by the pooled metric and keeps the head of the list", () => {
  const payload = {
    files: [
      {
        label: "run",
        region_statistics: {
          small: { count: 1, total_duration_seconds: 1 },
          big: { count: 4, total_duration_seconds: 9 },
          middle: { count: 2, total_duration_seconds: 5 },
        },
      },
    ],
  };
  const figure = buildRegionSummaryFigure(payload, { topN: 2 });
  assert.deepEqual(figure.data[0].y, ["big", "middle"]);
  assert.deepEqual(figure.data[0].x, [9, 5]);
  assert.deepEqual(figure.data[0].customdata, [4, 2]);
  assert.deepEqual(
    buildRegionSummaryFigure(payload, { metric: "count" }).data[0].y,
    ["big", "middle", "small"],
  );
  assert.deepEqual(
    buildRegionSummaryFigure(payload, {
      filterRegion: (region) => region !== "big",
    }).data[0].y,
    ["middle", "small"],
  );
});

test("callgraph builds a sankey from either export shape and drops cycles", () => {
  const compact = buildCallgraphFigure({
    regions: [
      { name: "step", depth: 0, total_duration: 4 },
      { name: "solve", depth: 1, total_duration: 3 },
    ],
    edges: [
      { parent: "step", child: "solve" },
      { parent: "solve", child: "solve" },
    ],
  });
  assert.equal(compact.data[0].type, "sankey");
  assert.deepEqual(compact.data[0].node.label, ["step", "solve"]);
  assert.deepEqual(compact.data[0].link.source, [0]);
  assert.deepEqual(compact.data[0].link.value, [3]);
  const full = buildCallgraphFigure({
    calls: [
      { call_id: 0, parent_id: null, name: "step", depth: 0 },
      { call_id: 1, parent_id: 0, name: "solve", depth: 1 },
      { call_id: 2, parent_id: 0, name: "solve", depth: 1 },
    ],
  });
  assert.deepEqual(full.data[0].link.value, [2]);
  assert.match(
    buildCallgraphFigure({ regions: [{ name: "step", depth: 0 }], edges: [] })
      .layout.annotations[0].text,
    /No data/,
  );
  assert.throws(() => buildCallgraphFigure({ points: [] }), /regions or calls/);
});

test("likwid groups one hardware-counter metric by series", () => {
  const figure = buildLikwidFigure(
    {
      metric: "DP MFLOP/s",
      bars: [
        { series: "rank 0", region: "solve", value: 900 },
        { series: "rank 1", region: "solve", value: 850 },
      ],
    },
    { logScale: true },
  );
  assert.deepEqual(
    figure.data.map((trace) => trace.name),
    ["rank 0", "rank 1"],
  );
  assert.deepEqual(figure.data[0].x, ["solve"]);
  assert.equal(figure.layout.yaxis.title, "DP MFLOP/s");
  assert.equal(figure.layout.yaxis.type, "log");
});

test("roofline keeps region points and its machine ceiling distinct", () => {
  const figure = buildRooflineFigure({
    colors: { run: "#123456" },
    points: [
      {
        file: "run",
        region: "solve",
        rank: 0,
        arithmetic_intensity_flops_per_byte: 2,
        performance_gflops: 80,
        bandwidth_gbs: 40,
      },
    ],
    roofline: [
      { arithmetic_intensity_flops_per_byte: 1, performance_gflops: 40 },
      { arithmetic_intensity_flops_per_byte: 4, performance_gflops: 100 },
    ],
  });
  assert.equal(figure.data[0].type, "scatter");
  assert.deepEqual(figure.data[0].x, [2]);
  assert.equal(figure.data[1].name, "roofline");
  assert.equal(figure.layout.xaxis.type, "log");
  assert.equal(
    inferPlotKind({
      points: [{ arithmetic_intensity_flops_per_byte: 2, rank: 0 }],
    }),
    "roofline",
  );
});

test("weak-scaling efficiency shares the efficiency column but not its title", () => {
  const points = [
    { region: "solve", num_ranks: 2, efficiency: 1 },
    { region: "solve", num_ranks: 4, efficiency: 0.7 },
  ];
  const figure = buildWeakScalingEfficiencyFigure({
    options: { baseline: 2 },
    points,
  });
  assert.deepEqual(figure.data[0].y, [1, 0.7]);
  assert.equal(figure.layout.yaxis.title, "Weak-scaling efficiency");
  // Its ideal is flat, not the rank-proportional line speedup draws.
  assert.deepEqual(figure.data[1].y, [1, 1]);

  // The document's own kind has to pick the reading, since the rows cannot:
  // both efficiencies are stored under the same column.
  assert.equal(
    buildFigure({ plot: "weak_scaling_efficiency", options: {}, points }).layout
      .yaxis.title,
    "Weak-scaling efficiency",
  );
  // Without one, an efficiency payload still infers the strong-scaling
  // reading it always did.
  assert.equal(inferPlotKind({ points }), "scaling_efficiency");
});

test("comparison keeps every shared region, vertically, for the two runs named", () => {
  const payload = {
    files: [
      {
        label: "before",
        region_statistics: {
          solve: { count: 2, total_duration_seconds: 9 },
          setup: { count: 1, total_duration_seconds: 1 },
          only_before: { count: 1, total_duration_seconds: 5 },
        },
      },
      {
        label: "after",
        region_statistics: {
          solve: { count: 2, total_duration_seconds: 4 },
          setup: { count: 1, total_duration_seconds: 1 },
        },
      },
      { label: "unrelated", region_statistics: { solve: {} } },
    ],
  };
  const figure = buildComparisonFigure(payload, { files: ["before", "after"] });
  assert.equal(figure.data.length, 2);
  assert.equal(figure.data[0].name, "before");
  assert.equal(figure.data[1].name, "after");
  // Vertical bars, and a region only one run recorded is dropped rather than
  // drawn as a zero against a real measurement.
  assert.equal(figure.data[0].orientation, undefined);
  assert.deepEqual(figure.data[0].x, ["solve", "setup"]);
  assert.deepEqual(figure.data[0].y, [9, 1]);
  assert.deepEqual(figure.data[1].y, [4, 1]);

  // Short metric names resolve to the fields region_statistics stores.
  assert.deepEqual(
    buildComparisonFigure(payload, { files: [0, 1], metric: "total" }).data[0]
      .y,
    [9, 1],
  );
});

test("a theme colours the chrome, and auto leaves it to the host page", () => {
  const payload = {
    options: { x_label: "MPI ranks" },
    points: [{ region: "solve", num_ranks: 2, speedup: 1 }],
  };
  const auto = buildSpeedupFigure(payload);
  assert.equal(auto.layout.font.color, undefined);
  assert.equal(typeof auto.layout.xaxis.title, "string");

  const dark = buildSpeedupFigure(payload, { theme: "dark" });
  assert.equal(dark.layout.font.color, resolveTheme("dark").text);
  assert.equal(dark.layout.hoverlabel.bgcolor, resolveTheme("dark").hoverBg);
  assert.equal(dark.layout.xaxis.gridcolor, resolveTheme("dark").grid);
  // An axis title passed as a bare string still picks up the muted colour.
  assert.equal(dark.layout.xaxis.title.text, "MPI ranks");
  assert.equal(dark.layout.xaxis.title.font.color, resolveTheme("dark").muted);

  // A page with a toggle sets the default once instead of passing it around.
  setTheme("dark");
  try {
    assert.equal(buildSpeedupFigure(payload).layout.font.color, "#e5e7eb");
    // Per-call options still win over the default.
    assert.equal(
      buildSpeedupFigure(payload, { theme: "light" }).layout.font.color,
      resolveTheme("light").text,
    );
    // And a caller can supply its own tokens.
    assert.equal(
      buildSpeedupFigure(payload, { theme: { text: "#abcdef" } }).layout.font
        .color,
      "#abcdef",
    );
  } finally {
    setTheme("auto");
  }
  assert.equal(buildSpeedupFigure(payload).layout.font.color, undefined);
});

test("a region name cannot rewrite or inject into a hover label", () => {
  // Region and file names come from the profiled application. Plotly
  // substitutes %{...} inside a hovertemplate and renders the rest as HTML.
  const hostile = "%{y} <img src=x onerror=alert(1)>";
  const templates = [
    buildDurationsFigure({
      bars: [{ region: "a", metric: "total", value_seconds: 1, file: hostile }],
    }).data[0].hovertemplate,
    buildGanttFigure({
      intervals: [
        { region: hostile, rank: 0, start_seconds: 0, end_seconds: 1 },
      ],
    }).data[0].hovertemplate,
    buildLikwidFigure({
      bars: [{ series: hostile, region: "a", value: 1 }],
    }).data[0].hovertemplate,
    buildHistogramFigure({
      bins: [
        {
          region: hostile,
          bin_low_seconds: 0,
          bin_high_seconds: 1,
          bin_center_seconds: 0.5,
          count: 2,
        },
      ],
    }).data[0].hovertemplate,
  ];
  for (const template of templates) {
    assert.ok(!template.includes("<img"), `unescaped markup in ${template}`);
    // The name's own "%{" is neutralised; any %{...} still in the template is
    // the builder's own substitution, not one the region smuggled in.
    assert.ok(
      template.includes("%&#123;y}"),
      `a region rewrote the template: ${template}`,
    );
  }
  // The flame chart passes plain hovertext, which Plotly also renders as HTML.
  const flame = buildFlameFigure({
    calls: [
      {
        call_id: 1,
        parent_call_id: null,
        region: hostile,
        start_seconds: 0,
        end_seconds: 1,
      },
    ],
  });
  assert.ok(!flame.data[0].hovertext[1].includes("<img"));
});

test("an empty payload keeps the caller's annotations and hides the grid", () => {
  const figure = buildGanttFigure(
    { intervals: [] },
    { layout: { annotations: [{ text: "mine" }] } },
  );
  assert.deepEqual(
    figure.layout.annotations.map((note) => note.text),
    ["mine", "No data to display."],
  );
  assert.equal(figure.layout.xaxis.visible, false);
  assert.equal(figure.layout.yaxis.visible, false);
});

test("density puts every cell at its own bin centre", () => {
  // Two runs binned into the same number of bins over different durations.
  const points = [
    ...[0, 1, 2].map((bin) => ({
      file: "short",
      region: "solve",
      bin_start_seconds: bin,
      bin_end_seconds: bin + 1,
      occupied_seconds: 0.5,
    })),
    ...[0, 1, 2].map((bin) => ({
      file: "long",
      region: "solve",
      bin_start_seconds: bin * 10,
      bin_end_seconds: bin * 10 + 10,
      occupied_seconds: 5,
    })),
  ];
  const figure = buildDensityFigure({ points });
  assert.deepEqual(figure.data[0].x, [0.5, 1.5, 2.5, 5, 15, 25]);
  // Every point lands in a cell; none is dropped onto another run's centres.
  const filled = figure.data[0].z.flat().filter((value) => value !== null);
  assert.equal(filled.length, points.length);
});

test("the roofline ceiling follows the theme instead of a fixed near-black", () => {
  const payload = {
    points: [
      {
        region: "solve",
        rank: 0,
        arithmetic_intensity_flops_per_byte: 1,
        performance_gflops: 2,
        bandwidth_gbs: 3,
      },
    ],
    roofline: [
      { arithmetic_intensity_flops_per_byte: 1, performance_gflops: 9 },
    ],
  };
  const dark = buildRooflineFigure(payload, { theme: "dark" });
  const light = buildRooflineFigure(payload, { theme: "light" });
  assert.equal(dark.data[1].line.color, resolveTheme("dark").text);
  assert.equal(light.data[1].line.color, resolveTheme("light").text);
  // It is the only figure with a title, so it needs room the shared top
  // margin does not leave.
  assert.ok(dark.layout.margin.t > 32);
});

test("durations accepts a colour keyed by the bare rank it labels", () => {
  const bars = [0, 1].map((rank) => ({
    rank,
    region: "solve",
    metric: "total",
    value_seconds: 1,
  }));
  const figure = buildDurationsFigure({ bars }, { colors: { 1: "#abcdef" } });
  const byName = new Map(
    figure.data.map((trace) => [trace.name, trace.marker.color]),
  );
  assert.equal(byName.get("rank 1"), "#abcdef");
  assert.notEqual(byName.get("rank 0"), "#abcdef");
});

test("durations and the heatmap rank their categories by pooled cost", () => {
  const bars = [
    { file: "one", region: "cheap", metric: "total", value_seconds: 1 },
    { file: "one", region: "costly", metric: "total", value_seconds: 9 },
  ];
  assert.deepEqual(buildDurationsFigure({ bars }).data[0].x, [
    "costly",
    "cheap",
  ]);
  const points = [
    { rank: 0, region: "cheap", total_duration_seconds: 1 },
    { rank: 0, region: "costly", total_duration_seconds: 9 },
  ];
  assert.deepEqual(buildRankHeatmapFigure({ points }).data[0].x, [
    "costly",
    "cheap",
  ]);
});

test("an unknown region-statistics metric is named, not drawn as nulls", () => {
  const payload = {
    files: [{ label: "one", region_statistics: { solve: { count: 2 } } }],
  };
  assert.throws(
    () => buildRegionSummaryFigure(payload, { metric: "median" }),
    /Unknown region-statistics metric "median"/,
  );
  // Both the short alias and the stored field name still work.
  assert.equal(SUMMARY_METRICS.count, "count");
  for (const metric of ["count", "total", "total_duration_seconds"])
    assert.ok(buildRegionSummaryFigure(payload, { metric }).data.length > 0);
});

test("updateFigure prefers react, and falls back to newPlot without one", () => {
  const figure = buildGanttFigure({ intervals: [] });
  const calls = [];
  const both = {
    newPlot: (...args) => calls.push(["newPlot", args[3]]),
    react: (...args) => calls.push(["react", args[3]]),
  };
  updateFigure(both, "#chart", figure);
  renderFigure(both, "#chart", figure);
  assert.deepEqual(
    calls.map(([name]) => name),
    ["react", "newPlot"],
  );
  assert.equal(calls[0][1].responsive, true);
  assert.equal(calls[0][1].displaylogo, false);

  const legacy = { newPlot: (...args) => calls.push(["legacy", args[3]]) };
  updateFigure(legacy, "#chart", figure, { staticPlot: true });
  assert.equal(calls[2][0], "legacy");
  assert.equal(calls[2][1].staticPlot, true);

  assert.throws(
    () => updateFigure({}, "#chart", figure),
    /react\(\) or newPlot/,
  );
});
