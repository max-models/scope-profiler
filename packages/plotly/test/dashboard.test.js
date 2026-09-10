// The browser glue, driven without a browser: the registry takes a Plotly
// stand-in and containers that are plain objects, which is enough to pin down
// what it remembers, what it redraws, and what it forgets.
import test from "node:test";
import assert from "node:assert/strict";
import {
  createFigureRegistry,
  documentTheme,
  matchesRegionFilter,
  parseRegionFilter,
  regionFilter,
} from "../src/dashboard.js";
import { buildComparisonFigure, resolveTheme, setTheme } from "../src/index.js";

const payload = {
  plot: "durations",
  bars: [
    { file: "one", region: "prop: push", metric: "total", value_seconds: 3 },
    {
      file: "one",
      region: "setup prop: push",
      metric: "total",
      value_seconds: 1,
    },
    { file: "one", region: "solve", metric: "total", value_seconds: 2 },
  ],
};

function fakePlotly() {
  const calls = [];
  return {
    calls,
    newPlot: (element, data, layout, config) =>
      calls.push({ how: "newPlot", element, data, layout, config }),
    react: (element, data, layout, config) =>
      calls.push({ how: "react", element, data, layout, config }),
  };
}

test("a filter is comma-separated substrings, with ^ anchoring a term", () => {
  assert.deepEqual(parseRegionFilter(" prop:, Setup: total ,, "), [
    "prop:",
    "setup: total",
  ]);
  assert.deepEqual(parseRegionFilter(null), []);
  const terms = parseRegionFilter("^prop:");
  assert.ok(matchesRegionFilter("prop: push", terms));
  assert.ok(!matchesRegionFilter("setup prop: push", terms));
  assert.ok(
    matchesRegionFilter("setup prop: push", parseRegionFilter("prop:")),
  );
});

test("an empty filter box is no filter at all, not a predicate that says yes", () => {
  assert.equal(regionFilter(""), undefined);
  assert.equal(regionFilter("   ,  "), undefined);
  assert.equal(typeof regionFilter("solve"), "function");
});

test("the registry filters, and redraws through react once a figure exists", async () => {
  const plotly = fakePlotly();
  const figures = createFigureRegistry(plotly, { theme: "dark" });
  const container = {};

  await figures.render(container, payload, { regionFilter: "^prop:" });
  assert.equal(plotly.calls.length, 1);
  assert.equal(plotly.calls[0].how, "newPlot");
  // "^prop:" keeps the propagator but not the region that merely mentions it.
  assert.deepEqual(plotly.calls[0].data[0].x, ["prop: push"]);
  assert.equal(plotly.calls[0].layout.font.color, "#e5e7eb");

  // A second render of the same container reacts into it, keeping the view.
  await figures.render(container, payload, { regionFilter: "" });
  assert.equal(plotly.calls[1].how, "react");
  assert.deepEqual(plotly.calls[1].data[0].x, [
    "prop: push",
    "solve",
    "setup prop: push",
  ]);
});

test("refresh rebuilds every live figure under the theme of the moment", async () => {
  const plotly = fakePlotly();
  let theme = "light";
  const figures = createFigureRegistry(plotly, { theme: () => theme });
  const first = {};
  const second = {};
  await figures.render(first, payload);
  await figures.render(second, payload);

  theme = "dark";
  await figures.refresh();
  assert.equal(figures.size, 2);
  const redrawn = plotly.calls.slice(2);
  assert.deepEqual(
    redrawn.map((call) => call.how),
    ["react", "react"],
  );
  for (const call of redrawn) assert.equal(call.layout.font.color, "#e5e7eb");
  // refresh also moves the module default, so a figure built outside the
  // registry picks up the same theme.
  assert.equal(resolveTheme().text, "#e5e7eb");
  setTheme("auto");
});

test("a container that has left the page is dropped rather than redrawn", async () => {
  const plotly = fakePlotly();
  const figures = createFigureRegistry(plotly);
  const detached = { isConnected: false };
  const live = { isConnected: true };
  await figures.render(detached, payload);
  await figures.render(live, payload);
  assert.equal(figures.size, 2);

  await figures.refresh();
  assert.equal(figures.size, 1);
  assert.equal(plotly.calls.at(-1).element, live);

  assert.equal(figures.forget(live), true);
  assert.equal(figures.size, 0);
});

test("forgetting a figure tears its plot down where the bundle can", async () => {
  const plotly = fakePlotly();
  const purged = [];
  plotly.purge = (element) => purged.push(element);
  const figures = createFigureRegistry(plotly);
  const first = {};
  const second = {};
  await figures.render(first, payload);
  await figures.render(second, payload);

  figures.clear();
  assert.deepEqual(purged, [first, second]);
  assert.equal(figures.size, 0);
  // A bundle without purge is not an error, just no teardown.
  const bare = { newPlot: () => {} };
  const plain = createFigureRegistry(bare);
  await plain.render({}, payload);
  plain.clear();
  assert.equal(plain.size, 0);
});

test("a figure no plot kind names is drawn by its own builder", async () => {
  const plotly = fakePlotly();
  const figures = createFigureRegistry(plotly);
  const statistics = {
    plot: "region_statistics",
    files: [
      {
        label: "a",
        region_statistics: { solve: { total_duration_seconds: 2 } },
      },
      {
        label: "b",
        region_statistics: { solve: { total_duration_seconds: 3 } },
      },
    ],
  };
  await figures.render({}, statistics, { build: buildComparisonFigure });
  // The comparison reading is vertical bars over the regions both runs share.
  assert.equal(plotly.calls[0].data.length, 2);
  assert.deepEqual(plotly.calls[0].data[0].x, ["solve"]);
});

test("watch returns the unsubscribe, and is inert without an event target", async () => {
  const plotly = fakePlotly();
  const figures = createFigureRegistry(plotly, { theme: "light" });
  const listeners = new Map();
  const target = {
    addEventListener: (event, listener) => listeners.set(event, listener),
    removeEventListener: (event) => listeners.delete(event),
  };
  const stop = figures.watch(target);
  assert.ok(listeners.has("themechanged"));
  stop();
  assert.equal(listeners.size, 0);
  assert.equal(typeof figures.watch({}), "function");
});

test("documentTheme falls back to auto outside a browser", () => {
  assert.equal(documentTheme(), "auto");
});
