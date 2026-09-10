import { test, expect } from "@playwright/test";
import { readdirSync } from "node:fs";
const fixtures = readdirSync(new URL("../fixtures", import.meta.url)).filter(
  (name) => name.endsWith(".json"),
);

test.beforeEach(async ({ page }, testInfo) => {
  await page.goto(`/test/browser/index.html?bundle=${testInfo.project.name}`);
  await page.waitForFunction(() => window.ready);
});

test("every exported fixture produces a real chart in both themes", async ({
  page,
}, testInfo) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  for (const theme of ["light", "dark"])
    for (const name of fixtures) {
      const result = await page.evaluate(
        async ({ name, theme }) => {
          const payload = await fetch(`/test/fixtures/${name}`).then(
            (response) => response.json(),
          );
          const figure = window.builders.buildFigure(payload, { theme });
          document.body.style.background =
            theme === "dark" ? "#171a21" : "white";
          try {
            await window.builders.renderFigure(window.Plotly, "chart", figure);
          } catch (error) {
            throw new Error(`${name} (${theme}): ${error.message}`);
          }
          const chart = document.getElementById("chart");
          return {
            traces: chart._fullData.length,
            nodes: chart.querySelectorAll("svg path, image").length,
          };
        },
        { name, theme },
      );
      expect(result.traces, name).toBeGreaterThan(0);
      expect(result.nodes, name).toBeGreaterThan(0);
    }
  expect(errors).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("last-fixture-dark.png") });
});

test("Gantt hover retains literal labels, identity, and zoom across updates", async ({
  page,
}, testInfo) => {
  const result = await page.evaluate(async () => {
    const payload = {
      intervals: [
        {
          region: "<b>solve</b> %{y}",
          file: "run",
          rank: 0,
          call_id: 7,
          start_seconds: 0,
          end_seconds: 2,
        },
      ],
    };
    const figure = window.builders.buildGanttFigure(payload, {
      layout: { uirevision: "same-profile" },
    });
    await window.builders.renderFigure(window.Plotly, "chart", figure);
    await window.Plotly.relayout("chart", { "xaxis.range": [0.5, 1.5] });
    await window.builders.updateFigure(
      window.Plotly,
      "chart",
      window.builders.buildGanttFigure(payload, {
        theme: "dark",
        layout: { uirevision: "same-profile" },
      }),
    );
    window.Plotly.Fx.hover("chart", [{ curveNumber: 0, pointNumber: 0 }]);
    const chart = document.getElementById("chart");
    return {
      range: chart._fullLayout.xaxis.range,
      identity: window.builders.getPointIdentity({
        customdata: chart.data[0].customdata[0],
      }),
    };
  });
  expect(result.range).toEqual([0.5, 1.5]);
  expect(result.identity.call_id).toBe(7);
  await expect(page.locator(".hoverlayer")).toContainText("<b>solve</b> %{y}");
  await expect(page.locator(".hoverlayer")).toContainText("run / rank 0");
  await page.screenshot({ path: testInfo.outputPath("gantt-hover.png") });
  await page.evaluate(() =>
    window.builders.disposeFigure(window.Plotly, "chart"),
  );
  await expect(page.locator("#chart svg")).toHaveCount(0);
});

test("variability bands, deltas and density edges render without invalid values", async ({
  page,
}, testInfo) => {
  const result = await page.evaluate(async () => {
    const b = window.builders;
    const figure = b.buildDurationTimeseriesFigure(
      {
        points: [0, 1, 2].map((time_seconds) => ({
          region: "solve",
          time_seconds,
          mean_duration_seconds: 2,
          min_duration_seconds: 1,
          max_duration_seconds: 3,
        })),
      },
      { variability: "band" },
    );
    await b.renderFigure(window.Plotly, "chart", figure);
    return {
      fills: document.querySelectorAll(".js-fill").length,
      groups: document
        .getElementById("chart")
        ._fullData.map((trace) => trace.legendgroup),
    };
  });
  expect(result.fills).toBeGreaterThan(0);
  expect(new Set(result.groups).size).toBe(1);
  await page.screenshot({ path: testInfo.outputPath("variability-band.png") });
  const delta = await page.evaluate(async () => {
    const figure = window.builders.buildComparisonFigure(
      {
        files: [
          { region_statistics: { solve: { count: 2 } } },
          { region_statistics: { solve: { count: 3 } } },
        ],
      },
      { metric: "count", comparison: "percent" },
    );
    await window.builders.renderFigure(window.Plotly, "chart", figure);
    return document.getElementById("chart").calcdata[0][0].s;
  });
  expect(delta).toBe(50);
  const density = await page.evaluate(async () => {
    const figure = window.builders.buildDensityFigure({
      points: [
        {
          region: "solve",
          bin_start_seconds: 0,
          bin_end_seconds: 1,
          occupied_seconds: 0.5,
        },
        {
          region: "solve",
          bin_start_seconds: 2,
          bin_end_seconds: 4,
          occupied_seconds: 1,
        },
      ],
    });
    await window.builders.renderFigure(window.Plotly, "chart", figure);
    return {
      x: document.getElementById("chart").calcdata[0][0].x,
      z: document.getElementById("chart").calcdata[0][0].z,
    };
  });
  expect(density.x).toEqual([0, 1, 2, 4]);
  expect(density.z[0].filter((value) => value != null)).toEqual([0.5, 0.5]);
  await page.screenshot({ path: testInfo.outputPath("density-gaps.png") });
});

test("large deterministic timeline records build, render and memory measurements", async ({
  page,
}, testInfo) => {
  const measurements = await page.evaluate(async () => {
    const intervals = Array.from({ length: 10000 }, (_, index) => ({
      region: `region-${index % 20}`,
      rank: index % 4,
      start_seconds: index / 1000,
      end_seconds: (index + 1) / 1000,
    }));
    const start = performance.now();
    const figure = window.builders.buildGanttFigure({ intervals });
    const built = performance.now();
    await window.builders.renderFigure(window.Plotly, "chart", figure, {
      staticPlot: true,
    });
    return {
      count: figure.data.reduce((sum, trace) => sum + trace.x.length, 0),
      buildMs: built - start,
      renderMs: performance.now() - built,
      heapBytes: performance.memory?.usedJSHeapSize ?? null,
    };
  });
  expect(measurements.count).toBe(10000);
  expect(measurements.buildMs).toBeLessThan(10000);
  expect(measurements.renderMs).toBeLessThan(20000);
  await testInfo.attach("performance.json", {
    body: JSON.stringify(measurements),
    contentType: "application/json",
  });
});
