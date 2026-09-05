import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

const packageDirectory = resolve(import.meta.dirname, "..");

test("the packed package installs and exposes its ESM API", () => {
  const consumerDirectory = mkdtempSync(
    join(tmpdir(), "scope-profiler-plotly-"),
  );
  let tarball;
  try {
    const packed = JSON.parse(
      execFileSync("npm", ["pack", "--json"], {
        cwd: packageDirectory,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "inherit"],
      }),
    );
    tarball = join(packageDirectory, packed[0].filename);
    execFileSync(
      "npm",
      ["install", "--ignore-scripts", "--no-audit", "--no-fund", tarball],
      {
        cwd: consumerDirectory,
        stdio: "inherit",
      },
    );
    const output = execFileSync(
      "node",
      [
        "--input-type=module",
        "--eval",
        `import { buildGanttFigure } from "@scope-profiler/plotly";
         const figure = buildGanttFigure({ intervals: [] });
         console.log(figure.layout.barmode);`,
      ],
      { cwd: consumerDirectory, encoding: "utf8" },
    );
    assert.equal(output.trim(), "overlay");
  } finally {
    if (tarball) rmSync(tarball, { force: true });
    rmSync(consumerDirectory, { recursive: true, force: true });
  }
});
