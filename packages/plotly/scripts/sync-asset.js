// The Python HTML report embeds this standalone ESM source verbatim.
import { copyFileSync } from "node:fs";
copyFileSync(new URL("../src/index.js", import.meta.url), new URL("../../../src/scope_profiler/_assets/scope-profiler-plotly-0.2.0.js", import.meta.url));
