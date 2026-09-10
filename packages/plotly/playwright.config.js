import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./test/browser",
  workers: 1,
  projects: [{ name: "plotly3" }, { name: "plotly4" }],
  use: {
    baseURL: "http://127.0.0.1:4178",
    viewport: { width: 1100, height: 800 },
  },
  webServer: {
    command: "node test/browser/server.js",
    url: "http://127.0.0.1:4178/test/browser/index.html",
    reuseExistingServer: false,
  },
});
