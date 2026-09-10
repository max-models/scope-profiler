import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
const root = resolve(import.meta.dirname, "../..");
createServer(async (request, response) => {
  const path = resolve(
    root,
    "." + new URL(request.url, "http://localhost").pathname,
  );
  if (!path.startsWith(root + "/")) {
    response.writeHead(403).end();
    return;
  }
  try {
    const body = await readFile(
      path.endsWith("/") ? path + "index.html" : path,
    );
    response.setHeader(
      "Content-Type",
      {
        ".js": "text/javascript",
        ".html": "text/html",
        ".json": "application/json",
      }[extname(path)] ?? "application/octet-stream",
    );
    response.end(body);
  } catch {
    response.writeHead(404).end();
  }
}).listen(4178, "127.0.0.1");
