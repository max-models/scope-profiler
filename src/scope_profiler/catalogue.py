"""Catalogue page linking several generated HTML reports."""

from __future__ import annotations

import html
import os
import re
from collections.abc import Sequence
from pathlib import Path


def _report_title(path: Path) -> str:
    """Return the ``<title>`` of an HTML report, or its file stem."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:200_000]
    except OSError:
        return path.stem
    match = re.search(r"<title>(.*?)</title>", head, re.S | re.I)
    title = html.unescape(match.group(1)).strip() if match else ""
    return title or path.stem


def create_catalogue(
    reports: Sequence[str | Path],
    filepath: str | Path,
    *,
    title: str = "Profiling reports",
) -> Path:
    """Write an index page linking several HTML reports.

    Each entry shows the report's title and file name and links to it by a
    path relative to the catalogue, so the reports and the catalogue can be
    moved together.
    """
    if not reports:
        raise ValueError("At least one report is required.")
    output = Path(filepath)
    output.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for report in reports:
        path = Path(report)
        if not path.is_file():
            raise FileNotFoundError(f"Report not found: {path}")
        href = Path(
            os.path.relpath(path.resolve(), output.parent.resolve())
        ).as_posix()
        items.append(
            f'<li><a href="{html.escape(href, quote=True)}">'
            f"{html.escape(_report_title(path))}</a> "
            f"<small>{html.escape(href)}</small></li>"
        )
    output.write_text(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:48rem;margin:2rem auto;"
        "padding:0 1rem}li{margin:.4rem 0}small{color:#666;margin-left:.5rem}"
        f"</style></head><body><h1>{html.escape(title)}</h1><ul>"
        + "".join(items)
        + "</ul></body></html>\n",
        encoding="utf-8",
    )
    return output
