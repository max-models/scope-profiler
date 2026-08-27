"""Self-contained HTML reports for profiling results.

The report deliberately has no plotting dependency: it is useful on a remote
machine immediately after a run, and can be opened locally in any browser.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

from scope_profiler.h5reader import read_h5
from scope_profiler.inspection import _json_safe
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import normalize_region_table_columns, region_rows

_STYLE = """
body { color: #1f2937; font: 15px/1.45 system-ui, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; }
h1, h2 { color: #111827; } section { margin: 2rem 0; }
.chart { min-height: 360px; margin: 1rem 0 2rem; }
.facts { display: flex; flex-wrap: wrap; gap: .75rem; }
.fact { background: #f3f4f6; border-radius: .4rem; padding: .5rem .75rem; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; }
th, td { border-bottom: 1px solid #d1d5db; padding: .45rem .6rem; text-align: right; }
th { background: #f9fafb; position: sticky; top: 0; } th:first-child, td:first-child { text-align: left; }
details { margin: .75rem 0; } summary { cursor: pointer; font-weight: 600; }
.muted { color: #6b7280; } code { overflow-wrap: anywhere; }
"""


def _text(value) -> str:
    """Convert a possibly numpy-backed value to safe HTML text."""
    value = _json_safe(value)
    if isinstance(value, (list, dict)):
        import json

        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def _seconds(value) -> str:
    return "-" if value is None else f"{value:.6g} s"


def _metadata_table(metadata: dict) -> str:
    if not metadata:
        return '<p class="muted">No metadata recorded.</p>'
    entries = "".join(
        f"<tr><th>{_text(key)}</th><td><code>{_text(value)}</code></td></tr>"
        for key, value in sorted(metadata.items())
    )
    return f"<table><tbody>{entries}</tbody></table>"


def _region_table(results, include, exclude, ranks, sort, columns) -> str:
    selected_columns = normalize_region_table_columns(columns)
    rows = region_rows(
        results, include=include, exclude=exclude, ranks=ranks, sort=sort
    )
    headers = "".join(f"<th>{_text(header)}</th>" for _, header in selected_columns)

    def cell(row, key) -> str:
        value = row["num_ranks"] if key == "ranks" else row[key]
        if key == "name":
            return _text(value)
        if isinstance(value, float):
            return _text(f"{value:.6g}")
        return _text(value)

    body = "".join(
        "<tr>"
        + "".join(f"<td>{cell(row, key)}</td>" for key, _ in selected_columns)
        + "</tr>"
        for row in rows
    )
    if not rows:
        body = (
            f'<tr><td colspan="{len(selected_columns)}">No regions recorded.</td></tr>'
        )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


def _chart_sections(runs, include, exclude, ranks) -> str:
    """Render Plotly charts inline, or explain why the optional extra is absent."""
    try:
        from scope_profiler.plotting_scripts import plot_durations, plot_gantt
    except ImportError:
        return (
            '<section><h2>Charts</h2><p class="muted">Charts require '
            "<code>scope-profiler[pproc]</code>; the statistics and metadata "
            "above remain available without it.</p></section>"
        )

    charts: list[tuple[str, object]] = []
    failures: list[str] = []
    for run in runs:
        try:
            charts.append(
                (
                    f"Timeline: {run.display_label}",
                    plot_gantt(
                        run,
                        include=include,
                        exclude=exclude,
                        ranks=ranks,
                        show=False,
                        verbose=False,
                        backend="plotly",
                        return_fig=True,
                    ),
                )
            )
        except (ImportError, ValueError) as exc:
            failures.append(f"Timeline for {run.display_label}: {exc}")

    try:
        charts.append(
            (
                "Region durations",
                plot_durations(
                    runs,
                    include=include,
                    exclude=exclude,
                    ranks=ranks,
                    show=False,
                    verbose=False,
                    backend="plotly",
                    return_fig=True,
                ),
            )
        )
    except (ImportError, ValueError) as exc:
        failures.append(f"Region durations: {exc}")

    fragments = []
    include_plotlyjs = True
    for title, figure in charts:
        if figure is None:
            continue
        fragments.append(
            f'<h3>{_text(title)}</h3><div class="chart">'
            f"{figure.to_html(full_html=False, include_plotlyjs=include_plotlyjs)}</div>"
        )
        include_plotlyjs = False
    if failures:
        fragments.append(
            '<p class="muted">Unavailable chart(s): '
            + _text("; ".join(failures))
            + "</p>"
        )
    if not fragments:
        fragments.append('<p class="muted">No charts could be rendered.</p>')
    return "<section><h2>Charts</h2>" + "".join(fragments) + "</section>"


def create_html_report(
    profiling_data: (
        ProfilingResults | str | Path | Sequence[ProfilingResults | str | Path]
    ),
    filepath: str | Path,
    *,
    include=None,
    exclude=None,
    ranks: list[int] | None = None,
    sort: str = "total",
    columns=None,
    include_charts: bool = True,
) -> Path:
    """Write a standalone HTML summary for one or more profiling results."""
    if isinstance(profiling_data, (ProfilingResults, str, Path)):
        profiling_data = [profiling_data]
    runs = [
        item if isinstance(item, ProfilingResults) else read_h5(item)
        for item in profiling_data
    ]
    if not runs:
        raise ValueError("At least one profiling result is required.")

    sections = []
    for results in runs:
        facts = [
            ("File", str(Path(results.file_path).resolve())),
            ("Ranks", results.num_ranks),
            ("Regions", len(results.get_regions(include=include, exclude=exclude))),
            ("Profiled window", _seconds(results.time_span)),
            ("Setup to finalize", _seconds(results.total_time)),
        ]
        facts_html = "".join(
            f'<div class="fact"><strong>{_text(name)}:</strong> {_text(value)}</div>'
            for name, value in facts
        )
        sections.append(
            f'<section><h2>{_text(results.display_label)}</h2><div class="facts">{facts_html}</div>'
            f"<h3>Region statistics</h3>{_region_table(results, include, exclude, ranks, sort, columns)}"
            f"<details><summary>Metadata</summary>{_metadata_table(results.metadata)}</details></section>"
        )

    charts = _chart_sections(runs, include, exclude, ranks) if include_charts else ""
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>scope-profiler report</title><style>"
        + _STYLE
        + "</style></head><body><h1>scope-profiler report</h1>"
        + "".join(sections)
        + charts
        + "</body></html>\n"
    )
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
