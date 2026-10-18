"""Astro/Starlight content for a catalogue of profiling runs.

``scope-profiler site MANIFEST -o DOCS`` reads a YAML manifest that lists cases
(each a set of profiling files) and writes the pages and chart data of a
catalogue into an Astro project, where the site's own build turns them into a
website. Nothing here renders HTML: a page is an MDX file of components plus a
JSON data file holding what it shows (the models of
:mod:`scope_profiler.report_model`, the same ones the HTML report is drawn
from), and charts are the plot-data JSON the HTML report embeds, drawn in the
browser by ``@scope-profiler/plotly``.

The manifest::

    title: Profiling catalogue
    description: Runs of my code.
    base: /my-catalogue          # the site's base path: "/" plus the repo name
    defaults: {scaling: strong, speedup_x: auto}
    cases:
      - id: poisson              # default: the slug of the title
        title: Poisson solver
        description: Strong scaling of the solver.
        group: Solvers           # optional: nests the case in a sidebar group
        tags: [cpu, scaling]
        scaling: strong          # strong, weak, both or none
        metadata: minimal        # or "all": publish every metadata field
        info: {commit: 843a82cd}
        runs:
          - {path: data/np1.h5, label: 1 rank}
          - {path: "data/np*.h5"}   # globs, relative to the manifest
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scope_profiler.report_data import (
    SPEEDUP_X_CHOICES,
    SpeedupAxisError,
    _check_speedup_x,
    collect_chart_payloads,
)

#: Written into the directories this module generates, so a later run knows
#: they are its own and may be replaced.
MARKER = ".scope-profiler-site"

#: Where the generated pages and chart data go, below the Astro project.
CONTENT_DIR = Path("src/content/docs/catalogue")
PUBLIC_DIR = Path("public/catalogue")
#: The components the generated pages are made of.
COMPONENTS = Path("src/components/report")

SCALING_CHOICES = ("strong", "weak", "both", "none")

#: Metadata shown on a run's page. A profile can carry much more (host name,
#: user, environment); a published catalogue only shows what identifies the run.
METADATA_KEYS = (
    "timestamp",
    "platform",
    "chip_information",
    "python_version",
    "scope_profiler_version",
    "omp_num_threads",
    "mpi_size",
    "total_cores",
    "num_nodes",
)

#: A case page's comparison of runs is computed in the browser; the report's own is shipped for this
#: many runs after the first, as the reference that computation is checked against.
REFERENCE_RUNS = 4

#: A chart whose data is larger than this is left out, with a note.
MAX_CHART_BYTES = 8_000_000
#: Rows of the hotspot tables.
HOTSPOTS = 15

_TOP_KEYS = {"title", "description", "base", "defaults", "cases"}
_CASE_KEYS = {
    "id",
    "title",
    "description",
    "group",
    "tags",
    "scaling",
    "speedup_x",
    "info",
    "runs",
    "include",
    "exclude",
    "ranks",
    "charts",
    "metadata",
}
_DEFAULT_KEYS = {
    "scaling",
    "speedup_x",
    "include",
    "exclude",
    "ranks",
    "charts",
    "metadata",
}
_RUN_KEYS = {"path", "label"}


class SiteError(ValueError):
    """The manifest is invalid, or the site cannot be written."""


@dataclass(frozen=True)
class RunSpec:
    """One profiling file of a case."""

    path: Path
    label: str | None = None


@dataclass(frozen=True)
class Case:
    """A set of runs shown together."""

    id: str
    title: str
    runs: tuple[RunSpec, ...]
    description: str = ""
    group: str | None = None
    tags: tuple[str, ...] = ()
    scaling: str = "strong"
    speedup_x: str = "auto"
    info: dict = field(default_factory=dict)
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    ranks: tuple[int, ...] | None = None
    charts: bool = True
    metadata: str = "minimal"


@dataclass(frozen=True)
class Manifest:
    """The parsed catalogue manifest."""

    title: str
    cases: tuple[Case, ...]
    description: str = ""
    base: str | None = None


@dataclass
class SiteResult:
    """What :func:`build_site` wrote."""

    files: list[Path]
    cases: int
    runs: int
    notes: list[str] = field(default_factory=list)


def slugify(text: str) -> str:
    """Lower-case ``text`` into a URL- and file-name-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _natural_key(path: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def _check_keys(mapping: dict, allowed: set, where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SiteError(
            f"{where}: unknown key(s) {', '.join(map(repr, unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}",
        )


def _mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise SiteError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _text(value, where: str) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise SiteError(f"{where} must be a string, got {value!r}")
    return str(value)


def _patterns(value, where: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SiteError(f"{where} must be a pattern or a list of patterns")
    for pattern in value:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise SiteError(f"{where}: {pattern!r} is not a regular expression: {exc}")
    return tuple(value)


def _rank_list(value, where: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list) or not all(
        isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0
        for rank in value
    ):
        raise SiteError(f"{where} must be a rank or a list of ranks (integers >= 0)")
    return tuple(value)


def _expand_runs(runs, where: str, root: Path) -> tuple[RunSpec, ...]:
    if not isinstance(runs, list) or not runs:
        raise SiteError(f"{where}.runs must be a non-empty list")
    specs: list[RunSpec] = []
    for index, entry in enumerate(runs):
        here = f"{where}.runs[{index}]"
        if isinstance(entry, str):
            entry = {"path": entry}
        entry = _mapping(entry, here)
        _check_keys(entry, _RUN_KEYS, here)
        if "path" not in entry:
            raise SiteError(f"{here}: 'path' is required")
        pattern = _text(entry["path"], f"{here}.path")
        label = entry.get("label")
        label = None if label is None else _text(label, f"{here}.label")
        full = pattern if os.path.isabs(pattern) else str(root / pattern)
        if glob.has_magic(full):
            matches = sorted(
                (m for m in glob.glob(full, recursive=True) if os.path.isfile(m)),
                key=_natural_key,
            )
            if not matches:
                raise SiteError(f"{here}: no file matches {pattern!r}")
            if label is not None and len(matches) > 1:
                raise SiteError(
                    f"{here}: a label cannot be given to a pattern that matches "
                    f"{len(matches)} files",
                )
            specs.extend(RunSpec(Path(match), label) for match in matches)
        else:
            if not os.path.isfile(full):
                raise SiteError(f"{here}: {pattern!r} is not a file")
            specs.append(RunSpec(Path(full), label))
    return tuple(specs)


def _base_path(value, where: str) -> str:
    """A site base path: ``""`` for a site at the root, else ``/name`` or ``/a/b``."""
    base = _text(value, where).strip().rstrip("/")
    if base and not base.startswith("/"):
        raise SiteError(f"{where} must start with '/', got {value!r}")
    return base


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate a catalogue manifest.

    Run paths are resolved against the manifest's directory and globs are
    expanded (in natural order, so ``np2`` precedes ``np10``). Every error names
    the case and field it comes from.
    """
    path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise SiteError(
            "reading a manifest needs PyYAML: pip install 'scope-profiler[site]'",
        ) from exc
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SiteError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SiteError(f"{path} is not valid YAML: {exc}") from exc
    document = _mapping(document if document is not None else {}, str(path))
    _check_keys(document, _TOP_KEYS, "manifest")

    defaults = _mapping(document.get("defaults") or {}, "defaults")
    _check_keys(defaults, _DEFAULT_KEYS, "defaults")

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SiteError("manifest: 'cases' must be a non-empty list")

    root = path.resolve().parent
    cases: list[Case] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        raw = _mapping(raw, f"cases[{index}]")
        _check_keys(raw, _CASE_KEYS, f"cases[{index}]")
        if "title" not in raw and "id" not in raw:
            raise SiteError(f"cases[{index}]: 'title' or 'id' is required")
        title = _text(raw.get("title", raw.get("id")), f"cases[{index}].title")
        case_id = slugify(_text(raw.get("id", title), f"cases[{index}].id"))
        if not case_id:
            raise SiteError(f"cases[{index}]: cannot make an id from {title!r}")
        where = f"case {case_id!r}"
        if case_id in seen:
            raise SiteError(f"{where}: the id is used by another case")
        seen.add(case_id)

        merged = {**defaults, **{k: v for k, v in raw.items() if k in _DEFAULT_KEYS}}
        scaling = _text(merged.get("scaling", "strong"), f"{where}.scaling")
        if scaling not in SCALING_CHOICES:
            raise SiteError(
                f"{where}.scaling must be one of {', '.join(SCALING_CHOICES)}, "
                f"got {scaling!r}",
            )
        speedup_x = _text(merged.get("speedup_x", "auto"), f"{where}.speedup_x")
        if speedup_x not in SPEEDUP_X_CHOICES:
            raise SiteError(
                f"{where}.speedup_x must be one of {', '.join(SPEEDUP_X_CHOICES)}, "
                f"got {speedup_x!r}",
            )
        charts = merged.get("charts", True)
        if not isinstance(charts, bool):
            raise SiteError(f"{where}.charts must be true or false")
        metadata = _text(merged.get("metadata", "minimal"), f"{where}.metadata")
        if metadata not in ("minimal", "all"):
            raise SiteError(
                f"{where}.metadata must be 'minimal' or 'all', got {metadata!r}"
            )
        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            raise SiteError(f"{where}.tags must be a list")
        info = _mapping(raw.get("info") or {}, f"{where}.info")
        group = raw.get("group")
        if group is not None:
            group = _text(group, f"{where}.group")
            if not slugify(group):
                raise SiteError(f"{where}.group: cannot make a name from {group!r}")
        cases.append(
            Case(
                id=case_id,
                title=title,
                runs=_expand_runs(raw.get("runs"), where, root),
                description=_text(raw.get("description", ""), f"{where}.description"),
                group=group,
                tags=tuple(_text(tag, f"{where}.tags") for tag in tags),
                scaling=scaling,
                speedup_x=speedup_x,
                info={str(key): value for key, value in info.items()},
                include=_patterns(merged.get("include"), f"{where}.include"),
                exclude=_patterns(merged.get("exclude"), f"{where}.exclude"),
                ranks=_rank_list(merged.get("ranks"), f"{where}.ranks"),
                charts=charts,
                metadata=metadata,
            ),
        )
    return Manifest(
        title=_text(document.get("title", "Profiling catalogue"), "title"),
        description=_text(document.get("description", ""), "description"),
        cases=tuple(cases),
        base=(
            _base_path(document["base"], "base")
            if document.get("base") is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# MDX


def _mdx(value) -> str:
    """Escape ``value`` for MDX prose or a table cell."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return re.sub(r"([\\`*_{}\[\]<>|$&#])", r"\\\1", text)


def _code(value) -> str:
    """A region name or similar as inline code, which MDX leaves alone."""
    return (
        "`" + re.sub(r"\s+", " ", str(value)).replace("`", "'").replace("|", "¦") + "`"
    )


def _yaml_string(value) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _seconds(value) -> str:
    if value is None:
        return "-"
    return f"{value:.4g} s"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _frontmatter(title, description="", label=None, order=None, hidden=False) -> str:
    lines = ["---", f"title: {_yaml_string(title)}"]
    if description:
        lines.append(f"description: {_yaml_string(description)}")
    if label is not None or order is not None or hidden:
        lines.append("sidebar:")
        if label is not None:
            lines.append(f"  label: {_yaml_string(label)}")
        if order is not None:
            lines.append(f"  order: {order}")
        if hidden:
            lines.append("  hidden: true")
    lines.append("---")
    return "\n".join(lines)


def _value_text(value) -> str:
    from scope_profiler.inspection import _json_safe

    value = _json_safe(value)
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return _mdx(value)


@dataclass
class _LoadedRun:
    slug: str
    label: str
    results: object
    path: Path


def _load_runs(case: Case) -> list[_LoadedRun]:
    from scope_profiler.profile_io import read_profile

    loaded: list[_LoadedRun] = []
    for spec in case.runs:
        try:
            results = read_profile(spec.path)
        except Exception as exc:  # unreadable or foreign files: say which
            raise SiteError(
                f"case {case.id!r}: cannot read {spec.path}: {exc}"
            ) from exc
        if spec.label:
            results.label = spec.label
        label = results.display_label
        slug = slugify(label) or "run"
        if any(run.slug == slug for run in loaded):
            raise SiteError(
                f"case {case.id!r}: two runs are called {label!r}; give them "
                "different labels",
            )
        loaded.append(_LoadedRun(slug, label, results, spec.path))
    return loaded


def _finite(value, drop_rows: bool = False):
    """``value`` without non-finite numbers, which are not JSON.

    A chart over runs lists a region for every run, with ``NaN`` where a run
    did not record it; the HTML report embeds that as JavaScript, where it is
    legal, but a ``.json`` file a browser fetches must not hold it. With
    ``drop_rows`` (chart data), rows that miss a measurement are left out, which
    is what the chart builders draw a missing measurement as anyway; any other
    non-finite number becomes ``null``.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [
            _finite(item, drop_rows)
            for item in value
            if not (drop_rows and isinstance(item, dict) and _has_non_finite(item))
        ]
    if isinstance(value, dict):
        return {key: _finite(item, drop_rows) for key, item in value.items()}
    return value


def _has_non_finite(row: dict) -> bool:
    return any(isinstance(v, float) and not math.isfinite(v) for v in row.values())


def _dump_json(value, *, drop_rows: bool = False) -> str:
    """``value`` as compact, strictly valid JSON (no ``NaN``), numpy-aware."""

    def default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"{type(item).__name__} is not JSON serializable")

    return (
        json.dumps(
            _finite(value, drop_rows=drop_rows),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=default,
        )
        + "\n"
    )


def _write_panels(panels, directory: Path, url_base: str, files, notes) -> list[dict]:
    """Write each chart panel's data next to the pages, and describe the panels.

    A panel with several x-axes has a file per axis. The records are what the
    ``Charts`` component draws: the panel's text and options and the URLs of its
    data, from the site root (the component adds the base path).
    """
    records = []
    for index, panel in enumerate(panels, start=1):
        stem = f"{index:02d}-{slugify(panel['title']) or 'chart'}"
        variants = panel["variants"]
        payloads = [v["payload"] for v in variants] if variants else [panel["payload"]]
        texts = [_dump_json(payload, drop_rows=True) for payload in payloads]
        biggest = max(len(text.encode("utf-8")) for text in texts)
        if biggest > MAX_CHART_BYTES:
            notes.append(
                f"{panel['title']}: left out, its data is {biggest / 1e6:.1f} MB"
            )
            continue
        urls = []
        for position, text in enumerate(texts):
            name = (
                f"{stem}.plot.json" if not variants else f"{stem}.{position}.plot.json"
            )
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
            files.append(target)
            urls.append(f"{url_base}/{name}")
        default = panel["default_variant"] or 0
        records.append(
            {
                "id": panel["id"],
                "title": panel["title"],
                "src": urls[default],
                "options": panel["options"],
                "description": panel["description"],
                "axis": panel["axis"],
                "open": panel["open"],
                "duration": panel["duration"],
                "default_variant": panel["default_variant"],
                "variants": (
                    [
                        {"label": variant["label"], "src": url}
                        for variant, url in zip(variants, urls)
                    ]
                    if variants
                    else None
                ),
            }
        )
    return records


def _chart_data(
    case, results, *, comparison, scaling_field, directory, url_base, files
):
    """The charts of a page: panels, why some are missing, and the filter data."""
    from scope_profiler.report_data import _TIMELINE_REGIONS, _region_index
    from scope_profiler.report_model import chart_panels

    try:
        collected, failures = collect_chart_payloads(
            results,
            list(case.include) if case.include else None,
            list(case.exclude) if case.exclude else None,
            list(case.ranks) if case.ranks else None,
            comparison=comparison,
            scaling_field=scaling_field,
            scaling=case.scaling if case.scaling != "none" else "both",
        )
    except ImportError as exc:
        raise SiteError(
            "charts need scope-profiler[pproc] (pip install 'scope-profiler[pproc]'), "
            f"or pass --no-charts: {exc}",
        ) from exc
    notes = list(failures)
    panels = _write_panels(
        chart_panels(
            collected, comparison, case.scaling if case.scaling != "none" else "both"
        ),
        directory,
        url_base,
        files,
        notes,
    )
    region_index = _region_index(
        results,
        list(case.include) if case.include else None,
        list(case.exclude) if case.exclude else None,
        list(case.ranks) if case.ranks else None,
    )
    depths = [depth for _, depth in region_index.values() if depth is not None]
    return {
        "panels": panels,
        "notes": notes,
        "region_index": region_index,
        "limits": {
            "regions": len(region_index),
            "max_depth": max(depths, default=0),
            "top": _TIMELINE_REGIONS if len(region_index) > _TIMELINE_REGIONS else None,
        },
    }


def _import_lines(page_dir: Path, components: Path, names) -> str:
    """MDX imports of the report components, relative to the page."""
    relative = os.path.relpath(components, page_dir).replace(os.sep, "/")
    if not relative.startswith("."):
        relative = "./" + relative
    return "\n".join(
        f'import {name} from {_yaml_string(f"{relative}/{name}.astro")};'
        for name in names
    )


def _section(heading: str, component: str, level: int = 2) -> str:
    return f"{'#' * level} {heading}\n\n<{component} data={{data}} />"


def _run_mdx(
    case, run, neighbours, page_dir, components, case_link, report, charts
) -> str:
    """A run's page: the report's sections, each drawn by a component."""
    previous, following = neighbours
    trail = [f"[← {_mdx(case.title)}]({case_link})"]
    if previous:
        trail.append(f"[‹ {_mdx(previous.label)}]({case_link}{previous.slug}/)")
    if following:
        trail.append(f"[{_mdx(following.label)} ›]({case_link}{following.slug}/)")
    body = [
        "<ReportFilter data={data} />",
        "<RunMeta data={data} />",
        " · ".join(trail),
    ]
    names = ["ReportFilter", "RunMeta"]

    def add(heading, component, level=2):
        names.append(component)
        body.append(_section(heading, component, level))

    if not report["summary"]["empty"]:
        add("Summary", "Summary")
    if report["hotspots"]["items"]:
        add("Hotspots", "Hotspots")
    add("Region statistics", "RegionTable")
    if report["balance"]:
        add("Load balance", "LoadBalance")
        add("Rank comparison", "RankMatrix", 3)
    if report["line_profiles"]:
        add("Line profile", "LineProfiles")
    if report["hardware"]:
        add("Hardware counters", "Hardware")
    if charts is not None and (charts["panels"] or charts["notes"]):
        add("Charts", "Charts")
    if report["metadata"]:
        add("Metadata", "Metadata")
    return (
        # The runs of a study can be hundreds: the sidebar lists the case, and the case
        # page lists its runs.
        _frontmatter(f"{case.title}: {run.label}", label=run.label, hidden=True)
        + "\n\n"
        + f'import data from "./{run.slug}.data.json";\n'
        + _import_lines(page_dir, components, names)
        + "\n\n"
        + "\n\n".join(body)
        + "\n"
    )


def _case_mdx(
    case, runs, order, page_dir, components, report, charts, case_link, data_name
) -> str:
    """A case's page: what it is, its runs, what changed and the comparison charts."""
    names: list[str] = []
    body = []
    if case.description:
        body.append(_mdx(case.description))
    if case.tags:
        body.append("Tags: " + ", ".join(_code(tag) for tag in case.tags))
    if case.info:
        body.append("## Information")
        body.append(
            _table(
                ["Field", "Value"],
                [[_mdx(key), _value_text(value)] for key, value in case.info.items()],
            ),
        )
    if report is None:
        body.append("## Runs")
        body.append(
            "\n".join(f"- [{_mdx(run.label)}]({case_link}{run.slug}/)" for run in runs)
        )
        data_import = ""
    else:
        names.append("ReportFilter")
        body.insert(0, "<ReportFilter data={data} />")

        def add(heading, component):
            names.append(component)
            body.append(_section(heading, component))

        add("Runs", "RunsTable")
        add("What changed", "CompareControls")
        add("Durations", "CompareTable")
        if charts is not None and (charts["panels"] or charts["notes"]):
            add("Charts", "Charts")
        data_import = f'import data from "./{data_name}";\n'
    return (
        _frontmatter(case.title, case.description, label=case.title, order=order)
        + "\n\n"
        + data_import
        + _import_lines(page_dir, components, names)
        + "\n\n"
        + "\n\n".join(body)
        + "\n"
    )


def _index_page(manifest: Manifest, summaries: list[dict], catalogue_link: str) -> str:
    rows = []
    for summary in summaries:
        case = summary["case"]
        group = f"{slugify(case.group)}/" if case.group else ""
        ranks = sorted({run["ranks"] for run in summary["runs"]})
        rank_text = str(ranks[0]) if len(ranks) == 1 else f"{ranks[0]}-{ranks[-1]}"
        rows.append(
            [
                f"[{_mdx(case.title)}]({catalogue_link}{group}{case.id}/)",
                _mdx(case.group or ""),
                ", ".join(_code(tag) for tag in case.tags),
                str(len(case.runs)),
                rank_text,
            ],
        )
    parts = [
        _frontmatter(manifest.title, manifest.description, label="Overview", order=0),
        _mdx(manifest.description) if manifest.description else "",
        _table(["Case", "Group", "Tags", "Runs", "Ranks"], rows),
    ]
    return "\n\n".join(part for part in parts if part) + "\n"


def _prepare(directory: Path, clean: bool) -> None:
    marker = directory / MARKER
    if directory.exists() and clean:
        if marker.exists():
            shutil.rmtree(directory)
        elif any(directory.iterdir()):
            raise SiteError(
                f"{directory} exists and was not written by scope-profiler site "
                f"(no {MARKER} file); remove it, or pass --no-clean to write into it",
            )
    directory.mkdir(parents=True, exist_ok=True)
    marker.write_text("generated by scope-profiler site\n", encoding="utf-8")


def _write(path: Path, text: str, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    files.append(path)


def _write(path: Path, text: str, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    files.append(path)


def build_site(
    manifest: Manifest,
    output: str | Path = "docs",
    *,
    base: str | None = None,
    charts: bool = True,
    clean: bool = True,
) -> SiteResult:
    """Write the catalogue's pages and chart data into the Astro project ``output``.

    Pages go to ``src/content/docs/catalogue`` (an MDX file and a JSON data file
    per page) and chart data to ``public/catalogue``; both are replaced on every
    run, but only when a previous run wrote them (``clean=False`` writes into
    them as they are).

    ``base`` is the site's base path (``/repo`` on GitHub Pages), which the
    links between pages need; it defaults to the manifest's ``base``. Raises
    :class:`SiteError` without one, for runs that cannot be read, and for
    ``charts=True`` without ``scope-profiler[pproc]``.
    """
    from scope_profiler.report_model import comparison_page_model, run_page_model

    site_base = _base_path(base, "base") if base is not None else manifest.base
    if site_base is None:
        raise SiteError(
            "the site's base path is needed for the links between pages: set "
            '\'base\' in the manifest ("/" plus the repository name, or "" for a '
            "site at the root of its domain), or pass --base"
        )
    catalogue_link = f"{site_base}/{PUBLIC_DIR.name}/"
    output = Path(output)
    content = output / CONTENT_DIR
    public = output / PUBLIC_DIR
    components = output / COMPONENTS
    _prepare(content, clean)
    _prepare(public, clean)

    files: list[Path] = []
    notes: list[str] = []
    summaries: list[dict] = []
    run_count = 0

    for case_order, case in enumerate(manifest.cases, start=1):
        runs = _load_runs(case)
        run_count += len(runs)
        group = slugify(case.group) if case.group else None
        relative = Path(group, case.id) if group else Path(case.id)
        case_dir = content / relative
        case_url = f"/{PUBLIC_DIR.name}/{relative.as_posix()}"
        case_public = public / relative
        case_link = f"{catalogue_link}{relative.as_posix()}/"
        include = list(case.include) if case.include else None
        exclude = list(case.exclude) if case.exclude else None
        ranks = list(case.ranks) if case.ranks else None
        # A published page shows what identifies a run, not the host or user.
        private = case.metadata != "all"
        host_keys = ("chip_information",) if private else None
        metadata_keys = METADATA_KEYS if private else None
        with_charts = charts and case.charts

        # The case is one page beside its runs' folder, so the sidebar lists it once.
        case_page = case_dir.parent / f"{case.id}.mdx"
        data_name = f"{case.id}.data.json"
        report = chart_data = None
        if len(runs) > 1:
            report = comparison_page_model(
                [run.results for run in runs],
                include,
                exclude,
                ranks,
                reference=REFERENCE_RUNS,
            )
            if private:
                for row in report["runs"]["rows"]:
                    row["host"] = None
                    if row["file"]:
                        row["file"] = {"name": row["file"]["name"], "path": None}
            if with_charts and case.scaling != "none":
                try:
                    field_name = _check_speedup_x(
                        [run.results for run in runs], case.speedup_x
                    )
                except SpeedupAxisError as exc:
                    raise SiteError(f"case {case.id!r}: {exc}") from exc
                chart_data = _chart_data(
                    case,
                    [run.results for run in runs],
                    comparison=True,
                    scaling_field=field_name,
                    directory=case_public,
                    url_base=case_url,
                    files=files,
                )
            elif with_charts:
                chart_data = _chart_data(
                    case,
                    [run.results for run in runs],
                    comparison=True,
                    scaling_field=None,
                    directory=case_public,
                    url_base=case_url,
                    files=files,
                )
            _write(
                case_page.with_name(data_name),
                _dump_json(
                    {
                        "report": report,
                        "charts": chart_data,
                        "links": {"runs": [f"{case_link}{run.slug}/" for run in runs]},
                    }
                ),
                files,
            )
            if chart_data:
                notes.extend(f"{case.id}: {note}" for note in chart_data["notes"])
        _write(
            case_page,
            _case_mdx(
                case,
                runs,
                case_order,
                case_page.parent,
                components,
                report,
                chart_data,
                case_link,
                data_name,
            ),
            files,
        )

        for position, run in enumerate(runs, start=1):
            run_report = run_page_model(
                run.results,
                include,
                exclude,
                ranks,
                host_keys=host_keys,
                metadata_keys=metadata_keys,
            )
            if private and run_report["meta"]["file"]:
                run_report["meta"]["file"]["path"] = None
            run_charts = None
            if with_charts:
                run_charts = _chart_data(
                    case,
                    [run.results],
                    comparison=False,
                    scaling_field=None,
                    directory=case_public / run.slug,
                    url_base=f"{case_url}/{run.slug}",
                    files=files,
                )
                notes.extend(
                    f"{case.id}/{run.slug}: {note}" for note in run_charts["notes"]
                )
            _write(
                case_dir / f"{run.slug}.data.json",
                _dump_json({"report": run_report, "charts": run_charts}),
                files,
            )
            _write(
                case_dir / f"{run.slug}.mdx",
                _run_mdx(
                    case,
                    run,
                    (
                        runs[position - 2] if position > 1 else None,
                        runs[position] if position < len(runs) else None,
                    ),
                    case_dir,
                    components,
                    case_link,
                    run_report,
                    run_charts,
                ),
                files,
            )
        summaries.append(
            {
                "case": case,
                "runs": [
                    {
                        "label": run.label,
                        "slug": run.slug,
                        "ranks": run.results.num_ranks,
                        "nodes": run.results.num_nodes,
                        "wall_time": run.results.time_span,
                    }
                    for run in runs
                ],
            },
        )

    _write(
        content / "index.mdx", _index_page(manifest, summaries, catalogue_link), files
    )
    index = {
        "title": manifest.title,
        "description": manifest.description,
        "cases": [
            {
                "id": summary["case"].id,
                "title": summary["case"].title,
                "group": summary["case"].group,
                "tags": list(summary["case"].tags),
                "info": summary["case"].info,
                "runs": summary["runs"],
            }
            for summary in summaries
        ],
    }
    _write(
        public / "index.json",
        json.dumps(index, ensure_ascii=False, indent=2, default=str) + "\n",
        files,
    )
    return SiteResult(
        files=files, cases=len(manifest.cases), runs=run_count, notes=notes
    )
