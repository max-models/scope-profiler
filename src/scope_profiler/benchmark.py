"""Repeatable benchmark workflows for coding agents and CI.

The benchmark format is TOML so it works with Python 3.10+ without adding a
configuration dependency.  A benchmark run produces a JSON manifest containing
the individual profile paths, robust summary statistics, and correctness status.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from scope_profiler.h5reader import read_h5_summary


class BenchmarkError(Exception):
    """A user-facing benchmark configuration or execution error."""


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    script: str
    args: tuple[str, ...] = ()
    runs: int = 5
    warmups: int = 1
    timeout_seconds: float = 300.0
    only_user_code: bool = True
    output_dir: str = ".scope-profiler"
    correctness_command: tuple[str, ...] = ()
    correctness_timeout_seconds: float = 300.0
    threshold_pct: float = 2.0


def load_config(path: str | os.PathLike[str]) -> BenchmarkConfig:
    """Load and validate a benchmark TOML file."""
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except OSError as exc:
        raise BenchmarkError(
            f"Could not read benchmark config {path!r}: {exc}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise BenchmarkError(f"Invalid benchmark TOML {path!r}: {exc}") from exc

    bench = raw.get("benchmark", raw)
    script = bench.get("script")
    if not script:
        raise BenchmarkError("Benchmark config must define benchmark.script")
    script_path = (config_path.parent / script).resolve()
    if not script_path.is_file():
        raise BenchmarkError(f"Benchmark script does not exist: {script_path}")

    correctness = raw.get("correctness", {})
    command = correctness.get("command", bench.get("correctness_command", []))
    if isinstance(command, str):
        command = [command]
    config = BenchmarkConfig(
        name=str(bench.get("name", config_path.stem)),
        script=str(script_path),
        args=tuple(str(x) for x in bench.get("args", [])),
        runs=int(bench.get("runs", 5)),
        warmups=int(bench.get("warmups", 1)),
        timeout_seconds=float(bench.get("timeout_seconds", 300.0)),
        only_user_code=bool(bench.get("only_user_code", True)),
        output_dir=str(
            (config_path.parent / bench.get("output_dir", ".scope-profiler")).resolve()
        ),
        correctness_command=tuple(str(x) for x in command),
        correctness_timeout_seconds=float(correctness.get("timeout_seconds", 300.0)),
        threshold_pct=float(bench.get("threshold_pct", 2.0)),
    )
    if config.runs < 2:
        raise BenchmarkError("benchmark.runs must be at least 2")
    if config.warmups < 0 or config.timeout_seconds <= 0:
        raise BenchmarkError(
            "warmups must be non-negative and timeout_seconds positive"
        )
    if not 0 <= config.threshold_pct:
        raise BenchmarkError("benchmark.threshold_pct must be non-negative")
    return config


def _profile_once(config: BenchmarkConfig, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "scope_profiler", "run", "-q", "-o", str(path)]
    if not config.only_user_code:
        command.append("--all")
    command.extend([config.script, *config.args])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkError(
            f"Benchmark run exceeded {config.timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        raise BenchmarkError(
            f"Benchmark run failed with exit code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    if not path.exists():
        raise BenchmarkError(f"Benchmark run produced no profile: {path}")
    results = read_h5_summary(path, include_likwid=False, include_line_profile=False)
    total = results.total_time
    if total is None or not math.isfinite(total):
        raise BenchmarkError(f"Profile has no finite total time: {path}")
    regions = {
        region.name: float(region.total_duration) for region in results.get_regions()
    }
    return {"path": str(path), "total_time_seconds": float(total), "regions": regions}


def _run_correctness(config: BenchmarkConfig) -> dict:
    if not config.correctness_command:
        return {
            "configured": False,
            "passed": True,
            "command": [],
            "stdout_tail": "",
            "stderr_tail": "",
        }
    command = list(config.correctness_command)
    if command[0] == "python":
        command[0] = sys.executable
    try:
        completed = subprocess.run(
            command,
            cwd=Path(config.script).parent,
            capture_output=True,
            text=True,
            timeout=config.correctness_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "configured": True,
            "passed": False,
            "command": command,
            "error": "timeout",
        }
    return {
        "configured": True,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_benchmark(config: BenchmarkConfig, label: str = "candidate") -> dict:
    """Run warmups, correctness, and repeated measured profiles."""
    output_dir = Path(config.output_dir) / config.name / label
    for index in range(config.warmups):
        _profile_once(config, output_dir / f"warmup-{index}.h5")
    correctness = _run_correctness(config)
    profiles = [
        _profile_once(config, output_dir / f"run-{index}.h5")
        for index in range(config.runs)
    ]
    totals = [item["total_time_seconds"] for item in profiles]
    manifest = {
        "format": 1,
        "name": config.name,
        "label": label,
        "config": asdict(config),
        "correctness": correctness,
        "profiles": profiles,
        "total_time_seconds": _stats(totals),
    }
    manifest_path = output_dir / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def compare_benchmarks(baseline: dict | str, candidate: dict | str) -> dict:
    """Compare benchmark manifests using medians and report an agent decision."""

    def load(value):
        if isinstance(value, dict):
            return value
        return json.loads(Path(value).read_text(encoding="utf-8"))

    base, cand = load(baseline), load(candidate)
    base_median = base["total_time_seconds"]["median"]
    cand_median = cand["total_time_seconds"]["median"]
    change_pct = (
        ((cand_median - base_median) / base_median * 100) if base_median else None
    )
    threshold = float(cand.get("config", {}).get("threshold_pct", 2.0))
    faster = bool(cand_median < base_median and change_pct <= -threshold)
    correctness_passed = bool(cand.get("correctness", {}).get("passed", False))
    decision = "keep" if faster and correctness_passed else "reject"
    return {
        "baseline": {"label": base.get("label"), "stats": base["total_time_seconds"]},
        "candidate": {"label": cand.get("label"), "stats": cand["total_time_seconds"]},
        "relative_change_pct": change_pct,
        "speedup": base_median / cand_median if cand_median else None,
        "faster_beyond_threshold": faster,
        "correctness_passed": correctness_passed,
        "decision": decision,
        "threshold_pct": threshold,
    }
