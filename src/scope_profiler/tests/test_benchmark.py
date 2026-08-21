import json

import pytest

from scope_profiler.benchmark import (
    BenchmarkConfig,
    BenchmarkError,
    compare_benchmarks,
    load_config,
    run_benchmark,
)


def test_load_config_resolves_script_and_defaults(tmp_path):
    script = tmp_path / "bench.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    config = tmp_path / "bench.toml"
    config.write_text(
        '[benchmark]\nscript = "bench.py"\nruns = 2\n' "[correctness]\ncommand = []\n",
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded.script == str(script.resolve())
    assert loaded.runs == 2
    assert loaded.correctness_command == ()


def test_missing_script_is_rejected(tmp_path):
    config = tmp_path / "bench.toml"
    config.write_text('[benchmark]\nscript = "missing.py"\n', encoding="utf-8")

    with pytest.raises(BenchmarkError, match="does not exist"):
        load_config(config)


def test_run_benchmark_writes_repeated_manifest(tmp_path):
    script = tmp_path / "bench.py"
    script.write_text(
        "from scope_profiler import ProfileManager\n"
        "with ProfileManager.profile_region('work'):\n"
        "    pass\n",
        encoding="utf-8",
    )
    config = BenchmarkConfig(
        name="test",
        script=str(script),
        runs=2,
        warmups=0,
        timeout_seconds=30.0,
        output_dir=str(tmp_path / "out"),
    )

    manifest = run_benchmark(config, label="baseline")

    assert manifest["correctness"]["passed"] is True
    assert len(manifest["profiles"]) == 2
    manifest_path = tmp_path / "out" / "test" / "baseline" / "benchmark.json"
    assert json.loads(manifest_path.read_text())["format"] == 1


def test_compare_benchmarks_requires_speedup_and_correctness():
    baseline = {"label": "baseline", "total_time_seconds": {"median": 10.0}}
    candidate = {
        "label": "candidate",
        "total_time_seconds": {"median": 9.0},
        "correctness": {"passed": True},
        "config": {"threshold_pct": 2.0},
    }

    result = compare_benchmarks(baseline, candidate)

    assert result["speedup"] == pytest.approx(10 / 9)
    assert result["decision"] == "keep"
