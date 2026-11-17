"""Tests of scope_profiler.jobs: cases, schedulers, cluster presets, TOML configs, sweeps and the results store."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scope_profiler.__main__ import main as cli_main
from scope_profiler.jobs import (
    ClusterPreset,
    GitResultsStore,
    ProfilingCase,
    detect_cluster,
    get_cluster,
    load_case_config,
    load_clusters,
    register_cluster,
    render_sbatch_script,
    run_sweep,
    variant_entries,
)
from scope_profiler.jobs import case as case_module
from scope_profiler.jobs import clusters as clusters_module
from scope_profiler.jobs.case import _render_command
from scope_profiler.jobs.schedulers import SlurmScheduler, detect_scheduler

# A stand-in for the profiled program: writes a "profile" and a log line, and
# fails if asked to.
APP = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path

    if "--fail" in sys.argv:
        sys.exit(3)
    run_dir = Path(os.environ["SCOPE_PROFILER_RUN_DIR"])
    Path(os.environ["SCOPE_PROFILER_PROFILE"]).write_bytes(b"not really hdf5")
    (run_dir / "run_metadata.json").write_text(json.dumps({"argv": sys.argv[1:], "mode": os.environ.get("MODE")}))
    print("ran with", sys.argv[1:])
    """,
)

PPROC = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    run_dir = Path(sys.argv[1])
    (run_dir / "results" / "figure.png").write_bytes(b"png")
    (run_dir / "results" / "ignored.bin").write_bytes(b"bin")
    (run_dir / "results" / "args.txt").write_text(" ".join(sys.argv[2:]))
    """,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Fresh registry and sweep state, a clean environment, and the tmp dir as cwd."""
    clusters_module._reset_registry()
    case_module._PREPARED_ROOTS.clear()
    case_module._SWEEP_CASES = None
    for name in (
        "SCOPE_PROFILER_CLUSTERS",
        "SCOPE_PROFILER_RESULTS_REPO",
        "SCOPE_PROFILER_JOBS_LOCAL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")
    monkeypatch.chdir(tmp_path)
    yield
    clusters_module._reset_registry()
    case_module._SWEEP_CASES = None


@pytest.fixture
def app(tmp_path):
    (tmp_path / "app.py").write_text(APP, encoding="utf-8")
    (tmp_path / "pproc.py").write_text(PPROC, encoding="utf-8")
    return tmp_path


def _case(tmp_path, **kwargs):
    defaults = {
        "label": "demo case",
        "command": [sys.executable, "{case_dir}/app.py"],
        "inputs": [tmp_path / "app.py"],
        "mpi": False,
        "scheduler": "local",
        "work_dir": tmp_path / "work",
        "record_pip_freeze": False,
    }
    return ProfilingCase(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Command rendering and batch scripts
# ---------------------------------------------------------------------------
def test_render_command_list_quotes_and_splices_args():
    values = {"run_dir": "/r d", "profile": "/r d/p.h5", "label": "x"}
    line = _render_command(
        ["app", "{run_dir}", "{args}", "--json={a: 1}"], values, ["--n", "a b"]
    )
    assert line == "app '/r d' --n 'a b' '--json={a: 1}'"
    assert _render_command(["app"], values, ["-v"]) == "app -v"
    assert _render_command(["app"], values, ["-v"], append_args=False) == "app"


def test_render_command_string_is_used_as_written():
    values = {"run_dir": "/r", "profile": "/r/p.h5"}
    assert (
        _render_command("app {profile} | tee log", values, ["-v"])
        == "app /r/p.h5 | tee log -v"
    )
    assert _render_command("app {args} | tee log", values, ["-v"]) == "app -v | tee log"
    assert (
        _render_command("app {unknown}", values, [], append_args=False)
        == "app {unknown}"
    )


def test_render_sbatch_script():
    text = render_sbatch_script(
        {
            "job_name": "x",
            "ntasks_per_node": 4,
            "exclusive": True,
            "account": None,
            "requeue": False,
        },
        ["echo hi"],
    )
    assert text.splitlines() == [
        "#!/bin/bash",
        "#SBATCH --job-name=x",
        "#SBATCH --ntasks-per-node=4",
        "#SBATCH --exclusive",
        "",
        "echo hi",
    ]


def test_detect_scheduler(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert detect_scheduler().name == "local"
    assert detect_scheduler("dry-run").name == "dry-run"
    assert detect_scheduler("slurm").name == "slurm"
    with pytest.raises(ValueError, match="scheduler must be"):
        detect_scheduler("pbs")


# ---------------------------------------------------------------------------
# Cluster presets
# ---------------------------------------------------------------------------
def test_cluster_presets_from_toml_and_detection(tmp_path, monkeypatch):
    path = tmp_path / "clusters.toml"
    path.write_text(
        textwrap.dedent(
            """
            [clusters.big]
            cpus_per_node = 64
            gpus_per_node = 4
            setup = ["module load mpi"]
            detect = { MY_CLUSTER = "big" }
            sbatch = { partition = "p", time = "00:05:00" }
            """,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCOPE_PROFILER_CLUSTERS", str(path))
    preset = get_cluster("big")
    assert preset.sbatch == {"partition": "p", "time": "00:05:00"}
    assert preset.nodes_for(1) == 1
    assert preset.nodes_for(64) == 1
    assert preset.nodes_for(65) == 2
    assert detect_cluster({"MY_CLUSTER": "small"}) is None
    assert detect_cluster({"MY_CLUSTER": "the-big-one"}).name == "big"
    with pytest.raises(KeyError, match="no cluster preset 'nope'"):
        get_cluster("nope")

    path.write_text("[clusters.bad]\nnodes = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_clusters(path)


def test_register_cluster_keywords():
    preset = register_cluster(name="laptop", cpus_per_node=8, detect={"HOSTNAME": ""})
    assert get_cluster("laptop") is preset
    assert ClusterPreset(name="none").nodes_for(1000) == 1
    with pytest.raises(TypeError):
        register_cluster(preset, name="other")


def test_cli_clusters(tmp_path, capsys):
    path = tmp_path / "c.toml"
    path.write_text(
        '[clusters.x]\ncpus_per_node = 2\nsbatch = { time = "1" }\n', encoding="utf-8"
    )
    assert cli_main(["jobs", "clusters", "--config", str(path)]) == 0
    assert "x" in capsys.readouterr().out
    assert cli_main(["jobs", "clusters", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["clusters"]["x"]["cpus_per_node"] == 2


# ---------------------------------------------------------------------------
# Local runs and packaging
# ---------------------------------------------------------------------------
def test_local_case_runs_and_packages(app):
    case = _case(
        app,
        postprocess=[sys.executable, "{case_dir}/pproc.py", "{run_dir}", "{args}"],
        inputs=[app / "app.py", app / "pproc.py"],
        env={"MODE": "base"},
        metadata={"code": "demo"},
    )
    first = case.launch(1, args=["--size", "8"], metadata={"variant": "a"})
    second = case.launch(1, env={"MODE": "other"})
    assert first["state"] == "COMPLETED"
    assert first["args"] == ["--size", "8"]
    case.finalize()

    package = case.destination_dir
    assert package.parent == app / "work" / "packaged"
    assert package.name.endswith("demo_case")
    metadata = json.loads((package / "case_metadata.json").read_text())
    assert metadata["general_information"]["label"] == "demo case"
    assert metadata["metadata"] == {"code": "demo"}
    assert metadata["scheduler"] == "local"
    assert metadata["inputs"] == ["inputs/app.py", "inputs/pproc.py"]
    assert (package / "inputs" / "app.py").is_file()
    assert [run["folder"] for run in metadata["runs"]] == ["run01", "run02"]

    run1 = json.loads((package / "run01" / "run_metadata.json").read_text())
    assert run1["packaged_files"]["profiles"] == ["run01/profiling_data.h5"]
    assert sorted(run1["packaged_files"]["results"]) == [
        "run01/args.txt",
        "run01/figure.png",
    ]
    assert run1["application_metadata"] == {"argv": ["--size", "8"], "mode": "base"}
    assert run1["metadata"] == {"variant": "a"}
    assert (package / "run01" / "args.txt").read_text() == "--size 8"
    assert not (package / "run01" / "ignored.bin").exists()
    run2 = json.loads((package / "run02" / "run_metadata.json").read_text())
    assert run2["application_metadata"]["mode"] == "other"
    assert second["job_script"].startswith("#!/bin/bash")


def test_failed_local_run_is_listed_as_failed(app, capsys):
    case = _case(app)
    info = case.launch(1, args=["--fail"])
    assert info["state"] == "FAILED"
    assert info["returncode"] == 3
    case.finalize()
    out = capsys.readouterr().out
    assert "listed as failed" in out
    (run,) = json.loads((case.destination_dir / "case_metadata.json").read_text())[
        "runs"
    ]
    assert run["failed"] is True
    assert run["state"] == "FAILED"
    assert run["profiles"] == []
    assert not (case.destination_dir / "run01").exists()


def test_local_run_skips_rank_counts_above_the_core_count(app, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    case = _case(app, mpi=True, launcher="echo")
    assert case.launch(4) is None
    assert case.job_infos == []


def test_mpi_launcher_prefix_and_missing_launcher(app):
    case = _case(app, mpi=True, launcher="mpirun --oversubscribe", scheduler="dry-run")
    info = case.launch(2)
    assert "mpirun --oversubscribe -n 2 " in info["job_script"]

    case = _case(app, mpi=True)
    case.launcher = None
    assert case.launch(1) is not None  # one rank runs without a launcher
    assert case.launch(2) is None  # skipped locally
    case.scheduler = "dry-run"
    with pytest.raises(RuntimeError, match="no MPI launcher"):
        case.launch(2)


def test_missing_input_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _case(tmp_path, inputs=[tmp_path / "missing.py"])


def test_dry_run_writes_sbatch_script_with_preset(app):
    register_cluster(
        name="c",
        cpus_per_node=4,
        setup=["module load x"],
        sbatch={"partition": "p", "time": "00:10:00"},
    )
    case = _case(
        app,
        scheduler="dry-run",
        cluster="c",
        mpi=True,
        launcher="srun",
        setup=["echo setup"],
    )
    info = case.launch(8, sbatch={"time": "00:20:00"})
    script = Path(info["job_script_path"]).read_text()
    assert "#SBATCH --partition=p" in script
    assert "#SBATCH --time=00:20:00" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks-per-node=4" in script
    assert "#SBATCH --job-name=demo_case_01" in script
    assert (
        script.index("module load x")
        < script.index("set -e")
        < script.index("echo setup")
    )
    assert "srun -n 8 " in script
    with pytest.raises(ValueError, match="not divisible"):
        case.launch(6, nodes=4)
    case.finalize()
    assert not any(case.destination_dir.glob("run*"))


# ---------------------------------------------------------------------------
# SLURM, with stand-ins for sbatch/squeue/sacct
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    bin_dir = tmp_path / "fake_slurm"
    state = bin_dir / "state"
    state.mkdir(parents=True)
    scripts = {
        "sbatch": f"""
            n=$(( $(cat {state}/counter 2>/dev/null || echo 100) + 1 ))
            echo $n > {state}/counter
            touch {state}/submitted_$n
            cp "${{@: -1}}" {state}/script_$n.sh
            ( bash "${{@: -1}}"; echo $? > {state}/done_$n ) > /dev/null 2>&1 &
            echo "$n;cluster"
        """,
        "squeue": f"""
            for f in {state}/submitted_*; do
                [ -e "$f" ] || continue
                n=${{f##*_}}
                [ -e {state}/done_$n ] || echo "$n RUNNING"
            done
        """,
        "sacct": f"""
            ids="${{@: -1}}"
            for n in ${{ids//,/ }}; do
                if [ "$(cat {state}/done_$n)" = "0" ]; then echo "$n|COMPLETED"; else echo "$n|FAILED"; fi
            done
        """,
    }
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return state


@pytest.mark.slow
def test_slurm_case_submits_polls_and_packages(app, fake_slurm):
    register_cluster(
        name="fake",
        cpus_per_node=2,
        sbatch={"partition": "debug"},
        detect={"FAKE_CLUSTER": "yes"},
    )
    os.environ["FAKE_CLUSTER"] = "yes"
    try:
        case = _case(app, scheduler=None)
    finally:
        del os.environ["FAKE_CLUSTER"]
    assert case.scheduler == "slurm"
    assert case.cluster.name == "fake"
    ok = case.launch(1)
    bad = case.launch(1, args=["--fail"])
    assert ok["job_id"] == "101"
    assert bad["job_id"] == "102"
    assert "#SBATCH --partition=debug" in (fake_slurm / "script_101.sh").read_text()

    case.finalize(poll_interval=0.05)
    assert ok["state"] == "COMPLETED"
    assert bad["state"] == "FAILED"
    runs = json.loads((case.destination_dir / "case_metadata.json").read_text())["runs"]
    assert [run["launch_id"] for run in runs] == [1, 2]
    assert [run.get("failed", False) for run in runs] == [False, True]


def test_slurm_job_states_without_sacct(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert SlurmScheduler.job_states(["1", "2"]) == {"1": None, "2": None}


# ---------------------------------------------------------------------------
# Results repository
# ---------------------------------------------------------------------------
@pytest.fixture
def results_repo(tmp_path):
    bare = tmp_path / "results.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(bare), str(seed)], check=True)
    (seed / "README.md").write_text("results\n")
    subprocess.run(["git", "-C", str(seed), "add", "."], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-q", "-m", "init"], check=True)
    subprocess.run(
        ["git", "-C", str(seed), "push", "-q", "origin", "HEAD:main"], check=True
    )
    return bare


def _repo_files(bare):
    out = subprocess.run(
        ["git", "--git-dir", str(bare), "ls-tree", "-r", "--name-only", "main"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.split()


def test_upload_pushes_each_run(app, results_repo, monkeypatch):
    monkeypatch.setenv("SCOPE_PROFILER_RESULTS_REPO", str(results_repo))
    case = _case(app, upload=True)
    case.launch(1)
    case.finalize()
    files = _repo_files(results_repo)
    name = case.destination_dir.name
    assert f"{name}/case_metadata.json" in files
    assert f"{name}/run01/profiling_data.h5" in files
    log = subprocess.run(
        ["git", "--git-dir", str(results_repo), "log", "--format=%s", "main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Add profiling data for demo case" in log


def test_upload_into_a_directory_of_the_repository(app, results_repo):
    store = GitResultsStore(url=str(results_repo), path="mycode/{label}")
    case = _case(app, upload=True, store=store)
    case.launch(1)
    case.finalize()
    name = case.destination_dir.name
    assert f"mycode/demo_case/{name}/run01/profiling_data.h5" in _repo_files(
        results_repo
    )


def test_cli_upload_into_a_directory(app, results_repo):
    case = _case(app)
    case.launch(1)
    case.finalize()
    argv = ["jobs", "upload", str(case.destination_dir), "--repo", str(results_repo)]
    assert cli_main([*argv, "--path", "mycode/{label}"]) == 0
    name = case.destination_dir.name
    assert f"mycode/demo_case/{name}/case_metadata.json" in _repo_files(results_repo)
    assert cli_main([*argv, "--path", "../outside"]) == 2


REAL_APP = textwrap.dedent(
    """
    import os
    from scope_profiler.tests.test_post_processing import _write_sample_h5

    ranks = int(os.environ["SCOPE_PROFILER_NTASKS"])
    data = {rank: {"solve": ([0], [1_000_000 // ranks])} for rank in range(ranks)}
    _write_sample_h5(os.environ["SCOPE_PROFILER_PROFILE"], data, {"mpi_size": ranks})
    """,
)


def test_a_catalogue_reads_the_uploaded_runs(app, results_repo, tmp_path):
    pytest.importorskip("yaml")
    pytest.importorskip("plotly")
    from scope_profiler.astro_site import build_site, load_manifest

    (app / "real.py").write_text(REAL_APP, encoding="utf-8")
    store = GitResultsStore(url=str(results_repo), path="mycode/{label}")
    case = _case(
        app,
        label="heat",
        command=[sys.executable, "{case_dir}/real.py"],
        inputs=[app / "real.py"],
        upload=True,
        store=store,
    )
    for ntasks in (1, 2):
        case.launch(ntasks)
    case.finalize()
    data = tmp_path / "data"
    subprocess.run(["git", "clone", "-q", str(results_repo), str(data)], check=True)
    (data / "profiling.yaml").write_text(
        "base: /mycode\ncases:\n  - title: Heat\n    executions:\n"
        '      path: "mycode/heat/{execution}/run*/profiling_data.h5"\n'
    )

    manifest = load_manifest(data / "profiling.yaml")
    from dataclasses import replace

    from scope_profiler.astro_site import _load_runs

    (execution,) = manifest.cases[0].executions
    runs = _load_runs(replace(manifest.cases[0], runs=execution.runs))
    assert [run.label for run in runs] == ["np1", "np2"]
    build_site(manifest, tmp_path / "docs", section="profiling")
    assert (tmp_path / "docs/src/content/docs/profiling/heat.mdx").is_file()


def test_placeholders_in_env_setup_and_sbatch(app):
    case = _case(
        app,
        scheduler="dry-run",
        env={"APP_LOG": "{run_dir}/app.log"},
        setup=["echo setting up {label} with {ntasks} ranks"],
    )
    info = case.launch(2, sbatch={"time": "00:{ntasks}0:00", "comment": "{label}"})
    script = info["job_script"]
    run_dir = info["run_directory"]
    assert f"export APP_LOG={run_dir}/app.log" in script
    assert "echo setting up demo case with 2 ranks" in script
    assert info["sbatch"]["time"] == "00:20:00"
    assert info["sbatch"]["comment"] == "demo case"


def test_prepare_runs_once_before_the_first_launch(app):
    case = _case(app, prepare="echo built {label} > built.txt; echo done")
    case.launch(1)
    case.launch(1)
    built = (case.case_dir / "built.txt").read_text().strip()
    assert built == "built demo case"
    assert "done" in (case.case_dir / "prepare.out").read_text()
    case.finalize()
    metadata = json.loads((case.destination_dir / "case_metadata.json").read_text())
    assert metadata["prepare"]["returncode"] == 0
    assert metadata["prepare"]["seconds"] >= 0


def test_a_failing_prepare_stops_the_case(app):
    case = _case(app, prepare=["bash", "-c", "echo broken build; exit 4"])
    with pytest.raises(RuntimeError, match=r"exit code 4\)[\s\S]*broken build"):
        case.launch(1)
    assert case.job_infos == []


def test_redacted_metadata_leaves_out_hosts_and_private_paths(app, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "secret-node7")
    case = _case(app, redact=True)
    case.launch(1)
    case.finalize()
    text = (case.destination_dir / "case_metadata.json").read_text()
    run_text = (case.destination_dir / "run01" / "run_metadata.json").read_text()
    for written in (text, run_text):
        assert "secret-node7" not in written
        assert str(case.work_dir) not in written
    assert "<work_dir>" in run_text
    assert json.loads(text)["hardware_information"]["submit_host"] == "<host>"


def test_a_public_store_redacts_by_default(app, results_repo):
    store = GitResultsStore(url=str(results_repo), public=True)
    assert _case(app, upload=True, store=store).redacting
    assert not _case(app, upload=True, store=store, redact=False).redacting
    assert not _case(app).redacting


def test_after_push_commands_run_after_a_push(app, results_repo, tmp_path):
    marker = tmp_path / "pushed.txt"
    store = GitResultsStore(
        url=str(results_repo),
        after_push=[f"echo {{label}} {{commit}} >> {marker}", "exit 3"],
    )
    case = _case(app, upload=True, store=store)
    case.launch(1)
    case.finalize()
    lines = marker.read_text().splitlines()
    assert lines and all(line.startswith("demo case") for line in lines)


def test_upload_requires_a_repository(app):
    with pytest.raises(RuntimeError, match="no results repository"):
        _case(app, upload=True)


def test_cli_upload_of_a_packaged_case(app, results_repo, capsys):
    case = _case(app)
    case.launch(1)
    case.finalize()
    assert (
        cli_main(
            ["jobs", "upload", str(case.destination_dir), "--repo", str(results_repo)]
        )
        == 0
    )
    assert f"{case.destination_dir.name}/run01/run_metadata.json" in _repo_files(
        results_repo
    )
    # Unchanged on a second upload: nothing to push.
    assert GitResultsStore(url=str(results_repo)).upload(case.destination_dir) is False
    assert cli_main(["jobs", "upload", str(app), "--repo", str(results_repo)]) == 2


# ---------------------------------------------------------------------------
# TOML case files and sweeps
# ---------------------------------------------------------------------------
CASE_TOML = """
[case]
label = "toml_case"
command = ["{python}", "{{case_dir}}/app.py"]
inputs = ["app.py"]
mpi = false
work_dir = "work"
record_pip_freeze = false
env = {{ MODE = "toml" }}

[[launch]]
ntasks = [1, 1]
args = ["--n", "4"]

[variants.a]
args = ["--a"]
[variants.b]
args = ["--b"]
env = {{ MODE = "b" }}
description = "variant b"
"""


def _write_case_toml(app):
    path = app / "case.toml"
    path.write_text(CASE_TOML.format(python=sys.executable), encoding="utf-8")
    return path


def test_case_toml_runs_every_variant(app):
    path = _write_case_toml(app)
    assert cli_main(["jobs", "run", str(path), "--scheduler", "local"]) == 0
    packages = sorted((app / "work" / "packaged").iterdir())
    assert [p.name.rsplit("-", 1)[-1] for p in packages] == [
        "toml_case_a",
        "toml_case_b",
    ]
    run = json.loads((packages[1] / "run02" / "run_metadata.json").read_text())
    assert run["application_metadata"] == {"argv": ["--b", "--n", "4"], "mode": "b"}
    case_meta = json.loads((packages[1] / "case_metadata.json").read_text())
    assert case_meta["metadata"] == {"variant": "b"}
    assert case_meta["general_information"]["description"] == "variant b"


def test_case_toml_validation(app, capsys):
    path = _write_case_toml(app)
    assert (
        cli_main(["jobs", "run", str(path), "--variant", "zzz", "--scheduler", "local"])
        == 2
    )
    assert "unknown variant" in capsys.readouterr().out
    bad = app / "bad.toml"
    bad.write_text(
        '[case]\nlabel = "x"\ncommand = ["x"]\nbogus = 1\n', encoding="utf-8"
    )
    assert cli_main(["jobs", "run", str(bad)]) == 2
    bad.write_text('[case]\nlabel = "x"\n', encoding="utf-8")
    assert cli_main(["jobs", "run", str(bad)]) == 2
    config = load_case_config(path)
    assert config["case"]["inputs"] == [str(app / "app.py")]


SCRIPT = """
import argparse, sys
from scope_profiler.jobs import ProfilingCase

VARIANTS = {{"x": ["--x"], "y": ["--y"]}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    case = ProfilingCase(
        label=f"script_{{args.variant}}",
        command=[{python!r}, {app!r}],
        mpi=False,
        scheduler="local",
        work_dir={work!r},
        record_pip_freeze=False,
    )
    case.launch(1, args=VARIANTS[args.variant])
    case.finalize()


if __name__ == "__main__":
    main()
"""


def test_sweep_of_scripts_and_toml_files(app, capsys):
    script = app / "submit.py"
    script.write_text(
        SCRIPT.format(
            python=sys.executable, app=str(app / "app.py"), work=str(app / "work")
        ),
        encoding="utf-8",
    )
    broken = app / "broken.py"
    broken.write_text(
        'if __name__ == "__main__":\n    raise SystemExit(4)\n', encoding="utf-8"
    )
    toml = _write_case_toml(app)

    entries = variant_entries(script)
    assert entries == [(script, ["--variant", "x"]), (script, ["--variant", "y"])]
    assert variant_entries(broken) == [(broken, [])]
    assert len(variant_entries(toml, ["--scheduler", "local"])) == 2

    assert run_sweep(entries, dry_run=True) == 0
    n_failed = run_sweep(
        [*entries, (broken, []), (toml, ["--variant", "a", "--scheduler", "local"])],
        poll_interval=0.01,
    )
    assert n_failed == 1
    out = capsys.readouterr().out
    assert "packaging is left to the sweep" in out
    names = sorted(
        p.name.rsplit("-", 1)[-1] for p in (app / "work" / "packaged").iterdir()
    )
    assert names == ["script_x", "script_y", "toml_case_a"]
    assert case_module._SWEEP_CASES is None


def test_cli_sweep_with_variants(app):
    toml = _write_case_toml(app)
    assert (
        cli_main(
            [
                "jobs",
                "sweep",
                str(toml),
                "--variants",
                "--arg=--scheduler",
                "--arg=local",
                "--poll-interval",
                "0",
            ]
        )
        == 0
    )
    assert len(list((app / "work" / "packaged").iterdir())) == 2


def test_push_rebases_on_a_concurrent_push(tmp_path, results_repo, monkeypatch):
    from scope_profiler.jobs import store as store_module

    monkeypatch.setattr(store_module.time, "sleep", lambda seconds: None)
    store = GitResultsStore(url=str(results_repo), branch="main")
    clone = store.clone(tmp_path / "clone")
    other = store.clone(tmp_path / "other")
    (other / "other.txt").write_text("other\n")
    assert store.push(other, label="other", commit="abc")
    (clone / "mine.txt").write_text("mine\n")
    assert store.push(clone, label="mine", commit="def")
    assert {"other.txt", "mine.txt"} <= set(_repo_files(results_repo))
    assert store.push(clone) is False


def test_case_toml_with_single_launch_table_and_store(app, results_repo):
    path = app / "single.toml"
    path.write_text(
        textwrap.dedent(
            f"""
            [case]
            label = "single"
            command = ["{sys.executable}", "{{case_dir}}/app.py"]
            inputs = ["app.py"]
            mpi = false
            work_dir = "work"
            repo = "."
            collect = ["*.h5"]
            results_suffixes = [".txt"]
            record_pip_freeze = false

            [launch]
            ntasks = 1

            [store]
            url = "{results_repo}"
            branch = "main"

            [clusters.inline]
            cpus_per_node = 3
            """,
        ),
        encoding="utf-8",
    )
    config = load_case_config(path)
    assert config["launch"] == [{"ntasks": 1}]
    assert config["case"]["collect"] == ("*.h5",)
    assert get_cluster("inline").cpus_per_node == 3
    assert cli_main(["jobs", "run", str(path), "--scheduler", "local", "--upload"]) == 0
    assert any(
        name.endswith("run01/profiling_data.h5") for name in _repo_files(results_repo)
    )


def test_launch_does_not_do_a_reverse_dns_lookup(app, monkeypatch):
    # socket.getfqdn() stalls for ~30 s per call on hosts without reverse DNS
    # (GitHub's macOS runners), which made this module take 18 minutes there.
    def fail():
        raise AssertionError("socket.getfqdn() must not be called")

    monkeypatch.setattr(case_module.socket, "getfqdn", fail)
    case = _case(app)
    case.launch(1)
    case.finalize()


# ---------------------------------------------------------------------------
# Looking after a results repository
# ---------------------------------------------------------------------------
def _package(root, label, stamp, runs=("run01",), metadata=None):
    from scope_profiler.tests.test_post_processing import _write_sample_h5

    folder = root / label / f"{stamp.replace('-', '').replace(':', '')}-{label}"
    listed = []
    for index, name in enumerate(runs, start=1):
        (folder / name).mkdir(parents=True)
        _write_sample_h5(
            folder / name / "profiling_data.h5", {0: {"a": ([0], [10])}}, {}
        )
        (folder / name / "run_metadata.json").write_text("{}")
        listed.append(
            {
                "launch_id": index,
                "folder": name,
                "run_metadata": f"{name}/run_metadata.json",
                "profiles": [f"{name}/profiling_data.h5"],
            }
        )
    (folder / "case_metadata.json").write_text(
        json.dumps(
            metadata
            or {
                "general_information": {"label": label, "time_date_utc": stamp},
                "runs": listed,
            }
        )
    )
    return folder


def test_check_finds_broken_packages(tmp_path, capsys):
    from scope_profiler.jobs.maintenance import check_results

    good = _package(tmp_path, "heat", "2026-11-01T00:00:00+00:00")
    assert check_results(tmp_path) == []
    (good / "run01" / "profiling_data.h5").write_text("not hdf5")
    broken = _package(tmp_path, "wave", "2026-11-02T00:00:00+00:00")
    shutil.rmtree(broken / "run01")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "case_metadata.json").write_text("{")

    problems = check_results(tmp_path, max_bytes=100)

    text = "\n".join(problems)
    assert "cannot be opened" in text
    assert "folder 'run01' is missing" in text
    assert "not valid JSON" in text
    assert "larger than" in text
    assert cli_main(["jobs", "check", str(tmp_path)]) == 1
    assert "problem(s)" in capsys.readouterr().out


def test_prune_keeps_the_newest_executions(tmp_path, capsys):
    from scope_profiler.jobs.maintenance import prune_results

    days = [f"2026-11-{day:02d}T00:00:00+00:00" for day in (1, 2, 3, 4)]
    folders = [_package(tmp_path, "heat", day) for day in days]
    other = _package(tmp_path, "wave", days[0])

    assert prune_results(tmp_path, keep=2) == folders[:2]
    assert all(folder.exists() for folder in folders)
    from datetime import datetime, timezone

    now = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    assert prune_results(tmp_path, keep=1, older_than_days=2, now=now) == folders[:2]
    assert cli_main(["jobs", "prune", str(tmp_path), "--keep", "2", "--yes"]) == 0
    assert [folder.exists() for folder in folders] == [False, False, True, True]
    assert other.exists()
    assert cli_main(["jobs", "prune", str(tmp_path), "--keep", "0"]) == 2
