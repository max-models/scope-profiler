"""``scope-profiler jobs ...``: launch profiling cases, run sweeps, upload packaged results.

- ``scope-profiler jobs run case.toml [--variant KEY] [--upload]`` -- launch
  every run of a case file and package them as they finish.
- ``scope-profiler jobs sweep A.py B.toml ... [--variants] [--upload]`` --
  run several case scripts/files from one process.
- ``scope-profiler jobs upload <packaged case folder> [--repo URL]`` -- push
  a case packaged without ``--upload``, unchanged.
- ``scope-profiler jobs clusters [--config FILE]`` -- list the cluster
  presets and the one detected on this machine.
- ``scope-profiler jobs check DIR`` -- check a results repository: metadata,
  missing runs, profiles that do not open, files that are too large.
- ``scope-profiler jobs prune DIR --keep N [--older-than DAYS] [--yes]`` --
  remove all but the newest executions of every case.
"""

from __future__ import annotations

import argparse
import json

from scope_profiler.jobs.clusters import detect_cluster, list_clusters, load_clusters
from scope_profiler.jobs.config import CaseConfigError, run_case_config
from scope_profiler.jobs.store import GitResultsStore
from scope_profiler.jobs.sweep import run_sweep, variant_entries


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--upload",
        action="store_true",
        help="push the packaged runs to the results repository",
    )
    parser.add_argument(
        "--cluster-config",
        action="append",
        default=[],
        metavar="FILE",
        help="TOML file of cluster presets to load (repeatable; also $SCOPE_PROFILER_CLUSTERS)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="seconds between polls of the SLURM queue (default: 30)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scope-profiler jobs",
        description="Launch profiling jobs (locally or with SLURM), package their output and push it to a "
        "results repository.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser(
        "run", help="launch the runs of a case TOML file and package them"
    )
    run.add_argument("config", help="case TOML file")
    run.add_argument(
        "--variant",
        action="append",
        metavar="KEY",
        help="run only this variant (repeatable)",
    )
    run.add_argument(
        "--scheduler",
        choices=["local", "slurm", "dry-run"],
        help="how to run the jobs (default: SLURM where sbatch exists, else local); "
        "dry-run only writes the job scripts",
    )
    _add_common(run)

    sweep = sub.add_parser(
        "sweep", help="run several case scripts or TOML files from one process"
    )
    sweep.add_argument(
        "entries", nargs="+", help="Python case scripts or case TOML files"
    )
    sweep.add_argument(
        "--variants",
        action="store_true",
        help="one entry per variant (VARIANTS dict of a script, [variants] of a TOML file)",
    )
    sweep.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="ARG",
        help="argument passed to every script (repeatable), e.g. --arg=--language --arg=c",
    )
    sweep.add_argument("--dry-run", action="store_true", help="only print the entries")
    _add_common(sweep)

    upload = sub.add_parser(
        "upload", help="push a packaged case folder to the results repository"
    )
    upload.add_argument(
        "case_dir", help="packaged case folder (it contains case_metadata.json)"
    )
    upload.add_argument(
        "--repo", help="results repository URL (default: $SCOPE_PROFILER_RESULTS_REPO)"
    )
    upload.add_argument(
        "--branch", help="branch to push to (default: the remote's default)"
    )
    upload.add_argument(
        "--path",
        default="",
        help="directory in the repository to put the folder into; {label} is the "
        "case's label (e.g. mycode/{label}; default: the top)",
    )

    check = sub.add_parser(
        "check",
        help="check a results repository (metadata, missing runs, profiles, file sizes)",
    )
    check.add_argument("root", help="a checkout of the results repository")
    check.add_argument(
        "--max-mb",
        type=float,
        default=50.0,
        help="largest file allowed, in MB (default: 50; GitHub refuses 100)",
    )
    check.add_argument(
        "--no-open", action="store_true", help="do not open the .h5 files"
    )

    prune = sub.add_parser(
        "prune", help="remove all but the newest executions of every case"
    )
    prune.add_argument("root", help="a checkout of the results repository")
    prune.add_argument(
        "--keep", type=int, required=True, help="executions to keep per case"
    )
    prune.add_argument(
        "--older-than",
        type=float,
        metavar="DAYS",
        help="only remove executions older than this",
    )
    prune.add_argument(
        "--yes",
        action="store_true",
        help="delete the folders (without it, only list them)",
    )

    presets = sub.add_parser(
        "clusters", help="list the cluster presets and the detected one"
    )
    presets.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="TOML file of presets",
    )
    presets.add_argument("--json", action="store_true", help="print JSON")
    return parser


def main(argv: list[str]) -> int:
    """Run ``scope-profiler jobs`` with ``argv`` (without the ``jobs``)."""
    args = _parser().parse_args(argv)

    for path in getattr(args, "cluster_config", []):
        load_clusters(path)

    if args.action == "run":
        try:
            run_case_config(
                args.config,
                variants=args.variant,
                upload=args.upload,
                scheduler=args.scheduler,
                poll_interval=args.poll_interval,
            )
        except (CaseConfigError, FileNotFoundError, KeyError) as exc:
            print(f"error: {exc}")
            return 2
        return 0

    if args.action == "sweep":
        entries = []
        for entry in args.entries:
            if args.variants:
                entries += variant_entries(entry, args.arg)
            else:
                entries.append((entry, list(args.arg)))
        return (
            1
            if run_sweep(
                entries,
                upload=args.upload,
                dry_run=args.dry_run,
                poll_interval=args.poll_interval,
            )
            else 0
        )

    if args.action == "upload":
        store = (
            GitResultsStore(url=args.repo, branch=args.branch, path=args.path)
            if args.repo
            else GitResultsStore.from_env(branch=args.branch, path=args.path)
        )
        try:
            store.upload(args.case_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return 2
        return 0

    if args.action == "check":
        from scope_profiler.jobs.maintenance import check_results, find_packages

        problems = check_results(
            args.root, max_bytes=int(args.max_mb * 1e6), open_profiles=not args.no_open
        )
        for problem in problems:
            print(problem)
        print(
            f"{len(find_packages(args.root))} packaged case folder(s), {len(problems)} problem(s)"
        )
        return 1 if problems else 0

    if args.action == "prune":
        from scope_profiler.jobs.maintenance import prune_results

        try:
            removed = prune_results(
                args.root, args.keep, older_than_days=args.older_than, apply=args.yes
            )
        except ValueError as exc:
            print(f"error: {exc}")
            return 2
        for path in removed:
            print(("removed " if args.yes else "would remove ") + str(path))
        if not args.yes and removed:
            print("Pass --yes to delete them, then commit the removal.")
        print(f"{len(removed)} execution(s) {'removed' if args.yes else 'to remove'}")
        return 0

    # clusters
    for path in args.config:
        load_clusters(path)
    presets = list_clusters()
    detected = detect_cluster()
    if args.json:
        print(
            json.dumps(
                {
                    "detected": detected.name if detected else None,
                    "clusters": {
                        name: preset.to_dict() for name, preset in presets.items()
                    },
                },
                indent=2,
            ),
        )
        return 0
    if not presets:
        print(
            "No cluster presets registered. Load some with --config FILE or $SCOPE_PROFILER_CLUSTERS."
        )
        return 0
    for name, preset in presets.items():
        marker = "*" if detected is not None and detected.name == name else " "
        hardware = f"{preset.cpus_per_node or '?'} cpus, {preset.gpus_per_node or 0} gpus per node"
        sbatch = ", ".join(f"{key}={value}" for key, value in preset.sbatch.items())
        print(f"{marker} {name:24s} {hardware:30s} {sbatch}")
    print(
        "(* detected on this machine)"
        if detected
        else "(no preset matches this machine)"
    )
    return 0
