"""The heat.py case of case.toml, written in Python instead: useful when the launches need logic.

python examples/jobs/submit_heat.py --variant small [--upload]
scope-profiler jobs sweep examples/jobs/submit_heat.py --variants   # every variant, one process
"""

import argparse
from pathlib import Path

from scope_profiler.jobs import ProfilingCase

HERE = Path(__file__).resolve().parent

# variant -> extra arguments of heat.py; `scope-profiler jobs sweep --variants` reads this dict.
VARIANTS = {
    "small": ["--cells", "100000"],
    "large": ["--cells", "10000000"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument(
        "--upload",
        action="store_true",
        help="push the results to $SCOPE_PROFILER_RESULTS_REPO",
    )
    args = parser.parse_args()

    case = ProfilingCase(
        label=f"heat_strong_scaling_{args.variant}",
        name="Heat equation, strong scaling",
        command=[
            "python",
            "{case_dir}/heat.py",
            "--profile-out",
            "{profile}",
            "--steps",
            "200",
        ],
        inputs=[HERE / "heat.py"],
        env={"OMP_NUM_THREADS": "1"},
        metadata={"code": "heat.py", "variant": args.variant},
        upload=args.upload,
    )
    for ntasks in (1, 2, 4):
        case.launch(ntasks, args=VARIANTS[args.variant])
    case.finalize()


if __name__ == "__main__":
    main()
