"""Profile a deterministic Node workload through the shared benchmark runner."""

import subprocess
from pathlib import Path

subprocess.run(["node", str(Path(__file__).with_name("build.mjs"))], check=True)
