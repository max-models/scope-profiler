"""Profile a deterministic Node workload through the shared benchmark runner."""
from pathlib import Path
import subprocess

subprocess.run(["node", str(Path(__file__).with_name("build.mjs"))], check=True)
