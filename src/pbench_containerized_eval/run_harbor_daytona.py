r"""Run Harbor with the native Daytona environment integration.

This wrapper simply forwards args to `run_harbor.py` while forcing `--env daytona`,
so Harbor uses Daytona sandboxes natively (no custom upload/sandbox management).

Example:
    uv run python src/pbench_containerized_eval/run_harbor_daytona.py \\
      jobs start -c out/harbor/supercon-mini-v2/ground-template/job.yaml \\
      -a gemini-cli -m gemini/gemini-2.5-flash

"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser."""
    if not path.exists():
        return {}

    env: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
        if key:
            env[key] = value
    return env


def main(argv: list[str]) -> int:
    """Invoke Harbor with Daytona enforced on the environment."""
    if not argv:
        print(
            "Usage: run_harbor_daytona.py <harbor args...>\n"
            "Example: run_harbor_daytona.py jobs start -c out/harbor/supercon-mini-v2/ground-template/job.yaml -a oracle",
            file=sys.stderr,
        )
        return 2

    dotenv = _parse_dotenv(_repo_root() / ".env")
    for key, value in dotenv.items():
        os.environ.setdefault(key, value)

    if "DAYTONA_API_KEY" not in os.environ:
        raise SystemExit("DAYTONA_API_KEY not found in environment or .env")

    if "--env" not in argv:
        argv = [*argv, "--env", "daytona"]

    command = [sys.executable, "src/pbench_containerized_eval/run_harbor.py", *argv]
    result = subprocess.run(command, check=False, cwd=_repo_root())
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
