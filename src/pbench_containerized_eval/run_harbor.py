r"""Run Harbor with secrets loaded from the repo's `.env`.

Harbor agents (including `gemini-cli`) read API keys from environment variables in the
*host* process and forward them into the container. `uv run harbor ...` does not load
`.env` automatically, so Gemini runs can fail even when you have `GOOGLE_API_KEY=...`
in `.env`.

This helper:
1) Parses `.env` from the repository root.
2) Populates `os.environ`.
3) Sets `GEMINI_API_KEY` from `GOOGLE_API_KEY` when needed (the Gemini CLI expects `GEMINI_API_KEY`).
4) Delegates to Harbor's CLI entrypoint.

Example:
    uv run python examples/containerized-extraction/run_harbor.py trials start \\
      -p out/harbor/supercon-mini/tc/easy/tasks/jac2980051--tc \\
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
    """Parse a minimal `.env` file.

    Supports lines like:
      - KEY=value
      - export KEY=value
      - Quoted values: KEY="value" or KEY='value'

    Ignores blank lines and `#` comments.
    """
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
            # Strip inline comments for unquoted values: KEY=value # comment
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()

        if key:
            env[key] = value

    return env


def main() -> int:
    """Docstring for main

    :return: Description
    :rtype: int
    """
    argv = sys.argv[1:]
    if not argv:
        print("Usage: run_harbor.py <harbor args...>", file=sys.stderr)
        print("Example: run_harbor.py trials start -p <task> -a gemini-cli -m ...")
        return 2

    dotenv = _parse_dotenv(_repo_root() / ".env")
    for key, value in dotenv.items():
        os.environ.setdefault(key, value)

    # common issue: having both might confuse some clients
    # enforce GOOGLE_API_KEY as primary
    if "GOOGLE_API_KEY" in os.environ:
       os.environ.pop("GEMINI_API_KEY", None)
    elif "GEMINI_API_KEY" in os.environ:
       os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    # Claude Code authentication: Harbor's `claude-code` agent looks for these vars.
    if "ANTHROPIC_API_KEY" not in os.environ and "CLAUDE_API_KEY" in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]

    if (
        "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ
        and "CLAUDE_CODE_TOKEN" in os.environ
    ):
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["CLAUDE_CODE_TOKEN"]

    if (
        "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ
        and "CLAUDE_CODE_API_TOKEN" in os.environ
    ):
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["CLAUDE_CODE_API_TOKEN"]

    command = [sys.executable, "-m", "harbor.cli.main", *argv]
    result = subprocess.run(command, check=False, cwd=_repo_root())
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
