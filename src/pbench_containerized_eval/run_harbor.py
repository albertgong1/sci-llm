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
    uv run python src/pbench_containerized_eval/run_harbor.py trials start \\
      -p out/harbor/supercon-mini-v2/ground-template/tasks/jac2980051 \\
      -a gemini-cli -m gemini/gemini-2.5-flash

"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import json
from typing import Any


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
    """Run Harbor with env prep + a small failure summary."""
    argv = sys.argv[1:]
    if not argv:
        print("Usage: run_harbor.py <harbor args...>", file=sys.stderr)
        print("Example: run_harbor.py trials start -p <task> -a gemini-cli -m ...")
        return 2

    if _reject_daytona(argv):
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

    argv = _rewrite_override_storage(argv)
    argv = _rewrite_gemini_model(argv)
    argv = _rewrite_env_flag(argv)

    env = os.environ.copy()
    src_path = str(_repo_root() / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    if src_path not in existing_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else src_path
        )
    command = [sys.executable, "-m", "harbor.cli.main", *argv]
    result = subprocess.run(command, check=False, cwd=_repo_root(), env=env)
    if result.returncode != 0:
        _summarize_failure(argv, repo=_repo_root())
    return int(result.returncode)


def _rewrite_override_storage(argv: list[str]) -> list[str]:
    """Map deprecated --override-storage to Harbor's --override-storage-mb."""
    rewritten: list[str] = []
    for arg in argv:
        if arg == "--override-storage":
            print(
                "Note: --override-storage is not a Harbor flag; using --override-storage-mb instead.",
                file=sys.stderr,
            )
            rewritten.append("--override-storage-mb")
            continue
        if arg.startswith("--override-storage="):
            value = arg.split("=", 1)[1]
            print(
                "Note: --override-storage is not a Harbor flag; using --override-storage-mb instead.",
                file=sys.stderr,
            )
            rewritten.append(f"--override-storage-mb={value}")
            continue
        rewritten.append(arg)
    return rewritten


def _rewrite_env_flag(argv: list[str]) -> list[str]:
    """Map --env to --environment-type for trials (jobs already support --env)."""
    if "trials" not in argv or "jobs" in argv:
        return argv

    rewritten: list[str] = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--env":
            rewritten.append("--environment-type")
            skip_next = True
            if idx + 1 < len(argv):
                rewritten.append(argv[idx + 1])
            continue
        if arg.startswith("--env="):
            rewritten.append(arg.replace("--env=", "--environment-type=", 1))
            continue
        rewritten.append(arg)
    return rewritten


def _reject_daytona(argv: list[str]) -> bool:
    """Block Daytona env usage now that Modal is the only supported backend."""

    def _is_daytona(value: str) -> bool:
        return value.strip().lower() == "daytona"

    for idx, arg in enumerate(argv):
        if arg == "--env" and idx + 1 < len(argv) and _is_daytona(argv[idx + 1]):
            break
        if arg.startswith("--env=") and _is_daytona(arg.split("=", 1)[1]):
            break
        if (
            arg == "--environment-type"
            and idx + 1 < len(argv)
            and _is_daytona(argv[idx + 1])
        ):
            break
        if arg.startswith("--environment-type=") and _is_daytona(arg.split("=", 1)[1]):
            break
    else:
        return False

    print(
        "Daytona support has been removed. Use `--env modal` "
        "(or `run_harbor_modal.py`).",
        file=sys.stderr,
    )
    return True


def _rewrite_gemini_model(argv: list[str]) -> list[str]:
    """Ensure Gemini model names include the provider prefix."""

    def _fix(value: str) -> str:
        if "/" in value:
            return value
        if value.startswith("gemini-"):
            return f"gemini/{value}"
        return value

    rewritten: list[str] = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if arg in {"-m", "--model", "--model-name"}:
            rewritten.append(arg)
            if idx + 1 < len(argv):
                rewritten.append(_fix(argv[idx + 1]))
            skip_next = True
            continue

        if arg.startswith("--model="):
            rewritten.append(f"--model={_fix(arg.split('=', 1)[1])}")
            continue
        if arg.startswith("--model-name="):
            rewritten.append(f"--model-name={_fix(arg.split('=', 1)[1])}")
            continue

        rewritten.append(arg)

    return rewritten


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find_latest_run_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _summarize_failure(argv: list[str], repo: Path) -> None:
    """Print a small hint about where Harbor wrote errors."""
    if "jobs" in argv or "run" in argv:
        jobs_dir = repo / "jobs"
        latest = _find_latest_run_dir(jobs_dir)
        if not latest:
            return
        result_path = latest / "result.json"
        result = _load_json(result_path)
        print(f"Harbor failed. Latest job: {result_path}", file=sys.stderr)
        if not result:
            return
        stats = result.get("stats") or {}
        n_errors = stats.get("n_errors")
        print(f"Job errors: {n_errors}", file=sys.stderr)
        evals = stats.get("evals") or {}
        if evals:
            first_eval = next(iter(evals.values()))
            exception_stats = first_eval.get("exception_stats") or {}
            if exception_stats:
                exc_name, trials = next(iter(exception_stats.items()))
                print(
                    f"Top exception: {exc_name} (count: {len(trials)})", file=sys.stderr
                )
                if trials:
                    trial = trials[0]
                    exc_path = latest / trial / "exception.txt"
                    if exc_path.exists():
                        exc_text = exc_path.read_text()
                        snippet = "\n".join(exc_text.splitlines()[:6])
                        print(
                            f"Sample exception ({exc_path}):\n{snippet}",
                            file=sys.stderr,
                        )
                        if "Total disk limit exceeded" in exc_text:
                            print(
                                "Hint: Disk quota exceeded. Lower --override-storage-mb "
                                "or delete old sandboxes.",
                                file=sys.stderr,
                            )
                        if "Total memory limit exceeded" in exc_text:
                            print(
                                "Hint: Memory quota exceeded. Lower --override-memory-mb "
                                "or reduce --n-concurrent.",
                                file=sys.stderr,
                            )
        return

    if "trials" in argv:
        trials_dir = repo / "trials"
        latest = _find_latest_run_dir(trials_dir)
        if not latest:
            return
        print(
            f"Harbor failed. Latest trial: {latest / 'result.json'}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
