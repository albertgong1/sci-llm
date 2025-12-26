"""Build and smoke-test Harbor tasks with Gemini and CC agents.

This script is a local verification that:
1) Task "compilation" (casual use of the word) works for a given prompt template.
2) Harbor can run a single task with different agents (ex. `gemini-cli` or `claude-code`).

Secrets:
  - Gemini: set `GOOGLE_API_KEY=...` in the repo root `.env`
  - Claude Code: set `ANTHROPIC_API_KEY=...` or `CLAUDE_CODE_OAUTH_TOKEN=...` in `.env`

Example:
  uv run python src/pbench_containerized_eval/smoke_harbor.py --refno PR05001178

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal `.env` file."""
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
        if not key:
            continue

        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()

        env[key] = value

    return env


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=repo_root())


def slugify(value: str) -> str:
    """Normalize strings for file-safe task IDs (mirrors the compiler)."""
    return (
        value.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
    )


def build_tasks(
    *, task: str | None, template: str, refno: str | None, force: bool
) -> Path:
    """Compile tasks and return the tasks directory path."""
    compiler = (
        repo_root() / "src" / "pbench_containerized_eval" / "prepare_harbor_tasks.py"
    )
    out_dir = repo_root() / "out" / "harbor" / "supercon-mini-v2"
    task_root = out_dir / template if task is None else out_dir / task / template
    tasks_dir = task_root / "tasks"

    if tasks_dir.exists() and any(tasks_dir.iterdir()) and not force:
        return tasks_dir

    cmd = [
        sys.executable,
        str(compiler),
        "--template",
        template,
        "--output-dir",
        str(out_dir),
        "--write-job-config",
    ]
    if task:
        cmd.extend(["--task", task])
    if refno:
        cmd.extend(["--refno", refno])
    if force:
        cmd.append("--force")

    _run(cmd)
    return tasks_dir


def run_trial(*, task_path: Path, agent: str, model: str | None) -> None:
    """Run a Harbor trial for a single local task directory."""
    runner = repo_root() / "src" / "pbench_containerized_eval" / "run_harbor.py"

    cmd = [
        sys.executable,
        str(runner),
        "trials",
        "start",
        "-p",
        str(task_path),
        "-a",
        agent,
    ]
    if model:
        cmd.extend(["-m", model])
    _run(cmd)


def main() -> int:
    """Docstring for main

    :return: Description
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Smoke test Harbor task variants + agents."
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task alias to filter properties (e.g., tc). Omit to include all.",
    )
    parser.add_argument(
        "--refno",
        default="PR05001178",
        help="Which refno to test (must exist in PDFs + HF split).",
    )
    parser.add_argument(
        "--templates",
        default="ground-template",
        help="Comma-separated list: ground-template,prompted-template,... (default: ground-template).",
    )
    parser.add_argument(
        "--agents",
        default="gemini-cli,claude-code",
        help="Comma-separated list: gemini-cli,claude-code,oracle,... (default: gemini-cli,claude-code).",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini/gemini-2.5-flash",
        help="Model for gemini-cli (must be provider/model).",
    )
    parser.add_argument(
        "--claude-model",
        default=None,
        help="Optional model for claude-code (provider/model). If omitted, uses Claude Code defaults.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the generated task directories.",
    )
    args = parser.parse_args()

    dotenv = _parse_dotenv(repo_root() / ".env")
    merged_env = {**dotenv, **os.environ}

    templates = [t.strip() for t in str(args.templates).split(",") if t.strip()]
    agents = [a.strip() for a in str(args.agents).split(",") if a.strip()]

    for template in templates:
        tasks_dir = build_tasks(
            task=str(args.task) if args.task else None,
            template=template,
            refno=str(args.refno) if args.refno else None,
            force=bool(args.force),
        )
        task_suffix = f"--{slugify(str(args.task))}" if args.task else ""
        task_id = f"{slugify(str(args.refno))}{task_suffix}"
        task_path = tasks_dir / task_id
        if not task_path.exists():
            raise FileNotFoundError(f"Expected task directory missing: {task_path}")

        for agent in agents:
            if agent == "gemini-cli":
                if not (
                    merged_env.get("GOOGLE_API_KEY") or merged_env.get("GEMINI_API_KEY")
                ):
                    print(
                        "Skipping gemini-cli: missing GOOGLE_API_KEY/GEMINI_API_KEY in env/.env"
                    )
                    continue
                run_trial(
                    task_path=task_path, agent=agent, model=str(args.gemini_model)
                )
                continue

            if agent == "claude-code":
                if not (
                    merged_env.get("ANTHROPIC_API_KEY")
                    or merged_env.get("CLAUDE_CODE_OAUTH_TOKEN")
                    or merged_env.get("CLAUDE_API_KEY")
                    or merged_env.get("CLAUDE_CODE_TOKEN")
                    or merged_env.get("CLAUDE_CODE_API_TOKEN")
                ):
                    print(
                        "Skipping claude-code: missing ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN in env/.env"
                    )
                    continue
                run_trial(task_path=task_path, agent=agent, model=args.claude_model)
                continue

            run_trial(task_path=task_path, agent=agent, model=None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
