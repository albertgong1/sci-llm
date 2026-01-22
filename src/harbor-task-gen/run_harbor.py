r"""Run Harbor with secrets loaded from the repo's `.env`.

Harbor agents (including `gemini-cli`) read API keys from environment variables in the
*host* process and forward them into the container. `uv run harbor ...` does not load
`.env` automatically, so Gemini runs can fail even when you have `GOOGLE_API_KEY=...`
in `.env`.

This helper:
1) Parses `.env` from the repository root.
2) Populates `os.environ`.
3) Sets `GEMINI_API_KEY` from `GOOGLE_API_KEY` when needed (the Gemini CLI expects
   `GEMINI_API_KEY`).
4) Delegates to Harbor's CLI entrypoint.

Extras bundled into this wrapper:
- HF tasks support: `--hf-tasks-repo ...` rewrites Harbor args to pull tasks from HF.
- Modal convenience: `--modal` forces `--env modal` and enables cleanup defaults.
- Post-run hooks: `--compile-run` compiles the latest run into a portable bundle, and
  `--push-run-to-hf` uploads that bundle to a HF dataset repo.

Example:
    uv run python src/harbor-task-gen/run_harbor.py trials start \\
      -p out/harbor/supercon-mini-v2/ground-template/tasks/jac2980051 \\
      -a gemini-cli -m gemini/gemini-2.5-flash

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlparse, urlunparse

import asyncio

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_workspace_root() -> Path:
    return _repo_root() / "examples" / "harbor-workspace"


def _resolve_workspace_root(value: str | None) -> Path:
    root = Path(value).expanduser() if value else _default_workspace_root()
    return root.resolve()


def _extract_workspace_arg(argv: list[str]) -> tuple[list[str], Path, bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", type=str, default=None)
    args, remaining = parser.parse_known_args(argv)
    return (
        remaining,
        _resolve_workspace_root(args.workspace),
        args.workspace is not None,
    )


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


@dataclass(frozen=True)
class PostRunPlan:
    """Configuration for optional compile/push actions after a Harbor run."""

    compile_run: bool
    push_to_hf: bool
    compile_out_dir: Path
    compile_name: str | None
    compile_force: bool
    hf_repo_id: str | None
    hf_repo_type: str
    hf_path_in_repo: str | None
    hf_private: bool | None
    hf_write_root_readme: bool
    hf_force_root_readme: bool

    @property
    def enabled(self) -> bool:
        """Return True when any post-run action is requested."""
        return self.compile_run or self.push_to_hf


def _extract_post_run_args(
    argv: list[str], *, workspace: Path
) -> tuple[list[str], PostRunPlan]:
    """Strip post-run flags from argv and return a structured plan."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compile-run", action="store_true")
    parser.add_argument(
        "--compile-out-dir",
        default=str(workspace / "out" / "harbor-runs"),
    )
    parser.add_argument("--compile-name", default=None)
    parser.add_argument("--compile-force", action="store_true")
    parser.add_argument("--push-run-to-hf", action="store_true")
    parser.add_argument("--hf-runs-repo", default=None)
    parser.add_argument(
        "--hf-runs-repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
    )
    parser.add_argument("--hf-runs-path-in-repo", default=None)
    parser.add_argument("--hf-runs-private", action="store_true")
    parser.add_argument("--hf-runs-public", action="store_true")
    parser.add_argument("--hf-runs-write-root-readme", action="store_true")
    parser.add_argument("--hf-runs-force-root-readme", action="store_true")
    args, remaining = parser.parse_known_args(argv)

    if args.hf_runs_private and args.hf_runs_public:
        raise SystemExit("Pass at most one of --hf-runs-private/--hf-runs-public.")

    private: bool | None
    if args.hf_runs_private:
        private = True
    elif args.hf_runs_public:
        private = False
    else:
        private = None

    compile_run = bool(args.compile_run or args.push_run_to_hf)
    if args.push_run_to_hf and not args.hf_runs_repo:
        raise SystemExit("--hf-runs-repo is required when using --push-run-to-hf.")

    plan = PostRunPlan(
        compile_run=compile_run,
        push_to_hf=bool(args.push_run_to_hf),
        compile_out_dir=Path(args.compile_out_dir),
        compile_name=args.compile_name,
        compile_force=bool(args.compile_force),
        hf_repo_id=args.hf_runs_repo,
        hf_repo_type=str(args.hf_runs_repo_type),
        hf_path_in_repo=args.hf_runs_path_in_repo,
        hf_private=private,
        hf_write_root_readme=bool(args.hf_runs_write_root_readme),
        hf_force_root_readme=bool(args.hf_runs_force_root_readme),
    )

    return remaining, plan


def _extract_modal_args(argv: list[str]) -> tuple[list[str], bool]:
    """Strip the wrapper-only `--modal` flag and return whether it was requested."""
    cleaned: list[str] = []
    modal_requested = False
    for arg in argv:
        if arg == "--modal":
            modal_requested = True
            continue
        cleaned.append(arg)
    return cleaned, modal_requested


def _patch_modal_download_logs(timeout_sec: int) -> None:
    """Guard Modal log downloads so stalled gRPC calls can't freeze the job."""
    try:
        from grpclib.exceptions import StreamTerminatedError
        from harbor.environments.modal import ModalEnvironment
        from harbor.trial.trial import Trial
    except Exception:
        return

    try:
        from modal.exception import ClientClosed
    except Exception:
        ClientClosed = None  # type: ignore[assignment]

    original = Trial._maybe_download_logs
    if getattr(original, "_harbor_task_gen_patched", False):
        return

    async def patched(self: Any, source_dir: str, target_dir: Path) -> None:
        if not isinstance(self._environment, ModalEnvironment):
            return await original(self, source_dir, target_dir)

        if self._environment.is_mounted or self._are_agent_logs_downloaded:
            return

        try:
            await asyncio.wait_for(
                self._environment.download_dir(
                    source_dir=source_dir, target_dir=target_dir
                ),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            self._logger.warning(
                "Modal download_dir timed out after %ss; skipping log download.",
                timeout_sec,
            )
        except StreamTerminatedError as exc:
            self._logger.warning(
                "Modal download_dir failed (StreamTerminatedError): %s", exc
            )
        except AttributeError as exc:
            self._logger.warning("Modal download_dir failed (AttributeError): %s", exc)
        except Exception as exc:
            if ClientClosed is not None and isinstance(exc, ClientClosed):
                self._logger.warning("Modal download_dir failed (ClientClosed).")
            else:
                self._logger.warning("Modal download_dir failed: %s", exc)
        finally:
            self._are_agent_logs_downloaded = True

    patched._harbor_task_gen_patched = True  # type: ignore[attr-defined]
    Trial._maybe_download_logs = patched  # type: ignore[assignment]


def _extract_agent_setup_timeout(
    argv: list[str],
) -> tuple[list[str], int | None]:
    """Strip --agent-setup-timeout-sec from argv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agent-setup-timeout-sec", type=int, default=None)
    args, remaining = parser.parse_known_args(argv)
    return remaining, args.agent_setup_timeout_sec


def _patch_agent_setup_timeout(timeout_sec: int) -> None:
    """Override Harbor's default agent setup timeout (seconds)."""
    try:
        from harbor.trial.trial import Trial
    except Exception:
        return

    Trial._AGENT_SETUP_TIMEOUT_SEC = timeout_sec


def _run_harbor_cli_in_process(argv: list[str], *, workspace: Path) -> int:
    """Invoke Harbor's Typer app in-process so runtime patches take effect."""
    from harbor.cli.main import app

    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    try:
        os.chdir(workspace)
        sys.argv = ["harbor", *argv]
        try:
            app()
            return 0
        except SystemExit as exc:
            return int(exc.code or 0)
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def main() -> int:
    """Run Harbor or helper utilities with env prep + a small failure summary."""
    argv = sys.argv[1:]
    if not argv:
        print(
            "Usage: run_harbor.py [--workspace PATH] <harbor args...>", file=sys.stderr
        )
        print(
            "Example: run_harbor.py trials start -p <task> -a gemini-cli -m ...",
            file=sys.stderr,
        )
        print(
            "Default workspace: examples/harbor-workspace (override with --workspace).",
            file=sys.stderr,
        )
        print(
            "Utilities: run_harbor.py compile-run ... | run_harbor.py push-run-to-hf ...",
            file=sys.stderr,
        )
        return 2

    argv, workspace, workspace_explicit = _extract_workspace_arg(argv)
    if not argv:
        print(
            "Usage: run_harbor.py [--workspace PATH] <harbor args...>", file=sys.stderr
        )
        print(
            "Example: run_harbor.py trials start -p <task> -a gemini-cli -m ...",
            file=sys.stderr,
        )
        print(
            "Default workspace: examples/harbor-workspace (override with --workspace).",
            file=sys.stderr,
        )
        print(
            "Utilities: run_harbor.py compile-run ... | run_harbor.py push-run-to-hf ...",
            file=sys.stderr,
        )
        return 2

    if argv[0] == "compile-run":
        forwarded = (
            [*argv[1:], "--workspace", str(workspace)]
            if workspace_explicit
            else argv[1:]
        )
        return compile_run_cli(forwarded)
    if argv[0] == "push-run-to-hf":
        forwarded = (
            [*argv[1:], "--workspace", str(workspace)]
            if workspace_explicit
            else argv[1:]
        )
        return push_run_to_hf_cli(forwarded)
    if workspace.exists() and not workspace.is_dir():
        raise SystemExit(f"--workspace must be a directory: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    argv, post_run = _extract_post_run_args(argv, workspace=workspace)
    argv, hf_args = _extract_hf_args(argv)
    argv, modal_requested = _extract_modal_args(argv)
    argv, agent_setup_override = _extract_agent_setup_timeout(argv)

    dotenv = _parse_dotenv(_repo_root() / ".env")
    for key, value in dotenv.items():
        os.environ.setdefault(key, value)

    # Keep both key names in sync so Gemini CLI and SDK clients work.
    if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    elif "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
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

    # Ensure OpenAI/Codex API key is present (no mapping needed, just a check/log if missing?)
    # Harbor's codex agent reads OPENAI_API_KEY directly.

    argv = _rewrite_override_storage(argv)
    argv = _rewrite_gemini_model(argv)
    argv = _rewrite_openai_model(argv)
    argv = _rewrite_qwen_model(argv)
    _apply_qwen_coder_defaults(argv)
    argv = _apply_modal_defaults(argv, modal_requested)
    argv = _rewrite_env_flag(argv)
    argv = _apply_hf_args(argv, hf_args, workspace=workspace)

    modal_active = _detect_environment_type(argv) == "modal"

    run_roots = _run_roots_from_args(argv, workspace=workspace)
    pre_run_mtime = _latest_run_mtime(run_roots)

    env = os.environ.copy()
    src_path = str(_repo_root() / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    if src_path not in existing_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else src_path
        )
    if modal_active:
        timeout_raw = env.get("HARBOR_MODAL_LOG_DOWNLOAD_TIMEOUT_SEC", "300").strip()
        if timeout_raw and timeout_raw != "0":
            try:
                timeout_sec = int(timeout_raw)
            except ValueError:
                timeout_sec = 300
            _patch_modal_download_logs(timeout_sec)
        setup_timeout_raw = (
            str(agent_setup_override)
            if agent_setup_override is not None
            else env.get("HARBOR_AGENT_SETUP_TIMEOUT_SEC", "900")
        ).strip()
        if setup_timeout_raw and setup_timeout_raw != "0":
            try:
                setup_timeout_sec = int(setup_timeout_raw)
            except ValueError:
                setup_timeout_sec = 900
            _patch_agent_setup_timeout(setup_timeout_sec)
        exit_code = _run_harbor_cli_in_process(argv, workspace=workspace)
    else:
        command = [sys.executable, "-m", "harbor.cli.main", *argv]
        result = subprocess.run(command, check=False, cwd=workspace, env=env)
        exit_code = int(result.returncode)

    if exit_code != 0:
        _summarize_failure(argv, workspace=workspace)
    
    # Try to score external trials (Codex/Terminus) if script exists
    _maybe_score_external_trials(workspace, run_roots, pre_run_mtime)

    if post_run.enabled:
        exit_code = _run_post_actions(
            argv=argv,
            plan=post_run,
            workspace=workspace,
            pre_run_mtime=pre_run_mtime,
            exit_code=exit_code,
        )

    return exit_code


def _maybe_score_external_trials(
    workspace: Path, 
    run_roots: list[Path], 
    pre_run_mtime: float | None
) -> None:
    """Invoke score_harbor_trials.py if present to fix up Codex/Terminus results."""
    # Assume scoring script is in examples/harbor-workspace
    # This path construction assumes standard repo layout
    scoring_script = _repo_root() / "examples" / "harbor-workspace" / "score_harbor_trials.py"
    if not scoring_script.exists():
        return

    # Find new trials created during this run
    new_trials: list[Path] = []
    
    # Check roots (usually workspace/trials or workspace/jobs)
    for root in run_roots:
        if not root.exists():
            continue
            
        # Helper to decide if a path is new
        def is_new(p: Path) -> bool:
            return pre_run_mtime is None or p.stat().st_mtime > pre_run_mtime

        # If searching 'jobs' dir, trials are nested one level deeper (job/trial)
        if root.name == "jobs":
            for job_dir in root.iterdir():
                if not job_dir.is_dir():
                    continue
                # If job dir itself is new, scan its children
                # If job dir is old, check if children are new (e.g. retries?)
                # Simplification: just scan all children of new/modified jobs
                if is_new(job_dir):
                    for trial_dir in job_dir.iterdir():
                        if trial_dir.is_dir():
                            new_trials.append(trial_dir)
        else:
            # Assuming 'trials' dir, direct children are trials
            for trial_dir in root.iterdir():
                if trial_dir.is_dir() and is_new(trial_dir):
                    new_trials.append(trial_dir)
    
    if not new_trials:
        return

    print(f"Running external scoring verification on {len(new_trials)} new trials...")
    subprocess.run(
        [
            sys.executable, 
            str(scoring_script), 
            "--trial-paths", 
            *[str(p) for p in new_trials]
        ],
        check=False,
    )


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


def _apply_modal_defaults(argv: list[str], modal_requested: bool) -> list[str]:
    """Apply Modal defaults when the run targets Modal."""
    env_value = _detect_environment_type(argv)
    modal_active = modal_requested or env_value == "modal"

    if modal_requested and env_value and env_value != "modal":
        raise SystemExit(
            f"--modal conflicts with --env/--environment-type={env_value}."
        )

    if modal_active and not env_value:
        argv = [*argv, "--env", "modal"]

    if modal_active and _should_force_delete(argv):
        argv = [*argv, "--delete"]

    return argv


def _detect_environment_type(argv: list[str]) -> str | None:
    """Return the environment type specified in argv (jobs/trials), if any."""
    for idx, arg in enumerate(argv):
        if arg == "--env" and idx + 1 < len(argv):
            return argv[idx + 1].strip().lower()
        if arg.startswith("--env="):
            return arg.split("=", 1)[1].strip().lower()
        if arg == "--environment-type" and idx + 1 < len(argv):
            return argv[idx + 1].strip().lower()
        if arg.startswith("--environment-type="):
            return arg.split("=", 1)[1].strip().lower()
    return None


def _should_force_delete(argv: list[str]) -> bool:
    """Return True when a run should delete its environment by default."""
    if "--delete" in argv or "--no-delete" in argv:
        return False
    if "start" not in argv:
        return False
    return "jobs" in argv or "trials" in argv


def _run_roots_from_args(argv: list[str], *, workspace: Path) -> list[Path]:
    """Infer which Harbor output roots (jobs/trials) a command will use."""
    if "trials" in argv and "jobs" not in argv:
        return [workspace / "trials"]
    if "jobs" in argv or "run" in argv:
        return [workspace / "jobs"]
    return [workspace / "jobs", workspace / "trials"]


def _latest_run_mtime(roots: Iterable[Path]) -> float | None:
    """Return the latest modified time among run directories in the given roots."""
    mtimes: list[float] = []
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir():
                mtimes.append(child.stat().st_mtime)
    return max(mtimes) if mtimes else None


def _select_run_dir_after(roots: Iterable[Path], since: float | None) -> Path | None:
    """Pick the most recent run dir, preferring ones newer than the given timestamp."""
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend([p for p in root.iterdir() if p.is_dir()])

    if not candidates:
        return None

    if since is not None:
        newer = [p for p in candidates if p.stat().st_mtime > since]
        if newer:
            return max(newer, key=lambda p: p.stat().st_mtime)

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_post_actions(
    *,
    argv: list[str],
    plan: PostRunPlan,
    workspace: Path,
    pre_run_mtime: float | None,
    exit_code: int,
) -> int:
    """Compile and/or upload the most recent Harbor run after execution."""
    if not plan.enabled:
        return exit_code

    run_root_candidates = _run_roots_from_args(argv, workspace=workspace)
    run_dir = _select_run_dir_after(run_root_candidates, pre_run_mtime)
    if run_dir is None:
        print("Post-run hook: no Harbor run directory found.", file=sys.stderr)
        return exit_code

    compile_out_dir = plan.compile_out_dir
    if not compile_out_dir.is_absolute():
        compile_out_dir = (workspace / compile_out_dir).resolve()

    try:
        compiled = compile_run(
            run_dir=run_dir,
            out_dir=compile_out_dir,
            name=plan.compile_name,
            force=plan.compile_force,
        )
        print(f"Compiled run bundle -> {compiled.bundle_dir}")
    except Exception as exc:
        print(f"Post-run compile failed: {exc}", file=sys.stderr)
        return exit_code or 1

    if plan.push_to_hf:
        try:
            _upload_bundle_to_hf(
                bundle_dir=compiled.bundle_dir,
                repo_id=str(plan.hf_repo_id),
                repo_type=str(plan.hf_repo_type),
                path_in_repo=plan.hf_path_in_repo,
                private=plan.hf_private,
                write_root_readme=plan.hf_write_root_readme,
                force_root_readme=plan.hf_force_root_readme,
                token=_infer_hf_token(),
            )
        except Exception as exc:
            print(f"Post-run upload failed: {exc}", file=sys.stderr)
            return exit_code or 1

    return exit_code


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

        if arg.startswith("--model-name="):
            rewritten.append(f"--model-name={_fix(arg.split('=', 1)[1])}")
            continue

        rewritten.append(arg)

    return rewritten


def _rewrite_openai_model(argv: list[str]) -> list[str]:
    """Ensure OpenAI/GPT model names include the provider prefix.

    This is mostly for consistency in logs; Harbor's codex agent strips the prefix anyway.
    """

    def _fix(value: str) -> str:
        if "/" in value:
            return value
        if value.startswith("gpt-") or value.startswith("o1-") or value.startswith("o3-"):
            return f"openai/{value}"
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


def _rewrite_qwen_model(argv: list[str]) -> list[str]:
    """Ensure Qwen model names are routed via OpenRouter (user preference)."""

    # Check if we are running the native qwen-coder agent
    is_native_agent = False
    for i, arg in enumerate(argv):
        if arg in ("-a", "--agent") and i + 1 < len(argv) and argv[i + 1] == "qwen-coder":
            is_native_agent = True
            break
        if arg.startswith("--agent=") and "qwen-coder" in arg:
            is_native_agent = True
            break

    def _fix(value: str) -> str:
        # If Native Agent, DO NOT prefix. OpenRouter expects "qwen/model", not "openrouter/qwen/model"
        if is_native_agent:
            return value

        # If Terminus/LiteLLM, we DO need the prefix so LiteLLM knows to route to OpenRouter.
        # If already has a provider prefix (like openrouter/), leave it
        if value.startswith("openrouter/") or value.startswith("dashscope/"):
            return value
        
        # If it looks like a Qwen model ID (starts with qwen or qwen/)
        if value.lower().startswith("qwen"):
            if "/" in value:
                return f"openrouter/{value}"
            return f"openrouter/{value}"
            
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


def _apply_qwen_coder_defaults(argv: list[str]) -> None:
    """If using qwen-coder agent, ensure OpenAI-compatible env vars point to OpenRouter."""
    print(f"DEBUG: argv={argv}") 
    is_qwen_code = False
    for i, arg in enumerate(argv):
        if arg in ("-a", "--agent") and i + 1 < len(argv) and argv[i + 1] == "qwen-coder":
            is_qwen_code = True
            break
        if arg.startswith("--agent=") and "qwen-coder" in arg:
            is_qwen_code = True
            break

    # Also detect if we are RESUMING a job that uses qwen-coder
    # argv might be like: jobs resume -p out/harbor/jobs/2026-01-18__19-17-22
    if not is_qwen_code and "resume" in argv:
        path_idx = -1
        if "-p" in argv:
            path_idx = argv.index("-p") + 1
        elif "--path" in argv:
             path_idx = argv.index("--path") + 1
        
        if path_idx > 0 and path_idx < len(argv):
             job_path = Path(argv[path_idx])
             config_path = job_path / "config.json"
             if config_path.exists():
                 try:
                     with open(config_path) as f:
                        config = json.load(f)
                        # Config has "agents": [{"name": "qwen-coder", ...}]
                        agents = config.get("agents", [])
                        if isinstance(agents, list):
                            for agent in agents:
                                if agent.get("name") == "qwen-coder":
                                    is_qwen_code = True
                                    print(f"DEBUG: Detected qwen-coder agent in resumed job config: {config_path}")
                                    break
                 except Exception as e:
                     print(f"DEBUG: Failed to read config {config_path}: {e}")

    if is_qwen_code:
        print(f"DEBUG: Detected qwen-coder agent. Checking env vars...")
        
        # If we have an OpenRouter key, we should USE it as the OpenAI key for this agent,
        # because we are likely defaulting the Base URL to OpenRouter.
        # This overrides any native OpenAI key that might be present (e.g. for Codex).
        if "OPENROUTER_API_KEY" in os.environ:
            print("DEBUG: Overwriting OPENAI_API_KEY with OPENROUTER_API_KEY (to match OpenRouter URL)")
            os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        elif "OPENAI_API_KEY" in os.environ:
             print("DEBUG: Using existing OPENAI_API_KEY (No OpenRouter key found).")
        else:
             print("DEBUG: Warning: No API key found (OPENAI_API_KEY or OPENROUTER_API_KEY missing).")

        # If User didn't specify a base URL, default to OpenRouter
        if "OPENAI_BASE_URL" not in os.environ:
            print("DEBUG: Setting OPENAI_BASE_URL to OpenRouter default")
            os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        else:
            print(f"DEBUG: OPENAI_BASE_URL already set: {os.environ['OPENAI_BASE_URL']}")


def _extract_hf_args(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """Strip HF task flags from argv and return them separately."""
    hf_args: dict[str, str] = {}
    cleaned: list[str] = []
    i = 0
    flag_map = {
        "--hf-tasks-repo": "repo_id",
        "--hf-repo-type": "repo_type",
        "--hf-tasks-dataset": "dataset_name",
        "--hf-tasks-version": "dataset_version",
        "--hf-registry-url": "registry_url",
        "--hf-registry-path": "registry_path",
        "--hf-task": "task_id",
        "--hf-task-path": "task_path",
        "--hf-tasks-path": "tasks_path",
        "--hf-task-commit": "task_commit",
    }

    while i < len(argv):
        arg = argv[i]
        matched = False
        for flag, key in flag_map.items():
            if arg == flag:
                if i + 1 >= len(argv):
                    raise SystemExit(f"Missing value for {flag}")
                hf_args[key] = argv[i + 1]
                i += 2
                matched = True
                break
            if arg.startswith(f"{flag}="):
                hf_args[key] = arg.split("=", 1)[1]
                i += 1
                matched = True
                break
        if not matched:
            cleaned.append(arg)
            i += 1

    return cleaned, hf_args


def _apply_hf_args(
    argv: list[str], hf_args: dict[str, str], *, workspace: Path
) -> list[str]:
    """Rewrite HF task flags into Harbor CLI arguments."""
    if not hf_args:
        return argv

    repo_id = hf_args.get("repo_id")
    if not repo_id:
        raise SystemExit("--hf-tasks-repo is required to use HF task flags.")

    repo_type = hf_args.get("repo_type", "dataset")
    registry_path = hf_args.get("registry_path", "registry.json").lstrip("/")
    registry_url = hf_args.get("registry_url")

    local_snapshot = None
    if not _has_git_lfs():
        local_snapshot = _download_hf_snapshot(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=hf_args.get("task_commit"),
            token=_infer_hf_token(),
        )
        print(
            "git-lfs not found; using a local HF snapshot for tasks instead of git clone.",
            file=sys.stderr,
        )

    if _argv_has_any(argv, {"jobs", "run"}):
        if _argv_has_any(
            argv, {"--dataset", "-d", "--registry-url", "--registry-path", "--path"}
        ):
            raise SystemExit(
                "Cannot combine --hf-tasks-repo with --dataset/--registry/--path."
            )
        tasks_path = hf_args.get("tasks_path", "tasks").strip("/")
        if local_snapshot is not None:
            local_tasks_root = (
                local_snapshot / tasks_path if tasks_path else local_snapshot
            )
            return [*argv, "--path", str(local_tasks_root)]

        dataset_name = hf_args.get("dataset_name", repo_id)
        dataset_version = hf_args.get("dataset_version", "head")
        registry_path_local = None
        if registry_url is None:
            registry_path_local = _download_hf_registry(
                repo_id=repo_id,
                repo_type=repo_type,
                registry_path=registry_path,
                token=_infer_hf_token(),
                workspace=workspace,
            )
        if registry_path_local is not None:
            return [
                *argv,
                "--dataset",
                f"{dataset_name}@{dataset_version}",
                "--registry-path",
                str(registry_path_local),
            ]
        registry_url = registry_url or _hf_resolve_url(
            repo_id, repo_type, registry_path
        )
        return [
            *argv,
            "--dataset",
            f"{dataset_name}@{dataset_version}",
            "--registry-url",
            registry_url,
        ]

    if _argv_has_any(argv, {"trials"}):
        if _argv_has_any(argv, {"--path", "-p", "--task-git-url"}):
            raise SystemExit(
                "Cannot combine --hf-tasks-repo with --path/--task-git-url."
            )
        task_path = hf_args.get("task_path")
        if not task_path:
            task_id = hf_args.get("task_id")
            if not task_id:
                raise SystemExit("--hf-task or --hf-task-path is required for trials.")
            tasks_root = hf_args.get("tasks_path", "tasks").strip("/")
            task_path = f"{tasks_root}/{task_id}" if tasks_root else task_id
        if local_snapshot is not None:
            local_task_path = local_snapshot / task_path
            return [*argv, "--path", str(local_task_path)]
        task_git_url = _hf_git_url(repo_id, repo_type)
        argv = [*argv, "--task-git-url", task_git_url, "--path", task_path]
        if "task_commit" in hf_args:
            argv.extend(["--task-git-commit", hf_args["task_commit"]])
        return argv

    return argv


def _argv_has_any(argv: list[str], needles: set[str]) -> bool:
    return any(arg in needles for arg in argv)


def _hf_repo_url(repo_id: str, repo_type: str) -> str:
    base = "https://huggingface.co"
    if repo_type == "dataset":
        return f"{base}/datasets/{repo_id}"
    if repo_type == "space":
        return f"{base}/spaces/{repo_id}"
    return f"{base}/{repo_id}"


def _hf_git_url(repo_id: str, repo_type: str) -> str:
    return f"{_hf_repo_url(repo_id, repo_type)}.git"


def _hf_resolve_url(repo_id: str, repo_type: str, path_in_repo: str) -> str:
    """Return a direct raw URL for a file in a HF repo."""
    return f"{_hf_repo_url(repo_id, repo_type)}/raw/main/{path_in_repo}"


def _has_git_lfs() -> bool:
    """Return True when git-lfs is available on PATH."""
    return shutil.which("git-lfs") is not None


def _download_hf_snapshot(
    *,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    token: str | None,
) -> Path:
    """Download a HF repo snapshot (used when git-lfs is unavailable)."""
    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    except Exception as exc:
        raise SystemExit(
            "Failed to download tasks from HF via snapshot. "
            "Install git-lfs or ensure HF_TOKEN is set. "
            f"Original error: {exc}"
        ) from exc

    return Path(snapshot_dir)


def _download_hf_registry(
    *,
    repo_id: str,
    repo_type: str,
    registry_path: str,
    token: str | None,
    workspace: Path,
) -> Path | None:
    """Download registry.json from HF and return a local path, if possible."""
    try:
        registry_file = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=registry_path,
            token=token,
        )
    except Exception as exc:
        raise SystemExit(
            "Failed to download registry.json from HF. "
            "For private repos, ensure HF_TOKEN (or HUGGINGFACE_HUB_TOKEN/HF_API_TOKEN) "
            f"is set and that {repo_id}/{registry_path} exists. Original error: {exc}"
        ) from exc

    registry_data = json.loads(Path(registry_file).read_text())
    if token:
        registry_data = _rewrite_registry_git_urls(registry_data, token=token)

    cache_dir = workspace / "out" / "harbor" / "registry-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_repo = repo_id.replace("/", "__")
    local_path = cache_dir / f"{safe_repo}__{Path(registry_path).name}"
    local_path.write_text(json.dumps(registry_data, indent=2))
    return local_path


def _rewrite_registry_git_urls(registry_data: Any, *, token: str) -> Any:
    """Embed HF tokens into git_url fields for private task clones."""
    if not isinstance(registry_data, list):
        return registry_data
    for dataset in registry_data:
        tasks = dataset.get("tasks") if isinstance(dataset, dict) else None
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            git_url = task.get("git_url")
            if not isinstance(git_url, str):
                continue
            task["git_url"] = _embed_hf_token_in_git_url(git_url, token=token)
    return registry_data


def _embed_hf_token_in_git_url(git_url: str, *, token: str) -> str:
    """Embed a HF token into an HTTPS git URL if it targets huggingface.co."""
    parsed = urlparse(git_url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        return git_url
    if "@" in parsed.netloc:
        return git_url
    safe_token = quote(token, safe="")
    netloc = f"hf:{safe_token}@{parsed.netloc}"
    return urlunparse(parsed._replace(netloc=netloc))


def _safe_read_json(path: Path) -> Any | None:
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


def _summarize_failure(argv: list[str], workspace: Path) -> None:
    """Print a small hint about where Harbor wrote errors."""
    if "jobs" in argv or "run" in argv:
        jobs_dir = workspace / "jobs"
        latest = _find_latest_run_dir(jobs_dir)
        if not latest:
            return
        result_path = latest / "result.json"
        result = _safe_read_json(result_path)
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
        trials_dir = workspace / "trials"
        latest = _find_latest_run_dir(trials_dir)
        if not latest:
            return
        print(
            f"Harbor failed. Latest trial: {latest / 'result.json'}",
            file=sys.stderr,
        )


def _utc_now_iso() -> str:
    """Return an ISO timestamp in UTC with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_git(args: list[str]) -> str | None:
    """Run a git command in the repo root, returning stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            cwd=_repo_root(),
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    """Capture minimal git metadata for compiled run bundles."""
    head = _run_git(["rev-parse", "HEAD"])
    is_dirty = bool(_run_git(["status", "--porcelain"]))
    origin = _run_git(["remote", "get-url", "origin"])
    return {"head": head, "is_dirty": is_dirty, "origin": origin}


def _find_latest_run_dir_in(*roots: Path) -> Path:
    """Return the most recently modified run dir under the given roots."""
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend([p for p in root.iterdir() if p.is_dir()])

    if not candidates:
        raise FileNotFoundError(
            f"No Harbor runs found under {', '.join(str(r) for r in roots)}."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _looks_like_job_dir(path: Path) -> bool:
    """Return True if a run directory appears to be a Harbor job."""
    if (path / "job.log").exists():
        return True
    result = _safe_read_json(path / "result.json")
    return isinstance(result, dict) and "n_total_trials" in result


def _iter_trial_dirs(run_dir: Path) -> Iterable[Path]:
    """Yield trial directories from a job or a single trial run."""
    if _looks_like_job_dir(run_dir):
        for child in sorted(run_dir.iterdir()):
            if child.is_dir() and (child / "result.json").exists():
                yield child
        return

    if (run_dir / "result.json").exists():
        yield run_dir


def _sha256(path: Path) -> str:
    """Return the sha256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CompiledRun:
    """Result metadata for a compiled Harbor run bundle."""

    bundle_dir: Path
    run_name: str
    run_type: str  # "job" or "trial"
    harbor_run_rel: str  # relative path in bundle (posix)


def compile_run(
    *,
    run_dir: Path,
    out_dir: Path,
    name: str | None = None,
    force: bool = False,
) -> CompiledRun:
    """Compile a Harbor run directory into a portable bundle."""
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Harbor run dir not found: {run_dir}")

    run_name = name or run_dir.name
    run_type = "job" if _looks_like_job_dir(run_dir) else "trial"

    bundle_dir = out_dir.resolve() / run_name
    if bundle_dir.exists():
        if not force:
            raise FileExistsError(
                f"Bundle already exists at {bundle_dir}. Re-run with --force to overwrite."
            )
        shutil.rmtree(bundle_dir)

    (bundle_dir / "harbor").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "index").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "meta").mkdir(parents=True, exist_ok=True)

    harbor_run_dir = bundle_dir / "harbor" / run_name
    shutil.copytree(run_dir, harbor_run_dir, dirs_exist_ok=False)

    git = _git_metadata()
    bundle = {
        "bundle_version": 1,
        "compiled_at": _utc_now_iso(),
        "source": {"run_dir": str(run_dir), "run_type": run_type},
        "git": git,
        "harbor": {"run_name": run_name, "run_type": run_type},
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    (bundle_dir / "meta" / "git.json").write_text(json.dumps(git, indent=2))
    (bundle_dir / "meta" / "system.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "compiled_at": bundle["compiled_at"],
            },
            indent=2,
        )
    )

    _write_trial_indexes(
        bundle_dir=bundle_dir,
        run_name=run_name,
        run_type=run_type,
        run_dir=run_dir,
        harbor_run_dir=harbor_run_dir,
    )
    _write_file_manifest(bundle_dir=bundle_dir)
    _write_bundle_readme(bundle_dir=bundle_dir, run_name=run_name)

    return CompiledRun(
        bundle_dir=bundle_dir,
        run_name=run_name,
        run_type=run_type,
        harbor_run_rel=(Path("harbor") / run_name).as_posix(),
    )


def _write_trial_indexes(
    *,
    bundle_dir: Path,
    run_name: str,
    run_type: str,
    run_dir: Path,
    harbor_run_dir: Path,
) -> None:
    """Write JSONL + CSV indexes for trials in the compiled bundle."""
    trials_jsonl_path = bundle_dir / "index" / "trials.jsonl"
    trials_csv_path = bundle_dir / "index" / "trials.csv"
    job_result_path = bundle_dir / "index" / "job_result.json"

    if run_type == "job":
        original_job_result = _safe_read_json(run_dir / "result.json")
        if original_job_result is not None:
            job_result_path.write_text(json.dumps(original_job_result, indent=2))

    def rel_in_bundle(path: Path) -> str:
        return path.relative_to(bundle_dir).as_posix()

    trials: list[dict[str, Any]] = []
    for trial_src_dir in _iter_trial_dirs(run_dir):
        trial_name = trial_src_dir.name
        trial_dst_dir = (
            harbor_run_dir if run_type == "trial" else harbor_run_dir / trial_name
        )
        trial_result = _safe_read_json(trial_src_dir / "result.json") or {}

        verifier_details_path = trial_dst_dir / "verifier" / "details.json"
        verifier_details = _safe_read_json(verifier_details_path)

        agent_logs: list[str] = []
        agent_dir = trial_dst_dir / "agent"
        if agent_dir.exists():
            for txt in sorted(agent_dir.glob("*.txt")):
                agent_logs.append(rel_in_bundle(txt))

        trial_entry = {
            "trial_name": trial_result.get("trial_name") or trial_name,
            "task_name": trial_result.get("task_name"),
            "agent_name": ((trial_result.get("agent_info") or {}).get("name")),
            "model_name": (
                ((trial_result.get("agent_info") or {}).get("model_info") or {}).get(
                    "name"
                )
            ),
            "reward": (
                (trial_result.get("verifier_result") or {}).get("rewards") or {}
            ).get("reward"),
            "exception_type": ((trial_result.get("exception_info") or {}) or {}).get(
                "type"
            ),
            "exception_message": ((trial_result.get("exception_info") or {}) or {}).get(
                "message"
            ),
            "started_at": trial_result.get("started_at"),
            "finished_at": trial_result.get("finished_at"),
            "verifier_details": verifier_details,
            "paths": {
                "trial_dir": rel_in_bundle(trial_dst_dir),
                "trial_result_json": rel_in_bundle(trial_dst_dir / "result.json"),
                "trial_config_json": rel_in_bundle(trial_dst_dir / "config.json"),
                "trial_log": rel_in_bundle(trial_dst_dir / "trial.log")
                if (trial_dst_dir / "trial.log").exists()
                else None,
                "verifier_reward": rel_in_bundle(
                    trial_dst_dir / "verifier" / "reward.txt"
                )
                if (trial_dst_dir / "verifier" / "reward.txt").exists()
                else None,
                "verifier_details": rel_in_bundle(verifier_details_path)
                if verifier_details_path.exists()
                else None,
                "agent_logs": agent_logs,
            },
        }
        trials.append(trial_entry)

    with trials_jsonl_path.open("w") as f:
        for row in trials:
            json.dump(row, f)
            f.write("\n")

    with trials_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_name",
                "task_name",
                "agent_name",
                "model_name",
                "reward",
                "exception_type",
                "exception_message",
                "started_at",
                "finished_at",
            ],
        )
        writer.writeheader()
        for row in trials:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})


def _write_file_manifest(*, bundle_dir: Path) -> None:
    """Write a sha256 manifest for every file in the bundle."""
    manifest_path = bundle_dir / "index" / "files.jsonl"

    def rel(path: Path) -> str:
        return path.relative_to(bundle_dir).as_posix()

    with manifest_path.open("w") as f:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.name == ".DS_Store":
                continue
            stat = path.stat()
            json.dump(
                {
                    "path": rel(path),
                    "bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    .replace(microsecond=0)
                    .isoformat(),
                    "sha256": _sha256(path),
                },
                f,
            )
            f.write("\n")


def _write_bundle_readme(*, bundle_dir: Path, run_name: str) -> None:
    """Write a README explaining the bundle layout."""
    readme = f"""\
# Harbor run bundle: `{run_name}`

This folder is a export of a Harbor run directory plus a few index files.

## Layout

- `bundle.json`: compile metadata (source path, git info, timestamps)
- `harbor/{run_name}/`: full copied Harbor run directory (agent logs, verifier outputs, configs, etc.)
- `index/trials.jsonl`: one JSON object per trial with convenient pointers into `harbor/...`
- `index/trials.csv`: table view of `index/trials.jsonl`
- `index/files.jsonl`: sha256 manifest of every file in this bundle

## Quick query example (Python)

```python
from pathlib import Path
import json

run_dir = Path(".")  # this bundle directory
trials = [json.loads(line) for line in (run_dir / "index" / "trials.jsonl").read_text().splitlines()]
failures = [t for t in trials if (t.get("reward") or 0.0) < 1.0 or t.get("exception_type")]

print("n_trials:", len(trials))
print("n_failures:", len(failures))
if failures:
    t = failures[0]
    print("first failure trial:", t["trial_name"])
    print("agent logs:", t["paths"]["agent_logs"])
```
"""
    (bundle_dir / "README.md").write_text(readme)


def _infer_hf_token() -> str | None:
    """Return an HF auth token from common environment variables."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


def _ensure_hf_repo(
    api: HfApi, *, repo_id: str, repo_type: str, private: bool | None
) -> None:
    """Create or re-use the HF repo for run bundles."""
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )


def _maybe_upload_root_readme(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    force: bool,
    token: str | None,
) -> None:
    """Ensure the HF repo root has a README describing the run bundles."""
    try:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    except Exception:
        files = set()

    if "README.md" in files and not force:
        return

    readme = """\
# Harbor run artifacts

This dataset repository stores Harbor job/trial artifacts for the `src/harbor-task-gen` benchmark.

## Layout

- `runs/<run-name>/`: one compiled run bundle per Harbor execution
  - `bundle.json`: compile metadata
  - `harbor/<run-name>/`: full copied Harbor run directory (agent logs, verifier outputs, configs, etc.)
  - `index/trials.jsonl`: one JSON object per trial (easy to query)
  - `index/files.jsonl`: sha256 manifest of every uploaded file

## Query example (Python)

```python
from huggingface_hub import snapshot_download
from pathlib import Path
import json

repo_id = "YOUR_ORG/sci-llm-harbor-runs"
local = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))

# Pick a run (replace with the run folder you want)
run = local / "runs" / "<run-name>"

trials = [json.loads(line) for line in (run / "index" / "trials.jsonl").read_text().splitlines()]
print("n_trials:", len(trials))
print("mean_reward:", sum((t.get("reward") or 0.0) for t in trials) / max(len(trials), 1))

# Inspect one trial's raw Harbor outputs
t = trials[0]
trial_dir = run / t["paths"]["trial_dir"]
print("trial_dir:", trial_dir)
print("agent logs:", t["paths"]["agent_logs"])
```
"""
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        commit_message="Add/update root README for Harbor runs",
        token=token,
    )


def _upload_bundle_to_hf(
    *,
    bundle_dir: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str | None,
    private: bool | None,
    write_root_readme: bool,
    force_root_readme: bool,
    token: str | None,
) -> None:
    """Upload a compiled run bundle to a Hugging Face repo."""
    api = HfApi(token=token)
    _ensure_hf_repo(
        api,
        repo_id=repo_id,
        repo_type=repo_type,
        private=True if private is None else private,
    )

    if write_root_readme or force_root_readme:
        _maybe_upload_root_readme(
            api,
            repo_id=repo_id,
            repo_type=repo_type,
            force=bool(force_root_readme),
            token=token,
        )

    run_name = bundle_dir.name
    target_path = path_in_repo
    if target_path is None:
        target_path = f"runs/{run_name}"

    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(bundle_dir),
        path_in_repo=target_path if target_path != "" else None,
        commit_message=f"Upload Harbor run bundle: {run_name}",
        token=token,
    )


def compile_run_cli(argv: list[str]) -> int:
    """CLI entrypoint for compiling Harbor runs."""
    parser = argparse.ArgumentParser(
        description="Compile a Harbor job/trial run directory into a portable bundle."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root that contains jobs/ and trials/ (default: examples/harbor-workspace).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a Harbor run directory (jobs/<name> or trials/<name>). If omitted, picks the latest run under jobs/ or trials/.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write bundled run folders (default: <workspace>/out/harbor-runs).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override the output bundle folder name (default: basename of --run-dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output bundle directory if it already exists.",
    )
    args = parser.parse_args(argv)
    workspace = _resolve_workspace_root(args.workspace)
    out_dir = Path(args.out_dir) if args.out_dir else workspace / "out" / "harbor-runs"

    jobs_dir = workspace / "jobs"
    trials_dir = workspace / "trials"

    if args.run_dir is None:
        run_dir = _find_latest_run_dir_in(jobs_dir, trials_dir)
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (workspace / run_dir).resolve()

    compiled = compile_run(
        run_dir=run_dir,
        out_dir=out_dir,
        name=args.name,
        force=bool(args.force),
    )

    print(compiled.bundle_dir)
    return 0


def push_run_to_hf_cli(argv: list[str]) -> int:
    """CLI entrypoint for compiling + uploading a run bundle to HF."""
    parser = argparse.ArgumentParser(
        description="Compile and upload Harbor run artifacts to the Hugging Face Hub."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root that contains jobs/ and trials/ (default: examples/harbor-workspace).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, e.g. ORG/sci-llm-harbor-runs",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type (default: dataset).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Harbor run directory (jobs/<name> or trials/<name>). If omitted, picks latest.",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Already-compiled bundle directory. If provided, skips compilation.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write compiled bundles (default: <workspace>/out/harbor-runs).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination path inside the HF repo (default: runs/<run-name>). Use '' to upload to repo root.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public if it does not exist.",
    )
    parser.add_argument(
        "--write-root-readme",
        action="store_true",
        help="Create README.md at repo root if missing (or overwrite with --force-root-readme).",
    )
    parser.add_argument(
        "--force-root-readme",
        action="store_true",
        help="Overwrite README.md at repo root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local bundle directory if it already exists.",
    )
    args = parser.parse_args(argv)
    workspace = _resolve_workspace_root(args.workspace)
    out_dir = Path(args.out_dir) if args.out_dir else workspace / "out" / "harbor-runs"

    if args.private and args.public:
        raise SystemExit("Pass at most one of --private/--public.")

    private: bool | None
    if args.private:
        private = True
    elif args.public:
        private = False
    else:
        private = True

    token = _infer_hf_token()

    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir)
        if not bundle_dir.is_absolute():
            bundle_dir = workspace / bundle_dir
        bundle_dir = bundle_dir.resolve()
        if not bundle_dir.exists():
            raise FileNotFoundError(f"--bundle-dir not found: {bundle_dir}")
    else:
        if args.run_dir:
            run_dir = Path(args.run_dir)
            if not run_dir.is_absolute():
                run_dir = workspace / run_dir
            run_dir = run_dir.resolve()
        else:
            run_dir = _find_latest_run_dir_in(workspace / "jobs", workspace / "trials")
        compiled = compile_run(
            run_dir=run_dir,
            out_dir=out_dir,
            name=None,
            force=bool(args.force),
        )
        bundle_dir = compiled.bundle_dir

    _upload_bundle_to_hf(
        bundle_dir=bundle_dir,
        repo_id=str(args.repo_id),
        repo_type=str(args.repo_type),
        path_in_repo=args.path_in_repo,
        private=private,
        write_root_readme=bool(args.write_root_readme),
        force_root_readme=bool(args.force_root_readme),
        token=token,
    )

    print(
        f"Uploaded {bundle_dir} -> {args.repo_id}:{args.path_in_repo or f'runs/{bundle_dir.name}'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
