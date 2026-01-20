r"""Run Harbor with secrets loaded from the repo's `.env`.

Harbor agents (including `gemini-cli`) read API keys from environment variables in the
*host* process and forward them into the container. `uv run harbor ...` does not load
`.env` automatically, so Gemini runs can fail even when you have `GOOGLE_API_KEY=...`
in `.env`.

This helper:
1) Loads `.env` from the repository root.
2) Populates `os.environ`.
3) Sets `GEMINI_API_KEY` from `GOOGLE_API_KEY` when needed (the Gemini CLI expects
   `GEMINI_API_KEY`).
4) Delegates to Harbor's CLI entrypoint.

Extras bundled into this wrapper:
- HF tasks support: `--hf-tasks-repo ...` rewrites Harbor args to pull tasks from HF.
- Modal convenience: `--modal` forces `--env modal` and enables cleanup defaults.

Example:
    uv run python src/harbor-task-gen/run_harbor.py trials start \\
      -p out/harbor/supercon-mini-v2/ground-template/tasks/jac2980051 \\
      -a gemini-cli -m gemini/gemini-2.5-flash

"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import asyncio

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download


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


def _extract_batch_args(
    argv: list[str],
) -> tuple[list[str], int, int | None, int | None, bool]:
    """Strip --batch-size, --batch-number, --seed, and --force from argv and return them separately."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-number", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    return remaining, args.batch_size, args.batch_number, args.seed, args.force


def _get_registry_path_from_argv(argv: list[str]) -> Path | None:
    """Extract --registry-path value from argv."""
    for idx, arg in enumerate(argv):
        if arg == "--registry-path" and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if arg.startswith("--registry-path="):
            return Path(arg.split("=", 1)[1])
    return None


def _replace_registry_path_in_argv(argv: list[str], new_path: str) -> list[str]:
    """Replace --registry-path value in argv with a new path."""
    result: list[str] = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--registry-path":
            result.append(arg)
            result.append(new_path)
            skip_next = True
            continue
        if arg.startswith("--registry-path="):
            result.append(f"--registry-path={new_path}")
            continue
        result.append(arg)
    return result


def _compute_total_batches(registry_data: list[dict[str, Any]], batch_size: int) -> int:
    """Compute total number of batches for a registry."""
    all_tasks = _collect_all_tasks(registry_data)
    total_tasks = len(all_tasks)
    return (total_tasks + batch_size - 1) // batch_size  # ceil division


def _shuffle_registry_tasks(
    registry_data: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Return a new registry with tasks shuffled using the given seed.

    The shuffle is deterministic: same seed always produces same order.
    """
    all_tasks = _collect_all_tasks(registry_data)

    # Shuffle with deterministic seed
    rng = random.Random(seed)
    rng.shuffle(all_tasks)

    # Rebuild registry structure with shuffled task order
    # We flatten all tasks into a single dataset to preserve shuffle order
    if not all_tasks:
        return registry_data

    # Use metadata from first dataset
    first_dataset = registry_data[0] if registry_data else {}
    shuffled_registry: list[dict[str, Any]] = [
        {
            "name": first_dataset.get("name", ""),
            "version": first_dataset.get("version", ""),
            "description": first_dataset.get("description", ""),
            "tasks": [task for _, task in all_tasks],
        }
    ]
    return shuffled_registry


def _get_agent_model_from_argv(argv: list[str]) -> tuple[str | None, str | None]:
    """Extract agent name and model name from command line arguments."""
    agent_name: str | None = None
    model_name: str | None = None

    for idx, arg in enumerate(argv):
        # Agent: -a or --agent
        if arg in {"-a", "--agent"} and idx + 1 < len(argv):
            agent_name = argv[idx + 1]
        elif arg.startswith("--agent="):
            agent_name = arg.split("=", 1)[1]

        # Model: -m or --model or --model-name
        if arg in {"-m", "--model", "--model-name"} and idx + 1 < len(argv):
            model_name = argv[idx + 1]
        elif arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
        elif arg.startswith("--model-name="):
            model_name = arg.split("=", 1)[1]

    return agent_name, model_name


def _get_batch_job_name(
    argv: list[str], batch_size: int, batch_number: int, seed: int | None
) -> str:
    """Generate the job name that would be used for a batch.

    This mirrors the logic in _apply_batching_to_argv to determine
    if a batch has already been processed.
    """
    agent_name, model_name = _get_agent_model_from_argv(argv)
    agent_part = agent_name or "agent"
    # Clean model name: remove provider prefix and special chars
    if model_name:
        model_part = model_name.split("/")[-1].replace(":", "-")
    else:
        model_part = "model"
    seed_part = f"-s{seed}" if seed is not None else ""
    return f"bn{batch_number}-bs{batch_size}-{agent_part}-{model_part}{seed_part}"


def _batch_already_processed(
    workspace: Path,
    argv: list[str],
    batch_size: int,
    batch_number: int,
    seed: int | None,
) -> Path | None:
    """Check if a batch has already been processed by looking for the job directory."""
    job_name = _get_batch_job_name(argv, batch_size, batch_number, seed)

    # Determine where jobs are stored based on command type
    run_roots = _run_roots_from_args(argv, workspace=workspace)

    # Check if a job directory with this name exists in any of the run roots
    for root in run_roots:
        if not root.exists():
            continue
        job_dir = root / job_name
        if job_dir.exists():
            return job_dir

    return None


def _apply_batching_to_argv(
    argv: list[str],
    batch_size: int,
    batch_number: int,
    workspace: Path,
    seed: int | None = None,
    shuffled_registry_data: list[dict[str, Any]] | None = None,
) -> tuple[list[str], int]:
    """Apply batching to registry path in argv.

    Args:
        argv: Command line arguments
        batch_size: Number of tasks per batch
        batch_number: 1-indexed batch number
        workspace: Workspace directory
        seed: Random seed (used in filename for cache identification)
        shuffled_registry_data: Pre-shuffled registry data (if None, reads from file)

    Returns:
        Tuple of (modified_argv, total_batches)

    """
    registry_path = _get_registry_path_from_argv(argv)
    if registry_path is None:
        raise SystemExit(
            "Batching requires --registry-path. Use --hf-tasks-repo or provide --registry-path directly."
        )

    if not registry_path.is_absolute():
        registry_path = (workspace / registry_path).resolve()

    if shuffled_registry_data is not None:
        registry_data = shuffled_registry_data
    else:
        if not registry_path.exists():
            raise SystemExit(f"Registry file not found: {registry_path}")
        registry_data = json.loads(registry_path.read_text())
        if not isinstance(registry_data, list):
            raise SystemExit(f"Invalid registry format in {registry_path}")

    # Generate batch registry filename (include seed in name if shuffled)
    cache_dir = workspace / "out" / "harbor" / "registry-cache"
    base_name = registry_path.stem.replace("__registry", "")
    seed_suffix = f"__seed{seed}" if seed is not None else ""
    batch_registry_path = (
        cache_dir / f"{base_name}{seed_suffix}__batch{batch_number}__registry.json"
    )

    _, total_batches = _create_batch_registry(
        registry_data=registry_data,
        batch_size=batch_size,
        batch_number=batch_number,
        output_path=batch_registry_path,
    )

    new_argv = _replace_registry_path_in_argv(argv, str(batch_registry_path))

    # Set job name: bn{batch_number}-bs{batch_size}-{agent_name}-{model_name}{seed_suffix}
    agent_name, model_name = _get_agent_model_from_argv(argv)
    agent_part = agent_name or "agent"
    # Clean model name: remove provider prefix and special chars
    if model_name:
        model_part = model_name.split("/")[-1].replace(":", "-")
    else:
        model_part = "model"
    seed_part = f"-s{seed}" if seed is not None else ""
    job_name = f"bn{batch_number}-bs{batch_size}-{agent_part}-{model_part}{seed_part}"

    # Add --job-name argument for the job name (always use the auto-generated name)
    new_argv = [*new_argv, "--job-name", job_name]

    return new_argv, total_batches


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
            "Usage: run_harbor.py [--workspace PATH] [--batch-size N] [--batch-number N] [--seed N] [--force] <harbor args...>",
            file=sys.stderr,
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
            "Batching: --batch-size (default: 10), --batch-number (1-indexed, run all if omitted), --seed (shuffle tasks), --force (reprocess batches).",
            file=sys.stderr,
        )
        return 2

    argv, workspace, workspace_explicit = _extract_workspace_arg(argv)
    if not argv:
        print(
            "Usage: run_harbor.py [--workspace PATH] [--batch-size N] [--batch-number N] [--seed N] [--force] <harbor args...>",
            file=sys.stderr,
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
            "Batching: --batch-size (default: 10), --batch-number (1-indexed, run all if omitted), --seed (shuffle tasks), --force (reprocess batches).",
            file=sys.stderr,
        )
        return 2

    if workspace.exists() and not workspace.is_dir():
        raise SystemExit(f"--workspace must be a directory: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    argv, hf_args = _extract_hf_args(argv)
    argv, modal_requested = _extract_modal_args(argv)
    argv, agent_setup_override = _extract_agent_setup_timeout(argv)
    argv, batch_size, batch_number, seed, force = _extract_batch_args(argv)

    load_dotenv()

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

    argv = _rewrite_override_storage(argv)
    argv = _apply_modal_defaults(argv, modal_requested)
    argv = _rewrite_env_flag(argv)
    argv = _apply_hf_args(argv, hf_args, workspace=workspace)
    modal_active = _detect_environment_type(argv) == "modal"

    env = os.environ.copy()

    def _run_single_harbor(argv_for_run: list[str]) -> int:
        """Run a single Harbor invocation."""
        if modal_active:
            timeout_raw = env.get(
                "HARBOR_MODAL_LOG_DOWNLOAD_TIMEOUT_SEC", "300"
            ).strip()
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
            return _run_harbor_cli_in_process(argv_for_run, workspace=workspace)
        else:
            command = [sys.executable, "-m", "harbor.cli.main", *argv_for_run]
            result = subprocess.run(command, check=False, cwd=workspace, env=env)
            return int(result.returncode)

    if batch_size is not None:
        # Determine total batches by reading the registry
        registry_path = _get_registry_path_from_argv(argv)
        if registry_path and not registry_path.is_absolute():
            registry_path = (workspace / registry_path).resolve()

        registry_data: list[dict[str, Any]] | None = None
        if registry_path and registry_path.exists():
            registry_data = json.loads(registry_path.read_text())

            # Shuffle tasks once if seed is provided (before any batching)
            if seed is not None:
                print(f"Shuffling tasks with seed {seed}...", file=sys.stderr)
                registry_data = _shuffle_registry_tasks(registry_data, seed)

            total_batches = _compute_total_batches(registry_data, batch_size)
        else:
            total_batches = 1

        # Determine which batches to run
        if batch_number is not None:
            batch_indices = [batch_number]
        else:
            batch_indices = list(range(1, total_batches + 1))

        print(
            f"Batching {total_batches} batch(es) of up to {batch_size} tasks each.",
            file=sys.stderr,
        )

        exit_code = 0
        for bn in batch_indices:
            # Check if batch has already been processed (unless --force is set)
            if job_dir := _batch_already_processed(
                workspace, argv, batch_size, bn, seed
            ):
                if force:
                    print(
                        f"\n=== Reprocessing batch {bn}/{total_batches} (deleting existing: {job_dir}) ===",
                        file=sys.stderr,
                    )
                    shutil.rmtree(job_dir)
                else:
                    print(
                        f"\n=== Skipping batch {bn}/{total_batches} (already processed: {job_dir}) ===",
                        file=sys.stderr,
                    )
                    continue

            print(
                f"\n=== Running batch {bn}/{total_batches} ===",
                file=sys.stderr,
            )
            batch_argv, _ = _apply_batching_to_argv(
                argv,
                batch_size,
                bn,
                workspace,
                seed=seed,
                shuffled_registry_data=registry_data,
            )
            exit_code = _run_single_harbor(batch_argv)
    else:
        exit_code = _run_single_harbor(argv)

    return exit_code


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


def _get_jobs_dir_from_argv(argv: list[str]) -> str | None:
    """Extract --jobs-dir value from argv if provided."""
    for idx, arg in enumerate(argv):
        if arg == "--jobs-dir" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--jobs-dir="):
            return arg.split("=", 1)[1]
    return None


def _run_roots_from_args(argv: list[str], *, workspace: Path) -> list[Path]:
    """Infer which Harbor output roots (jobs/trials) a command will use."""
    if "trials" in argv and "jobs" not in argv:
        return [workspace / "trials"]

    # Check for custom jobs directory
    jobs_dir_arg = _get_jobs_dir_from_argv(argv)
    if "jobs" in argv or "run" in argv:
        if jobs_dir_arg:
            jobs_dir = Path(jobs_dir_arg)
            if not jobs_dir.is_absolute():
                jobs_dir = workspace / jobs_dir
            return [jobs_dir]
        return [workspace / "jobs"]

    # Default case
    if jobs_dir_arg:
        jobs_dir = Path(jobs_dir_arg)
        if not jobs_dir.is_absolute():
            jobs_dir = workspace / jobs_dir
        return [jobs_dir, workspace / "trials"]
    return [workspace / "jobs", workspace / "trials"]


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


def _collect_all_tasks(registry_data: list[dict[str, Any]]) -> list[tuple[int, dict]]:
    """Collect all tasks from registry with their dataset index."""
    all_tasks: list[tuple[int, dict]] = []
    for dataset_idx, dataset in enumerate(registry_data):
        tasks = dataset.get("tasks") if isinstance(dataset, dict) else None
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if isinstance(task, dict):
                all_tasks.append((dataset_idx, task))
    return all_tasks


def _create_batch_registry(
    registry_data: list[dict[str, Any]],
    batch_size: int,
    batch_number: int,
    output_path: Path,
) -> tuple[Path, int]:
    """Create a registry with only the tasks in the specified batch.

    Args:
        registry_data: Full registry data (list of dataset dicts)
        batch_size: Number of tasks per batch
        batch_number: 1-indexed batch number
        output_path: Path to write the batch registry

    Returns:
        Tuple of (output_path, total_batches)

    """
    all_tasks = _collect_all_tasks(registry_data)
    total_tasks = len(all_tasks)
    total_batches = (total_tasks + batch_size - 1) // batch_size  # ceil division

    if batch_number < 1 or batch_number > total_batches:
        raise SystemExit(
            f"--batch-number {batch_number} out of range (1-{total_batches} for {total_tasks} tasks)."
        )

    # Get tasks for this batch
    start_idx = (batch_number - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_tasks)
    batch_tasks = all_tasks[start_idx:end_idx]

    # Build new registry with only batch tasks, preserving dataset structure
    # Group tasks by their original dataset index
    tasks_by_dataset: dict[int, list[dict]] = {}
    for dataset_idx, task in batch_tasks:
        if dataset_idx not in tasks_by_dataset:
            tasks_by_dataset[dataset_idx] = []
        tasks_by_dataset[dataset_idx].append(task)

    # Create new registry with only datasets that have tasks in this batch
    batch_registry: list[dict[str, Any]] = []
    for dataset_idx, tasks in sorted(tasks_by_dataset.items()):
        original_dataset = registry_data[dataset_idx]
        batch_dataset = {
            "name": original_dataset.get("name", ""),
            "version": original_dataset.get("version", ""),
            "description": original_dataset.get("description", ""),
            "tasks": tasks,
        }
        batch_registry.append(batch_dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(batch_registry, indent=2))

    return output_path, total_batches


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


def _infer_hf_token() -> str | None:
    """Return an HF auth token from common environment variables."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


if __name__ == "__main__":
    raise SystemExit(main())
