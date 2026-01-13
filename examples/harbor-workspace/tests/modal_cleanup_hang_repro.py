r"""Reproduce Modal download_dir hangs / connection errors under Harbor.

This harness builds a tiny Harbor task set, then runs a Harbor Job in-process
using the Modal environment. It injects a fault into
`ModalEnvironment.download_dir()` to simulate stale gRPC connections (or a
hard hang) and demonstrates how the job runner can freeze during cleanup.

Usage (hang repro, small + fast):
  uv run python examples/harbor-workspace/tests/modal_cleanup_hang_repro.py \\
    --n-tasks 4 --n-concurrent 2 --sleep-sec 10 --timeout-sec 3 \\
    --fault-mode hang --hang-sec 120 --max-runtime-sec 30

Usage (error repro):
  uv run python examples/harbor-workspace/tests/modal_cleanup_hang_repro.py \\
    --fault-mode stream

This script loads `.env` from the repo root to access MODAL_TOKEN_* values.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

from grpclib.exceptions import StreamTerminatedError
from typing import Any

from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig, LocalDatasetConfig, OrchestratorConfig
from harbor.models.orchestrator_type import OrchestratorType
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, VerifierConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
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
        if key and key not in os.environ:
            os.environ[key] = value


def _write_task_files(task_dir: Path, *, sleep_sec: int, timeout_sec: int) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"
    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    task_toml = f"""\
version = "1.0"

[metadata]
author_name = "harbor-task-gen"
author_email = "n/a"
difficulty = "debug"
category = "diagnostic"
tags = ["modal", "cleanup", "hang"]

[verifier]
timeout_sec = {timeout_sec}

[agent]
timeout_sec = {timeout_sec}

[environment]
cpus = 1
memory_mb = 256
storage_mb = 512
"""
    (task_dir / "task.toml").write_text(task_toml)
    (task_dir / "instruction.md").write_text(
        "Repro task: oracle sleeps longer than timeout to trigger cleanup."
    )

    (env_dir / "task_meta.json").write_text(
        json.dumps({"task_id": task_dir.name, "sleep_sec": sleep_sec}, indent=2)
    )

    dockerfile = """\
FROM python:3.13-slim
WORKDIR /app
RUN mkdir -p /logs/agent /logs/verifier /app/output
COPY task_meta.json /app/task_meta.json
"""
    (env_dir / "Dockerfile").write_text(dockerfile)

    solve_sh = f"""\
#!/bin/bash
set -euo pipefail
sleep {sleep_sec}
mkdir -p /app/output
printf '[]' > /app/output/predictions.json
"""
    solve_path = solution_dir / "solve.sh"
    solve_path.write_text(solve_sh)
    solve_path.chmod(0o755)

    check_prediction = '''\
"""Write a reward file if predictions.json exists."""
from pathlib import Path
import json

def main() -> None:
    output = Path("/app/output/predictions.json")
    reward = 1.0 if output.exists() else 0.0
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier") / "reward.txt").write_text(str(reward))
    details = {"reward": reward, "predictions_path": str(output)}
    (Path("/logs/verifier") / "details.json").write_text(json.dumps(details))

if __name__ == "__main__":
    main()
'''
    (tests_dir / "check_prediction.py").write_text(check_prediction)

    test_sh = """\
#!/bin/bash
set -euo pipefail
python /tests/check_prediction.py 2>&1 | tee /logs/verifier/log.txt
"""
    test_path = tests_dir / "test.sh"
    test_path.write_text(test_sh)
    test_path.chmod(0o755)


def _inject_fault(fault_mode: str, hang_sec: int) -> None:
    from harbor.environments.modal import ModalEnvironment

    if fault_mode == "none":
        return

    original = ModalEnvironment.download_dir
    state = {"used": False}

    async def patched(self: Any, source_dir: str, target_dir: Path) -> None:
        if not state["used"]:
            state["used"] = True
            if fault_mode == "hang":
                await asyncio.sleep(hang_sec)
            elif fault_mode == "stream":
                raise StreamTerminatedError()
            elif fault_mode == "attr":
                raise AttributeError("Connection object has no attribute '_transport'")
        await original(self, source_dir, target_dir)

    ModalEnvironment.download_dir = patched  # type: ignore[assignment]


async def _run_job(
    *,
    tasks_dir: Path,
    workspace: Path,
    n_concurrent: int,
    max_runtime_sec: int,
) -> int:
    config = JobConfig(
        jobs_dir=workspace / "jobs",
        orchestrator=OrchestratorConfig(
            type=OrchestratorType.LOCAL, n_concurrent_trials=n_concurrent
        ),
        environment=EnvironmentConfig(type=EnvironmentType.MODAL, delete=True),
        agents=[AgentConfig(name="oracle")],
        verifier=VerifierConfig(disable=False),
        datasets=[LocalDatasetConfig(path=tasks_dir)],
    )

    job = Job(config)
    try:
        await asyncio.wait_for(job.run(), timeout=max_runtime_sec)
        return 0
    except asyncio.TimeoutError:
        return 2


def main() -> int:
    """Run the Modal cleanup hang repro with optional fault injection."""
    parser = argparse.ArgumentParser(
        description="Reproduce Modal download_dir hangs via fault injection."
    )
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--n-tasks", type=int, default=4)
    parser.add_argument("--n-concurrent", type=int, default=2)
    parser.add_argument("--sleep-sec", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=3)
    parser.add_argument(
        "--fault-mode",
        choices=["none", "hang", "stream", "attr"],
        default="hang",
        help="Fault to inject into Modal download_dir.",
    )
    parser.add_argument("--hang-sec", type=int, default=120)
    parser.add_argument("--max-runtime-sec", type=int, default=30)
    args = parser.parse_args()

    _load_dotenv(_repo_root() / ".env")

    workspace = (args.workspace or _default_workspace_root()).resolve()
    if not workspace.exists():
        raise SystemExit(f"Workspace not found: {workspace}")

    repro_root = workspace / "out" / "harbor" / "modal-download-dir-repro"
    tasks_dir = repro_root / "tasks"

    if repro_root.exists():
        shutil.rmtree(repro_root)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.n_tasks):
        _write_task_files(
            tasks_dir / f"hang-{idx:03d}",
            sleep_sec=args.sleep_sec,
            timeout_sec=args.timeout_sec,
        )

    _inject_fault(args.fault_mode, args.hang_sec)

    start = time.time()
    result = asyncio.run(
        _run_job(
            tasks_dir=tasks_dir,
            workspace=workspace,
            n_concurrent=args.n_concurrent,
            max_runtime_sec=args.max_runtime_sec,
        )
    )
    elapsed = time.time() - start

    log_path = repro_root / "repro.log"
    log_path.write_text(
        json.dumps(
            {
                "fault_mode": args.fault_mode,
                "hang_sec": args.hang_sec,
                "elapsed_sec": round(elapsed, 2),
                "max_runtime_sec": args.max_runtime_sec,
                "result_code": result,
                "tasks_dir": str(tasks_dir),
            },
            indent=2,
        )
    )

    if result == 2:
        print(
            f"Repro timed out after {args.max_runtime_sec}s (likely hang).",
            file=sys.stderr,
        )
    else:
        print(f"Repro finished in {elapsed:.1f}s with code {result}.")

    print(f"Log: {log_path}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
