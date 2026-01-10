"""Bundle a Harbor job/trial directory into a Hugging Face-friendly folder.

Typical usage:
  uv run python src/pbench_containerized_eval/compile_harbor_run.py \
    --run-dir jobs/2025-12-18__15-40-56

Outputs:
  out/harbor-runs/<run-name>/
    bundle.json
    harbor/<run-name>/...   # full copy of the Harbor run directory
    index/trials.jsonl      # 1 JSON object per trial
    index/trials.csv        # shallow table view
    index/files.jsonl       # file manifest (size + sha256)
    README.md               # "how to read this run" notes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except Exception:
        return None


def _safe_read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _run_git(args: list[str]) -> str | None:
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
    head = _run_git(["rev-parse", "HEAD"])
    is_dirty = bool(_run_git(["status", "--porcelain"]))
    origin = _run_git(["remote", "get-url", "origin"])
    return {"head": head, "is_dirty": is_dirty, "origin": origin}


def _find_latest_run_dir(*, jobs_dir: Path, trials_dir: Path) -> Path:
    candidates: list[Path] = []
    for root in [jobs_dir, trials_dir]:
        if not root.exists():
            continue
        candidates.extend([p for p in root.iterdir() if p.is_dir()])

    if not candidates:
        raise FileNotFoundError(
            f"No Harbor runs found under {jobs_dir} or {trials_dir}."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _looks_like_job_dir(path: Path) -> bool:
    if (path / "job.log").exists():
        return True
    result = _safe_read_json(path / "result.json")
    return isinstance(result, dict) and "n_total_trials" in result


def _iter_trial_dirs(run_dir: Path) -> Iterable[Path]:
    if _looks_like_job_dir(run_dir):
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            if (child / "result.json").exists():
                yield child
        return

    # Trial run directory: the run dir itself is the only trial.
    if (run_dir / "result.json").exists():
        yield run_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CompiledRun:
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
    readme = f"""\
# Harbor run bundle: `{run_name}`

This folder is a export of a Harbor run directory plus a few index files.

## Layout

- `bundle.json`: compile metadata (source path, git info, timestamps)
- `harbor/{run_name}/`: full copied Harbor run directory (trials, agent logs, verifier outputs)
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


def main() -> int:
    """Docstring for main

    :return: Description
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Compile a Harbor job/trial run directory into a portable bundle."
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
        default=str(_repo_root() / "out" / "harbor-runs"),
        help="Where to write bundled run folders (default: out/harbor-runs).",
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
    args = parser.parse_args()

    jobs_dir = _repo_root() / "jobs"
    trials_dir = _repo_root() / "trials"

    if args.run_dir is None:
        run_dir = _find_latest_run_dir(jobs_dir=jobs_dir, trials_dir=trials_dir)
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (_repo_root() / run_dir).resolve()

    compiled = compile_run(
        run_dir=run_dir,
        out_dir=Path(args.out_dir),
        name=args.name,
        force=bool(args.force),
    )

    print(compiled.bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
