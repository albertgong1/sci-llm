"""Compile and upload Harbor run artifacts to a Hugging Face dataset repository.

This is intended to be run *after* a Harbor job/trial completes.

Example:
  uv run python src/pbench_containerized_eval/push_harbor_run_to_hf.py \
    --repo-id YOUR_ORG/sci-llm-harbor-runs \
    --run-dir jobs/2025-12-18__15-40-56

"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

from huggingface_hub import HfApi

import compile_harbor_run


def _infer_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


def _ensure_repo(
    api: HfApi, *, repo_id: str, repo_type: str, private: bool | None
) -> None:
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
    try:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    except Exception:
        files = set()

    if "README.md" in files and not force:
        return

    readme = """\
# Harbor run artifacts

This dataset repository stores Harbor job/trial artifacts for the `src/pbench_containerized_eval` benchmark.

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


def main() -> int:
    """Docstring for main

    :return: Description
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Compile and upload Harbor run artifacts to the Hugging Face Hub."
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
        default=str(compile_harbor_run._repo_root() / "out" / "harbor-runs"),
        help="Where to write compiled bundles (default: out/harbor-runs).",
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
    args = parser.parse_args()

    if args.private and args.public:
        raise SystemExit("Pass at most one of --private/--public.")
    private: bool | None
    if args.private:
        private = True
    elif args.public:
        private = False
    else:
        private = True  # safer default for logs/artifacts

    token = _infer_token()
    api = HfApi(token=token)
    _ensure_repo(
        api, repo_id=str(args.repo_id), repo_type=str(args.repo_type), private=private
    )

    if args.write_root_readme or args.force_root_readme:
        _maybe_upload_root_readme(
            api,
            repo_id=str(args.repo_id),
            repo_type=str(args.repo_type),
            force=bool(args.force_root_readme),
            token=token,
        )

    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir)
        if not bundle_dir.is_absolute():
            bundle_dir = compile_harbor_run._repo_root() / bundle_dir
        bundle_dir = bundle_dir.resolve()
        if not bundle_dir.exists():
            raise FileNotFoundError(f"--bundle-dir not found: {bundle_dir}")
        run_name = bundle_dir.name
    else:
        run_dir = None
        if args.run_dir:
            run_dir = Path(args.run_dir)
            if not run_dir.is_absolute():
                run_dir = compile_harbor_run._repo_root() / run_dir
            run_dir = run_dir.resolve()
        compiled = compile_harbor_run.compile_run(
            run_dir=run_dir
            if run_dir is not None
            else compile_harbor_run._find_latest_run_dir(
                jobs_dir=compile_harbor_run._repo_root() / "jobs",
                trials_dir=compile_harbor_run._repo_root() / "trials",
            ),
            out_dir=Path(args.out_dir),
            name=None,
            force=bool(args.force),
        )
        bundle_dir = compiled.bundle_dir
        run_name = compiled.run_name

    path_in_repo = args.path_in_repo
    if path_in_repo is None:
        path_in_repo = f"runs/{run_name}"

    api.upload_folder(
        repo_id=str(args.repo_id),
        repo_type=str(args.repo_type),
        folder_path=str(bundle_dir),
        path_in_repo=path_in_repo if path_in_repo != "" else None,
        commit_message=f"Upload Harbor run bundle: {run_name}",
        token=token,
    )

    print(f"Uploaded {bundle_dir} -> {args.repo_id}:{path_in_repo or '/'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
