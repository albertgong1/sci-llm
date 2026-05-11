r"""Compile Harbor tasks for property extraction from a folder of PDFs.

This "task compiler" turns a (PDF, ground-truth) dataset into Harbor task directories,
each with:
  - `environment/`: Docker build context with the paper PDF
  - `instruction.md`: a single prompt/instruction file shared across tasks via a template
  - `tests/`: verifier that scores predictions using rubric tolerances
  - `solution/`: an oracle solution used by Harbor's built-in `oracle` agent
  - `registry.json`: a local task registry for Harbor to discover and load tasks

The ground truth source is specified via --gt-hf-repo, --gt-hf-split, and optionally
--gt-hf-revision (defaults to main).

Optional: pass `--upload-hf` to upload the generated tasks to a Hugging Face repo
so Harbor can pull tasks directly from the Hub.

Example usage:

uv run python prepare_harbor_tasks.py --templates-dir targeted-stoichiometric-template --force \
    --gt-hf-repo kilian-group/supercon-extraction --gt-hf-split full --gt-hf-revision v0.0.0

"""

import argparse
import io
import json
import os
import random
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset
from harbor.models.task.paths import TaskPaths
from huggingface_hub import HfApi
from slugify import slugify
import logging

import llm_utils as _llm_utils

_LLM_UTILS_SRC = Path(_llm_utils.__file__).parent


logger = logging.getLogger(__name__)


_LBRACE_SENTINEL = "\0LBRACE\0"
_RBRACE_SENTINEL = "\0RBRACE\0"


def _format_template(template: str, values: Mapping[str, Any]) -> str:
    """Render a prompt/template with optional placeholders.

    This repository's prompt templates often include JSON examples with `{ ... }`.
    Using `str.format(...)` on such templates is fragile because unescaped braces in
    JSON will be interpreted as format placeholders.

    This renderer is intentionally conservative:
    - Only `{name}` placeholders are substituted, where `name` matches
      `[A-Za-z_][A-Za-z0-9_]*`.
    - Missing values do NOT raise; unresolved placeholders are left unchanged.
    - `{{` and `}}` are treated as escaped literal braces for compatibility with
      existing `str.format`-style templates.

    Args:
        template: Raw template string (may contain JSON/LaTeX braces).
        values: Mapping of placeholder names to values (converted to `str`).

    Returns:
        Rendered string.

    """
    protected = template.replace("{{", _LBRACE_SENTINEL).replace("}}", _RBRACE_SENTINEL)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            return match.group(0)
        value = values[key]
        if value is None:
            return ""
        return str(value)

    rendered = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, protected)
    return rendered.replace(_LBRACE_SENTINEL, "{").replace(_RBRACE_SENTINEL, "}")


def dockerfile_contents(templates_dir: Path) -> str:
    """Render the task environment Dockerfile.

    The environment always includes the PDF at `/app/paper.pdf`.
    The container includes `pdftotext` (poppler-utils) so agents can extract text
    from the PDF on their own.
    """
    install_pdf_tools = (
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "    ca-certificates \\\n"
        "    poppler-utils \\\n"
        "    procps \\\n"
        "  && rm -rf /var/lib/apt/lists/*"
    )

    return _format_template(
        (templates_dir / "environment/Dockerfile").read_text(),
        {"install_pdf_tools": install_pdf_tools},
    )


def write_local_registry(
    task_dirs: list[Path],
    registry_path: Path,
    *,
    dataset_name: str,
    dataset_version: str = "local",
    description: str = "Locally generated Harbor tasks.",
) -> None:
    """Write a local registry.json for the generated tasks.

    This registry can be used by Harbor to discover and load tasks from the local
    filesystem without requiring HuggingFace upload.
    """
    tasks = []
    for task_dir in task_dirs:
        tasks.append(
            {
                "name": task_dir.name,
                "path": task_dir.as_posix(),
            }
        )
    registry = [
        {
            "name": dataset_name,
            "version": dataset_version,
            "description": description,
            "tasks": tasks,
        }
    ]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))


def build_task(
    task_dir: Path,
    *,
    pdf_path: Path,
    refno: str,
    rows: list[dict[str, str]],
    templates_dir: Path,
) -> None:
    """Build a single Harbor task directory (one paper, many questions)."""
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"

    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pdf_path, env_dir / "paper.pdf")

    expected = {
        "refno": refno,
        "ground_truth": rows,
    }
    (tests_dir / "expected.json").write_text(json.dumps(expected, indent=2))

    task_meta = {
        "refno": refno,
        "pdf_path": "/app/paper.pdf",
        "predictions_path": "/app/predictions.json",
    }
    (env_dir / "task_meta.json").write_text(json.dumps(task_meta, indent=2))

    gemini_at_commands = "`@paper.pdf`"
    paper_at_command = "@paper.pdf"
    claude_file_examples = "`/app/paper.pdf`"

    instruction_template = (templates_dir / "instruction.md.template").read_text()
    instruction_values = {
        # Identifiers
        "task_id": task_dir.name,
        "refno": refno,
        # Standard in-container paths
        "pdf_path": "/app/paper.pdf",
        "meta_path": "/app/task_meta.json",
        "predictions_path": "/app/predictions.json",
        # Agent affordances (optional)
        "paper_at_command": paper_at_command,
        "gemini_at_commands": gemini_at_commands,
        "claude_file_examples": claude_file_examples,
    }
    instruction = _format_template(instruction_template, instruction_values)
    (task_dir / "instruction.md").write_text(textwrap.dedent(instruction))

    task_toml_template = (templates_dir / "task.toml.template").read_text()
    task_toml = _format_template(
        task_toml_template, {"task_name": task_dir.name, "refno": refno}
    )
    (task_dir / "task.toml").write_text(task_toml)

    (env_dir / "Dockerfile").write_text(dockerfile_contents(templates_dir))
    # Bundle the monorepo-local llm_utils package into the build context so the
    # verifier Dockerfile's `COPY llm_utils /opt/llm_utils` step can find it.
    llm_utils_dst = env_dir / "llm_utils"
    if llm_utils_dst.exists():
        shutil.rmtree(llm_utils_dst)
    shutil.copytree(
        _LLM_UTILS_SRC, llm_utils_dst, ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(
        templates_dir / "tests/check_prediction.py", tests_dir / "check_prediction.py"
    )
    shutil.copy2(templates_dir / "tests/test.sh", tests_dir / "test.sh")
    shutil.copy2(
        templates_dir / "tests/si_conversion_factors.csv",
        tests_dir / "si_conversion_factors.csv",
    )

    solution_script = f"""\
#!/bin/bash
set -euo pipefail

cat > /app/predictions.json <<'EOF'
{json.dumps({"properties": rows}, indent=2)}
EOF
"""
    (solution_dir / "solve.sh").write_text(solution_script)

    for script in [tests_dir / "test.sh", solution_dir / "solve.sh"]:
        script.chmod(0o755)


def main() -> None:
    """Generate Harbor tasks for the benchmark.

    This is a multi-step pipeline:
      1) Load the HF dataset (refno -> properties).
      2) Materialize Harbor tasks on disk (env/tests/solution + prompt).
      3) Optionally upload tasks to HF and write a registry.json.

    """
    parser = argparse.ArgumentParser(
        description="Generate Harbor tasks for the superconductor extraction benchmark."
    )
    parser.add_argument(
        "--gt-hf-repo",
        type=str,
        required=False,
        help="Hugging Face repo name for ground truth dataset (e.g., kilian-group/supercon-extraction).",
    )
    parser.add_argument(
        "--gt-hf-split",
        type=str,
        required=False,
        help="Split of the ground truth HF dataset (e.g., full, test).",
    )
    parser.add_argument(
        "--gt-hf-revision",
        type=str,
        default="main",
        help="Revision/version of the ground truth HF dataset (e.g., v0.0.0, main). Defaults to main.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="targeted-template",
        help="Template folder under the workspace (default: targeted-template).",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=Path,
        default="out",
        help="Output directory for generated tasks (default: out).",
    )
    parser.add_argument(
        "--data-dir",
        "-dd",
        type=Path,
        default="data",
        help="Root data directory containing Paper_DB/ (default: data).",
    )
    parser.add_argument(
        "--refno",
        action="append",
        default=None,
        help="Only build tasks for specific refno(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output task directory if it already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling task order in the registry. If not set, tasks are sorted alphabetically.",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload the generated tasks to a Hugging Face repo (writes registry.json).",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Hugging Face repo id, e.g. ORG/supercon-harbor-tasks.",
    )
    parser.add_argument(
        "--hf-registry-path",
        default="registry.json",
        help="Registry JSON path inside the repo (default: registry.json).",
    )
    parser.add_argument(
        "--hf-dataset-name",
        default=None,
        help="Dataset name in registry.json (default: repo id).",
    )
    parser.add_argument(
        "--hf-dataset-version",
        default="head",
        help="Dataset version in registry.json (default: head).",
    )
    args = parser.parse_args()

    # Load dataset configuration from CLI args
    if not args.gt_hf_repo or not args.gt_hf_split:
        raise SystemExit("--gt-hf-repo and --gt-hf-split are required.")
    dataset_name = args.gt_hf_repo
    dataset_split = args.gt_hf_split
    dataset_revision = args.gt_hf_revision
    templates_dir = Path(__file__).parent / args.template
    pdf_dir = args.data_dir / "Paper_DB"

    task_root = args.output_dir / "harbor"
    tasks_dir = task_root / "tasks"
    if task_root.exists():
        if args.force:
            shutil.rmtree(task_root)
        elif any(task_root.iterdir()):
            raise FileExistsError(
                f"{task_root} already exists. Re-run with --force to rebuild tasks."
            )
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset from configuration
    print(
        f"Loading dataset: {dataset_name} (revision: {dataset_revision}, split: {dataset_split})"
    )
    grouped = load_dataset(
        dataset_name, split=dataset_split, revision=dataset_revision
    ).to_pandas()

    for _, row in grouped.iterrows():
        refno = row["refno"]
        if args.refno and refno not in args.refno:
            continue
        properties = list(row["properties"])
        pdf_path = pdf_dir / f"{refno}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF for refno {refno} at {pdf_path}")

        task_id = slugify(refno)
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        build_task(
            task_dir,
            pdf_path=pdf_path,
            refno=refno,
            rows=properties,
            templates_dir=templates_dir,
        )
        task_rel = task_dir
        print(f"Wrote task {task_id} -> {task_rel}")

    # -- Write local registry.json --
    # NOTE: the local registry JSON is consistent with the HF upload registry JSON,
    # just with local task paths.
    generated_task_dirs = _collect_task_dirs(tasks_dir)
    if args.seed is not None:
        random.Random(args.seed).shuffle(generated_task_dirs)
    if generated_task_dirs:
        registry_path = task_root / "registry.json"
        dataset_short_name = dataset_name.split("/")[-1]
        write_local_registry(
            generated_task_dirs,
            registry_path,
            dataset_name=dataset_short_name,
            dataset_version=dataset_revision,
            description=f"Harbor tasks for {dataset_name}.",
        )
        registry_rel = registry_path
        print(f"Wrote registry -> {registry_rel}")

    if args.upload_hf:
        if args.hf_repo_id is None:
            raise SystemExit("--hf-repo-id is required when --upload-hf is set.")
        summary = upload_tasks_to_hf(
            task_dirs=generated_task_dirs,
            tasks_root=task_root,
            repo_id=str(args.hf_repo_id),
            registry_path=str(args.hf_registry_path),
            dataset_name=args.hf_dataset_name or str(args.hf_repo_id),
            dataset_version=str(args.hf_dataset_version),
        )
        print(
            f"Uploaded {summary['task_count']} tasks to {args.hf_repo_id}:{summary['path_in_repo']}"
        )
        print(f"Registry URL: {summary['registry_url']}")
        print(f"Dataset name: {summary['dataset_name']}")


def _infer_hf_token() -> str | None:
    """Return an HF auth token from common environment variables."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


def _prompt_value(label: str, default: str | None = None) -> str:
    """Prompt for a string value with an optional default."""
    prompt = f"{label} [{default}]: " if default else f"{label}: "
    value = input(prompt).strip()
    return value or (default or "")


def _collect_task_dirs(tasks_root: Path) -> list[Path]:
    """Return Harbor-valid task directories under the tasks root."""
    return [
        child
        for child in sorted(tasks_root.iterdir())
        if child.is_dir() and TaskPaths(child).is_valid()
    ]


def _hf_repo_url(repo_id: str, repo_type: str) -> str:
    """Return the https URL for a HF repo."""
    base = "https://huggingface.co"
    if repo_type == "dataset":
        return f"{base}/datasets/{repo_id}"
    if repo_type == "space":
        return f"{base}/spaces/{repo_id}"
    return f"{base}/{repo_id}"


def _hf_git_url(repo_id: str, repo_type: str) -> str:
    """Return the git URL for a HF repo."""
    return f"{_hf_repo_url(repo_id, repo_type)}.git"


def _hf_resolve_url(repo_id: str, repo_type: str, path_in_repo: str) -> str:
    """Return a resolve URL for a file in a HF repo."""
    return f"{_hf_repo_url(repo_id, repo_type)}/resolve/main/{path_in_repo}"


def _build_registry(
    *,
    task_dirs: list[Path],
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    dataset_name: str,
    dataset_version: str,
    description: str,
) -> list[dict[str, object]]:
    """Build a Harbor registry.json payload for a list of tasks."""
    git_url = _hf_git_url(repo_id, repo_type)
    tasks = []
    for task_dir in task_dirs:
        task_path = (Path(path_in_repo) / task_dir.name).as_posix()
        tasks.append(
            {
                "name": task_dir.name,
                "git_url": git_url,
                "git_commit_id": None,
                "path": task_path,
            }
        )
    return [
        {
            "name": dataset_name,
            "version": dataset_version,
            "description": description,
            "tasks": tasks,
        }
    ]


def upload_tasks_to_hf(
    *,
    task_dirs: list[Path],
    tasks_root: Path,
    repo_id: str,
    repo_type: str = "dataset",
    registry_path: str = "registry.json",
    dataset_name: str | None = None,
    dataset_version: str = "head",
    description: str = "Harbor tasks uploaded from a local tasks directory.",
    create: bool = True,
    private: bool | None = None,
    token: str | None = None,
) -> dict[str, str]:
    """Upload Harbor tasks and registry.json to a Hugging Face repo.

    Returns a small summary dict for logging.
    """
    dataset_name = dataset_name or repo_id
    registry_path = str(registry_path).strip("/")

    hf_token = token or _infer_hf_token()
    api = HfApi(token=hf_token)

    if create:
        api.create_repo(
            repo_id=str(repo_id),
            repo_type=str(repo_type),
            private=False if private is None else private,
            exist_ok=True,
        )
    else:
        try:
            api.list_repo_files(repo_id=str(repo_id), repo_type=str(repo_type))
        except Exception as exc:
            raise SystemExit(f"Repo not found or not accessible: {repo_id}") from exc

    # NOTE: upload_large_folder does not support path_in_repo or commit_message.
    # If path_in_repo is needed, local folder structure must match the desired repo path.
    api.upload_large_folder(
        repo_id=str(repo_id),
        repo_type=str(repo_type),
        folder_path=str(tasks_root),
    )

    registry = _build_registry(
        task_dirs=task_dirs,
        repo_id=str(repo_id),
        repo_type=str(repo_type),
        path_in_repo="tasks",
        dataset_name=dataset_name,
        dataset_version=str(dataset_version),
        description=str(description),
    )

    api.upload_file(
        repo_id=str(repo_id),
        repo_type=str(repo_type),
        path_or_fileobj=io.BytesIO(json.dumps(registry, indent=2).encode("utf-8")),
        path_in_repo=registry_path,
        commit_message="Add/update Harbor registry.json",
        token=hf_token,
    )

    return {
        "task_count": str(len(task_dirs)),
        "registry_url": _hf_resolve_url(str(repo_id), str(repo_type), registry_path),
        "dataset_name": f"{dataset_name}@{dataset_version}",
        "path_in_repo": "/",
    }


if __name__ == "__main__":
    main()
