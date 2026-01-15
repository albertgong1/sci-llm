r"""Compile Harbor tasks for property extraction from a folder of PDFs.

This "task compiler" turns a (PDF, ground-truth) dataset into Harbor task directories,
each with:
  - `environment/`: Docker build context with the paper PDF
  - `instruction.md`: a single prompt/instruction file shared across tasks via a template
  - `tests/`: verifier that scores predictions using rubric tolerances
  - `solution/`: an oracle solution used by Harbor's built-in `oracle` agent

The ground truth source is specified via --gt-hf-repo, --gt-hf-split, and optionally
--gt-hf-revision (defaults to main).

Optional: pass `--upload-hf` to upload the generated tasks to a Hugging Face repo
and write a `registry.json` so Harbor can pull tasks directly from the Hub.

By default this script writes tasks under
`examples/harbor-workspace/out/harbor/<dataset>/<template>/` so the
repository stays clean until you build.

Example (from repo root):
    uv run python src/harbor-task-gen/prepare_harbor_tasks.py --task tc --force \
      --gt-hf-repo kilian-group/supercon-extraction --gt-hf-split full --gt-hf-revision v0.0.0
    uv run python src/harbor-task-gen/run_harbor.py jobs start \
      -c out/harbor/supercon-extraction/ground-template/job.yaml -a oracle

    # To use the guided template:
    uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
      --task tc --template ground-template-easy --force \
      --gt-hf-repo kilian-group/supercon-extraction --gt-hf-split full
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from datasets import load_dataset
from harbor.models.task.paths import TaskPaths
from huggingface_hub import HfApi


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def default_workspace_root() -> Path:
    """Return the default Harbor workspace location."""
    return repo_root() / "examples" / "harbor-workspace"


def workspace_root() -> Path:
    """Return the configured Harbor workspace root."""
    return _WORKSPACE_ROOT or default_workspace_root()


def templates_dir() -> Path:
    """Return the directory containing files copied into generated Harbor tasks."""
    return workspace_root() / _TEMPLATES_SUBDIR


_TEMPLATES_SUBDIR = "ground-template"
_WORKSPACE_ROOT: Path | None = None

_TASK_PROPERTY_FILTERS: dict[str, set[str]] = {
    # Default task: superconducting critical temperature recommended for the sample.
    "tc": {"Tc (of this sample) recommended"},
}


def read_template(relative_path: str) -> str:
    """Read a template file relative to the workspace template folder."""
    return (templates_dir() / relative_path).read_text()


def copy_template(relative_path: str, dest_path: Path) -> None:
    """Copy a template file relative to the workspace template folder."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(templates_dir() / relative_path, dest_path)


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


def slugify(value: str) -> str:
    """Normalize strings for file-safe task IDs."""
    return (
        value.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
    )


def load_rubric_mapping(rubric_path: Path) -> dict[str, str]:
    """Load the property_name -> rubric mapping from the rubric CSV."""
    mapping: dict[str, str] = {}
    with rubric_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            property_name = row.get("property_name")
            rubric = row.get("rubric")
            if property_name and rubric:
                mapping[property_name] = rubric
    return mapping


def load_definitions(rubric_path: Path) -> dict[str, str]:
    """Load property_name -> definition mapping from the rubric CSV (if present)."""
    definitions: dict[str, str] = {}
    with rubric_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            property_name = row.get("property_name")
            definition = row.get("definition") or ""
            if property_name:
                definitions[property_name] = definition
    return definitions


def dockerfile_contents() -> str:
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
        read_template("environment/Dockerfile"),
        {"install_pdf_tools": install_pdf_tools},
    )


def resolve_property_filter(task: str | None) -> set[str] | None:
    """Return the set of property_names to keep for a given task alias (or None for all)."""
    if task is None:
        return None
    return _TASK_PROPERTY_FILTERS.get(task.strip().lower())


def flatten_dataset(
    dataset: Iterable[dict[str, Any]],
    *,
    definitions: Mapping[str, str],
    property_filter: set[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Flatten HF rows (refno + properties list) into per-property rows grouped by refno."""
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in dataset:
        refno = str(row.get("refno") or "").strip()
        if not refno:
            continue

        props = row.get("properties") or []
        if not isinstance(props, list):
            continue

        for prop in props:
            if not isinstance(prop, dict):
                continue
            prop_name = str(prop.get("property_name") or "").strip()
            if not prop_name:
                continue
            if property_filter and prop_name not in property_filter:
                continue

            grouped[refno].append(
                {
                    "material": str(prop.get("material_or_system") or ""),
                    "property_name": prop_name,
                    # value_string already contains any unit; keep unit empty to avoid double-parsing.
                    "property_value": str(prop.get("value_string") or ""),
                    "property_unit": "",
                    "definition": definitions.get(prop_name, ""),
                }
            )
    return grouped


def write_job_config(tasks_dir: Path, job_path: Path, *, workspace: Path) -> None:
    """Write a Harbor job YAML pointing at the generated tasks."""
    if tasks_dir.is_absolute():
        try:
            tasks_rel = tasks_dir.relative_to(workspace)
        except ValueError:
            tasks_rel = tasks_dir
    else:
        tasks_rel = tasks_dir
    job_yaml = f"""\
jobs_dir: jobs
n_attempts: 1
timeout_multiplier: 1.0
orchestrator:
  type: local
  n_concurrent_trials: 2
  quiet: false
environment:
  type: docker
  force_build: true
  delete: true
agents:
  - name: oracle
datasets:
  - path: {tasks_rel.as_posix()}
"""
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(job_yaml)


def build_task(
    task_dir: Path,
    *,
    pdf_path: Path,
    task_name: str,
    refno: str,
    rows: list[dict[str, str]],
    rubric_mapping: dict[str, str],
) -> None:
    """Build a single Harbor task directory (one paper, many questions)."""
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"

    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pdf_path, env_dir / "paper.pdf")

    questions: list[dict[str, str]] = []
    expected_rows: list[dict[str, str]] = []
    for row in rows:
        rubric = rubric_mapping.get(row["property_name"], "categorical")
        questions.append(
            {
                "material": row["material"],
                "property_name": row["property_name"],
                "definition": row["definition"],
            }
        )
        expected_rows.append(
            {
                "material": row["material"],
                "property_name": row["property_name"],
                "property_value": str(row["property_value"]),
                "property_unit": str(row.get("property_unit", "")),
                "rubric": rubric,
            }
        )

    expected = {
        "task": task_name,
        "refno": refno,
        "ground_truth": expected_rows,
    }
    (tests_dir / "expected.json").write_text(json.dumps(expected, indent=2))

    task_meta = {
        "refno": refno,
        "pdf_path": "/app/paper.pdf",
        "predictions_path": "/app/output/predictions.json",
        "questions": questions,
    }
    (env_dir / "task_meta.json").write_text(json.dumps(task_meta, indent=2))

    question_blocks = "\n\n".join(
        textwrap.dedent(
            f"""\
            [{idx}]
            Question: What is the {item["property_name"]} recommended for {item["material"]}? Here, "{item["property_name"]}" is defined as "{item["definition"]}".
            Answer:
            """
        ).strip()
        for idx, item in enumerate(questions)
    )
    gemini_at_commands = "`@paper.pdf`"
    paper_at_command = "@paper.pdf"
    claude_file_examples = "`/app/paper.pdf`"

    instruction_template = read_template("instruction.md.template")
    instruction_values = {
        # Identifiers
        "task": task_name,
        "task_name": task_name,
        "task_id": task_dir.name,
        "refno": refno,
        # Standard in-container paths
        "pdf_path": "/app/paper.pdf",
        "meta_path": "/app/task_meta.json",
        "predictions_path": "/app/output/predictions.json",
        # Prompt building blocks (optional; templates may ignore these)
        "question_blocks": question_blocks,
        "questions_json": json.dumps(questions, indent=2),
        "task_meta_json": json.dumps(task_meta, indent=2),
        # Agent affordances (optional)
        "paper_at_command": paper_at_command,
        "gemini_at_commands": gemini_at_commands,
        "claude_file_examples": claude_file_examples,
    }
    instruction = _format_template(instruction_template, instruction_values)
    (task_dir / "instruction.md").write_text(textwrap.dedent(instruction))

    task_toml = _format_template(
        read_template("task.toml.template"),
        {"task_name": task_name, "task": task_name},
    )
    (task_dir / "task.toml").write_text(task_toml)

    (env_dir / "Dockerfile").write_text(dockerfile_contents())
    copy_template("tests/check_prediction.py", tests_dir / "check_prediction.py")
    copy_template("tests/test.sh", tests_dir / "test.sh")

    solution_predictions = [
        {
            "material": row["material"],
            "property_name": row["property_name"],
            "pred_value": row["property_value"],
            "pred_unit": row.get("property_unit", ""),
        }
        for row in expected_rows
    ]
    solution_script = f"""\
#!/bin/bash
set -euo pipefail

mkdir -p /app/output
cat > /app/output/predictions.json <<'EOF'
{json.dumps(solution_predictions, indent=2)}
EOF
"""
    (solution_dir / "solve.sh").write_text(solution_script)

    for script in [tests_dir / "test.sh", solution_dir / "solve.sh"]:
        script.chmod(0o755)


def main() -> None:
    """Generate Harbor tasks for the benchmark.

    This is a multi-step pipeline:
      1) Load the HF dataset (refno -> properties).
      2) Flatten rows into per-paper questions.
      3) Materialize Harbor tasks on disk (env/tests/solution + prompt).
      4) Optionally upload tasks to HF and write a registry.json.
    """
    global _TEMPLATES_SUBDIR

    parser = argparse.ArgumentParser(
        description="Generate Harbor tasks for the superconductor extraction benchmark."
    )
    parser.add_argument(
        "--gt-hf-repo",
        type=str,
        required=True,
        help="Hugging Face repo name for ground truth dataset (e.g., kilian-group/supercon-extraction).",
    )
    parser.add_argument(
        "--gt-hf-split",
        type=str,
        required=True,
        help="Split of the ground truth HF dataset (e.g., full, test).",
    )
    parser.add_argument(
        "--gt-hf-revision",
        type=str,
        default="main",
        help="Revision/version of the ground truth HF dataset (e.g., v0.0.0, main). Defaults to main.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root for templates/data/output (default: examples/harbor-workspace).",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="ground-template",
        help="Template folder under the workspace (default: ground-template).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task alias for filtering property_names (e.g., tc). If omitted, include all properties.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing PDFs named <refno>.pdf (default: <workspace>/data/Paper_DB).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write generated Harbor tasks "
            "(default: <workspace>/out/harbor/supercon-mini-v2)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of tasks (papers) to generate.",
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
        "--write-job-config",
        action="store_true",
        help="Also emit a Harbor job config pointing at the generated tasks.",
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
        "--hf-repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type (default: dataset).",
    )
    parser.add_argument(
        "--hf-path-in-repo",
        default="tasks",
        help="Where to place task folders inside the repo (default: tasks).",
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
    parser.add_argument(
        "--hf-description",
        default="Harbor tasks uploaded from a local tasks directory.",
        help="Dataset description for registry.json.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create the repo as private if it does not exist.",
    )
    parser.add_argument(
        "--hf-public",
        action="store_true",
        help="Create the repo as public if it does not exist.",
    )
    parser.add_argument(
        "--hf-create",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the repo if it does not exist (default: create).",
    )
    parser.add_argument(
        "--hf-no-input",
        action="store_true",
        help="Disable interactive prompts for HF upload settings.",
    )
    parser.add_argument(
        "--hf-tasks-root",
        default=None,
        help="Override the tasks root to upload (default: generated tasks dir).",
    )
    args = parser.parse_args()

    resolved_workspace = (args.workspace or default_workspace_root()).resolve()
    global _WORKSPACE_ROOT
    _WORKSPACE_ROOT = resolved_workspace
    _TEMPLATES_SUBDIR = str(args.template)

    if resolved_workspace.exists() and not resolved_workspace.is_dir():
        raise SystemExit(f"--workspace must be a directory: {resolved_workspace}")
    resolved_workspace.mkdir(parents=True, exist_ok=True)

    # Load dataset configuration from CLI args
    dataset_name = args.gt_hf_repo
    dataset_split = args.gt_hf_split
    dataset_revision = args.gt_hf_revision

    if args.pdf_dir is None:
        args.pdf_dir = resolved_workspace / "data" / "Paper_DB"
    if args.output_dir is None:
        # Use the dataset name (last part) for output directory
        dataset_short_name = dataset_name.split("/")[-1]
        args.output_dir = resolved_workspace / "out" / "harbor" / dataset_short_name
    if not args.pdf_dir.is_absolute():
        args.pdf_dir = resolved_workspace / args.pdf_dir
    if not args.output_dir.is_absolute():
        args.output_dir = resolved_workspace / args.output_dir

    template_root = templates_dir()
    if not template_root.exists():
        raise FileNotFoundError(
            f"Template not found: {template_root}. "
            "Create it under the workspace or pass --template."
        )

    if not args.pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found at {args.pdf_dir}")

    task_label = (args.task or "all").strip() or "all"
    # If no task alias is specified, drop the task segment to keep a flat layout.
    task_root = (
        args.output_dir / str(args.template)
        if args.task is None
        else args.output_dir / task_label / str(args.template)
    )
    tasks_dir = task_root / "tasks"
    if task_root.exists():
        if args.force:
            shutil.rmtree(task_root)
        elif any(task_root.iterdir()):
            raise FileExistsError(
                f"{task_root} already exists. Re-run with --force to rebuild tasks."
            )
    tasks_dir.mkdir(parents=True, exist_ok=True)

    rubric_path = resolved_workspace / "scoring" / "rubric.csv"
    rubric_mapping = load_rubric_mapping(rubric_path)
    definitions = load_definitions(rubric_path)

    # Load dataset from configuration
    print(
        f"Loading dataset: {dataset_name} (revision: {dataset_revision}, split: {dataset_split})"
    )
    dataset = load_dataset(dataset_name, split=dataset_split, revision=dataset_revision)
    property_filter = resolve_property_filter(args.task)
    grouped = flatten_dataset(
        cast(Iterable[dict[str, Any]], dataset),
        definitions=definitions,
        property_filter=property_filter,
    )

    refnos = list(grouped.keys())
    if args.refno:
        requested = {value.strip() for value in args.refno if value and value.strip()}
        missing = sorted(requested - set(refnos))
        if missing:
            raise ValueError(f"Unknown refno(s) requested: {missing}")
        refnos = [refno for refno in refnos if refno in requested]
    if args.limit is not None:
        refnos = refnos[: args.limit]

    for refno in refnos:
        pdf_path = args.pdf_dir / f"{refno}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF for refno {refno} at {pdf_path}")

        task_id = (
            f"{slugify(refno)}--{slugify(task_label)}"
            if args.task is not None
            else slugify(refno)
        )
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        rows = grouped.get(refno, [])
        if not rows:
            print(f"Skipping {refno}: no properties matched task '{args.task}'.")
            continue

        build_task(
            task_dir,
            pdf_path=pdf_path,
            task_name=task_label,
            refno=refno,
            rows=rows,
            rubric_mapping=rubric_mapping,
        )
        try:
            task_rel = task_dir.relative_to(resolved_workspace)
        except ValueError:
            task_rel = task_dir
        print(f"Wrote task {task_id} -> {task_rel}")

    if args.write_job_config:
        job_path = task_root / "job.yaml"
        write_job_config(tasks_dir, job_path, workspace=resolved_workspace)
        try:
            job_rel = job_path.relative_to(resolved_workspace)
        except ValueError:
            job_rel = job_path
        print(f"Wrote job config -> {job_rel}")

    if args.upload_hf:
        _upload_tasks_after_build(
            args=args,
            tasks_root=tasks_dir,
        )


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


def _resolve_tasks_root(path: Path) -> Path:
    """Allow passing either the tasks root or its parent directory."""
    if (path / "tasks").is_dir():
        return path / "tasks"
    return path


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
    tasks_root: Path,
    repo_id: str,
    repo_type: str = "dataset",
    path_in_repo: str = "tasks",
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
    resolved_root = _resolve_tasks_root(tasks_root).resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Tasks root not found: {resolved_root}")

    task_dirs = _collect_task_dirs(resolved_root)
    if not task_dirs:
        raise SystemExit(f"No valid Harbor task folders found under {resolved_root}.")

    dataset_name = dataset_name or repo_id
    path_in_repo = str(path_in_repo).strip("/")
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
    if path_in_repo and path_in_repo not in ("", ".", "/"):
        print(
            f"Warning: upload_large_folder ignores path_in_repo='{path_in_repo}'. "
            "Files will be uploaded to repo root."
        )
    api.upload_large_folder(
        repo_id=str(repo_id),
        repo_type=str(repo_type),
        folder_path=str(resolved_root),
    )

    registry = _build_registry(
        task_dirs=task_dirs,
        repo_id=str(repo_id),
        repo_type=str(repo_type),
        path_in_repo=path_in_repo or ".",
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
        "path_in_repo": path_in_repo or "/",
    }


def _upload_tasks_after_build(*, args: argparse.Namespace, tasks_root: Path) -> None:
    """Handle HF upload configuration + calls after tasks are generated."""
    if args.hf_private and args.hf_public:
        raise SystemExit("Pass at most one of --hf-private/--hf-public.")

    repo_id = args.hf_repo_id
    if repo_id is None and not args.hf_no_input:
        repo_id = _prompt_value("HF repo id (org/name)")

    if repo_id is None:
        raise SystemExit("--hf-repo-id is required when --upload-hf is set.")

    if args.hf_tasks_root is not None:
        tasks_root = Path(args.hf_tasks_root)
        if not tasks_root.is_absolute():
            tasks_root = workspace_root() / tasks_root

    private: bool | None
    if args.hf_private:
        private = True
    elif args.hf_public:
        private = False
    else:
        private = None

    summary = upload_tasks_to_hf(
        tasks_root=tasks_root,
        repo_id=str(repo_id),
        repo_type=str(args.hf_repo_type),
        path_in_repo=str(args.hf_path_in_repo),
        registry_path=str(args.hf_registry_path),
        dataset_name=args.hf_dataset_name or str(repo_id),
        dataset_version=str(args.hf_dataset_version),
        description=str(args.hf_description),
        create=bool(args.hf_create),
        private=private,
    )

    print(
        f"Uploaded {summary['task_count']} tasks to {repo_id}:{summary['path_in_repo']}"
    )
    print(f"Registry URL: {summary['registry_url']}")
    print(f"Dataset name: {summary['dataset_name']}")


if __name__ == "__main__":
    main()
