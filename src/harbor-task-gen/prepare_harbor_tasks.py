r"""Compile Harbor tasks for property extraction from a folder of PDFs.

This "task compiler" turns a (PDF, ground-truth) dataset into Harbor task directories,
each with:
  - `environment/`: Docker build context with the paper PDF
  - `instruction.md`: a single prompt/instruction file shared across tasks via a template
  - `tests/`: verifier that scores predictions using rubric tolerances
  - `solution/`: an oracle solution used by Harbor's built-in `oracle` agent
  - `registry.json`: a local task registry for Harbor to discover and load tasks

When `--no-score` is set, tasks are built from PDFs only (no dataset), and no
verifier/solution files are written. Use `--disable-verification` at runtime (or
the generated job.yaml with `verifier.disable=true`) to skip scoring.

The ground truth source is specified via --gt-hf-repo, --gt-hf-split, and optionally
--gt-hf-revision (defaults to main).

Optional: pass `--upload-hf` to upload the generated tasks to a Hugging Face repo
so Harbor can pull tasks directly from the Hub.

By default this script writes tasks under
`examples/harbor-workspace/out/harbor/<dataset>/<template>/` so the
repository stays clean until you build.

Example (from repo root):
    uv run python src/harbor-task-gen/prepare_harbor_tasks.py --task tc --force \
      --gt-hf-repo kilian-group/supercon-extraction --gt-hf-split full --gt-hf-revision v0.0.0
    uv run python src/harbor-task-gen/run_harbor.py jobs start \
      --registry out/harbor/supercon-extraction/tc/ground-template/registry.json -a oracle

    # To build tasks without a dataset (no scoring):
    uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
      --no-score --force
"""

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from datasets import load_dataset
from harbor.models.task.paths import TaskPaths
from huggingface_hub import HfApi
from slugify import slugify
import logging
from mineru import MinerUBundlePaths, MinerUConfig, ensure_mineru_bundle


logger = logging.getLogger(__name__)


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


def load_harbor_task_ordering(registry_path: Path) -> list[str]:
    """Load ordered task refnos from a Harbor registry.json file."""
    payload = json.loads(registry_path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid Harbor registry payload in {registry_path}")

    dataset = payload[0]
    if not isinstance(dataset, dict):
        raise ValueError(f"Invalid Harbor registry dataset entry in {registry_path}")

    tasks = dataset.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Missing tasks list in Harbor registry {registry_path}")

    ordered_refnos: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        refno = str(task.get("name") or "").strip()
        if refno:
            ordered_refnos.append(refno)
    return ordered_refnos


def build_pdf_lookup(pdf_dir: Path) -> dict[str, Path]:
    """Index PDFs by lowercase stem for case-insensitive refno resolution."""
    lookup: dict[str, Path] = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        key = pdf_path.stem.lower()
        if key in lookup and lookup[key] != pdf_path:
            raise ValueError(
                f"Duplicate PDF stems differing only by case under {pdf_dir}: "
                f"{lookup[key].name} and {pdf_path.name}"
            )
        lookup[key] = pdf_path
    return lookup


def resolve_pdf_path(pdf_lookup: Mapping[str, Path], refno: str) -> Path:
    """Return the PDF path for a refno using case-insensitive matching."""
    pdf_path = pdf_lookup.get(refno.lower())
    if pdf_path is None:
        raise FileNotFoundError(
            f"Missing PDF for refno {refno}. Expected a file named like "
            f"{refno}.pdf under the configured pdf dir (case-insensitive)."
        )
    return pdf_path


def dockerfile_contents(paper_source: str) -> str:
    """Render the task environment Dockerfile for the selected paper source."""
    install_pdf_tools = ""
    if paper_source != "mineru":
        install_pdf_tools = (
            "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            "    ca-certificates \\\n"
            "    poppler-utils \\\n"
            "    procps \\\n"
            "  && rm -rf /var/lib/apt/lists/*"
        )

    dockerfile = _format_template(
        read_template("environment/Dockerfile"),
        {"install_pdf_tools": install_pdf_tools},
    )
    if paper_source == "mineru":
        dockerfile = dockerfile.replace("COPY paper.pdf /app/paper.pdf\n", "")
        dockerfile = f"{dockerfile}\nCOPY paper_mineru /app/paper_mineru\n"
    return dockerfile


def prepare_task_paper_artifacts(
    env_dir: Path,
    *,
    pdf_path: Path,
    paper_source: str,
    mineru_bundle: MinerUBundlePaths | None,
) -> dict[str, str]:
    """Copy paper artifacts into the task environment and build prompt placeholders."""
    artifacts = {
        "pdf_path": "/app/paper.pdf",
        "paper_source": paper_source,
        "paper_source_path": "/app/paper.pdf",
        "paper_source_dir": "",
        "paper_at_command": "@paper.pdf",
        "pdf_at_command": "@paper.pdf",
        "output_at_command": "@paper.pdf",
        "paper_source_at_command": "@paper.pdf",
        "mineru_dir_path": "",
        "mineru_clean_markdown_path": "",
        "mineru_outline_path": "",
        "mineru_table_index_path": "",
        "mineru_tables_path": "",
        "mineru_captions_path": "",
        "mineru_raw_markdown_path": "",
        "mineru_primary_markdown_path": "",
        "paper_source_description": (
            "The paper is provided as the original PDF at `/app/paper.pdf`."
        ),
        "gemini_at_commands": "`@paper.pdf`",
        "claude_file_examples": "`/app/paper.pdf`",
    }

    if paper_source != "mineru":
        shutil.copy2(pdf_path, env_dir / "paper.pdf")
        return artifacts

    if mineru_bundle is None:
        raise ValueError("mineru_bundle is required when paper_source='mineru'.")

    bundle_dest = env_dir / "paper_mineru"
    bundle_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mineru_bundle.primary_markdown_path, bundle_dest / "primary.md")
    if (
        mineru_bundle.clean_markdown_path is not None
        and mineru_bundle.clean_markdown_path.exists()
    ):
        shutil.copy2(mineru_bundle.clean_markdown_path, bundle_dest / "clean.md")
    if mineru_bundle.outline_path is not None and mineru_bundle.outline_path.exists():
        shutil.copy2(mineru_bundle.outline_path, bundle_dest / "outline.md")
    if (
        mineru_bundle.table_index_path is not None
        and mineru_bundle.table_index_path.exists()
    ):
        shutil.copy2(mineru_bundle.table_index_path, bundle_dest / "table_index.md")
    if mineru_bundle.tables_path is not None and mineru_bundle.tables_path.exists():
        shutil.copy2(mineru_bundle.tables_path, bundle_dest / "tables.md")
    if mineru_bundle.captions_path is not None and mineru_bundle.captions_path.exists():
        shutil.copy2(mineru_bundle.captions_path, bundle_dest / "captions.md")
    if (
        mineru_bundle.raw_markdown_path is not None
        and mineru_bundle.raw_markdown_path.exists()
    ):
        shutil.copy2(mineru_bundle.raw_markdown_path, bundle_dest / "raw.md")
    images_dest = bundle_dest / "images"
    images_dest.mkdir(exist_ok=True)
    if mineru_bundle.images_dir is not None and mineru_bundle.images_dir.exists():
        shutil.copytree(
            mineru_bundle.images_dir,
            images_dest,
            dirs_exist_ok=True,
        )
    artifacts.update(
        {
            "pdf_path": "",
            "paper_source_path": "/app/paper_mineru/primary.md",
            "paper_source_dir": "/app/paper_mineru",
            "output_at_command": "@paper_mineru/primary.md",
            "paper_source_at_command": "@paper_mineru/primary.md",
            "mineru_dir_path": "/app/paper_mineru",
            "mineru_clean_markdown_path": "/app/paper_mineru/clean.md"
            if (bundle_dest / "clean.md").exists()
            else "",
            "mineru_outline_path": "/app/paper_mineru/outline.md"
            if (bundle_dest / "outline.md").exists()
            else "",
            "mineru_table_index_path": "/app/paper_mineru/table_index.md"
            if (bundle_dest / "table_index.md").exists()
            else "",
            "mineru_tables_path": "/app/paper_mineru/tables.md"
            if (bundle_dest / "tables.md").exists()
            else "",
            "mineru_captions_path": "/app/paper_mineru/captions.md"
            if (bundle_dest / "captions.md").exists()
            else "",
            "mineru_raw_markdown_path": "/app/paper_mineru/raw.md"
            if (bundle_dest / "raw.md").exists()
            else "",
            "mineru_primary_markdown_path": "/app/paper_mineru/primary.md",
            "paper_source_description": (
                "The paper has been preprocessed with MinerU. Available text views are "
                "`/app/paper_mineru/primary.md` (full page-ordered reconstruction), "
                "`/app/paper_mineru/clean.md` (clean reading view), "
                "`/app/paper_mineru/outline.md` (page/section/figure/table map), and "
                "`/app/paper_mineru/table_index.md` (compact table preview). "
                "Additional structured views are "
                "`/app/paper_mineru/tables.md` (full table TSV blocks), "
                "`/app/paper_mineru/captions.md` (figure/table captions), and "
                "`/app/paper_mineru/raw.md` (raw MinerU markdown). "
                "Inspect `/app/paper_mineru/images/` for extracted figure, table, and equation "
                "crops when they help resolve a value."
            ),
            "gemini_at_commands": (
                "`@paper_mineru/primary.md`, "
                "`@paper_mineru/clean.md`, "
                "`@paper_mineru/outline.md`, "
                "`@paper_mineru/table_index.md`, "
                "`@paper_mineru/tables.md`, "
                "`@paper_mineru/captions.md`, "
                "`@paper_mineru/raw.md`"
            ),
            "claude_file_examples": (
                "`/app/paper_mineru/primary.md`, "
                "`/app/paper_mineru/clean.md`, "
                "`/app/paper_mineru/outline.md`, "
                "`/app/paper_mineru/table_index.md`, "
                "`/app/paper_mineru/tables.md`, "
                "`/app/paper_mineru/captions.md`, "
                "`/app/paper_mineru/raw.md`, "
                "`/app/paper_mineru/images/`"
            ),
        }
    )
    return artifacts


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


def write_job_config(
    tasks_dir: Path,
    job_path: Path,
    *,
    workspace: Path,
    disable_verification: bool = False,
    agent_name: str = "oracle",
) -> None:
    """Write a Harbor job YAML pointing at the generated tasks."""
    if tasks_dir.is_absolute():
        try:
            tasks_rel = tasks_dir.relative_to(workspace)
        except ValueError:
            tasks_rel = tasks_dir
    else:
        tasks_rel = tasks_dir
    verifier_block = "verifier:\n  disable: true\n" if disable_verification else ""
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
{verifier_block}agents:
  - name: {agent_name}
datasets:
  - path: {tasks_rel.as_posix()}
"""
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(job_yaml)


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
    paper_source: str,
    mineru_bundle: MinerUBundlePaths | None,
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

    paper_artifacts = prepare_task_paper_artifacts(
        env_dir,
        pdf_path=pdf_path,
        paper_source=paper_source,
        mineru_bundle=mineru_bundle,
    )

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
        "paper_source": paper_source,
        "paper_source_path": paper_artifacts["paper_source_path"],
        "paper_source_dir": paper_artifacts["paper_source_dir"] or None,
        "pdf_path": paper_artifacts["pdf_path"],
        "mineru_dir_path": paper_artifacts["mineru_dir_path"] or None,
        "mineru_clean_markdown_path": (
            paper_artifacts["mineru_clean_markdown_path"] or None
        ),
        "mineru_outline_path": paper_artifacts["mineru_outline_path"] or None,
        "mineru_table_index_path": paper_artifacts["mineru_table_index_path"] or None,
        "mineru_tables_path": paper_artifacts["mineru_tables_path"] or None,
        "mineru_captions_path": paper_artifacts["mineru_captions_path"] or None,
        "mineru_raw_markdown_path": (
            paper_artifacts["mineru_raw_markdown_path"] or None
        ),
        "mineru_primary_markdown_path": (
            paper_artifacts["mineru_primary_markdown_path"] or None
        ),
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

    instruction_template = read_template("instruction.md.template")
    instruction_values = {
        # Identifiers
        "task": task_name,
        "task_name": task_name,
        "task_id": task_dir.name,
        "refno": refno,
        # Standard in-container paths
        "pdf_path": paper_artifacts["pdf_path"],
        "meta_path": "/app/task_meta.json",
        "predictions_path": "/app/output/predictions.json",
        # Prompt building blocks (optional; templates may ignore these)
        "question_blocks": question_blocks,
        "questions_json": json.dumps(questions, indent=2),
        "task_meta_json": json.dumps(task_meta, indent=2),
        # Agent affordances (optional)
        "paper_at_command": paper_artifacts["paper_at_command"],
        "pdf_at_command": paper_artifacts["pdf_at_command"],
        "output_at_command": paper_artifacts["output_at_command"],
        "paper_source_at_command": paper_artifacts["paper_source_at_command"],
        "paper_source": paper_artifacts["paper_source"],
        "paper_source_path": paper_artifacts["paper_source_path"],
        "paper_source_dir": paper_artifacts["paper_source_dir"],
        "mineru_dir_path": paper_artifacts["mineru_dir_path"],
        "mineru_clean_markdown_path": paper_artifacts["mineru_clean_markdown_path"],
        "mineru_outline_path": paper_artifacts["mineru_outline_path"],
        "mineru_table_index_path": paper_artifacts["mineru_table_index_path"],
        "mineru_tables_path": paper_artifacts["mineru_tables_path"],
        "mineru_captions_path": paper_artifacts["mineru_captions_path"],
        "mineru_raw_markdown_path": paper_artifacts["mineru_raw_markdown_path"],
        "mineru_primary_markdown_path": paper_artifacts["mineru_primary_markdown_path"],
        "paper_source_description": paper_artifacts["paper_source_description"],
        "gemini_at_commands": paper_artifacts["gemini_at_commands"],
        "claude_file_examples": paper_artifacts["claude_file_examples"],
    }
    instruction = _format_template(instruction_template, instruction_values)
    (task_dir / "instruction.md").write_text(textwrap.dedent(instruction))

    task_toml = _format_template(
        read_template("task.toml.template"),
        {"task_name": task_name, "task": task_name},
    )
    (task_dir / "task.toml").write_text(task_toml)

    (env_dir / "Dockerfile").write_text(dockerfile_contents(paper_source))
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


def build_task_no_score(
    task_dir: Path,
    *,
    pdf_path: Path,
    paper_source: str,
    mineru_bundle: MinerUBundlePaths | None,
    task_name: str,
    refno: str,
) -> None:
    """Build a task without ground-truth, verifier, or solution."""
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    paper_artifacts = prepare_task_paper_artifacts(
        env_dir,
        pdf_path=pdf_path,
        paper_source=paper_source,
        mineru_bundle=mineru_bundle,
    )

    task_meta = {
        "refno": refno,
        "paper_source": paper_source,
        "paper_source_path": paper_artifacts["paper_source_path"],
        "paper_source_dir": paper_artifacts["paper_source_dir"] or None,
        "pdf_path": paper_artifacts["pdf_path"],
        "mineru_dir_path": paper_artifacts["mineru_dir_path"] or None,
        "mineru_clean_markdown_path": (
            paper_artifacts["mineru_clean_markdown_path"] or None
        ),
        "mineru_outline_path": paper_artifacts["mineru_outline_path"] or None,
        "mineru_table_index_path": paper_artifacts["mineru_table_index_path"] or None,
        "mineru_tables_path": paper_artifacts["mineru_tables_path"] or None,
        "mineru_captions_path": paper_artifacts["mineru_captions_path"] or None,
        "mineru_raw_markdown_path": (
            paper_artifacts["mineru_raw_markdown_path"] or None
        ),
        "mineru_primary_markdown_path": (
            paper_artifacts["mineru_primary_markdown_path"] or None
        ),
        "predictions_path": "/app/output/predictions.json",
        "questions": [],
    }
    (env_dir / "task_meta.json").write_text(json.dumps(task_meta, indent=2))

    instruction_template = read_template("instruction.md.template")
    instruction_values = {
        # Identifiers
        "task": task_name,
        "task_name": task_name,
        "task_id": task_dir.name,
        "refno": refno,
        # Standard in-container paths
        "pdf_path": paper_artifacts["pdf_path"],
        "meta_path": "/app/task_meta.json",
        "predictions_path": "/app/output/predictions.json",
        # Prompt building blocks (optional; templates may ignore these)
        "question_blocks": "No required properties for this run.",
        "questions_json": "[]",
        "task_meta_json": json.dumps(task_meta, indent=2),
        # Agent affordances (optional)
        "paper_at_command": paper_artifacts["paper_at_command"],
        "pdf_at_command": paper_artifacts["pdf_at_command"],
        "output_at_command": paper_artifacts["output_at_command"],
        "paper_source_at_command": paper_artifacts["paper_source_at_command"],
        "paper_source": paper_artifacts["paper_source"],
        "paper_source_path": paper_artifacts["paper_source_path"],
        "paper_source_dir": paper_artifacts["paper_source_dir"],
        "mineru_dir_path": paper_artifacts["mineru_dir_path"],
        "mineru_clean_markdown_path": paper_artifacts["mineru_clean_markdown_path"],
        "mineru_outline_path": paper_artifacts["mineru_outline_path"],
        "mineru_table_index_path": paper_artifacts["mineru_table_index_path"],
        "mineru_tables_path": paper_artifacts["mineru_tables_path"],
        "mineru_captions_path": paper_artifacts["mineru_captions_path"],
        "mineru_raw_markdown_path": paper_artifacts["mineru_raw_markdown_path"],
        "mineru_primary_markdown_path": paper_artifacts["mineru_primary_markdown_path"],
        "paper_source_description": paper_artifacts["paper_source_description"],
        "gemini_at_commands": paper_artifacts["gemini_at_commands"],
        "claude_file_examples": paper_artifacts["claude_file_examples"],
    }
    instruction = _format_template(instruction_template, instruction_values)
    (task_dir / "instruction.md").write_text(textwrap.dedent(instruction))

    task_toml = _format_template(
        read_template("task.toml.template"),
        {"task_name": task_name, "task": task_name},
    )
    (task_dir / "task.toml").write_text(task_toml)

    (env_dir / "Dockerfile").write_text(dockerfile_contents(paper_source))


def main() -> None:
    """Generate Harbor tasks for the benchmark.

    This is a multi-step pipeline:
      1) Load the HF dataset (refno -> properties).
      2) Flatten rows into per-paper questions.
      3) Materialize Harbor tasks on disk (env/tests/solution + prompt).
      4) Optionally upload tasks to HF and write a registry.json.
    With `--no-score`, steps 1-2 are skipped and tasks are built from PDFs only.
    """
    global _TEMPLATES_SUBDIR

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
        "--no-score",
        action="store_true",
        help=(
            "Build tasks from PDFs only (no dataset, no verifier/solution). "
            "Run Harbor with --disable-verification or use the generated job.yaml."
        ),
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
        default="targeted-template",
        help="Template folder under the workspace (default: targeted-template).",
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
        "--paper-source",
        type=str,
        default="pdf",
        choices=["pdf", "mineru"],
        help=(
            "Primary paper artifact exposed to agents. `pdf` preserves the current "
            "behavior, while `mineru` preprocesses each PDF into a MinerU bundle "
            "and points prompts at the normalized markdown output."
        ),
    )
    parser.add_argument(
        "--mineru-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for normalized MinerU bundles "
            "(default: <workspace>/out/mineru/<dataset>)."
        ),
    )
    parser.add_argument(
        "--mineru-binary",
        type=str,
        default="mineru",
        help="MinerU CLI binary name (default: mineru).",
    )
    parser.add_argument(
        "--mineru-backend",
        type=str,
        default="hybrid-auto-engine",
        choices=[
            "pipeline",
            "hybrid-auto-engine",
            "hybrid-http-client",
            "vlm-auto-engine",
            "vlm-http-client",
        ],
        help="MinerU backend (default: hybrid-auto-engine).",
    )
    parser.add_argument(
        "--mineru-method",
        type=str,
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="MinerU parsing method (default: auto).",
    )
    parser.add_argument(
        "--mineru-lang",
        type=str,
        default=None,
        help="Optional MinerU OCR language hint.",
    )
    parser.add_argument(
        "--mineru-source",
        type=str,
        default=None,
        choices=["huggingface", "modelscope", "local"],
        help="Optional MinerU model source override.",
    )
    parser.add_argument(
        "--mineru-device",
        type=str,
        default=None,
        help="Optional MinerU device override such as cpu, mps, or cuda:0.",
    )
    parser.add_argument(
        "--mineru-extra-arg",
        action="append",
        default=None,
        help="Extra raw CLI arg to pass through to MinerU. Repeat as needed.",
    )
    parser.add_argument(
        "--mineru-formula",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MinerU formula parsing (default: enabled).",
    )
    parser.add_argument(
        "--mineru-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MinerU table parsing (default: enabled).",
    )
    parser.add_argument(
        "--mineru-force",
        action="store_true",
        help="Rebuild MinerU bundles even if the cache already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write generated Harbor tasks "
            "(default: <workspace>/out/harbor/<dataset>)."
        ),
    )
    parser.add_argument(
        "--limit",
        "--max-num-papers",
        "--max_num_papers",
        dest="limit",
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
        "--harbor-task-ordering-registry-path",
        type=Path,
        default=None,
        help=(
            "Optional Harbor registry.json whose task names define the paper order. "
            "Useful for reproducing the same first-N paper slice across direct and "
            "Harbor evaluation flows."
        ),
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
    if args.no_score:
        if args.gt_hf_repo or args.gt_hf_split:
            print(
                "Note: --gt-hf-* flags are ignored when --no-score is set.",
                file=sys.stderr,
            )
        dataset_name = "no-score"
        dataset_split = "none"
        dataset_revision = "none"
    else:
        if not args.gt_hf_repo or not args.gt_hf_split:
            raise SystemExit(
                "--gt-hf-repo and --gt-hf-split are required unless --no-score is set."
            )
        dataset_name = args.gt_hf_repo
        dataset_split = args.gt_hf_split
        dataset_revision = args.gt_hf_revision

    if args.pdf_dir is None:
        args.pdf_dir = resolved_workspace / "data" / "Paper_DB"
    dataset_short_name = dataset_name.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = resolved_workspace / "out" / "harbor" / dataset_short_name
    if not args.pdf_dir.is_absolute():
        args.pdf_dir = resolved_workspace / args.pdf_dir
    if not args.output_dir.is_absolute():
        args.output_dir = resolved_workspace / args.output_dir
    if args.harbor_task_ordering_registry_path is not None and (
        not args.harbor_task_ordering_registry_path.is_absolute()
    ):
        args.harbor_task_ordering_registry_path = (
            resolved_workspace / args.harbor_task_ordering_registry_path
        )
    if args.mineru_cache_dir is None:
        args.mineru_cache_dir = (
            resolved_workspace / "out" / "mineru" / dataset_short_name
        )
    if not args.mineru_cache_dir.is_absolute():
        args.mineru_cache_dir = resolved_workspace / args.mineru_cache_dir

    mineru_config: MinerUConfig | None = None
    if args.paper_source == "mineru":
        mineru_config = MinerUConfig(
            binary=args.mineru_binary,
            backend=args.mineru_backend,
            method=args.mineru_method,
            formula=bool(args.mineru_formula),
            table=bool(args.mineru_table),
            lang=args.mineru_lang,
            source=args.mineru_source,
            device=args.mineru_device,
            extra_args=list(args.mineru_extra_arg or []),
            force=bool(args.mineru_force),
        )
        args.mineru_cache_dir.mkdir(parents=True, exist_ok=True)

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

    rubric_mapping: dict[str, str] = {}
    definitions: dict[str, str] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}

    if not args.no_score:
        rubric_path = resolved_workspace / "scoring" / "rubric.csv"
        rubric_mapping = load_rubric_mapping(rubric_path)
        definitions = load_definitions(rubric_path)

        # Load dataset from configuration
        print(
            f"Loading dataset: {dataset_name} (revision: {dataset_revision}, split: {dataset_split})"
        )
        dataset = load_dataset(
            dataset_name, split=dataset_split, revision=dataset_revision
        )
        property_filter = resolve_property_filter(args.task)
        grouped = flatten_dataset(
            cast(Iterable[dict[str, Any]], dataset),
            definitions=definitions,
            property_filter=property_filter,
        )

        refnos = list(grouped.keys())
    else:
        pdf_paths = sorted(
            path for path in args.pdf_dir.iterdir() if path.suffix.lower() == ".pdf"
        )
        refnos = [path.stem for path in pdf_paths]
    pdf_lookup = build_pdf_lookup(args.pdf_dir)
    refno_lookup = {refno.lower(): refno for refno in refnos}
    limit_applied = False
    if args.refno:
        requested = [value.strip() for value in args.refno if value and value.strip()]
        missing = sorted(
            value for value in requested if value.lower() not in refno_lookup
        )
        if missing:
            raise ValueError(f"Unknown refno(s) requested: {missing}")
        requested_order: list[str] = []
        seen: set[str] = set()
        for value in requested:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            requested_order.append(refno_lookup[key])
        refnos = requested_order
    elif args.harbor_task_ordering_registry_path is not None:
        requested = load_harbor_task_ordering(args.harbor_task_ordering_registry_path)
        if args.limit is not None:
            requested = requested[: args.limit]
            limit_applied = True
        missing = sorted(
            value for value in requested if value.lower() not in refno_lookup
        )
        if missing:
            raise ValueError(
                "The ordering registry references papers that are missing from the "
                f"current dataset/pdf-dir: {missing[:20]}"
            )
        requested_order = []
        seen = set()
        for value in requested:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            requested_order.append(refno_lookup[key])
        refnos = requested_order
    if args.limit is not None and not limit_applied:
        refnos = refnos[: args.limit]

    if args.no_score and not refnos:
        raise FileNotFoundError(f"No PDFs found under {args.pdf_dir}")

    written_task_dirs: list[Path] = []
    for refno in refnos:
        pdf_path = resolve_pdf_path(pdf_lookup, refno)
        mineru_bundle: MinerUBundlePaths | None = None
        if mineru_config is not None:
            mineru_bundle = ensure_mineru_bundle(
                pdf_path=pdf_path,
                bundle_root=args.mineru_cache_dir,
                config=mineru_config,
            )

        task_id = (
            f"{slugify(refno)}--{slugify(task_label)}"
            if args.task is not None
            else slugify(refno)
        )
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        if args.no_score:
            build_task_no_score(
                task_dir,
                pdf_path=pdf_path,
                paper_source=args.paper_source,
                mineru_bundle=mineru_bundle,
                task_name=task_label,
                refno=refno,
            )
        else:
            rows = grouped.get(refno, [])
            if not rows:
                print(f"Skipping {refno}: no properties matched task '{args.task}'.")
                continue

            build_task(
                task_dir,
                pdf_path=pdf_path,
                paper_source=args.paper_source,
                mineru_bundle=mineru_bundle,
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
        written_task_dirs.append(task_dir)

    if args.write_job_config:
        job_path = task_root / "job.yaml"
        write_job_config(
            tasks_dir,
            job_path,
            workspace=resolved_workspace,
            disable_verification=bool(args.no_score),
            agent_name="gemini-cli" if args.no_score else "oracle",
        )
        try:
            job_rel = job_path.relative_to(resolved_workspace)
        except ValueError:
            job_rel = job_path
        print(f"Wrote job config -> {job_rel}")

    # -- Write local registry.json --
    # NOTE: the local registry JSON is consistent with the HF upload registry JSON,
    # just with local task paths.
    generated_task_dirs = _filter_valid_task_dirs(
        written_task_dirs, disable_verification=bool(args.no_score)
    )
    if not generated_task_dirs:
        generated_task_dirs = _collect_task_dirs(
            tasks_dir, disable_verification=bool(args.no_score)
        )
    if args.seed is not None:
        generated_task_dirs = _shuffle_task_dirs(generated_task_dirs, args.seed)
    if generated_task_dirs:
        registry_path = task_root / "registry.json"
        dataset_short_name = dataset_name.split("/")[-1]
        write_local_registry(
            generated_task_dirs,
            registry_path,
            dataset_name=dataset_short_name,
            dataset_version=dataset_revision,
            description=f"Harbor tasks for {dataset_name} ({task_label}).",
        )
        try:
            registry_rel = registry_path.relative_to(resolved_workspace)
        except ValueError:
            registry_rel = registry_path
        print(f"Wrote registry -> {registry_rel}")

    if args.no_score:
        print(
            "No-score mode: tasks omit verifier/solution. Run Harbor with "
            "--disable-verification (or use the generated job.yaml)."
        )

    if args.upload_hf:
        # import pdb; pdb.set_trace()
        _upload_tasks_after_build(
            args=args,
            tasks_root=task_root,
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


def _collect_task_dirs(
    tasks_root: Path, *, disable_verification: bool = False
) -> list[Path]:
    """Return Harbor-valid task directories under the tasks root."""
    return [
        child
        for child in sorted(tasks_root.iterdir())
        if child.is_dir()
        and TaskPaths(child).is_valid(disable_verification=disable_verification)
    ]


def _filter_valid_task_dirs(
    task_dirs: Iterable[Path], *, disable_verification: bool = False
) -> list[Path]:
    """Keep Harbor-valid task directories while preserving the caller's order."""
    return [
        task_dir
        for task_dir in task_dirs
        if task_dir.is_dir()
        and TaskPaths(task_dir).is_valid(disable_verification=disable_verification)
    ]


def _shuffle_task_dirs(task_dirs: list[Path], seed: int) -> list[Path]:
    """Shuffle task directories with a deterministic seed."""
    if True:
        logger.warning("Using custom shuffling with always-include papers.")
        # HACK: include papers that Chao and Fatmagul have already validated at the start.
        REFNOS_ALWAYS_INCLUDE = [
            "0304328",
            "0505463",
            "0804.1930",
            "0807.2541",
            "0811.0342",
            "0812.1214",
            "0903.4018",
            "0908.0518",
            "1312.5475",
            "1401.0712",
            "1401.1975",
            "1602.07983",
            "1612.04105",
            "1711.09143",
            "1906.07149",
            "1910.05526",
            "2001.05649",
            "2111.01152",
            "2302.10031",
            "9902061",
            "9907030",
            "9912178",
        ]
        always_include_dirs = [
            d for d in task_dirs if d.name.replace("-", ".") in REFNOS_ALWAYS_INCLUDE
        ]
        remaining_dirs = [
            d
            for d in task_dirs
            if d.name.replace("-", ".") not in REFNOS_ALWAYS_INCLUDE
        ]
        random.Random(seed).shuffle(remaining_dirs)
        return always_include_dirs + remaining_dirs
    else:
        shuffled = list(task_dirs)
        random.Random(seed).shuffle(shuffled)
        return shuffled


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
    registry_path: str = "registry.json",
    dataset_name: str | None = None,
    dataset_version: str = "head",
    description: str = "Harbor tasks uploaded from a local tasks directory.",
    create: bool = True,
    private: bool | None = None,
    token: str | None = None,
    disable_verification: bool = False,
    seed: int | None = None,
) -> dict[str, str]:
    """Upload Harbor tasks and registry.json to a Hugging Face repo.

    Returns a small summary dict for logging.
    """
    task_dirs = _collect_task_dirs(
        tasks_root / "tasks", disable_verification=disable_verification
    )
    if seed is not None:
        task_dirs = _shuffle_task_dirs(task_dirs, seed)

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
        registry_path=str(args.hf_registry_path),
        dataset_name=args.hf_dataset_name or str(repo_id),
        dataset_version=str(args.hf_dataset_version),
        description=str(args.hf_description),
        create=bool(args.hf_create),
        private=private,
        disable_verification=bool(getattr(args, "no_score", False)),
        seed=getattr(args, "seed", None),
    )

    print(
        f"Uploaded {summary['task_count']} tasks to {repo_id}:{summary['path_in_repo']}"
    )
    print(f"Registry URL: {summary['registry_url']}")
    print(f"Dataset name: {summary['dataset_name']}")


if __name__ == "__main__":
    main()
