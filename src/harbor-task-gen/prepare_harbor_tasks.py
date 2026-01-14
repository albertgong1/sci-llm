r"""Compile Harbor tasks from a (PDF, ground-truth) dataset.

This "task compiler" turns a (PDF, ground-truth) dataset into Harbor task directories,
each with:
  - `environment/`: Docker build context with the paper PDF
  - `instruction.md`: a single prompt/instruction file shared across tasks via a template
  - `tests/`: verifier that scores predictions using rubric tolerances
  - `solution/`: an oracle solution used by Harbor's built-in `oracle` agent

By default the dataset is pulled from Hugging Face (currently
`kilian-group/supercon-mini-v2`), grouped by `refno` (one Harbor task per paper).
All dataset/schema fields are configurable via CLI flags so the framework can
support other task families.

Optional: pass `--upload-hf` to upload the generated tasks to a Hugging Face repo
and write a `registry.json` so Harbor can pull tasks directly from the Hub.

By default this script writes tasks under
`examples/harbor-workspace/out/harbor/<dataset>/<task>/<template>/` so the
repository stays clean until you build.

Example (from repo root):
    # Default template is `ground-template`.
    uv run python src/harbor-task-gen/prepare_harbor_tasks.py --task tc --force
    uv run python src/harbor-task-gen/run_harbor.py jobs start \\
      -c out/harbor/<dataset>/ground-template/job.yaml -a oracle

    # To use a custom template:
    uv run python src/harbor-task-gen/prepare_harbor_tasks.py \\
      --task tc --template my-template --force
    uv run python src/harbor-task-gen/run_harbor.py jobs start \\
      -c out/harbor/<dataset>/my-template/job.yaml -a oracle
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
from dataclasses import dataclass
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

DEFAULT_TASK_PROPERTY_FILTERS: dict[str, set[str]] = {
    # SuperCon-specific alias: superconducting critical temperature recommended for the sample.
    "tc": {"Tc (of this sample) recommended"},
}

DEFAULT_QUESTION_TEMPLATE = (
    "[{index}]\n"
    "Question: What is the {property_name} for {material}?{definition_text}\n"
    "Answer:"
)


def _resolve_template_path(path: str | Path) -> Path:
    """Resolve a template path relative to the template root unless absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return templates_dir() / candidate


def read_template(path: str | Path) -> str:
    """Read a template file (relative to template root unless absolute)."""
    return _resolve_template_path(path).read_text()


def copy_template(source_path: str | Path, dest_path: Path) -> None:
    """Copy a template file (relative to template root unless absolute)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_resolve_template_path(source_path), dest_path)


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


def load_rubric_mapping(rubric_path: Path | None) -> dict[str, str]:
    """Load the property_name -> rubric mapping from the rubric CSV."""
    if rubric_path is None or not rubric_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with rubric_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            property_name = row.get("property_name")
            rubric = row.get("rubric")
            if property_name and rubric:
                mapping[property_name] = rubric
    return mapping


def load_definitions(rubric_path: Path | None) -> dict[str, str]:
    """Load property_name -> definition mapping from the rubric CSV (if present)."""
    if rubric_path is None or not rubric_path.exists():
        return {}
    definitions: dict[str, str] = {}
    with rubric_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            property_name = row.get("property_name")
            definition = row.get("definition") or ""
            if property_name:
                definitions[property_name] = definition
    return definitions


def dockerfile_contents_from_template(dockerfile_template: Path) -> str:
    """Render the task environment Dockerfile from the given template.

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
        read_template(dockerfile_template),
        {"install_pdf_tools": install_pdf_tools},
    )


def _load_property_filter_file(path: Path) -> set[str]:
    """Load property_name filters from a text or CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Property filter file not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            values = [
                str(row.get("property_name") or "").strip() for row in reader if row
            ]
        return {value for value in values if value}
    values = [line.strip() for line in path.read_text().splitlines()]
    return {value for value in values if value}


def _load_task_filter_map(path: Path) -> dict[str, set[str]]:
    """Load task -> property_name mapping from a JSON or CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Task filter map not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(
                "Task filter JSON must map task -> list of property names."
            )
        mapping: dict[str, set[str]] = {}
        for key, value in payload.items():
            if not key:
                continue
            if isinstance(value, list):
                props = {str(item).strip() for item in value if str(item).strip()}
            else:
                props = {str(value).strip()} if str(value).strip() else set()
            if props:
                mapping[str(key).strip().lower()] = props
        return mapping

    mapping = defaultdict(set)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = str(row.get("task") or "").strip().lower()
            prop = str(row.get("property_name") or "").strip()
            if task and prop:
                mapping[task].add(prop)
    return dict(mapping)


def resolve_property_filter(
    task: str | None,
    *,
    explicit_filter: set[str] | None,
    task_filter_map: dict[str, set[str]] | None,
) -> set[str] | None:
    """Return the set of property_names to keep (or None for all)."""
    if explicit_filter:
        return explicit_filter
    if task is None:
        return None
    mapping = task_filter_map or DEFAULT_TASK_PROPERTY_FILTERS
    return mapping.get(task.strip().lower())


@dataclass(frozen=True)
class DatasetSchema:
    """Field mapping for dataset rows and nested properties."""

    refno_field: str
    properties_field: str | None
    material_field: str
    property_name_field: str
    value_field: str
    unit_field: str | None
    definition_field: str | None


@dataclass(frozen=True)
class TemplateFiles:
    """Resolved template paths used when building tasks."""

    instruction: Path
    task_toml: Path
    dockerfile: Path
    check_prediction: Path
    test_script: Path


def _resolve_template_override(
    value: str,
    *,
    template_root: Path,
    label: str,
) -> Path:
    """Resolve and validate a template path override."""
    candidate = Path(value)
    resolved = candidate if candidate.is_absolute() else template_root / candidate
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def flatten_dataset(
    dataset: Iterable[dict[str, Any]],
    *,
    schema: DatasetSchema,
    definitions: Mapping[str, str],
    property_filter: set[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Flatten dataset rows into per-property rows grouped by refno."""
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in dataset:
        refno = str(row.get(schema.refno_field) or "").strip()
        if not refno:
            continue

        props: list[dict[str, Any]] = []
        if schema.properties_field:
            raw_props = row.get(schema.properties_field) or []
            if not isinstance(raw_props, list):
                continue
            props = [prop for prop in raw_props if isinstance(prop, dict)]
        else:
            if isinstance(row, dict):
                props = [row]
        if not props:
            continue

        for prop in props:
            prop_name = str(prop.get(schema.property_name_field) or "").strip()
            if not prop_name:
                continue
            if property_filter and prop_name not in property_filter:
                continue
            unit_value = (
                str(prop.get(schema.unit_field) or "").strip()
                if schema.unit_field
                else ""
            )
            definition = (
                str(prop.get(schema.definition_field) or "").strip()
                if schema.definition_field
                else ""
            )

            grouped[refno].append(
                {
                    "material": str(prop.get(schema.material_field) or ""),
                    "property_name": prop_name,
                    # value_string already contains any unit; keep unit empty to avoid double-parsing.
                    "property_value": str(prop.get(schema.value_field) or ""),
                    "property_unit": unit_value,
                    "definition": definition or definitions.get(prop_name, ""),
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
    question_template: str,
    template_files: TemplateFiles,
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

    question_blocks: list[str] = []
    for idx, item in enumerate(questions):
        definition = str(item.get("definition", "") or "").strip()
        definition_text = f' Definition: "{definition}".' if definition else ""
        question_blocks.append(
            textwrap.dedent(
                _format_template(
                    question_template,
                    {
                        "index": idx,
                        "idx": idx,
                        "property_name": item["property_name"],
                        "material": item["material"],
                        "definition": definition,
                        "definition_text": definition_text,
                        "task": task_name,
                        "refno": refno,
                    },
                )
            ).strip()
        )
    question_blocks = "\n\n".join(question_blocks)
    gemini_at_commands = "`@paper.pdf`"
    paper_at_command = "@paper.pdf"
    claude_file_examples = "`/app/paper.pdf`"

    instruction_template = read_template(template_files.instruction)
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
        read_template(template_files.task_toml),
        {"task_name": task_name, "task": task_name},
    )
    (task_dir / "task.toml").write_text(task_toml)

    (env_dir / "Dockerfile").write_text(
        dockerfile_contents_from_template(template_files.dockerfile)
    )
    copy_template(template_files.check_prediction, tests_dir / "check_prediction.py")
    copy_template(template_files.test_script, tests_dir / "test.sh")

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
        description="Generate Harbor tasks from a PDF + labeled dataset."
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
        "--instruction-template",
        type=str,
        default="instruction.md.template",
        help="Instruction template path (relative to template root unless absolute).",
    )
    parser.add_argument(
        "--task-toml-template",
        type=str,
        default="task.toml.template",
        help="task.toml template path (relative to template root unless absolute).",
    )
    parser.add_argument(
        "--dockerfile-template",
        type=str,
        default="environment/Dockerfile",
        help="Dockerfile template path (relative to template root unless absolute).",
    )
    parser.add_argument(
        "--scoring-script",
        type=str,
        default="tests/check_prediction.py",
        help=(
            "Verifier script path (relative to template root unless absolute). "
            "Copied to tests/check_prediction.py in each task."
        ),
    )
    parser.add_argument(
        "--test-script",
        type=str,
        default="tests/test.sh",
        help=(
            "Test runner script path (relative to template root unless absolute). "
            "Copied to tests/test.sh in each task."
        ),
    )
    parser.add_argument(
        "--question-template",
        type=str,
        default=None,
        help="Template for each question block (default: built-in).",
    )
    parser.add_argument(
        "--question-template-file",
        type=Path,
        default=None,
        help=(
            "Optional file containing a question template (overrides --question-template). "
            "Relative paths are resolved from the template root."
        ),
    )
    parser.add_argument(
        "--rubric-path",
        type=Path,
        default=None,
        help=(
            "Rubric CSV path used to annotate expected.json with rubrics/definitions. "
            "Defaults to <workspace>/<template>/rubric.csv (fallback: <workspace>/rubric.csv)."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Task alias for filtering property_names (e.g., tc). "
            "If omitted, include all properties."
        ),
    )
    parser.add_argument(
        "--task-filter-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON/CSV mapping of task alias -> property_name list. "
            "Overrides built-in task filters."
        ),
    )
    parser.add_argument(
        "--property-filter",
        action="append",
        default=None,
        help=(
            "Explicit property_name to include (repeatable). "
            "Overrides --task filter mapping when provided."
        ),
    )
    parser.add_argument(
        "--property-filter-file",
        type=Path,
        default=None,
        help="Path to a text/CSV file listing property_name values to include.",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="kilian-group/supercon-mini-v2",
        help="Hugging Face dataset repo id (default: kilian-group/supercon-mini-v2).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split name (default: test).",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=None,
        help="Dataset revision/commit (default: repo default).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset config name for multi-config datasets.",
    )
    parser.add_argument(
        "--dataset-refno-field",
        type=str,
        default="refno",
        help="Row field holding the paper refno (default: refno).",
    )
    parser.add_argument(
        "--dataset-properties-field",
        type=str,
        default="properties",
        help=(
            "Row field holding the list of properties (default: properties). "
            "Set to '-' to treat each row as a property."
        ),
    )
    parser.add_argument(
        "--dataset-material-field",
        type=str,
        default="material_or_system",
        help="Property field for material/system (default: material_or_system).",
    )
    parser.add_argument(
        "--dataset-property-name-field",
        type=str,
        default="property_name",
        help="Property field for property_name (default: property_name).",
    )
    parser.add_argument(
        "--dataset-value-field",
        type=str,
        default="value_string",
        help="Property field for value string (default: value_string).",
    )
    parser.add_argument(
        "--dataset-unit-field",
        type=str,
        default="",
        help=(
            "Property field for units (default: empty). "
            "Leave empty if units are embedded in the value string."
        ),
    )
    parser.add_argument(
        "--dataset-definition-field",
        type=str,
        default="",
        help=(
            "Property field for definitions (default: empty). "
            "If empty, definitions are pulled from the rubric CSV when available."
        ),
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
            "(default: <workspace>/out/harbor/<dataset>)."
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

    if args.pdf_dir is None:
        args.pdf_dir = resolved_workspace / "data" / "Paper_DB"
    if args.output_dir is None:
        dataset_slug = args.dataset_repo.split("/")[-1]
        args.output_dir = resolved_workspace / "out" / "harbor" / dataset_slug
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

    template_files = TemplateFiles(
        instruction=_resolve_template_override(
            args.instruction_template,
            template_root=template_root,
            label="Instruction template",
        ),
        task_toml=_resolve_template_override(
            args.task_toml_template,
            template_root=template_root,
            label="task.toml template",
        ),
        dockerfile=_resolve_template_override(
            args.dockerfile_template,
            template_root=template_root,
            label="Dockerfile template",
        ),
        check_prediction=_resolve_template_override(
            args.scoring_script,
            template_root=template_root,
            label="Scoring script",
        ),
        test_script=_resolve_template_override(
            args.test_script,
            template_root=template_root,
            label="Test script",
        ),
    )

    question_template = args.question_template or DEFAULT_QUESTION_TEMPLATE
    if args.question_template_file:
        question_path = (
            args.question_template_file
            if args.question_template_file.is_absolute()
            else template_root / args.question_template_file
        )
        if not question_path.exists():
            raise FileNotFoundError(
                f"Question template file not found: {question_path}"
            )
        question_template = question_path.read_text()

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

    rubric_path = None
    if args.rubric_path:
        rubric_path = (
            args.rubric_path
            if args.rubric_path.is_absolute()
            else resolved_workspace / args.rubric_path
        )
    else:
        template_rubric = template_root / "rubric.csv"
        workspace_rubric = resolved_workspace / "rubric.csv"
        if template_rubric.exists():
            rubric_path = template_rubric
        elif workspace_rubric.exists():
            rubric_path = workspace_rubric

    rubric_mapping = load_rubric_mapping(rubric_path)
    definitions = load_definitions(rubric_path)

    properties_field_raw = (args.dataset_properties_field or "").strip().lower()
    properties_field = (
        None
        if properties_field_raw in {"", "-", "none", "null"}
        else args.dataset_properties_field
    )
    unit_field = args.dataset_unit_field.strip() or None
    definition_field = args.dataset_definition_field.strip() or None

    schema = DatasetSchema(
        refno_field=args.dataset_refno_field,
        properties_field=properties_field,
        material_field=args.dataset_material_field,
        property_name_field=args.dataset_property_name_field,
        value_field=args.dataset_value_field,
        unit_field=unit_field,
        definition_field=definition_field,
    )

    dataset_kwargs: dict[str, Any] = {"split": args.dataset_split}
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config
    if args.dataset_revision:
        dataset_kwargs["revision"] = args.dataset_revision
    dataset = load_dataset(args.dataset_repo, **dataset_kwargs)

    explicit_filter: set[str] | None = None
    if args.property_filter:
        explicit_filter = {value.strip() for value in args.property_filter if value}
    if args.property_filter_file:
        filter_path = (
            args.property_filter_file
            if args.property_filter_file.is_absolute()
            else resolved_workspace / args.property_filter_file
        )
        explicit_filter = _load_property_filter_file(filter_path)

    task_filter_map = None
    if args.task_filter_map:
        map_path = (
            args.task_filter_map
            if args.task_filter_map.is_absolute()
            else resolved_workspace / args.task_filter_map
        )
        task_filter_map = _load_task_filter_map(map_path)

    property_filter = resolve_property_filter(
        args.task,
        explicit_filter=explicit_filter,
        task_filter_map=task_filter_map,
    )
    grouped = flatten_dataset(
        cast(Iterable[dict[str, Any]], dataset),
        schema=schema,
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
            question_template=question_template,
            template_files=template_files,
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
            private=True if private is None else private,
            exist_ok=True,
        )
    else:
        try:
            api.list_repo_files(repo_id=str(repo_id), repo_type=str(repo_type))
        except Exception as exc:
            raise SystemExit(f"Repo not found or not accessible: {repo_id}") from exc

    api.upload_folder(
        repo_id=str(repo_id),
        repo_type=str(repo_type),
        folder_path=str(resolved_root),
        path_in_repo=path_in_repo or None,
        commit_message=f"Upload Harbor tasks from {resolved_root.name}",
        token=hf_token,
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
