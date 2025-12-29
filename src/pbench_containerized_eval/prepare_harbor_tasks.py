r"""Compile Harbor tasks for SuperCon property extraction from a folder of PDFs.

This "task compiler" turns a (PDF, ground-truth) dataset into Harbor task directories,
each with:
  - `environment/`: Docker build context with the paper PDF
  - `instruction.md`: a single prompt/instruction file shared across tasks via a template
  - `tests/`: verifier that scores predictions using rubric tolerances
  - `solution/`: an oracle solution used by Harbor's built-in `oracle` agent

The source of truth for the benchmark is the Hugging Face dataset
`kilian-group/supercon-mini-v2`, grouped by `refno` (one Harbor task per paper).

By default this script writes tasks under `out/harbor/supercon-mini-v2/<task>/<template>/` so the
repository doesn't contain a persistent `harbor/` directory until you build.

Example (from repo root):
    # Default template is `ground-template`.
    uv run python src/pbench_containerized_eval/prepare_harbor_tasks.py --task tc --force
    uv run python src/pbench_containerized_eval/run_harbor.py jobs start \\
      -c out/harbor/supercon-mini-v2/ground-template/job.yaml -a oracle

    # To use the question-based template:
    uv run python src/pbench_containerized_eval/prepare_harbor_tasks.py --task tc --template prompted-template --force
    uv run python src/pbench_containerized_eval/run_harbor.py jobs start \\
      -c out/harbor/supercon-mini-v2/prompted-template/job.yaml -a oracle
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from datasets import load_dataset


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def templates_dir() -> Path:
    """Return the directory containing files copied into generated Harbor tasks."""
    return Path(__file__).parent / _TEMPLATES_SUBDIR


_TEMPLATES_SUBDIR = "ground-template"

_TASK_PROPERTY_FILTERS: dict[str, set[str]] = {
    # Default task: superconducting critical temperature recommended for the sample.
    "tc": {"Tc (of this sample) recommended"},
}


def read_template(relative_path: str) -> str:
    """Read a template file relative to `task_templates/`."""
    return (templates_dir() / relative_path).read_text()


def copy_template(relative_path: str, dest_path: Path) -> None:
    """Copy a template file relative to `task_templates/` to the destination path."""
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


def write_job_config(tasks_dir: Path, job_path: Path) -> None:
    """Write a Harbor job YAML pointing at the generated tasks."""
    tasks_rel = (
        tasks_dir.relative_to(repo_root()) if tasks_dir.is_absolute() else tasks_dir
    )
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
    """Generate Harbor tasks for the benchmark."""
    global _TEMPLATES_SUBDIR

    parser = argparse.ArgumentParser(
        description="Generate Harbor tasks for the superconductor extraction benchmark."
    )
    parser.add_argument(
        "--template",
        type=str,
        default="ground-template",
        choices=["ground-template", "prompted-template"],
        help="Which template bundle to use under src/pbench_containerized_eval/.",
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
        default=Path(__file__).parent / "data" / "Paper_DB",
        help="Directory containing PDFs named <refno>.pdf.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "out" / "harbor" / "supercon-mini-v2",
        help="Where to write generated Harbor tasks (default: out/harbor/supercon-mini-v2).",
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
    args = parser.parse_args()

    _TEMPLATES_SUBDIR = str(args.template)

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

    rubric_path = Path(__file__).parent / "rubric.csv"
    rubric_mapping = load_rubric_mapping(rubric_path)
    definitions = load_definitions(rubric_path)

    dataset = load_dataset(
        "kilian-group/supercon-mini-v2", split="test", revision="v2.0.1"
    )
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
        print(f"Wrote task {task_id} -> {task_dir.relative_to(repo_root())}")

    if args.write_job_config:
        job_path = task_root / "job.yaml"
        write_job_config(tasks_dir, job_path)
        print(f"Wrote job config -> {job_path.relative_to(repo_root())}")


if __name__ == "__main__":
    main()
