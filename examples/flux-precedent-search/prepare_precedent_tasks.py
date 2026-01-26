"""Generate Harbor tasks for the Flux precedent search (dev set)."""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

def templates_dir() -> Path:
    return Path("search-template")

def pbench_eval_dir() -> Path:
    return Path("../../src/pbench_eval")

def read_template(relative_path: str) -> str:
    return (templates_dir() / relative_path).read_text()

def copy_template(relative_path: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(templates_dir() / relative_path, dest_path)

def dockerfile_contents() -> str:
    return read_template("environment/Dockerfile")

def slugify(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
    )

def build_task(task_dir: Path, row: dict[str, str], task_name: str) -> None:
    # 1. Setup dirs
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"
    
    for d in [env_dir, tests_dir, solution_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Extract values
    material = row["material"]
    # Map yes/no to Yes/No
    raw_is_grown = row.get("is_grown_with_flux", "").lower()
    if raw_is_grown == "yes":
        is_grown = "Yes"
    elif raw_is_grown == "no":
        is_grown = "No"
    else:
        is_grown = "Unknown"

    # 3. Create Expected JSON (Ground Truth)
    expected_rows = [
        {
            "material": material,
            "property_name": "is_grown_with_flux",
            "property_value": is_grown,
            "rubric": "categorical"
        }
    ]
    
    expected = {
        "task": task_name,
        "refno": "flux-precedent-search",
        "ground_truth": expected_rows,
    }
    (tests_dir / "expected.json").write_text(json.dumps(expected, indent=2))
    
    # 4. Instruction
    instruction_template = read_template("instruction.md.template")
    # Simple substitution
    instruction = instruction_template.replace("{material}", material)
    instruction = instruction.replace("{predictions_path}", "/app/output/predictions.json")
    (task_dir / "instruction.md").write_text(instruction)
    
    # 5. Task TOML
    task_toml = read_template("task.toml.template").replace("{task_name}", task_name)
    (task_dir / "task.toml").write_text(task_toml)
    
    # 6. Dockerfile and Test Scripts
    (env_dir / "Dockerfile").write_text(dockerfile_contents())
    copy_template("tests/check_prediction.py", tests_dir / "check_prediction.py")
    copy_template("tests/test.sh", tests_dir / "test.sh")

    # Copy shared scoring utils
    utils_path = tests_dir / "pbench_eval_utils.py"
    shutil.copy2(pbench_eval_dir() / "utils.py", utils_path)
    
    # PATCH: Remove relative imports and dependencies
    content = utils_path.read_text()
    # Replace relative imports with mocks
    content = re.sub(r'from \.space_groups_normalized import\s*\([^)]+\)', 'SPACE_GROUPS = {}', content, flags=re.DOTALL)
    content = re.sub(r'from \.normalize_material import classify_and_normalize, strip_formula', 
                     'def classify_and_normalize(*args, **kwargs): return None, None, None\ndef strip_formula(s): return s, {}', 
                     content)
    utils_path.write_text(content)
    
    # 7. Oracle Solution (solve.sh)
    prediction_rows = []
    for er in expected_rows:
        prediction_rows.append({
            "material": er["material"],
            "property_name": er["property_name"],
            "value_string": er["property_value"] 
        })
        
    solution_json = { "properties": prediction_rows }
    
    solution_script = f"""#!/bin/bash
set -euo pipefail

mkdir -p /app/output
cat > /app/output/predictions.json <<'EOF'
{json.dumps(solution_json, indent=2)}
EOF
"""
    (solution_dir / "solve.sh").write_text(solution_script)
    (solution_dir / "solve.sh").chmod(0o755)
    (tests_dir / "test.sh").chmod(0o755)


def write_job_config(tasks_dir: Path, job_path: Path) -> None:
    repo_root = Path("../..").resolve()
    workspace_root = (repo_root / "examples/harbor-workspace").resolve()
    
    if not tasks_dir.is_absolute():
        tasks_full = (Path.cwd() / tasks_dir).resolve()
    else:
        tasks_full = tasks_dir.resolve()
        
    try:
        tasks_rel = tasks_full.relative_to(workspace_root)
    except ValueError:
        print(f"Warning: Tasks dir {tasks_full} is not within workspace {workspace_root}. Using repo-relative path.")
        tasks_rel = tasks_full.relative_to(repo_root)

    job_filename = "job.yaml"

    job_yaml = f"""\
jobs_dir: jobs
n_attempts: 1
timeout_multiplier: 1.0
orchestrator:
  type: local
  n_concurrent_trials: 10
  quiet: false
environment:
  type: docker
  force_build: true
  delete: true
agents:
  - name: oracle
datasets:
  - path: {tasks_rel}
"""
    job_config_path = tasks_dir.parent / job_filename
    job_config_path.parent.mkdir(parents=True, exist_ok=True)
    with job_config_path.open("w") as f:
        f.write(job_yaml)
    print(f"Wrote job config -> {job_config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("flux-material_dev-set.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("../../examples/harbor-workspace/out/harbor/precedent-search"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-job-config", action="store_true")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    
    input_csv = args.csv
    print(f"Using input CSV: {input_csv}")

    task_root = args.output_dir / "flux-precedent-search"
    tasks_dir_name = "tasks"
    tasks_dir = task_root / tasks_dir_name
    
    if task_root.exists():
        if args.force:
            shutil.rmtree(task_root)
        elif any(task_root.iterdir()):
            print(f"Directory {task_root} exists. Use --force to overwrite.")
            return

    tasks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV: {input_csv}")
    with input_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if args.limit:
        rows = rows[:args.limit]
        
    print(f"Generating tasks for {len(rows)} materials...")
    
    for row in rows:
        material = row["material"]
        # Sanitize material name
        safe_material = material.replace("+", "plus")
        safe_material = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", safe_material).lower()
        
        task_dir_name = safe_material
        task_dir = tasks_dir / task_dir_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        build_task(task_dir, row, "flux-precedent-search")
        
    print(f"All tasks built in {tasks_dir}")

    if args.write_job_config:
        job_path = task_root / "job.yaml"
        write_job_config(tasks_dir, job_path)
        print(f"Wrote job config -> {job_path}")

if __name__ == "__main__":
    main()
