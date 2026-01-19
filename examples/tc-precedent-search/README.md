# Tc precedent Search Evaluation

This directory contains the workflow for evaluating the "Precedent Search" task, where an agent must determine if a material has been reported as superconducting, and if so, what the highest Tc and lowest Tcn (non-superconducting temperature) are.

## Workflow

### 1. Generate Development Set
Download [SuperCon_Tc_Tcn - no NA.csv](https://drive.google.com/file/d/11mqYhvSbl_cCNIDiQOgnS5N1P-26BREC/view?usp=sharing) and save to `examples/tc-precedent-search/` directory.

> \[!NOTE\]
> 1. Total SuperCon rows: 21,153
> 2. Rows where `file` column is #N/A: 4,534
> 3. Rows where `file` column is blank: 14,226
> - `SuperCon_Tc_Tcn - no NA.csv` contains 16,619 rows (we just exclude the N/A rows)
> - The dev set is a subset of `SuperCon_Tc_Tcn - no NA.csv` with 200 randomly selected rows. 100 rows with materials that are superconducting (`Tc`) and 100 that are not (`Tcn`).

**Command:**
```bash
cd examples/tc-precedent-search/
python create_dev_set.py
```

**Inputs:**
- `assets/SuperCon_Tc_Tcn - no NA.csv`: Source SuperCon dataset.

**Outputs:**
- `examples/tc-precedent-search/SuperCon_Tc_Tcn_dev-set.csv`: The generated development set.
- You can verify the dev set is correct by comparing with the dev set [here](https://drive.google.com/file/d/13nb4HTU2p28b8oiEbQ87Y1CF4Jrcwlsa/view?usp=sharing).

### 2. Generate Harbor Tasks
Generate the task directories for Harbor evaluation. This script reads the dev set and creates a folder for each material, containing the instructions, scoring logic, and Docker environment.

**Command:**
```bash
cd examples/tc-precedent-search/
python prepare_precedent_tasks.py --force --write-job-config
```

**Options:**
- `--csv`: Path to the input CSV (default: `SuperCon_Tc_Tcn_dev-set.csv`).
- `--limit N`: Generate only N tasks (useful for testing).
- `--force`: Overwrite existing output directory.
- `--write-job-config`: Write the `job.yaml` file for Harbor.

**Inputs:**
- `SuperCon_Tc_Tcn_dev-set.csv`: The dev set.
- `search-template/`: Template files for the tasks.

**Outputs:**
- `examples/harbor-workspace/out/harbor/precedent-search/tc-precedent-search/tasks/`: Directory containing one task folder per material.
- `examples/harbor-workspace/out/harbor/precedent-search/tc-precedent-search/job.yaml`: Job configuration file.

> **Note on Docker Image Rebuilding:**
> Running this step creates fresh `Dockerfile`s in each task's `environment/` directory. When you run Step 3 (trials), Harbor will detect these new Dockerfiles and rebuild the Docker images. However, once built, these images are cached and reused for subsequent trial runs unless you re-run this step or modify the Dockerfile template.

### 3. Run Trials (Harbor)
Execute the agents on the generated tasks using Harbor.

**Command (Run one task):**
```bash
# From the root
python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace trials start \
  -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a gemini-cli -m gemini/gemini-3-pro-preview \
  --env modal
```
- `<task_name>` is just the name of the task directories in `examples/harbor-workspace/out/harbor/precedent-search/tc-precedent-search/tasks/`.

**Command (Run all tasks):**
```bash
# From the root
python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace jobs start \
  -c out/harbor/precedent-search/tc-precedent-search/job.yaml \
  -a gemini-cli -m gemini/gemini-3-pro-preview \
  --env modal
```

**Inputs:**
- Task directories in `examples/harbor-workspace/out/harbor/precedent-search/tc-precedent-search/tasks/`.
- `job.yaml` (for batch execution).

**Outputs:**
- `examples/harbor-workspace/out/harbor/precedent-search/trials/`: A directory containing execution logs and `predictions.json` for each run.

**Note:**
- If you want to check which modal tasks failed, you can find them with
  ```bash
  find examples/harbor-workspace/jobs/<TIMESTAMP_DIR> -name "exception.txt"
  ```
  where `<TIMESTAMP_DIR>` is the actual job folder (e.g., `2026-01-08__19-56-26`).
- However, to make sure errors don't happen, we set the number of attempts to 3 in the `job.yaml` config.

### More run examples (different agents)

Run the full job locally (Gemini):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  jobs start -c out/harbor/precedent-search/tc-precedent-search/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

Run the full job on Modal (Claude Code):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  jobs start -c out/harbor/precedent-search/tc-precedent-search/job.yaml \
  -a claude-code -m anthropic/claude-3-5-sonnet --modal --n-concurrent 4
```

Run one task (OpenRouter via Terminus 2):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  trials start -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a terminus-2 -m openrouter/qwen/qwen3-coder-plus
```

Run one task (Codex):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  trials start -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a codex -m openai/gpt-4o-mini
```


### 4. Collect Results
Consolidate the results from the `trials/` directory into a clean format for analysis.

**Command:**
```bash
# From the root
python examples/harbor-workspace/collect_harbor_results.py
```
*(Note: This script looks in `examples/harbor-workspace/out/harbor/precedent-search/trials/` and aggregates results based on the task metadata.) You can also pass the `--trials-dir` flag to point to a specific trials directory.*

**Outputs:**
- `examples/harbor-workspace/out/harbor/precedent-search/preds/*.json`: Consolidated prediction files.

### 5. Analyze Results
Score the predictions against the ground truth and generate report tables and figures.

**Command:**
```bash
# From the root
python src/pbench_eval/score_task.py \
  --domain precedent-search \
  --output_dir examples/harbor-workspace/out/harbor \
  --analyze
```

**Inputs:**
- `examples/harbor-workspace/out/harbor/precedent-search/preds/*.json`: The collected JSON results.
- `assets/hard/rubric.csv`: The scoring rubric.
- `src/pbench_eval/utils.py`: The scoring logic.

**Outputs:**
- `examples/harbor-workspace/out/harbor/precedent-search/scores/*.csv`: CSVs containing exact scores for every prediction.
- `examples/harbor-workspace/out/harbor/precedent-search/analysis/*.csv`: CSVs with aggregate statistics (Mean/Error by property).
- `examples/harbor-workspace/out/harbor/precedent-search/figures/*.pdf`: PDF plots of the analysis.

```bash
# Ensure you are running in an environment with 'pymatgen' and 'pandas' installed
PYTHONPATH=src python src/pbench_eval/score_task.py \
  --domain precedent-search \
  --output_dir examples/harbor-workspace/out/harbor \
  --analyze \
  --split ""
```
