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
- `out/harbor/precedent-search/tc-precedent-search/tasks/`: Directory containing one task folder per material.
- `out/harbor/precedent-search/tc-precedent-search/job.yaml`: Job configuration file.

> **Note on Docker Image Rebuilding:**
> Running this step creates fresh `Dockerfile`s in each task's `environment/` directory. When you run Step 3 (trials), Harbor will detect these new Dockerfiles and rebuild the Docker images. However, once built, these images are cached and reused for subsequent trial runs unless you re-run this step or modify the Dockerfile template.

### 3. Run Trials (Harbor)
Execute the agents on the generated tasks using Harbor.

**Command (Run one task):**
```bash
# From the root
python src/pbench_containerized_eval/run_harbor.py trials start \
  -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a gemini-cli -m gemini/gemini-3-pro-preview
```
- `<task_name>` is just the name of the task directories in `out/harbor/precedent-search/tc-precedent-search/tasks/`.

**Command (Run all tasks):**
```bash
# From the root
python src/pbench_containerized_eval/run_harbor.py trials start \
  -j out/harbor/precedent-search/tc-precedent-search/job.yaml \
  -a gemini-cli -m gemini/gemini-3-pro-preview
```

**Inputs:**
- Task directories in `out/harbor/precedent-search/tc-precedent-search/tasks/`.
- `job.yaml` (for batch execution).

**Outputs:**
- `trials/`: A directory containing execution logs and `predictions.json` for each run.

### 4. Collect Results
Consolidate the results from the `trials/` directory into a clean format for analysis.

**Command:**
```bash
# From the root
python src/pbench_containerized_eval/collect_harbor_results.py \
  --output-dir out/harbor/precedent-search/preds
```
*(Note: This script typically looks in `trials/` and aggregates results based on the task metadata.)*

- `--output-dir` is required. This is where the collected results will be saved.

**Outputs:**
- `out/harbor/precedent-search/preds/*.json`: Consolidated prediction files.

### 5. Analyze Results
Score the predictions against the ground truth and generate report tables and figures.

**Command:**
```bash
# From the root
python src/pbench_eval/score_task.py \
  --domain precedent-search \
  --output_dir out/harbor \
  --analyze
```

**Inputs:**
- `out/harbor/precedent-search/preds/*.json`: The collected JSON results will be used by default if no `--input_csv` is specified.
- `assets/hard/rubric.csv`: The scoring rubric.
- `src/pbench_eval/utils.py`: The scoring logic.

**Outputs:**
- `out/harbor/precedent-search/scores/*.csv`: CSVs containing exact scores for every prediction.
- `out/harbor/precedent-search/analysis/*.csv`: CSVs with aggregate statistics (Mean/Error by property).
- `out/harbor/precedent-search/figures/*.pdf`: PDF plots of the analysis.

```bash
# Ensure you are running in an environment with 'pymatgen' and 'pandas' installed
PYTHONPATH=src python src/pbench_eval/score_task.py \
  --domain precedent-search \
  --output_dir out/harbor \
  --analyze \
  --split ""
```