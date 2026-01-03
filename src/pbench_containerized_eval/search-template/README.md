# Precedent Search Evaluation Workflow

This directory contains the template for the Precedent Search evaluation task.

## Full Workflow

### 1. Generate Tasks
This step creates the task directories, putting the necessary scoring logic and "ground truth" files in place.
It copies `check_prediction.py` and the shared `utils.py` (renamed to `pbench_eval_utils.py`) into each task folder.

**Inputs:**
- `assets/SuperCon_Tc_Tcn_dev-set.csv`: The source data.
- `src/pbench_eval/utils.py`: Shared scoring logic.
- `src/pbench_containerized_eval/search-template/`: Template files.

**Outputs:**
- `out/harbor/precedent-search/search-template/tasks/`: Directory containing one folder per task.
- `out/harbor/precedent-search/search-template/job.yaml`: Configuration for Harbor/Modal.

```bash
# From repo root
python src/pbench_containerized_eval/prepare_precedent_tasks.py --force --write-job-config
```

### 2. Execute Trials (Harbor)
This runs the agents against the tasks in Docker (or Modal).
Results are saved to `trials/`.

**Inputs:**
- `out/harbor/precedent-search/search-template/tasks/`: The generated task directories.
- `out/harbor/precedent-search/search-template/job.yaml`: The job config.

**Outputs:**
- `trials/`: A directory for each trial run, containing logs and the agent's raw output (`predictions.json`).

```bash
# Run one trial manually (example)
python src/pbench_containerized_eval/run_harbor.py trials start \
  -p out/harbor/precedent-search/search-template/tasks/<task_name> \
  -a gemini-cli -m gemini/gemini-pro
```

### 3. Collect Results
This collects the `predictions.json` files from all the scattered trial folders in `trials/` and consolidates them into `out/harbor/precedent-search/preds/`.

**Inputs:**
- `trials/`: The raw output from the execution step.

**Outputs:**
- `out/harbor/precedent-search/preds/*.json`: Cleaned JSON result files (one per trial), standardized for analysis.

```bash
python src/pbench_containerized_eval/collect_harbor_results.py
```

### 4. Analyze Results
This runs the offline scorer (`score_task.py`) **locally on your host machine** (not in Docker) to generate aggregate tables and plots.
It uses the same underlying scoring logic (`utils.py`) as the online verifier.

**Inputs:**
- `out/harbor/precedent-search/preds/*.json`: The collected JSON results.
- `assets/precedent-search/rubric.csv`: The scoring rules.
- `src/pbench_eval/utils.py`: The scoring logic.

**Outputs:**
- `out/harbor/precedent-search/scores/`: CSVs containing exact scores for every prediction.
- `out/harbor/precedent-search/analysis/`: CSVs with aggregate statistics (Mean/Error by property).
- `out/harbor/precedent-search/figures/`: PDF plots of the analysis.

```bash
# Ensure you are running in an environment with 'pymatgen' and 'pandas' installed
PYTHONPATH=src python src/pbench_eval/score_task.py \
  --domain precedent-search \
  --output_dir out/harbor \
  --analyze \
  --split ""
```

**Outputs:**
- `out/harbor/precedent-search/scores/*.csv` (Row-level scores)
- `out/harbor/precedent-search/analysis/*.csv` (Aggregate tables)
- `out/harbor/precedent-search/figures/*.pdf` (Plots)
