# Tc Precedent Search Evaluation

This directory contains the workflow for evaluating the "Precedent Search" task, where an agent must determine if a material has been reported as superconducting, and if so, what the highest Tc and lowest Tcn (non-superconducting temperature) are.

## Quick Start

If you just want to run the full pipeline, skip to [Running Example](#running-example-gemini-cli--gemini-3-pro).

---

## Detailed Workflow

### 1. Generate Development Set (One-Time Setup)

Download [SuperCon_Tc_Tcn - no NA.csv](https://drive.google.com/file/d/11mqYhvSbl_cCNIDiQOgnS5N1P-26BREC/view?usp=sharing) and save to `examples/tc-precedent-search/`.

> [!NOTE]
>
> - Total SuperCon rows: 21,153
> - Rows where `file` column is #N/A: 4,534
> - Rows where `file` column is blank: 14,226
> - `SuperCon_Tc_Tcn - no NA.csv` contains 16,619 rows (excludes N/A rows)
> - The dev set is 200 randomly selected rows: 100 superconducting (`Tc`) and 100 non-superconducting (`Tcn`)

```bash
cd examples/tc-precedent-search/
python create_dev_set.py
```

**Inputs:**

- `assets/SuperCon_Tc_Tcn - no NA.csv`: Source SuperCon dataset

**Outputs:**

- `examples/tc-precedent-search/SuperCon_Tc_Tcn_dev-set.csv`: The generated development set

Verify against the reference dev set [here](https://drive.google.com/file/d/13nb4HTU2p28b8oiEbQ87Y1CF4Jrcwlsa/view?usp=sharing).

### 2. Generate Harbor Tasks

Creates a task folder for each material with instructions, scoring logic, and Docker environment.

```bash
cd examples/tc-precedent-search/
uv run python prepare_precedent_tasks.py --force --write-job-config
```

**Options:**

| Flag | Description |
|------|-------------|
| `--csv` | Path to input CSV (default: `SuperCon_Tc_Tcn_dev-set.csv`) |
| `--limit N` | Generate only N tasks (for testing) |
| `--force` | Overwrite existing output directory |
| `--write-job-config` | Write `job.yaml` for Harbor |

**Inputs:**

- `SuperCon_Tc_Tcn_dev-set.csv`: The dev set
- `search-template/`: Template files for the tasks

**Outputs:**

- `examples/harbor-workspace/out/harbor/precedent-search/tc-precedent-search/tasks/`: Directory containing one task folder per material
- `examples/harbor-workspace/out/harbor/precedent-search/tc-precedent-search/job.yaml`: Job configuration file

> [!NOTE]
> Running this step creates fresh `Dockerfile`s in each task's `environment/` directory. Harbor will rebuild Docker images on first run, then cache them for subsequent runs.

### 3. Run Trials

**Regime 1: Controlled Agent Leaderboard (Model Reasoning)**

Uses `terminus-2` agent where the model must explicitly plan and execute Linux commands.

```bash
# Gemini
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace trials start \
  -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a terminus-2 -m gemini/gemini-3-pro-preview \
  --env modal

# GPT-5.1
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace trials start \
  -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a terminus-2 -m gpt-5.1-2025-11-13 \
  --env modal
```

**Regime 2: Deployed-Stack Leaderboard (End-to-End System)**

Uses native agents (`gemini-cli`, `codex`).

```bash
# Gemini CLI
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace trials start \
  -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a gemini-cli -m gemini/gemini-3-pro-preview \
  --env modal

# Codex
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace trials start \
  -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a codex -m gpt-5.1-2025-11-13 \
  --env modal
```

**Run all tasks as a job:**

```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace jobs start \
  -c out/harbor/precedent-search/tc-precedent-search/job.yaml \
  -a gemini-cli -m gemini/gemini-3-pro-preview \
  --env modal
```

**Inputs:**

- Task directories in `examples/harbor-workspace/out/harbor/precedent-search/tc-precedent-search/tasks/`
- `job.yaml` (for batch execution)

**Outputs:**

- `examples/harbor-workspace/out/harbor/precedent-search/trials/`: Directory containing execution logs and `predictions.json` for each run

**Check for failed tasks:**

```bash
find examples/harbor-workspace/jobs/<TIMESTAMP_DIR> -name "exception.txt"
```

> [!NOTE]
> We set the number of attempts to 3 in `job.yaml` to handle transient failures.

### 4. Collect Results

Consolidate trial results into JSON files for analysis.

**For Gemini CLI / Terminus agents:**

```bash
uv run python examples/harbor-workspace/collect_harbor_results.py \
  --trials-dir examples/harbor-workspace/jobs/<JOB_ID> \
  --output-dir examples/harbor-workspace/out/harbor/precedent-search/<PRED_FOLDER_NAME>
```

> [!IMPORTANT]
> **For Codex results, you MUST use `recover_codex_results.py` instead:**
>
> ```bash
> uv run python examples/harbor-workspace/recover_codex_results.py \
>   --trials-dir examples/harbor-workspace/jobs/<JOB_TIMESTAMP_DIR> \
>   --output-dir examples/harbor-workspace/out/harbor/precedent-search/preds_codex-run3
> ```

**Outputs:**

- `examples/harbor-workspace/out/harbor/precedent-search/preds/*.json`: Consolidated prediction files
  - Naming convention: `<task_name>__<trial_id>__<agent>__<model>.json` (e.g., `mo3p1__abc123__codex__gpt-5.1-2025-11-13.json`)

### 5. Score Results

```bash
uv run python src/pbench_eval/score_task.py \
  --domain precedent-search \
  --input_preds_dir examples/harbor-workspace/out/harbor/precedent-search/<PRED_FOLDER_NAME> \
  --file_pattern "*gemini-cli*" \
  --output_tag <OUTPUT_TAG> \
  --output_dir examples/harbor-workspace/out/harbor
```

**Inputs:**

- `examples/harbor-workspace/out/harbor/precedent-search/preds/*.json`: The collected JSON results
- `assets/hard/rubric.csv`: The scoring rubric
- `src/pbench_eval/utils.py`: The scoring logic

**Outputs:**

- `examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>.csv`: CSV containing exact scores for every prediction

### 6. Enrich with Token Counts

```bash
uv run python examples/tc-precedent-search/add_token_counts.py \
  --input examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>.csv \
  --output examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_enriched.csv
```

**Outputs:**

- `examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_enriched.csv`

### 7. Rescore (0-1-2 Logic)

```bash
uv run python examples/tc-precedent-search/rescore_results.py \
  --input examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_enriched.csv \
  --output examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_final.csv
```

**Outputs:**

- `examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_final.csv`

### 8. Generate Plots

```bash
uv run python examples/tc-precedent-search/generate_plots.py --tag <OUTPUT_TAG>_final
```

**Outputs:**

- `examples/harbor-workspace/out/harbor/precedent-search/analysis/*.csv`: CSVs with aggregate statistics (Mean/Error by property)
- `examples/harbor-workspace/out/harbor/precedent-search/figures/*.pdf`: PDF plots of the analysis

---

## Running Example: Gemini CLI + Gemini 3 Pro

### Phase 1: Generation (Run Once)

```bash
cd examples/tc-precedent-search
uv run python prepare_precedent_tasks.py --force --write-job-config
cd ../..
```

### Phase 2: Execution & Analysis

#### Step 1: Run the Job

```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace jobs start \
  -c out/harbor/precedent-search/tc-precedent-search/job.yaml \
  -a gemini-cli -m gemini/gemini-3-pro-preview \
  --env modal
```

> [!TIP]
> **When the Job Hangs:**
>
> 1. Delete all the task folders you want to rerun
> 2. Resume with:
>
>    ```bash
>    uv run python src/harbor-task-gen/run_harbor.py \
>        --workspace examples/harbor-workspace \
>        jobs resume \
>        -p /Users/jjk297/repos/sci-llm/examples/harbor-workspace/jobs/<JOB_TIMESTAMP_DIR>
>    ```
>
> 3. Note the job timestamp directory (e.g., `jobs/2026-01-13__14-30-00`)

#### Step 2: Collect Results

```bash
uv run python examples/harbor-workspace/collect_harbor_results.py \
  --trials-dir examples/harbor-workspace/jobs/<JOB_ID> \
  --output-dir examples/harbor-workspace/out/harbor/precedent-search/<NEW_PRED_FOLDER_NAME>
```

*Creates JSON files in `.../out/harbor/precedent-search/<NEW_PRED_FOLDER_NAME>/`*

> [!IMPORTANT]
> **For Codex results, use `recover_codex_results.py` instead:**
>
> ```bash
> uv run python examples/harbor-workspace/recover_codex_results.py \
>   --trials-dir examples/harbor-workspace/jobs/2026-01-17__21-54-27 \
>   --output-dir examples/harbor-workspace/out/harbor/precedent-search/preds_codex-run3
> ```

#### Step 3: Score (Property Level)

```bash
uv run python src/pbench_eval/score_task.py \
  --domain precedent-search \
  --input_preds_dir examples/harbor-workspace/out/harbor/precedent-search/<NEW_PRED_FOLDER_NAME> \
  --file_pattern "*gemini-cli*" \
  --output_tag gemini_cli_pro_run2 \
  --output_dir examples/harbor-workspace/out/harbor
```

*Creates: `examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_gemini_cli_pro_run2.csv`*

> [!NOTE]
> The `--output_tag` value (e.g., `gemini_cli_pro_run2`) becomes `<OUTPUT_TAG>` for subsequent steps.

#### Step 4: Enrich (Token Counts + Job ID)

```bash
uv run python examples/tc-precedent-search/add_token_counts.py \
  --input examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>.csv \
  --output examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_enriched.csv
```

*Creates: `examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_enriched.csv`*

#### Step 5: Rescore (0-1-2 Logic)

```bash
uv run python examples/tc-precedent-search/rescore_results.py \
  --input examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_enriched.csv \
  --output examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_final.csv
```

*Creates: `examples/harbor-workspace/out/harbor/precedent-search/scores/scored_results_<OUTPUT_TAG>_final.csv`*

#### Step 6: Plot

```bash
uv run python examples/tc-precedent-search/generate_plots.py --tag <OUTPUT_TAG>_final
```

*Creates plots in `examples/harbor-workspace/out/harbor/precedent-search/analysis/plots/`*

---

## Getting Citation (DOI) Info

**Base Path:** `examples/harbor-workspace/out/harbor/precedent-search/analysis/final_results/citation_info/`

#### Step 1: Extract citation info from agent outputs

```bash
uv run python examples/tc-precedent-search/extract_source_dois.py
```

*Creates: `<base>/citation-info-*.csv`*

#### Step 2: Analyze citation quality against ground truth

```bash
uv run python examples/tc-precedent-search/analyze_citation_quality.py
```

*Creates:*

- `<base>/detailed_matches.csv`
- `<base>/title_or_doi_matches.csv`
- `<base>/high_score_no_citation_match.csv`

## LLM API Workflow

We can prompt LLMs with web search grounding to perform Tc precedent search as well. After creating the Tc dev set in step 1:

```bash
uv run python run_precedent_search_with_llms.py --server gemini -m gemini-3-pro-preview -od out --use_web_search --max_concurrent 50 --run run1
```

This creates a CSV at `out/precedent_search__model=gemini-3-pro-preview__web_search.csv` with LLM predictions and DOI citations for material Tc values. This CSV already contains ground-truth Tc values from SuperCon merged.

```bash
uv run python run_precedent_search_with_llms.py --server openai -m gpt-5.1-2025-11-13 -od out --use_web_search --max_concurrent 50 --reasoning_effort high --max_output_tokens 65536 --run run1
```

This creates a CSV at `out/precedent_search__model=gpt-5.1-2025-11-13__web_search.csv` with LLM predictions and DOI citations for material Tc values. This CSV already contains ground-truth Tc values from SuperCon merged.

### 2. Scoring and Evaluation

Use the scoring script to compare predictions against ground truth and extract structured citation data.

#### Run Scoring (Batch Mode)

The script will scan the `out/` directory and save results to `out/scores/`.

```bash
python score_web_search_csv_format.py --input out/
```

#### Run Scoring (Single File)

```bash
python score_web_search_csv_format.py --input out/your_results.csv --output out/your_results_scored.csv
```
