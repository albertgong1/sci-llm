# Flux Precedent Search Evaluation

This directory contains the workflow for evaluating the "Precedent Search" task, where an agent must determine if a material has been reported with flux growth.

## Detailed Workflow

### 1. Download Development Set (One-Time Setup)

Download [flux-material_dev-set.csv](https://drive.google.com/file/d/1lJrRh6D8Cj1Wq5Ta8w0_rUUSGh0wTtS3/view?usp=drive_link) and save to `examples/flux-precedent-search/`.

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

### Metrics and Logic

#### Scoring Logic

- **Score 1.0**: The predicted value matches the ground truth (case-insensitive).
- **Score 0.0**: The values do not match, or the model returned "Unknown" or an error occurred.

#### Extracted Data

- **Citation Extraction**: DOIs are extracted from the `sources` column and flattened into indexed columns (e.g., `is_grown_with_flux_source_1_doi`).
- **Tool Calls**: The `tool_calls` column is populated by counting the number of web search queries performed during the run.
- **Terminal Statistics**: The script prints a summary table of scores and the average number of `tool_calls` across all trials.
- **Relative Path Resolution**: Paths provided to `--input`, `--input-dir`, and `--output-dir` are resolved relative to the script's directory for easier usage within the example folder.
