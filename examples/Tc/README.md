# Tc Prediction

Search literature to see if a material is a known superconductor.

## Edison Platform API Query Script

This directory contains a script to query materials from `data/filtered_dataset.csv` using the Edison Platform API with the PRECEDENT job type.

### Setup

**Method 1: Conda environment variable (persists across sessions)**
```bash
conda env config vars set EDISON_API_KEY=your_api_key_here
conda activate sci-llm  # Reactivate to apply the environment variable
```

**Method 2: Current session only**
```bash
export EDISON_API_KEY="your_api_key_here"
```

### Usage

**Run full batch (all ~1200 materials):**
```bash
python query_materials_with_edison.py
```

**Test with first 10 materials:**
```bash
python query_materials_with_edison.py --limit 10
```

**Dry run to preview queries without submitting:**
```bash
python query_materials_with_edison.py --dry-run --limit 5
```

**Save to custom output directory:**
```bash
python query_materials_with_edison.py --output-dir my_results
```

**Verbose mode for debugging:**
```bash
python query_materials_with_edison.py --verbose --limit 3
```

### Output

Results are saved to the `out/` directory (or custom directory specified with `--output-dir`) with two files:

1. **CSV file**: `edison_precedent_results_YYYYMMDD_HHMMSS.csv`
   - Columns: `icsd_id`, `reduced_formula`, `query`, `task_id`, `answer`, `formatted_answer`, `has_successful_answer`, `status`, `error`

2. **JSON file**: `edison_precedent_results_YYYYMMDD_HHMMSS.json`
   - Contains full detailed responses from the API

### Query Format

For each material, the script submits:
```
Is {formula} a known superconductor? If so, what is the highest reported Tc and what is the paper source?
```

Where `{formula}` is the value from the `reduced_formula` column in the input CSV.