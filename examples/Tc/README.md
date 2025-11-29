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

The script processes materials in batches to handle large datasets efficiently.

**Process all materials (default batch size: 10):**
```bash
python query_materials_with_edison.py
```

**Process a specific batch only:**
```bash
python query_materials_with_edison.py --batch-number 3
```

**Save to custom output directory:**
```bash
python query_materials_with_edison.py --output-dir my_results
```

**Verbose mode for debugging:**
```bash
python query_materials_with_edison.py --verbose --batch-size 10 --batch-number 1
```

### Batch Processing

The script divides materials into batches and processes each batch sequentially:
- Default batch size: 100 materials
- Batches are numbered starting from 1
- Each batch creates separate output files
- Already processed batches are automatically skipped (delete output files to reprocess)

This approach allows for:
- Resuming interrupted processing
- Parallel processing across multiple machines (by specifying different batch numbers)
- Managing API rate limits effectively

### Output

Results are saved to the `out/` directory (or custom directory specified with `--output-dir`) with separate files for each batch:

1. **CSV file**: `edison_precedent_batch=N__bs=SIZE__ts=TIMESTAMP.csv`
   - Columns: `icsd_id`, `reduced_formula`, `query`, `task_id`, `answer`, `formatted_answer`, `has_successful_answer`, `status`, `error`

2. **JSON file**: `edison_precedent_batch=N__bs=SIZE__ts=TIMESTAMP.json`
   - Contains full detailed responses from the API

### Query Format

For each material, the script submits:
```
Is {formula} a known superconductor? If so, what is the highest reported Tc and what is the paper source?
```

Where `{formula}` is the value from the `reduced_formula` column in the input CSV.