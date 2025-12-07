# Tc Prediction

Search literature to see if a material is a known superconductor. This directory contains two scripts for querying different APIs:

1. **Edison Platform API** (`query_materials_with_edison.py`) - Uses the PRECEDENT job type
2. **Gemini API** (`pred.py`) - Uses Google's Gemini model

Both scripts process materials from CSV files and check if they are known superconductors.

---

## Edison Platform API Script

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

---

## Gemini API Script

This script (`pred.py`) queries the Google Gemini API to identify related superconductor materials mentioned in the `combined_predictions.csv` file.

### Setup

#### 1. Install Dependencies

```bash
# Install the package with the new dependency
uv pip install -e /home/ag2435/sci_llm/src/sci-llm
```

Or install the dependency directly:

```bash
uv pip install google-genai
```

**Note:** This script uses the new [Google Gen AI SDK](https://googleapis.github.io/python-genai/) (`google-genai` package), which is the modern replacement for `google-generativeai`.

#### 2. Set API Key

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or alternatively:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### Usage

#### Basic Usage

Process all batches with default settings (batch size of 100):

```bash
cd /home/ag2435/sci_llm/src/sci-llm/examples/Tc
python pred.py
```

#### Process a Specific Batch

```bash
python pred.py --batch-number 1 --batch-size 10
```

#### Test with a Small Batch

For testing, process just the first 5 rows:

```bash
python pred.py --batch-size 5 --batch-number 1
```

#### Custom Output Directory

```bash
python pred.py --output-dir results
```

#### Overwrite Existing Results

```bash
python pred.py --force
```

### Command-Line Arguments

- `--input PATH`: Input CSV file (default: `combined_predictions.csv`)
- `--output-dir PATH`: Output directory (default: `preds`)
- `--batch-size N`: Number of rows per batch (default: 100)
- `--batch-number N`: Process specific batch (1-indexed)
- `--model-name NAME`: Gemini model to use (default: `gemini-2.5-flash`)
- `--force`: Overwrite existing result files
- `--max-tokens N`: Maximum tokens in response (default: 1024)
- `--temperature F`: Temperature for generation (default: 0.7)
- `--top-k N`: Top-k sampling (default: 40)
- `--top-p F`: Top-p sampling (default: 0.95)
- `--seed N`: Random seed for reproducibility
- `--max-retries N`: Maximum API retries (default: 3)
- `--wait-seconds F`: Initial retry wait time (default: 1.0)

### Output Format

Results are saved as JSON files in the format:

```
preds/gemini_batch=1__bs=100.json
```

Each file contains a **list of dictionaries** (compatible with `combine_predictions.py`):

```json
[
  {
    "icsd_id": "141183",
    "reduced_formula": "CsBaB7O12",
    "batch_number": 1,
    "query": "...",
    "formatted_answer": "...",
    "prompt": "Does this description mention...",
    "gemini_response": "Response from Gemini API...",
    "error": null,
    "status": "success",
    "metadata": {
      "model": "gemini-2.5-flash",
      "batch_size": 100,
      "batch_number": 1,
      "row_index": "0"
    },
    "usage": {
      "prompt_token_count": 150,
      "response_token_count": 50,
      "total_token_count": 200,
      "cached_content_token_count": 0
    }
  }
]
```

### Features

- **Async Batch Processing**: Uses `asyncio.gather` to process multiple rows in parallel
- **Automatic Checkpointing**: Skips already-processed batches (use `--force` to override)
- **Rate Limiting**: Built-in rate limiting to avoid API throttling
- **Retry Logic**: Exponential backoff retry on API errors
- **Progress Tracking**: Clear console output showing progress
- **Compatible Output**: JSON format matches `query_materials_with_edison.py` for use with `combine_predictions.py`

### Architecture

The implementation follows the phantom-wiki project structure:

- **`gemini_utils.py`**: Contains `GeminiChat` class with rate limiting and `InferenceGenerationConfig` dataclass
  - Uses the new [Google Gen AI SDK](https://googleapis.github.io/python-genai/) (`google-genai` package)
  - Supports both sync and async API calls via `client.models.generate_content()` and `client.aio.models.generate_content()`
- **`pred.py`**: Main script with async batch processing logic
- Based on [phantom-wiki's __main__.py](https://github.com/kilian-group/phantom-wiki/blob/main/src/phantom_eval/__main__.py)

### Query Format

The prompt asks:
```
Does this description mention a closely related material that is a known superconductor? 
If so, name the material and the associated Tc using only the information in the description.
```

---

## Example Workflows

### Edison Workflow

```bash
# Set API key
export EDISON_API_KEY="your_api_key_here"

# Process all materials
python query_materials_with_edison.py

# Or process specific batches in parallel
python query_materials_with_edison.py --batch-number 1 &
python query_materials_with_edison.py --batch-number 2 &
```

### Gemini Workflow

```bash
# Set API key
export GEMINI_API_KEY="your-api-key-here"

# Test with a small batch first
python pred.py --batch-size 5 --batch-number 1

# Process all batches (38,862 rows with batch size 100 = 389 batches)
python pred.py --batch-size 100

# Or process specific batches in parallel (e.g., on a cluster)
python pred.py --batch-number 1 --batch-size 100 &
python pred.py --batch-number 2 --batch-size 100 &
```

---

## Combining Results

After processing all batches, you can combine them into a single CSV using the existing `combine_predictions.py` script:

```bash
# Combine all Gemini predictions into a CSV
python combine_predictions.py \
  --input-dir preds \
  --output gemini_combined_predictions.csv \
  --pattern "gemini_*.json"
```

This will create a CSV with all predictions from all batches.

---

## Notes

- Both scripts automatically handle API rate limits and retries on errors
- Each batch is saved independently, allowing for interrupted processing
- Output formats are compatible with `combine_predictions.py` for easy aggregation
