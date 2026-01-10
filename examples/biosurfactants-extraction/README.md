# Biosurfactants Dataset

1. Please follow the setup instructions at [README.md](../../README.md#setup-instructions).

2. If constructing the dataset, please follow the instructions under "Dataset Construction".

3. Otherwise, follow the steps under "Experiments". The already generated Harbor tasks will load from the HuggingFace repo at (insert link).

## Experiments

1. Generate predictions using Harbor/Modal:

```bash
```

## Reproducing the Dataset Construction

1. Download files (todo) and place in `data/Paper_DB`

2. Obtain candidate properties:

<details>
    <summary><b>If using the already extracted properties (for Jiashuo)</b></summary>

- Export the list of candidate properties from this Google Sheet by clicking "File" -> "Download" -> "CSV".
- Rename the CSV file to `extracted_properties_combined.csv`.
- Place the CSV file in `OUTPUT_DIR/candidates/`.

</details>

<details>
    <summary>If extracting properties from scratch</summary>

- Extract properties from PDFs using an LLM:
```bash
uv run pbench-extract --server gemini --model_name gemini-3-pro-preview -od OUTPUT_DIR -pp prompts/benchmark_soft_prompt_00.md
```

- Filter the irrelevant extracted properties so that we have a smaller set of properties to validate:

```bash
uv run pbench-filter -od OUTPUT_DIR
```

TODO:
- [ ] Move filtering logic to a script. (Note: currently we are just using a spreadsheet to filter the irrelevant extracted properties.)

</details>

3. Launch the validator app and accept/reject the candidates:

> \[!NOTE\]
> This step requires manual effort and is not fully reproducibile.

```bash
uv sync --group validator
uv run streamlit run ../../src/pbench_validator_app/app.py -- -od OUTPUT_DIR
```

## Sharing the dataset via HuggingFace

1. Construct Harbor tasks:

```bash

```

2. Push Harbor tasks to HuggingFace:

```bash
```
