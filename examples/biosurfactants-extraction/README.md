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

2. Extract all properties from each PDF using an LLM:

```bash
uv run pbench-extract --server gemini --model_name gemini-3-pro-preview -od OUTPUT_DIR -pp prompts/benchmark_soft_prompt_00.md
# add data_type column and save to OUTPUT_DIR/candidates/
uv run pbench-add-datatype -od OUTPUT_DIR
```

3. Launch the validator app and accept/reject the candidates.

> \[!NOTE\]
> This step requires manual effort and is not fully reproducibile.

```bash
uv sync --group validator
uv run streamlit run ../../src/pbench_validator_app/app.py -- -od OUTPUT_DIR
```

4. Construct Harbor tasks:

```bash

```

5. Push Harbor tasks to HuggingFace:

```bash
```
