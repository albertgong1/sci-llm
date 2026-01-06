# Biosurfactants Dataset

1. Please follow the setup instructions at [README.md](../../README.md#setup-instructions).

2. If constructing the dataset, please follow the instructions under "Dataset Construction".

3. Otherwise, follow the steps under "Experiments". The already generated Harbor tasks will load from the HuggingFace repo at (insert link).

## Experiments

1. Generate predictions using Harbor/Modal:

```bash
```

## Reproducing the Dataset Construction

1. Download files (todo) and place in `data/biosurfactants/Paper_DB`

2. Extract all properties from each PDF using an LLM:

<!-- ```bash
./src/pbench/mass_extract_properties_from_llm.py --domain biosurfactants --model_name gemini-3-pro-preview -od OUTPUT_DIR
``` -->
```bash
./src/pbench/mass_extract_properties_from_llm.py --model_name gemini-3-pro-preview -od OUTPUT_DIR
```

3. Launch the validator app and accept/reject the proposed properties. For more information, please see [VALIDATOR_GUIDE.md](../../docs/VALIDATOR_GUIDE.md).

> \[!NOTE\]
> This step requires manual effort and is not fully reproducibile.

<!-- ```bash
uv sync --group validator
uv run streamlit run src/pbench_validator_app/app.py -- --csv_folder OUTPUT_DIR/biosurfactants/unsupervised_llm_extraction --paper_folder data/biosurfactants/Paper_DB
``` -->
```bash
uv sync --group validator
uv run streamlit run src/pbench_validator_app/app.py -- --csv_folder OUTPUT_DIR --paper_folder data/Paper_DB
```

4. Construct Harbor tasks:

```bash

```

5. Push Harbor tasks to HuggingFace:

```bash
```
