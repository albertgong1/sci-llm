# Biosurfactants Dataset

1. Please follow the setup instructions at [README.md](../../README.md#setup-instructions).

2. If constructing the dataset, please follow the instructions under "Dataset Construction".

3. Otherwise, follow the steps under "Experiments". The already generated Harbor tasks will load from the HuggingFace repo at (insert link).

## Experiments

1. Generate predictions using Harbor/Modal:

```bash
```

## Reproducing the Dataset Construction

1. Download [Paper_DB.zip](https://drive.google.com/file/d/1XqosBMhqzUIx3U5Cfd0kO9co2KaKUErI/view?usp=share_link) and unzip to `data/Paper_DB`.

2. Obtain candidate properties:

<details>
    <summary><b>If using the already extracted properties (for Jiashuo)</b></summary>

- Export the list of candidate properties from [this Google Sheet](https://docs.google.com/spreadsheets/d/1IbiFwHcjTNi2tGCkQO_eOD_pTkL1Qetk_6r3xIq4B6c/edit?gid=0#gid=0) (ping Albert Gong on Slack or email ag2435@cornell.edu for access) by clicking "File" -> "Download" -> "CSV".
- Rename the CSV file to `extracted_properties_combined.csv`.
- Place the CSV file in `OUTPUT_DIR/candidates/` (you may need to create the directory if it doesn't exist).

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

4. Create HuggingFace dataset:

```bash
uv run python create_huggingface_dataset.py --output_dir out-0113-for-jiashuo --repo_name kilian-group/biosurfactants-mini --tag_name v0.0.0
```

## Constructing Harbor tasks

1. Create Harbor template:

```bash
# Copy the Harbor workspace template
cp -r ../harbor-workspace/ground-template .
# Add placeholder variable to the start of the prompt
{ echo '{paper_at_command}'; echo; cat prompts/benchmark_soft_prompt_00.md; } > ground-template/instruction.md.template
```

2. Instantiate the Harbor tasks using the template for all papers:

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py --write-job-config \
    --pdf-dir data/Paper_DB --output-dir out-0114-harbor --workspace . --domain biosurfactants \
    --force
```
