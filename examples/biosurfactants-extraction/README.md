# Biosurfactants Dataset

1. Please follow the setup instructions at [README.md](../../README.md#getting-started).

2. If constructing the dataset, please follow the instructions under "Dataset Construction".

3. Otherwise, follow the steps under "Experiments". The already generated Harbor tasks will load from the HuggingFace repo at (insert link).

## Experiments

1. Generate predictions using Harbor + Modal:

> \[!TIP\]
> To run on Modal, add `--modal` to the command. Note: this allows you to run more concurrent tasks (e.g., 10) than the default of 4.

```bash
uv run python ../../src/harbor-task-gen/run_harbor.py jobs start \
  -c out-0114-harbor/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-3-flash-preview \
  --workspace .
```

## Reproducing the Dataset Construction

1. Download PDFs and place them in `DATA_DIR/Paper_DB`.

Link: [Google Drive Folder](https://drive.google.com/drive/folders/1xjR5tQSpiKuLiuJyc3Khzmkjl0ie4htK?usp=share_link). Contains 52 PDFs.

<details>
    <summary>Download instructions for Lite version</summary>

Link: [Paper_DB.zip](https://drive.google.com/file/d/1XqosBMhqzUIx3U5Cfd0kO9co2KaKUErI/view?usp=share_link). Contains 5 PDFs.

```bash
# Assumes Paper_DB.zip is in the current directory
unzip Paper_DB.zip -d DATA_DIR
```

</details>

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
uv run pbench-extract --server gemini --model_name gemini-3-pro-preview -dd DATA_DIR -od OUTPUT_DIR -pp prompts/benchmark_soft_prompt_00.md
```

- Filter the irrelevant extracted properties so that we have a smaller set of properties to validate:

```bash
uv run pbench-filter -dd DATA_DIR -od OUTPUT_DIR
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

4. Create a local HuggingFace dataset `OUTPUT_DIR/SPLIT` for the papers that have PDFS in `DATA_DIR/Paper_DB`. Note: the dataset will also be shared at https://huggingface.co/datasets/kilian-group/biosurfactants-extraction.

> \[!NOTE\]
> Replace `SPLIT` with `lite` or `full` depending on the version of the dataset you want to create.

```bash
uv run python create_huggingface_dataset.py -dd DATA_DIR -od OUTPUT_DIR --filter_pdf \
    --repo_name kilian-group/biosurfactants-extraction --tag_name v0.0.0 --split SPLIT
```

## Constructing Harbor tasks

1. Create the Harbor tasks at `OUTPUT_DIR` by instantiating the Harbor template with the papers in `DATA_DIR/Paper_DB`. Note: the tasks will also be shared at https://huggingface.co/datasets/kilian-group/biosurfactants-extraction-harbor-tasks.

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py --write-job-config \
    --pdf-dir DATA_DIR/Paper_DB --output-dir OUTPUT_DIR --workspace . \
    --gt-hf-repo kilian-group/biosurfactants-extraction --gt-hf-split SPLIT --gt-hf-revision v0.0.0 \
    --force --upload-hf --hf-repo-id kilian-group/biosurfactants-extraction-harbor-tasks --hf-repo-type dataset --hf-dataset-version v0.0.0
```
