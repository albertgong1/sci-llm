# Biosurfactants Dataset

1. Please follow the setup instructions at [README.md](../../README.md#setup-instructions).

2. If constructing the dataset, please follow the instructions under "Dataset Construction".

3. Otherwise, follow the steps under "Experiments". The already generated Harbor tasks will load from the HuggingFace repo at (insert link).

## Experiments

1. Generate predictions using Harbor + Modal:

> \[!TIP\]
> To run on Modal, add `--modal` to the command. Note: this allows you to run more concurrent tasks (e.g., 10) than the default of 4.

```bash
uv run python ../../src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/biosurfactants-mini/biosurfactants-extraction-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --workspace .
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

Build tasks from `kilian-group/biosurfactants-mini`:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --domain biosurfactants \
  --workspace examples/biosurfactants-extraction \
  --template biosurfactants-extraction-template \
  --write-job-config --force
```

Run jobs (local or Modal):
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  --workspace examples/biosurfactants-extraction \
  -c out/harbor/biosurfactants-mini/biosurfactants-extraction-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

Add `--modal` to run on Modal (with cleanup defaults).

### More run examples

Run one task (oracle sanity check):
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  --workspace examples/biosurfactants-extraction \
  -p out/harbor/biosurfactants-mini/biosurfactants-extraction-template/tasks/<task-id> \
  -a oracle
```

Run job with OpenRouter via Terminus 2:
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  --workspace examples/biosurfactants-extraction \
  -c out/harbor/biosurfactants-mini/biosurfactants-extraction-template/job.yaml \
  -a terminus-2 -m openrouter/qwen/qwen3-coder-plus --modal
```

For the full flag reference, see:
- `src/harbor-task-gen/README.md`
