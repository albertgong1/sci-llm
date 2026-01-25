# Biosurfactants Dataset

1. Please follow the setup instructions at [README.md](../../README.md#getting-started).

2. If constructing the dataset, please follow the instructions under "Dataset Construction".

3. Otherwise, follow the steps under "Experiments". The already generated Harbor tasks will load from the HuggingFace repo at (insert link).

## Reproducing Experiments

1. Please run the following command to execute the Harbor tasks in batches (default batch size: 10):

> \[!IMPORTANT\]
> Adding the `--seed 1` flag will randomly shuffle the tasks.

```bash
uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start \
  --hf-tasks-repo kilian-group/biosurfactants-extraction-harbor-tasks --hf-tasks-version main \
  -a gemini-cli -m gemini/gemini-3-flash-preview \
  --workspace . --jobs-dir JOBS_DIR --seed 1 --batch-size 50
```

<details>
    <summary>Instructions for running Harbor tasks saved locally</summary>

```bash
uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start \
  --registry-path OUTPUT_DIR/ground-template/registry.json --dataset biosurfactants-extraction@v0.0.0 \
  -a gemini-cli -m gemini/gemini-3-flash-preview --modal --n-concurrent 4
  --seed 1 --jobs-dir JOBS_DIR
```

</details>

2. Compute task-average precision and recall by model:

```bash
# Generate embeddings for predicted property names
uv run pbench-pred-embeddings -jd JOBS_DIR -od OUTPUT_DIR

# Query LLM to determine best match between generated and ground-truth property name:
uv run pbench-generate-matches -jd JOBS_DIR -od OUTPUT_DIR -m gemini-2.5-flash \
    --hf_repo kilian-group/biosurfactants-extraction --hf_split full --hf_revision v0.0.0 \
    --prompt_path prompts/property_matching_prompt.md

# Compute precision (condition-based matching for biosurfactants)
uv run pbench-score-precision -jd JOBS_DIR -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode conditions

# Compute recall (condition-based matching for biosurfactants)
uv run pbench-score-recall -jd JOBS_DIR -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode conditions
```

### Using the LLM API (no Harbor)

1. Please run the following command to generate the predictions:

```bash
uv run pbench-eval -dd DATA_DIR --server gemini -m gemini-3-pro-preview \
    -pp prompts/benchmark_soft_prompt_01.md -od OUTPUT_DIR \
    --hf_repo kilian-group/biosurfactants-extraction --hf_split full --hf_revision v0.0.0
```

2. Compute task-average precision and recall by model:

```bash
# Generate embeddings for predicted property names
uv run pbench-pred-embeddings -od OUTPUT_DIR

# Query LLM to determine best match between generated and ground-truth property name
uv run pbench-generate-matches -od OUTPUT_DIR -m gemini-2.5-flash \
    --hf_repo kilian-group/biosurfactants-extraction --hf_split full --hf_revision v0.0.0 \
    --prompt_path prompts/property_matching_prompt.md

# Compute precision (condition-based matching for biosurfactants)
uv run pbench-score-precision -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode conditions

# Compute recall (condition-based matching for biosurfactants)
uv run pbench-score-recall -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode conditions
```

6. Compute task-average token usage, steps, and cost:

```bash
uv run python format_tokens.py -od OUTPUT_DIR
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
    --hf_repo kilian-group/biosurfactants-extraction --hf_revision v0.0.0 --hf_split SPLIT
```

5. Generate embeddings for the ground-truth property names for scoring:

```bash
uv run pbench-gt-embeddings --hf_repo kilian-group/biosurfactants-extraction --hf_revision v0.0.0 --hf_split SPLIT
```

<!-- Old command (deprecated):
```bash
uv run python generate_gt_embeddings.py --hf_repo kilian-group/biosurfactants-extraction --hf_revision v0.0.0 --hf_split SPLIT
```
-->

6. Create the Harbor tasks at `OUTPUT_DIR` by instantiating the Harbor template with the papers in `DATA_DIR/Paper_DB`. Note: the tasks will also be shared at https://huggingface.co/datasets/kilian-group/biosurfactants-extraction-harbor-tasks.

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
    --pdf-dir DATA_DIR/Paper_DB --output-dir OUTPUT_DIR --workspace . \
    --gt-hf-repo kilian-group/biosurfactants-extraction --gt-hf-split full --gt-hf-revision v0.0.0 \
    --force --upload-hf --hf-repo-id kilian-group/biosurfactants-extraction-harbor-tasks
```
