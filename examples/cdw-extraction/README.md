# Charge Density Wave Property Extraction

## Setup Instructions

1. Follow the setup instructions at [README.md](../../README.md#getting-started).

2. Additional setup instructions:

<details>
    <summary>Instructions for running Harbor locally</summary>

* Install Docker Desktop following [these](https://docs.docker.com/desktop/setup/install/mac-install/) instructions.

</details>

<details>
    <summary>Instructions for running Harbor on Modal</summary>

* Create a Modal API key at https://modal.com/settings/kilian-group/tokens (email ag2435@cornell.edu to be added to the group) and follow the onscreen instructions to activate it.

</details>

3. Optional: If running Harbor locally, launch Docker Desktop.

## Reproducing Experiments

> \[!IMPORTANT\]
> To obtain results incrementally, batching functionality is available. Simply specificy the `--batch-size` in the commands below. To obtain an unbiased estimate of the average accuracy across all tasks, please shuffle the tasks using the `--seed 1` flag.

> \[!TIP\]
> To run on Modal, simply add the `--modal` flag to any of the commands below.

1. Please run the following command to execute the Harbor tasks in batches of size 10:

```bash
uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start \
  --hf-tasks-repo kilian-group/cdw-extraction-harbor-tasks --hf-tasks-version v0.0.0 \
  -a gemini-cli -m gemini/gemini-3-flash-preview \
  --workspace . --jobs-dir JOBS_DIR --seed 1 --batch-size 10
```

<details>
    <summary>Instructions for running Harbor tasks saved locally</summary>

```bash
uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start \
  --registry-path OUTPUT_DIR/targeted-template/registry.json --dataset cdw-extraction@v0.0.0 \
  -a gemini-cli -m gemini/gemini-3-flash-preview \
  --workspace . --jobs-dir JOBS_DIR --seed 1 --batch-size 10
```

</details>

2. Compute accuracy across tasks:

```bash
uv run python format_accuracy.py -jd JOBS_DIR
```

## Constructing the CDW Dataset

1. Download PDFs from [Google Drive Folder](https://drive.google.com/drive/folders/1HYwG2V38DaOHH-Osn7NvxtQdZNLoeINt?usp=share_link) and place them in `data/Paper_DB`:

- [ ] @tempoxylophone: The subsequent steps assumes that that the PDF for refno is at `data/Paper_DB/refno.pdf`, so we (annoyingly) need to rename the PDFs to use refno instead of the arXiv ID. If a single refno maps to multiple arXiv IDs, then we can probably throw away that refno. See Step 1 [README.md](../supercon-extraction/README.md#constructing-the-dataset-from-supercon-original) for how I renamed the files. Let me know on Slack if you have thoughts on this / better workarounds.

2. Obtain candidate properties:

- Extract properties from PDFs using an LLM:

```bash
uv run --env-file=.env pbench-extract --server gemini --model_name gemini-3-pro-preview -od OUTPUT_DIR -pp prompts/targeted_extraction_prompt.md
```

- Add `data_type` column to the CSV. The resulting CSV will be saved to `OUTPUT_DIR/candidates`.

```bash
uv run pbench-filter -od OUTPUT_DIR
```

> \[!STOP\]
> Send the file at `OUTPUT_DIR/candidates/extracted_properties_combined.csv` on Slack to Albert.

3. Launch the validator app and accept/reject the candidates:

> \[!NOTE\]
> This step requires manual effort and is not fully reproducibile.

```bash
uv sync --group validator
uv run streamlit run ../../src/pbench_validator_app/app.py -- -od OUTPUT_DIR
```

4. Create a local HuggingFace dataset `OUTPUT_DIR/SPLIT` for the papers that have PDFS in `data/Paper_DB`. Note: the dataset will also be shared at https://huggingface.co/datasets/kilian-group/cdw-extraction.

> \[!NOTE\]
> Replace `SPLIT` with `lite` or `full` depending on the version of the dataset you want to create.

```bash
uv run python create_huggingface_dataset.py -od OUTPUT_DIR --filter_pdf \
    --tag_name v0.0.0 --repo_name kilian-group/cdw-extraction --split SPLIT
```

5. Create the Harbor tasks at `OUTPUT_DIR` by instantiating the Harbor template with the papers in `data/Paper_DB`. Note: the tasks will also be shared at https://huggingface.co/datasets/kilian-group/cdw-extraction-harbor-tasks.

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
    --pdf-dir data/Paper_DB --output-dir OUTPUT_DIR --workspace . --template targeted-template \
    --gt-hf-repo kilian-group/cdw-extraction --gt-hf-split SPLIT --gt-hf-revision v0.0.0 \
    --force --upload-hf --hf-repo-id kilian-group/cdw-extraction-harbor-tasks
```
