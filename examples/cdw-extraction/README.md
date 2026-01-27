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

### Using Harbor

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

2. Compute average precision and recall across tasks:

> \[!TIP\]
> If your Gemini account is on Tier 3, you can set `--max_concurrent 20`.

```bash
# Generate property name embeddings
uv run pbench-pred-embeddings -jd JOBS_DIR -od OUTPUT_DIR

# Query LLM to determine best match between generated and ground-truth property name:
uv run pbench-generate-matches -jd JOBS_DIR -od OUTPUT_DIR -m gemini-2.5-flash \
    --hf_repo kilian-group/cdw-extraction --hf_split full --hf_revision v0.0.0 \
    --prompt_path prompts/property_matching_prompt.md

# Compute precision (material-based matching for supercon)
uv run pbench-score-precision -jd JOBS_DIR -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode material --log_level ERROR

# Compute recall (condition-based matching for supercon)
uv run pbench-score-recall -jd JOBS_DIR -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode material --log_level ERROR
```

3. Aggregate accuracy and tokens across tasks:

```bash
# Use cost
uv run pbench-aggregate -jd jobs-cdw -od out-cdw -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode material --log_level ERROR --x-axis cost

# Use tokens
uv run pbench-aggregate -jd jobs-cdw -od out-cdw -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode material --log_level ERROR --x-axis tokens
```

### Using the LLM API (no Harbor)

> \[!WARNING\]
> These instructions are Claude generated. I have not tested each one yet. -  Albert

1. Please run the following command to generate the predictions:

> \[!IMPORTANT\]
> Registry and max num papers flags define an ordering to process the big list of papers and a limit. This script assumes that `registry_data.json` exists in this examples subdirectory. Ask ag2435@cornell.edu on Slack for a copy of this file.
> Remove these flags to process the full dataset in DATA_DIR=data.

```bash
uv run pbench-eval -dd DATA_DIR --server gemini -m gemini-3-pro-preview -pp prompts/targeted_extraction_prompt_03.md \
    --harbor_task_ordering_registry_path out-0126-harbor/targeted-stoichiometric-template/registry.json --max_num_papers 50 -od OUTPUT_DIR
```

2. Compute task-average precision and recall by model:

```bash
# Generate embeddings for predicted property names
uv run pbench-pred-embeddings -od OUTPUT_DIR

# Query LLM to determine best match between generated and ground-truth property name:
uv run pbench-generate-matches -od OUTPUT_DIR -m gemini-2.5-flash \
    --hf_repo kilian-group/cdw-extraction --hf_split full --hf_revision v0.0.0 \
    --prompt_path prompts/property_matching_prompt.md

# Compute precision (material-based matching for cdw)
uv run pbench-score-precision -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode material

# Compute recall (material-based matching for cdw)
uv run pbench-score-recall -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode material
```

3. Aggregate accuracy and tokens across tasks:

```bash
# Use cost
uv run pbench-aggregate -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode material --log_level ERROR --x-axis cost

# Use tokens
uv run pbench-aggregate -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric.csv \
    --matching_mode material --log_level ERROR --x-axis tokens
```

## Constructing the CDW Dataset

1. Download PDFs from [Google Drive Folder](https://drive.google.com/drive/folders/1HYwG2V38DaOHH-Osn7NvxtQdZNLoeINt?usp=share_link) and place them in `data/Paper_DB`:

- [x] @tempoxylophone: The subsequent steps assumes that that the PDF for refno is at `data/Paper_DB/refno.pdf`, so we (annoyingly) need to rename the PDFs to use refno instead of the arXiv ID. If a single refno maps to multiple arXiv IDs, then we can probably throw away that refno. See Step 1 [README.md](../supercon-extraction/README.md#constructing-the-dataset-from-supercon-original) for how I renamed the files. Let me know on Slack if you have thoughts on this / better workarounds.
    - we have resolved this. There is no `refno` for CDW, I (@tempoxylophone) had used the arXiv ID because it was a unique property that was already available and mapped clearly to a paper's filename.

2. Obtain candidate properties:

- Extract properties from PDFs using an LLM:

```bash
uv run --env-file=.env pbench-extract --server gemini --model_name gemini-3-pro-preview -od OUTPUT_DIR -pp examples/cdw-extraction/prompts/targeted_extraction_prompt_02.md
```
>Ensure that you have set the `GOOGLE_API_KEY` in the `.env` file before calling Gemini via the above command.

- Add `data_type` column to the CSV. The resulting CSV will be saved to `OUTPUT_DIR/candidates`.

```bash
uv run pbench-filter -od OUTPUT_DIR
```

3. Launch the validator app and accept/reject the candidates:

> \[!WARNING\]
> This step requires manual effort and is not fully reproducibile.

```bash
uv sync --group validator
uv run streamlit run ../../src/pbench_validator_app/app.py -- -od OUTPUT_DIR
```

4. Combine validation results from multiple annotators:

> \[!WARNING\]
> A CSV file will be created at `data/new-supercon-papers` with a column "validated_resolved". This will be auto-resolved if possible and set to "RESOLVE" if manual resolution is needed.

```bash
uv run python combine_validation_results.py \
    --output_dir1 out-0125-for-chao \
    --output_dir2 out-0125-for-fatmagul \
    --data_dir data
```

5. Create a local HuggingFace dataset `OUTPUT_DIR/SPLIT` for the papers that have PDFS in `data/Paper_DB`. Note: the dataset will also be shared at https://huggingface.co/datasets/kilian-group/cdw-extraction.

> \[!NOTE\]
> Replace `SPLIT` with `lite` or `full` depending on the version of the dataset you want to create.

```bash
uv run python create_huggingface_dataset.py \
    --data_dir data \
    --output_dir OUTPUT_DIR \
    --hf_repo kilian-group/cdw-extraction \
    --hf_split SPLIT \
    --hf_revision v0.0.0 \
    --filter_pdf
```

6. Generate embeddings for the ground-truth property names for scoring:

```bash
uv run pbench-gt-embeddings --hf_repo kilian-group/cdw-extraction --hf_revision v0.0.0 --hf_split full
```

7. Create the Harbor tasks at `OUTPUT_DIR` by instantiating the Harbor template with the papers in `data/Paper_DB`. Note: the tasks will also be shared at https://huggingface.co/datasets/kilian-group/cdw-extraction-harbor-tasks.

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
    --pdf-dir data/Paper_DB --output-dir OUTPUT_DIR --workspace . --template targeted-stoichiometric-template \
    --gt-hf-repo kilian-group/cdw-extraction --gt-hf-split SPLIT --gt-hf-revision v0.0.0 \
    --force --upload-hf --hf-repo-id kilian-group/cdw-extraction-harbor-tasks
```

### Evaluating Validation Accuracy

1. To compute the validation accuracy of a single annotator, please run the following command. This script assumes the validation results are at `OUTPUT_DIR`:

```bash
uv run python format_validation_accuracy.py -od OUTPUT_DIR
```
