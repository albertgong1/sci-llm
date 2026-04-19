# SuperCon Property Extraction

TODO:
- [ ] Share embeddings on HF.
- [ ] Fill in "category" field in GT dataset

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
  --hf-tasks-repo kilian-group/supercon-extraction-harbor-tasks --hf-tasks-version v0.0.0 \
  -a gemini-cli -m gemini/gemini-3-flash-preview \
  --workspace . --jobs-dir JOBS_DIR --seed 1 --batch-size 10
```

<details>
    <summary>Instructions for running Harbor tasks saved locally</summary>

```bash
uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start \
  --registry-path out-0121-harbor/targeted-stoichiometric-template/registry.json --dataset supercon-extraction@main \
  -a gemini-cli -m gemini/gemini-3-flash-preview \
  --workspace . --jobs-dir JOBS_DIR --seed 1 --batch-size 10
```

</details>

2. Use LLM judge for property name matching between extracted and ground-truth properties:

```bash
# Generate property name embeddings
uv run pbench-pred-embeddings -jd JOBS_DIR -od OUTPUT_DIR

# Query LLM to determine best match between generated and ground-truth property name:
uv run pbench-generate-matches -jd JOBS_DIR -od OUTPUT_DIR -m gemini-2.5-flash \
    --hf_repo kilian-group/supercon-post-2021-extraction --hf_split full --hf_revision v0.0.1 \
    --prompt_path prompts/property_matching_prompt.md
```

3. Compute evidence metrics for SuperCon Post-2021 only (since SuperCon original does not provide the ground-truth evidence):

```bash
uv run pbench-score-evidence -od out-post-2021-no-agent --hf_repo kilian-group/supercon-post-2021-extraction --hf_split full --hf_revision v0.0.1
```

### Using the LLM API (no Harbor)

1. Please run the following command to generate the predictions:

> \[!IMPORTANT\]
> Registry and max num papers flags define an ordering to process the big list of papers and a limit. This script assumes that `registry_data.json` exists in this examples subdirectory. Ask ag2435@cornell.edu on Slack for a copy of this file.
> Remove these flags to process the full dataset in DATA_DIR=data.

```bash
uv run pbench-eval -dd DATA_DIR --server gemini -m gemini-3-pro-preview -pp prompts/targeted_stoic_extraction_prompt.md \
    --harbor_task_ordering_registry_path registry_data.json --max_num_papers 50 -od OUTPUT_DIR
```

2. Use LLM judge for property name matching between extracted and ground-truth properties:

TODO:
- [X] Update these instructions with new domain-agnostic eval scripts
- [ ] Ensure that the results are similar as before using the preds at `/Users/ag2435/sci_llm/src/sci-llm/examples/supercon-extraction/out-0123`

```bash
uv run pbench-pred-embeddings -od OUTPUT_DIR

# Query LLM to determine best match between generated and ground-truth property name:
uv run pbench-generate-matches -od out-post-2021-no-agent -m gemini-2.5-flash \
    --hf_repo kilian-group/supercon-post-2021-extraction --hf_split full --hf_revision v0.0.1 \
    --prompt_path prompts/property_matching_prompt.md
```

3. Compute average F1, precision, recall scores using the following commands:

```bash
# Compute F1 using material-based matching
uv run pbench-score-f1 -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric_5.csv \
    --conversion_factors_path scoring/si_conversion_factors.csv \
    --matching_mode material --log_level ERROR

# Compute precision using material-based matching
uv run pbench-score-precision -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric_5.csv \
    --conversion_factors_path scoring/si_conversion_factors.csv \
    --matching_mode material --log_level ERROR

# Compute recall using material-based matching
uv run pbench-score-recall -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric_5.csv \
    --conversion_factors_path scoring/si_conversion_factors.csv \
    --matching_mode material --log_level ERROR
```

4. Aggregate accuracy and dollar/token cost:

TODO:
- [ ] Allow user to specify precision, recall, or F1. Currently, it will compute F1 scores.

```bash
# Total cost
uv run pbench-aggregate -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric_5.csv \
    --conversion_factors_path scoring/si_conversion_factors.csv \
    --matching_mode material --log_level ERROR --x-axis cost

# Total tokens
uv run pbench-aggregate -od OUTPUT_DIR -m gemini-2.5-flash \
    --rubric_path scoring/rubric_5.csv \
    --conversion_factors_path scoring/si_conversion_factors.csv \
    --matching_mode material --log_level ERROR --x-axis tokens
```

## Constructing the Post-2021 version of the SuperCon Dataset from Scratch

1. Download PDFs from [Google Drive Folder](https://drive.google.com/file/d/1yrZJkDAYQpLPpgqtEqU2gn6VGjrckXFa/view?usp=drive_link) and place them in `data/new-supercon-papers/Paper_DB`:
```bash
# Assumes new-supercon-papers.tar is in the current directory
tar -xvf new-supercon-papers.tar
mkdir -p data/new-supercon-papers/ && mv supercon-new-papers data/new-supercon-papers/Paper_DB
```

2. Obtain candidate properties (skip as Anmol did this step already):

> \[!NOTE\]
> Anmol created a `out-new-supercon-papers.zip` in [Google Drive](https://drive.google.com/file/d/164MrUNANseRpk88vdl35tDMKMORY3Lsk/view?usp=drive_link). Download it and place it in current directory.

- Extract properties from PDFs using an LLM:
- [ ] @anmolkabra: Replace the prompt with [targeted_extraction_prompt.md](prompts/targeted_extraction_prompt.md)

```bash
uv run --env-file=.env pbench-extract --server gemini --model_name gemini-3-pro-preview -dd data/new-supercon-papers -od out-new-supercon-papers -pp prompts/unsupervised_extraction_prompt.md
```

- Add `data_type` column to the CSV. The resulting CSV will be saved to `out-new-supercon-papers/candidates`.

```bash
uv run pbench-filter -dd data/new-supercon-papers -od out-new-supercon-papers
```

3. Launch the validator app and accept/reject the candidates:

> \[!WARNING\]
> This step requires manual effort and is not fully reproducibile.

```bash
# Assumes out-new-supercon-papers/ exists in this directory

uv sync --group validator
uv run streamlit run ../../src/pbench_validator_app/app.py -- -od out-new-supercon-papers
```

4. Combine validation results from multiple human experts:

> \[!WARNING\]
> A CSV file will be created at `data/new-supercon-papers` with a column "validated_resolved". This will be auto-resolved if possible and set to "RESOLVE" if manual resolution is needed.

```bash
uv run python combine_validation_results.py \
    --output_dir1 out-new-supercon-papers-stoic__for_validation-joshua \
    --output_dir2 out-new-supercon-papers-stoic__for_validation-aaditya \
    --data_dir data/new-supercon-papers
```

5. Create a local HuggingFace dataset `out-new-supercon-papers/full` for the papers that have PDFS in `data/new-supercon-papers/Paper_DB`. Note: the dataset will also be shared at https://huggingface.co/datasets/kilian-group/supercon-post-2021-extraction.

```bash
uv run python create_huggingface_dataset_post-2021.py -dd data/new-supercon-papers -od out-new-supercon-papers --filter_pdf \
    --hf_revision v0.0.0 --hf_repo kilian-group/supercon-post-2021-extraction --hf_split full
```

6. Generate embeddings for the ground-truth property names for scoring:

```bash
uv run pbench-gt-embeddings --hf_repo kilian-group/supercon-post-2021-extraction --hf_revision v0.0.1 --hf_split full
```

<!-- Old command (deprecated):
```bash
uv run python generate_gt_embeddings.py --hf_revision v0.0.1 --hf_repo kilian-group/supercon-post-2021-extraction --hf_split full
```
-->

7. Create the Harbor tasks at `out-new-supercon-papers` by instantiating the Harbor template with the papers in `data/new-supercon-papers/Paper_DB`. Note: the tasks will also be shared at https://huggingface.co/datasets/kilian-group/supercon-post-2021-extraction-harbor-tasks.

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
    --pdf-dir data/new-supercon-papers/Paper_DB --output-dir out-0125-harbor-post-2021 --workspace . --template targeted-stoichiometric-template \
    --gt-hf-repo kilian-group/supercon-post-2021-extraction --gt-hf-split full --gt-hf-revision v0.0.0 \
    --force --upload-hf --hf-repo-id kilian-group/supercon-post-2021-extraction-harbor-tasks
```

## Evaluating quality of dataset construction pipeline

### Inter-annotator agreement rate:

Assuming you have two separate validation results (one at `OUTPUT_DIR_HUMAN_1` and another at `OUTPUT_DIR_HUMAN_2`), please run the following command:

```bash
uv run python compute_cohens_kappa.py -od1 OUTPUT_DIR_HUMAN_1 -od2 OUTPUT_DIR_HUMAN_2
```

### Validation Accuracy

TODO:
- [ ] Combine the below two steps into a single script.

1. To compute the validation accuracy of a single annotator, please run the following command. This script assumes the validation results are at `OUTPUT_DIR`:

```bash
uv run python format_validation_accuracy.py -od OUTPUT_DIR
```

2. To compute the validation accuracy of the resolved validation results, please run the following command:

```bash
uv run python format_validation_accuracy_resolved.py -dd data/new-supercon-papers
```
