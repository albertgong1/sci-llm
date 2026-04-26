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

### Stable 200-paper PDF vs MinerU workflow

Build both registries with the existing compare-source builder:

```bash
./scripts/build_harbor_compare_sources.sh --force
```

Run the paired PDF vs MinerU workflow with the existing compare-source runner:

```bash
HARBOR_COMPARE_SCORE_ROOT=jobs-compare-stable-scores \
./scripts/run_harbor_compare_sources.sh jobs-compare-stable --modal --n-concurrent 10
```

This writes:
- jobs under `jobs-compare-stable/{pdf,mineru}`
- final scores under `jobs-compare-stable-scores/{pdf,mineru}`
- rolling status in each score directory via `progress_status.json`
- build/run/score manifests that pin the exact dataset and scoring configuration

If you want to score the finished jobs again without rerunning Harbor:

```bash
./scripts/score_harbor_compare_sources.sh jobs-compare-stable jobs-compare-stable-scores
```

For legacy `origin/main` comparability, set both:

```bash
export HARBOR_COMPARE_DATASET=supercon-extraction@main
export SUPERCON_GT_HF_REVISION=main
```

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
    --hf_repo kilian-group/supercon-extraction --hf_split full --hf_revision v0.2.1 \
    --prompt_path prompts/property_matching_prompt.md
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

```bash
# Generate embeddings for predicted property names
uv run pbench-pred-embeddings -od OUTPUT_DIR

# Query LLM to determine best match between generated and ground-truth property name:
uv run pbench-generate-matches -od OUTPUT_DIR -m gemini-2.5-flash \
    --hf_repo kilian-group/supercon-extraction --hf_split full --hf_revision v0.2.1 \
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

## Constructing the Dataset

1. Download the following from Google Drive (email ag2435@cornell.edu for access) and place in `DATA_DIR/Paper_DB`.

* Link: [Google Drive Folder](https://drive.google.com/drive/folders/1Kk6kZAzgLMNlmlsKPJcvqCoW5_IVuQKb?usp=sharing). Contains 1339 PDFs.

* Additionally, download the CSV of SuperCon refno to arXiv PDF name mapping by going to the following [Google Sheet](https://docs.google.com/spreadsheets/d/14MW-16wK7h4gOPJsexllRY_Zzx3WNEa4_pQ87oQrg14/edit?gid=933802094#gid=933802094) -> navigate to the Sheet named "Arxiv" -> click "File" -> "Download" -> "Comma Separated values (.csv)" and place at `DATA_DIR/SuperCon Property Extraction Dataset - Arxiv.csv`.

Rename the PDFs from arXiv IDs to paper_ids (refnos) based on the CSV mapping:

```bash
uv run python rename_arxiv_pdfs.py --data-dir DATA_DIR
```

<details>
    <summary>Download instructions for Lite version</summary>

Link: [Paper_DB.tar](https://drive.google.com/file/d/1Uq90PLAfUWSec_GusnSPWuVoLcRK5lP8/view?usp=sharing). Contains 15 PDFs.

```bash
# Assumes Paper_DB.tar is in the current directory
mkdir -p data && tar -xvf Paper_DB.tar -C DATA_DIR
```

</details>

2. Download [SuperCon.csv](https://drive.google.com/file/d/1Vod_pLOV3O8Sm4glyeSVc9AMbO_XEuxZ/view?usp=drive_link) and save to `DATA_DIR/SuperCon.csv`.

3. Generate mappings from properties to their corresponding units:

```bash
# The output will be saved to `property_unit_mappings.csv`
uv run python generate_property_unit_mappings.py
```

4. Create a local HuggingFace dataset `OUTPUT_DIR/SPLIT` for the papers that have PDFS in `DATA_DIR/Paper_DB`. Note: the dataset will also be shared at https://huggingface.co/datasets/kilian-group/supercon-extraction.

> \[!NOTE\]
> Replace `SPLIT` with `lite` or `full` depending on the version of the dataset you want to create.
> Also make sure to update HF_DATASET_NAME, HF_DATASET_REVISION, and HF_DATASET_SPLIT accordingly.

```bash
uv run python create_huggingface_dataset.py -dd data-arxiv -od out-0122-harbor --filter_pdf \
    --hf_revision v0.2.1 --hf_repo kilian-group/supercon-extraction --hf_split full
```

5. Generate embeddings for the ground-truth property names for scoring:

```bash
uv run pbench-gt-embeddings --hf_repo kilian-group/supercon-extraction --hf_revision v0.2.1 --hf_split full
```

<!-- Old command (deprecated):
```bash
uv run python generate_gt_embeddings.py
```
-->

6. Create the Harbor tasks at `OUTPUT_DIR` by instantiating the Harbor template with the papers in `DATA_DIR/Paper_DB`. Note: the tasks will also be shared at https://huggingface.co/datasets/kilian-group/supercon-extraction-harbor-tasks.

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
    --pdf-dir DATA_DIR/Paper_DB --output-dir OUTPUT_DIR --workspace . --template targeted-template \
    --gt-hf-repo kilian-group/supercon-extraction --gt-hf-split SPLIT --gt-hf-revision main \
    --force --upload-hf --hf-repo-id kilian-group/supercon-extraction-harbor-tasks --hf-repo-type dataset --hf-dataset-version v0.2.0
```

**Stoichiometric variant of prompt:**

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
    --pdf-dir data-arxiv/Paper_DB --output-dir out-0121-harbor --workspace . --template targeted-stoichiometric-template \
    --gt-hf-repo kilian-group/supercon-extraction --gt-hf-split full --gt-hf-revision main --force
```
