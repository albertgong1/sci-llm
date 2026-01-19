# SuperCon Property Extraction

> \[!NOTE\]
> Currently, we are using the original SuperCon dataset as the ground-truth, so please follow the instructions under [Constructing the Dataset from SuperCon original](#constructing-the-dataset-from-supercon-original) to construct the dataset.

TODO:
- [X] Move steps for generating GT property name embeddings to [Constructing the dataset](#constructing-the-dataset-from-supercon-original).
- [ ] Share embeddings on HF.
- [ ] Add instructions for computing interannotator agreement for post-2021 SuperCon.
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
uv run python ../../src/harbor-task-gen/run_harbor.py jobs start \
  --hf-tasks-repo kilian-group/supercon-extraction-harbor-tasks --hf-tasks-version v0.0.0 \
  -a gemini-cli -m gemini/gemini-3-flash-preview \
  --workspace . --jobs-dir JOBS_DIR --seed 1 --batch-size 10
```

<details>
    <summary>Instructions for running Harbor tasks saved locally</summary>

```bash
uv run python ../../src/harbor-task-gen/run_harbor.py jobs start \
  --registry-path OUTPUT_DIR/ground-template/registry.json --dataset supercon-extraction@v0.0.0 \
  -a gemini-cli -m gemini/gemini-3-flash-preview \
  --workspace . --jobs-dir JOBS_DIR --seed 1 --batch-size 10
```

</details>

2. Compute accuracy across tasks:

```bash
uv run python format_accuracy.py -jd JOBS_DIR
```

## Experiments using simple LLM API (for debugging only)

1. Generate predictions using `gemini-2.5-flash` for the `tc` (short for "Tc (of this sample) recommended") task.
This uses the existing huggingface repo https://huggingface.co/datasets/kilian-group/supercon-mini.

Outputs are stored at `out/supercon/preds/*.json`.

```bash
./src/pbench_eval/extract_properties.py \
    --domain supercon \
    --task tc \
    --server gemini \
    --model_name gemini-2.5-flash \
    -od out/
```
- [ ] Update these instructions

2. Generate embeddings for the predicted and ground-truth properties in SuperCon:

```bash
uv run python generate_pred_embeddings.py -od OUTPUT_DIR
```

3. Query LLM to determine best match between generated and ground-truth property name:

```bash
uv run python generate_property_name_matches.py -od OUTPUT_DIR -m gemini-3-flash-preview
```

4. Compute precision:

```bash
uv run python score_precision.py -od OUTPUT_DIR
```

5. Compute recall:

```bash
uv run python score_recall.py -od OUTPUT_DIR
```

## Constructing the Dataset from SuperCon original

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
uv run python create_huggingface_dataset.py -dd DATA_DIR -od OUTPUT_DIR --filter_pdf \
    --tag_name v0.0.0 --repo_name kilian-group/supercon-extraction --split SPLIT
```

5. Generate embeddings for the ground-truth property names for scoring:

```bash
uv run python generate_gt_embeddings.py
```

6. Create the Harbor tasks at `OUTPUT_DIR` by instantiating the Harbor template with the papers in `DATA_DIR/Paper_DB`. Note: the tasks will also be shared at https://huggingface.co/datasets/kilian-group/supercon-extraction-harbor-tasks.

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
    --pdf-dir DATA_DIR/Paper_DB --output-dir OUTPUT_DIR --workspace . \
    --gt-hf-repo kilian-group/supercon-extraction --gt-hf-split SPLIT --gt-hf-revision main \
    --force --upload-hf --hf-repo-id kilian-group/supercon-extraction-harbor-tasks --hf-repo-type dataset --hf-dataset-version v0.1.0
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

```bash
uv run --env-file=.env pbench-extract --server gemini --model_name gemini-3-pro-preview -dd data/new-supercon-papers -od out-new-supercon-papers -pp prompts/unsupervised_extraction_prompt.md
```

- Add `data_type` column to the CSV. The resulting CSV will be saved to `out-new-supercon-papers/candidates`.

```bash
uv run pbench-filter -dd data/new-supercon-papers -od out-new-supercon-papers
```

3. Launch the validator app and accept/reject the candidates:

> \[!NOTE\]
> This step requires manual effort and is not fully reproducibile.

```bash
# Assumes out-new-supercon-papers/ exists in this directory

uv sync --group validator
uv run streamlit run ../../src/pbench_validator_app/app.py -- -od out-new-supercon-papers
```

4. Create a local HuggingFace dataset `out-new-supercon-papers/SPLIT` for the papers that have PDFS in `data/new-supercon-papers/Paper_DB`. Note: the dataset will also be shared at https://huggingface.co/datasets/kilian-group/supercon-post-2021-extraction.

> \[!NOTE\]
> Replace `SPLIT` with `lite` or `full` depending on the version of the dataset you want to create.

```bash
uv run python create_huggingface_dataset.py -dd data/new-supercon-papers -od out-new-supercon-papers --filter_pdf \
    --tag_name v0.0.0 --repo_name kilian-group/supercon-post-2021-extraction --split SPLIT
```

5. Create the Harbor tasks at `out-new-supercon-papers` by instantiating the Harbor template with the papers in `data/new-supercon-papers/Paper_DB`. Note: the tasks will also be shared at https://huggingface.co/datasets/kilian-group/supercon-post-2021-extraction-harbor-tasks.

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
    --pdf-dir data/new-supercon-papers/Paper_DB --output-dir out-new-supercon-papers --workspace . \
    --gt-hf-repo kilian-group/supercon-post-2021-extraction --gt-hf-split SPLIT --gt-hf-revision v0.0.0 \
    --force --upload-hf --hf-repo-id kilian-group/supercon-post-2021-extraction-harbor-tasks
```
