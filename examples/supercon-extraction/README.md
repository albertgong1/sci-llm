# SuperCon Property Extraction

> \[!NOTE\]
> We only need to construct the dataset following [Construct the Dataset from Scratch](#constructing-the-dataset-from-scratch) once.
> For now, we will use the orignal SuperCon dataset as the ground-truth.

TODO:
- [ ] Move steps for generating GT property name embeddings to [Constructing the dataset](#constructing-the-dataset-from-supercon-original).
- [ ] Push GT property name embeddings to HF.

## Experiments

1. Run the tasks using Harbor/Modal:

```bash
```

2. Compute accuracy:

```bash
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

2. Compute the accuracy of the extracted information:

```bash
./src/pbench_eval/score_task.py \
    --domain supercon \
    --task tc \
    -od out/
```

## Constructing the Dataset from Scratch

1. Mass-extract properties on supercon papers with an LLM:

```bash
./src/pbench/mass_extract_properties_from_llm.py --domain supercon --server gemini --model_name gemini-2.5-flash -od out/
```

Outputs of the LLM are saved in `out/supercon/unsupervised_llm_extraction/*.csv`.
Once you verify the format of the CSV, you can move them to `assets/supercon/validate_csv/*.csv`.

2. Run the validator app with the following:

```bash
./src/pbench_validator_app/app.py --csv_folder out/supercon/unsupervised_llm_extraction/ --paper_folder data/supercon/Paper_DB/
```

It will save a copy of the CSV file with `_validated.csv` suffix under the same folder `/out/supercon/unsupervised/llm_extraction/`.
There are more instructions in `docs/VALIDATOR_GUIDE.md`.

### Validating the dataset construction (SuperCon only)

1. Generate embeddings for the predicted and ground-truth properties in SuperCon:

```bash
uv run python generate_pred_embeddings.py -od OUTPUT_DIR
uv run python generate_gt_embeddings.py -od OUTPUT_DIR
```

2. Query LLM to determine best match between generated and ground-truth property name:

```bash
uv run python generate_property_name_matches.py -od OUTPUT_DIR -m gemini-3-flash-preview
```

3. Compute recall:

```bash
uv run python score_recall.py -od OUTPUT_DIR
```

4. Compute precision:

```bash
uv run python score_precision.py -od OUTPUT_DIR
```

5. Compute inter-annotator agreement:

```bash
```

## Constructing the Dataset from SuperCon original

1. Download files [Paper_DB.tar](https://drive.google.com/file/d/1Uq90PLAfUWSec_GusnSPWuVoLcRK5lP8/view?usp=sharing) containing 15 PDFs and untar to `data`.

```bash
# Assumes Paper_DB.tar is in the current directory
mkdir -p data && tar -xvf Paper_DB.tar -C data
```

2. Download [SuperCon.csv](https://drive.google.com/file/d/1Vod_pLOV3O8Sm4glyeSVc9AMbO_XEuxZ/view?usp=drive_link) and save to `data/SuperCon.csv`.

3. Generate mappings from properties to their corresponding units:

```bash
# The output will be saved to `property_unit_mappings.csv`
uv run python generate_property_unit_mappings.py
```

3. Generate a CSV version of the SuperCon property extraction dataset for the PDFs present in `data/Paper_DB`:

```bash
uv run python create_huggingface_dataset.py \
    --output_dir OUTPUT_DIR \
    --filter_pdf
```

4. Generate Harbor tasks:

```bash
```
