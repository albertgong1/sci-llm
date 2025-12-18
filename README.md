# SciLLM

## Setup Instructions

For first time setup, run the following script to create a conda environment `scillm` with python dependencies and packages.

```bash
./scripts/setup_conda_environment.sh
```

Update the packages with:

```bash
uv sync --all-groups
```

## Run Property Extraction Experiments

Set your API keys in a file named `.env` in the root directory and add

```bash
GOOGLE_API_KEY=xxxxx
OPENAI_API_KEY=xxxxx
```

### SuperCon Domain

1. Download [Paper_DB.tar](https://drive.google.com/file/d/1Uq90PLAfUWSec_GusnSPWuVoLcRK5lP8/view?usp=sharing) containing 15 PDFs and untar to `data/supercon/`:

```bash
# Assumes Paper_DB.tar is in the current directory
mkdir -p data/supercon && tar -xvf Paper_DB.tar -C data/supercon/
```

2. Generate predictions using `gemini-2.5-flash` for the `tc` (short for "Tc (of this sample) recommended") task.
Outputs are stored at `out/supercon/preds/*.json`.

```bash
./src/pbench_eval/extract_properties.py \
    --domain supercon \
    --task tc \
    --server gemini \
    --model_name gemini-2.5-flash \
    -od out/
```

3. Compute the accuracy of the extracted information:

```bash
./src/pbench_eval/score_task.py \
    --domain supercon \
    --task tc \
    -od out/
```
