# SciLLM

## Setup Instructions

1. For first time setup, run the following script to create a conda environment `scillm` with python dependencies and packages.

```bash
./scripts/setup_conda_environment.sh
```

2. Update the packages with:

```bash
uv sync --all-groups
```

3. Set your API keys in a file named `.env` in the root directory and add

```bash
GOOGLE_API_KEY=xxxxx
OPENAI_API_KEY=xxxxx
```

## Dataset Construction

1. Download files:

<details>
    <summary>SuperCon</summary>

1. [Paper_DB.tar](https://drive.google.com/file/d/1Uq90PLAfUWSec_GusnSPWuVoLcRK5lP8/view?usp=sharing) containing 15 PDFs and untar to `data/supercon/`.
2. Download [SuperCon.csv](https://drive.google.com/file/d/1Vod_pLOV3O8Sm4glyeSVc9AMbO_XEuxZ/view?usp=drive_link) and save to `data/supercon/SuperCon.csv`.

```bash
# Assumes Paper_DB.tar is in the current directory
mkdir -p data/supercon && tar -xvf Paper_DB.tar -C data/supercon/
```

</details>

2. Extract all properties from the papers:

```bash
./src/pbench/mass_extract_properties_from_llm.py --domain supercon --model_name gemini-3-pro
# Move the outputs to `assets/supercon/validate_csv/*.csv`
```

3. Follow the instructions in [VALIDATOR_GUIDE.md](docs/VALIDATOR_GUIDE.md).

4. Share the dataset to HuggingFace using the following command:

```bash
./src/pbench/create_supercon_hf_dataset.py \
    --data_dir data/ \
    --output_dir out/ \
    --repo_name <repo_name> \
    --filter_pdf
```

## Harbor Evaluation

## Standalone LLM Evaluation (for debugging)

1. Extract all properties from each paper:

```bash
./src/pbench/mass_extract_properties_from_llm.py --domain supercon --model_name gemini-3-flash
```

Outputs of the LLM are saved in `out/supercon/unsupervised_llm_extraction/*.csv`.
