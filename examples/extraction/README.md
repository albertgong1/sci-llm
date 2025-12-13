# SuperCon Property Extraction

> \[!NOTE\]
> Currently, only Gemini is supported.

## Setup Instructions

1. Follow the instructions at [README.md](../../README.md#setup-instructions)

2. Install additional dependencies:

```bash
conda activate sci-llm
uv pip install google-genai pandas tqdm
# Setup Gemini API key
conda env config vars set GOOGLE_API_KEY="your-api-key-here"
```

<details>
    <summary>Create a HuggingFace dataset (from scratch)</summary>

>\[!IMPORTANT\]
> To push to huggingface, you need to authenticate using `hf auth login`. When creating a new token, set the permission level to "write".

1. Download [PaperDB.tar](https://drive.google.com/file/d/1Uq90PLAfUWSec_GusnSPWuVoLcRK5lP8/view?usp=sharing) containing 15 PDFs and untar to `data/`:

```bash
# Assumes Paper_DB.zip is in the current directory
mkdir data && tar -xvf Paper_DB.tar -C data
```

2. Run the following script to generate a HF dataset and push to `kilian-group/supercon-mini`:

```bash
python create_huggingface_dataset.py --repo_name kilian-group/supercon-mini
```

</details>

## Experiments

1. Generate predictions using `gemini-2.5-flash` for the `tc` (short for "Tc (of this sample) recommended") task:

```bash
python pred_gemini.py --task tc --output_dir out-MMDD
```

2. Compute the accuracy of the extracted information using the following command:

```bash
python score_task.py --task tc --output_dir out-MMDD --model gemini-2.5-flash
```
