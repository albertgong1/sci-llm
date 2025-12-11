# SuperCon Property Extraction

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

1. Download the collection of (15) papers from https://drive.google.com/file/d/1kJz-zlOdRF5_9gtt3oeoKN2-29GPBLNQ/view?usp=share_link to the current directory and unzip to `data/`:

```bash
# Assumes Paper_DB.zip is in the current directory
tar -xvf Paper_DB.zip -C data/
```

2. Run the following script to generate a HF dataset and push to `kilian-group/supercon-mini`:

```bash
python create_huggingface_dataset.py --repo_name kilian-group/supercon-mini
```

</details>

## Experiments

1. To extract properties from a paper, please run the following command:

We use the papers and properties listed in `assets/dataset.csv`.
With the following command, we can extract listed properties from the papers in `Paper_DB/` with Gemini-2.5-Flash.
The resulting CSV is stored at `results/preds__gemini-2.5-flash.csv`.

```bash
python extract_supercon_properties_w_gemini.py --save_results_csv_dir results --model_name gemini-2.5-flash
```

2. Evaluate the accuracy of the extracted information, use the following command:

```bash
python score.py preds__gemini-2.5-flash.csv
```
