# Property Extraction

## Experiments

>\[!IMPORTANT\]
> To push to huggingface, you need to authenticate using `hf auth login`. When creating a new token, set the permission level to "write".

1. Create a HuggingFace dataset containing the questions, context (i.e., path to PDF), and answer.

```bash
python create_huggingface_dataset.py --repo_name REPO_NAME
```

2. To extract properties from a paper, please run the following command:

We use the papers and properties listed in `assets/dataset.csv`.
With the following command, we can extract listed properties from the papers in `Paper_DB/` with Gemini-2.5-Flash.
The resulting CSV is stored at `results/preds__gemini-2.5-flash.csv`.

```bash
pip install uv
uv pip install google-genai pandas tqdm

conda env config vars set GOOGLE_API_KEY="your-api-key-here"

python extract_supercon_properties_w_gemini.py --save_results_csv_dir results --model_name gemini-2.5-flash
```

3. Evaluate the accuracy of the extracted information, use the following command:

```bash
```
