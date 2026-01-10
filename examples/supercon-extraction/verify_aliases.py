"""Script to match property names to ground truth labels

Below is our two-stage approach:
1. We first use the embeddings to obtain the top-k matches for each property name.
2. We then query an LLM to check if the property names are the same.

Example usage:
```bash
uv run python verify_aliases.py -m gemini-3-pro-preview -od out/
```
"""

# standard imports
import argparse
import json
import os
import time
import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from functools import cache

# llm imports
from google import genai
from google.genai import types

# pbench imports
import pbench

PROMPT_TEMPLATE = """
You are an expert material scientist with years of experience in material science and physics.
You have a multiple top-tier publications in high-impact journals like Science, Nature, and PNAS.
Your task is to determine if two property names refer to the exact same physical property.

Property 1: "{s1}"
Property 2: "{s2}"

Context:
- Property 1 was extracted from a scientific paper by an automated system.
- Property 2 is a standard label from a curated database (Ground Truth).
- They might differ in capitalization, abbreviations (e.g., "Tc" vs "Critical Temperature"), or terminology.
- However, if Property 1 is seemingly related but technically different (e.g. "Tc onset" vs "Tc zero", or "lattice constant a" vs "lattice constant c"), they are NOT the same.

Question: Are these two property names refer to the exact same physical property?
Return JSON only: {{ "is_match": boolean, "reason": "concise explanation" }}
"""


@cache
def check_if_same_property(s1: str, s2: str, model_name: str) -> dict:
    """Check if two property names are the same using an LLM.

    TODO (Albert):
    - pass in s1 and s2 in alphabetical order, so that we can save on LLM calls
    - Use async requests

    Args:
        s1: First property name
        s2: Second property name
        model_name: Name of the model to use

    Returns:
        dict: Dictionary containing the result of the check

    """
    # LLM Call
    try:
        prompt = PROMPT_TEMPLATE.format(s1=s1, s2=s2)

        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.0
            ),
        )

        if response.text:
            data = json.loads(response.text)
            is_match = data.get("is_match", False)
            reason = data.get("reason", "No reason provided")
        else:
            is_match = False
            reason = "Empty response from LLM"

        res = {
            "s1": s1,
            "s2": s2,
            "is_match": is_match,
            "reason": reason,
            "model": model_name,
        }
        return res

        # Rate limit
        # time.sleep(0.5)

    except Exception as e:
        logging.error(f"Error validating '{s1}' vs '{s2}': {e}")
        time.sleep(1)


#
# Parse arguments
#
parser = argparse.ArgumentParser(
    description="Verify alias candidates using Gemini LLM."
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()
# Setup logging
pbench.setup_logging(args.log_level)
logger = logging.getLogger(__name__)
# Check env
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment.")
# Load env variables
load_dotenv()

output_dir = args.output_dir
model_name = args.model_name
split = args.split
dataset = args.dataset
top_k = 3
force = args.force

#
# Load embeddings
#
pred_embeddings_dir = args.output_dir / "pred_embeddings"
gt_embeddings_dir = args.output_dir / "gt_embeddings"

# Get embeddings for predicted property names
pred_embeddings_files = list(pred_embeddings_dir.glob("*.parquet"))
if not pred_embeddings_files:
    raise FileNotFoundError(f"No embeddings files found in {pred_embeddings_dir}")

#
# Main Loop
#
matches_dir = args.output_dir / "matches"
matches_dir.mkdir(parents=True, exist_ok=True)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

results = []
for pred_file in pred_embeddings_files:
    df_pred = pd.read_parquet(pred_file)
    assert len(df_pred["refno"].unique()) == 1, "Expected only one refno per file"
    refno = df_pred["refno"].unique()[0]
    df_gt = pd.read_parquet(gt_embeddings_dir / f"{refno}.parquet")
    logger.info(f"Processing refno {refno}...")

    similarity_matrix = cosine_similarity(
        np.vstack(df_pred["embedding"].values), np.vstack(df_gt["embedding"].values)
    )

    # Get top-k matches for each predicted property name (to compute precision)
    pred_matches_path = matches_dir / f"{refno}_pred_matches.csv"
    if pred_matches_path.exists() and not force:
        logging.info(f"Skipping refno {refno} because pred matches already exist")
    else:
        top_k_matches_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][
            :, :top_k
        ]
        pred_results = []
        for i in range(len(df_pred)):
            pred_name = df_pred.iloc[i]["property_name"]
            top_k_matches = df_gt.iloc[top_k_matches_indices[i]][
                "property_name"
            ].tolist()
            for rank, match in enumerate(top_k_matches):
                # import pdb; pdb.set_trace()
                # always pass in pred_name and match in alphabetical order
                result = check_if_same_property(pred_name, match, model_name)
                result["rank"] = rank
                pred_results.append(result)
        # save pred_results to csv
        pd.DataFrame(pred_results).to_csv(pred_matches_path, index=False)
        logging.info(f"Saved pred matches to {pred_matches_path}")

    # Get top-k matches for each ground truth property name (to compute recall)
    # import pdb; pdb.set_trace()
    top_k_matches_indices = np.argsort(similarity_matrix.T, axis=1)[:, ::-1][:, :top_k]
    gt_matches_path = matches_dir / f"{refno}_gt_matches.csv"
    if gt_matches_path.exists() and not force:
        logging.info(f"Skipping refno {refno} because gt matches already exist")
    else:
        gt_results = []
        for i in range(len(df_gt)):
            gt_name = df_gt.iloc[i]["property_name"]
            top_k_matches = df_pred.iloc[top_k_matches_indices[i]][
                "property_name"
            ].tolist()
            for rank, match in enumerate(top_k_matches):
                # import pdb; pdb.set_trace()
                result = check_if_same_property(gt_name, match, model_name)
                result["rank"] = rank
                gt_results.append(result)
        # save gt_results to csv
        pd.DataFrame(gt_results).to_csv(gt_matches_path, index=False)
        logging.info(f"Saved gt matches to {gt_matches_path}")
