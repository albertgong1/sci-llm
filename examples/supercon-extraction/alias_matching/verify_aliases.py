import argparse
import json
import os
import time
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

PROMPT_TEMPLATE = """
You are an expert material scientist with years of experience in material science and physics.
You have a multiple top-tier publications in high-impact journals like Science, Nature, and PNAS.
Your task is to determine if two property names refer to the exact same physical property.

Property 1: "{pred}"
Property 2: "{gt}"

Context:
- Property 1 was extracted from a scientific paper by an automated system.
- Property 2 is a standard label from a curated database (Ground Truth).
- They might differ in capitalization, abbreviations (e.g., "Tc" vs "Critical Temperature"), or terminology.
- However, if Property 1 is seemingly related but technically different (e.g. "Tc onset" vs "Tc zero", or "lattice constant a" vs "lattice constant c"), they are NOT the same.

Question: Are these two property names refer to the exact same physical property?
Return JSON only: {{ "is_match": boolean, "reason": "concise explanation" }}
"""


def verify_pairs(
    df: pd.DataFrame, client: genai.Client, model_name: str, output_dir: Path
):
    """Verify alias pairs using LLM."""
    # 1. Deduplicate pairs to avoid re-querying
    # We care about unique (pred_name, gt_name) tuples
    unique_pairs = df[["pred_name", "gt_name"]].drop_duplicates()
    logging.info(f"Total rows in CSV: {len(df)}")
    logging.info(f"Unique pairs to verify: {len(unique_pairs)}")

    results = []

    # Check for existing results to resume
    safe_model_name = model_name.replace("/", "--")
    output_all_path = output_dir / f"alias_verification_all__{safe_model_name}.csv"
    seen_pairs = set()
    if output_all_path.exists():
        existing_df = pd.read_csv(output_all_path)
        for _, row in existing_df.iterrows():
            seen_pairs.add((row["pred_name"], row["gt_name"]))
            results.append(row.to_dict())
        logging.info(f"Resuming... Found {len(seen_pairs)} already verified pairs.")

    # 2. Iterate and Verify
    for idx, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs)):
        pred = row["pred_name"]
        gt = row["gt_name"]

        if (pred, gt) in seen_pairs:
            continue

        # Heuristic: Exact match (case-insensitive) is always True
        # Also check existing sim score if available? standardizing 1.0 check
        # THis is equivalent to a 1.0 sim score
        if str(pred).lower().strip() == str(gt).lower().strip():
            res = {
                "pred_name": pred,
                "gt_name": gt,
                "is_match": True,
                "reason": "Exact string match (auto-verified)",
                "model": "heuristic",
            }
            results.append(res)
            # Autosave periodically?
            continue

        # LLM Call
        try:
            prompt = PROMPT_TEMPLATE.format(pred=pred, gt=gt)

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
                "pred_name": pred,
                "gt_name": gt,
                "is_match": is_match,
                "reason": reason,
                "model": model_name,
            }
            results.append(res)

            # Rate limit
            time.sleep(0.5)

        except Exception as e:
            logging.error(f"Error validating '{pred}' vs '{gt}': {e}")
            time.sleep(1)

        # Incremental Save (Every 2 rows)
        if len(results) % 2 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_all_path, index=False)
            logging.info(
                f"Progress: {len(results)}/{len(unique_pairs)} pairs processed. Saved to {output_all_path}"
            )

    # 3. Save Results
    results_df = pd.DataFrame(results)

    # Merge back original info (counts, sim) for ALL results too
    full_results_df = pd.merge(results_df, df, on=["pred_name", "gt_name"], how="left")

    # Save the expanded results
    full_results_df.to_csv(output_all_path, index=False)
    logging.info(f"Saved all verification results to {output_all_path}")

    # 4. Create "Matches Only" CSV
    matches_df = results_df[results_df["is_match"] == True].copy()

    # Merge original metadata (count, avg_sim) back onto the matches
    # We merge matches_df (unique pairs) with the original df (which has counts)
    # Actually wait, the original df was unique pairs?
    # The input csv inputs are unique pairs with counts.
    # So we can just merge.

    full_info_matches = pd.merge(
        matches_df, df, on=["pred_name", "gt_name"], how="left"
    )

    output_matches_path = (
        output_dir / f"alias_verification_matches__{safe_model_name}.csv"
    )
    full_info_matches.to_csv(output_matches_path, index=False)
    logging.info(f"Saved confirmed matches to {output_matches_path}")

    # 5. Update property_aliases.json?
    # The user didn't explicitly ask to overwrite the json yet, just "create a csv".
    # But it would be useful to generate a new candidate json.
    # Let's stick to CSV for now as requested.


def main():
    parser = argparse.ArgumentParser(
        description="Verify alias candidates using Gemini LLM."
    )
    parser.add_argument(
        "candidates_csv", type=Path, help="Path to alias_candidates_review.csv"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("out_pr"), help="Output directory"
    )
    parser.add_argument(
        "--model", type=str, default="gemini-3-pro-preview", help="Gemini model to use"
    )

    args = parser.parse_args()

    if not args.candidates_csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.candidates_csv}")

    # Load input
    df = pd.read_csv(args.candidates_csv)

    # Check env
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY not found in environment.")

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    verify_aliases_logic(df, client, args.model, args.output_dir)


# Wrapper to allow importing logic if needed testing
def verify_aliases_logic(df, client, model, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_pairs(df, client, model, output_dir)


if __name__ == "__main__":
    main()
