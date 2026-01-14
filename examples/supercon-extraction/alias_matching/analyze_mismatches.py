import pandas as pd
from pathlib import Path
import argparse
import logging

# Use python-Levenshtein if installed, else fallback to standard lib implementation
try:
    from Levenshtein import ratio
except ImportError:
    # Fallback to simple ratio using difflib
    from difflib import SequenceMatcher

    def ratio(a, b):
        return SequenceMatcher(None, a, b).ratio()


logging.basicConfig(level=logging.INFO, format="%(message)s")


def analyze_mismatches_from_df(df: pd.DataFrame, output_dir: Path, top_n: int = 200):
    """Analyze mismatches from a pre-loaded DataFrame and generate alias candidates.

    Args:
        df: DataFrame containing 'status', 'refno', 'property_name'
        output_dir: Directory to save output files
        top_n: Number of top candidates to keep (default 200)

    """
    # Filter for matches that are NOT TP (i.e. FP or FN)
    mismatches = df[df["status"].isin(["FP", "FN"])]

    if mismatches.empty:
        logging.info("No mismatches found to analyze.")
        return

    logging.info(f"Analyzing {len(mismatches)} mismatch records...")

    # Mismatches per RefNo
    refnos = mismatches["refno"].unique()

    suggestions = []

    for refno in refnos:
        paper_df = mismatches[mismatches["refno"] == refno]

        # Get FP names (Predictions that didn't match)
        fps = paper_df[paper_df["status"] == "FP"]["property_name"].unique()

        # Get FN names (GT that wasn't found)
        # Note: In matches file, for FN, property_name is the GT label
        fns = paper_df[paper_df["status"] == "FN"]["property_name"].unique()

        for fp_name in fps:
            # Calculate similarity for ALL potential FN matches
            matches_for_fp = []
            for fn_name in fns:
                sim = ratio(str(fp_name).lower(), str(fn_name).lower())
                matches_for_fp.append(
                    {
                        "refno": refno,
                        "pred_name": fp_name,
                        "gt_name": fn_name,
                        "similarity": sim,
                    }
                )

            # Sort by similarity descending
            matches_for_fp.sort(key=lambda x: x["similarity"], reverse=True)

            # Take Top K (e.g. 3) best matches for this specific FP, regardless of score
            k = 3
            suggestions.extend(matches_for_fp[:k])

    # Aggregate suggestions
    sugg_df = pd.DataFrame(suggestions)
    if sugg_df.empty:
        logging.info("No obvious naming mismatches found.")
        return

    # Count frequencies of these pairs
    summary = (
        sugg_df.groupby(["pred_name", "gt_name"])
        .agg(count=("refno", "count"), avg_sim=("similarity", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # ---------------------------------------------------------
    # Generate Aliases Mapping
    # Logic:
    # 1. Sort by Count (desc) then Score (desc)
    # 2. For each unique Pred, pick the best GT match
    # 3. Save to JSON
    # ---------------------------------------------------------

    aliases = {}

    # Use top_n
    top_candidates = summary.head(top_n)

    for _, row in top_candidates.iterrows():
        p = row["pred_name"]
        g = row["gt_name"]

        # Simple heuristic: if we haven't mapped this Pred yet, take the most frequent valid match
        if p not in aliases:
            aliases[p] = g

    # Save Candidates CSV (for Expert Review, includes clashes)
    # User wants to see ALL candidates (Top 3 per FP per paper aggregated)
    # Sort by Predicted Name so matches for the same FP are grouped together
    summary_sorted = summary.sort_values(
        ["pred_name", "count"], ascending=[True, False]
    )

    cand_path = output_dir / "alias_candidates_review.csv"
    summary_sorted.to_csv(cand_path, index=False)
    logging.info(
        f"[Artifact Generated] All {len(summary_sorted)} candidate pairs saved to: {cand_path}"
    )

    # Save to JSON
    out_path = output_dir / "property_aliases.json"
    import json

    with open(out_path, "w") as f:
        json.dump(aliases, f, indent=2, sort_keys=True)

    logging.info(f"[Artifact Generated] Draft aliases saved to: {out_path}")


def analyze_mismatches(matches_file: Path):
    if not matches_file.exists():
        logging.error(f"File not found: {matches_file}")
        return

    df = pd.read_csv(matches_file)
    analyze_mismatches_from_df(df, matches_file.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matches_file", type=Path, help="Path to score_pr_matches.csv")
    args = parser.parse_args()

    analyze_mismatches(args.matches_file)
