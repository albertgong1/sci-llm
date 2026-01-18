"""Calculate precision scores for predicted material properties.

This script reads predicted property matches, validates material compositions
using pymatgen, and scores property values against ground truth.

Usage (from examples/supercon-extraction):
    python score_precision.py [--output_dir OUTPUT_DIR]

Example:
    python score_precision.py --output_dir ./output

"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

# pbench imports
import pbench
from pbench_eval.metrics import compute_precision_per_material_property
import logging

logger = logging.getLogger(__name__)

# Parse arguments
parser = ArgumentParser(
    description="Calculate precision scores for predicted material properties"
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

model_name = args.model_name

# Load all CSV files from output_dir/pred_matches
pred_matches_dir = args.output_dir / "pred_matches"

if not pred_matches_dir.exists():
    logger.error(f"Directory not found: {pred_matches_dir}")
    sys.exit(1)

csv_files = list(pred_matches_dir.glob("*.csv"))

if not csv_files:
    logger.error(f"No CSV files found in {pred_matches_dir}")
    sys.exit(1)

logger.info(f"Found {len(csv_files)} CSV file(s) in {pred_matches_dir}")

dfs = []
for csv_file in csv_files:
    logger.info(f"Loading {csv_file.name}")
    df = pd.read_csv(csv_file)
    dfs.append(df)

df_matches = pd.concat(dfs, ignore_index=True)
df_matches = df_matches[df_matches["model"] == model_name]
logger.info(
    f"Loaded {len(df_matches)} total rows using {model_name} for property matching"
)

# Filter for rows where is_match is True
original_count = len(df_matches)
df_matches = df_matches[df_matches["is_match"] == True]  # noqa: E712
logger.info(
    f"Filtered to {len(df_matches)} rows where is_match=True (from {original_count} total)"
)

# Load rubric
rubric_path = Path("scoring") / "rubric_2.csv"
logger.info(f"Loading rubric from {rubric_path}")
df_rubric = pd.read_csv(rubric_path)
logger.info(f"Loaded {len(df_rubric)} rows from rubric")

# Join matches with rubric to get scoring method
logger.info("Joining matches with rubric...")
df = df_matches.merge(
    df_rubric[["property_name", "rubric"]],
    left_on="property_name_gt",
    right_on="property_name",
    how="left",
)

# Load conversion factors
conversion_factors_path = Path("scoring") / "si_conversion_factors.csv"
logger.info(f"Loading conversion factors from {conversion_factors_path}")
conversion_df = pd.read_csv(conversion_factors_path, index_col=0)

# Check for missing rubrics
missing_rubric = df["rubric"].isna().sum()
if missing_rubric > 0:
    logger.warning(f"{missing_rubric} rows have no matching rubric")

df_results = compute_precision_per_material_property(df, conversion_df=conversion_df)

# Summary statistics
logger.info("=" * 60)
logger.info("SUMMARY")
logger.info("=" * 60)
print(f"Total (material, property) pairs: {len(df_results)}")
print(f"Pairs with matches: {(df_results['num_property_material_matches'] > 0).sum()}")
print(f"Average precision score: {df_results['precision_score'].mean():.3f}")
print(f"Median precision score: {df_results['precision_score'].median():.3f}")

# save results to csv
score_dir = args.output_dir / "scores"
score_dir.mkdir(parents=True, exist_ok=True)
score_path = score_dir / "score_precision.csv"
logger.info(f"Saving precision scores to {score_path}")
df_results.to_csv(score_path, index=False)
