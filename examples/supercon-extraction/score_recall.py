"""Calculate recall scores for ground truth material properties.

This script reads ground truth property matches, validates material compositions
using pymatgen, and scores property values against predictions.

Usage (from examples/supercon-extraction):
    python score_recall.py [--output_dir OUTPUT_DIR]

Example:
    python score_recall.py --output_dir ./output

"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from tabulate import tabulate
import pandas as pd

# pbench imports
import pbench
from pbench_eval.metrics import compute_recall_per_material_property
import logging

logger = logging.getLogger(__name__)

# Parse arguments
parser = ArgumentParser(
    description="Calculate recall scores for ground truth material properties"
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)
# model used for property matching
model_name = args.model_name

# Load all CSV files from output_dir/gt_matches
gt_matches_dir = args.output_dir / "gt_matches"

if not gt_matches_dir.exists():
    logger.error(f"Directory not found: {gt_matches_dir}")
    sys.exit(1)

csv_files = list(gt_matches_dir.glob("*.csv"))

if not csv_files:
    logger.error(f"No CSV files found in {gt_matches_dir}")
    sys.exit(1)

logger.info(f"Found {len(csv_files)} CSV file(s) in {gt_matches_dir}")

dfs = []
for csv_file in csv_files:
    logger.debug(f"Loading {csv_file.name}")
    df = pd.read_csv(csv_file)
    dfs.append(df)

df_matches = pd.concat(dfs, ignore_index=True)
df_matches = df_matches[df_matches["judge"] == model_name]
logger.info(
    f"Loaded {len(df_matches)} total rows using {model_name} for property matching"
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

df_results = compute_recall_per_material_property(df, conversion_df=conversion_df)

counta = lambda x: (x > 0).sum()  # noqa: E731
acc_by_refno = (
    df_results.groupby(["model", "refno"], dropna=False)
    .agg(
        recall_score=pd.NamedAgg(column="recall_score", aggfunc="mean"),
        matches=pd.NamedAgg(column="num_property_material_matches", aggfunc=counta),
    )
    .reset_index()
)
mean_sem = lambda x: f"{x.mean():.2f} ± {x.sem():.2f}"  # noqa: E731
acc = (
    acc_by_refno.groupby("model")
    .agg(
        avg_recall=pd.NamedAgg(column="recall_score", aggfunc=mean_sem),
        avg_matches=pd.NamedAgg(column="matches", aggfunc="sum"),
        count=pd.NamedAgg(column="model", aggfunc="count"),
    )
    .reset_index()
)
# print acc as table using the tabulate library with 'github' format
print(tabulate(acc, headers="keys", tablefmt="github", showindex=False))
