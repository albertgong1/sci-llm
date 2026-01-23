"""Compute Cohen's kappa between two sets of validations.

Usage:
    python compute_cohens_kappa.py --output_dir1 /path/to/output1 --output_dir2 /path/to/output2
"""

from argparse import ArgumentParser
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

import pbench

logger = logging.getLogger(__name__)

parser = ArgumentParser(
    description="Compute Cohen's kappa between two sets of validations"
)
parser.add_argument(
    "--output_dir1",
    "-od1",
    type=Path,
    required=True,
    help="Path to first output directory containing validated_candidates/",
)
parser.add_argument(
    "--output_dir2",
    "-od2",
    type=Path,
    required=True,
    help="Path to second output directory containing validated_candidates/",
)
parser.add_argument(
    "--log_level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Logging level",
)
args = parser.parse_args()
pbench.setup_logging(args.log_level)


def load_validated_candidates(output_dir: Path) -> pd.DataFrame:
    """Load and concatenate all validated candidate CSVs from an output directory."""
    validated_candidates_dir = output_dir / "validated_candidates"
    csv_files = list(validated_candidates_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {validated_candidates_dir}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} CSV file(s) in {validated_candidates_dir}")
    dfs = []
    for csv_file in csv_files:
        logger.debug(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} total rows from {output_dir}")
    return df


# Load both validation sets
df1 = load_validated_candidates(args.output_dir1)
df2 = load_validated_candidates(args.output_dir2)

# Use id column (unique per refno) combined with refno to match records
id_cols = ["refno", "id"]
for col in id_cols:
    if col not in df1.columns or col not in df2.columns:
        logger.error(f"Column '{col}' not found in one of the datasets")
        sys.exit(1)

df1["record_id"] = df1["refno"].astype(str) + "||" + df1["id"].astype(str)
df2["record_id"] = df2["refno"].astype(str) + "||" + df2["id"].astype(str)

# Filter for rows where "validated" is not nan in both datasets
df1_validated = df1[~df1["validated"].isna()][["record_id", "validated"]]
df2_validated = df2[~df2["validated"].isna()][["record_id", "validated"]]

# Merge on record_id to find common records
merged = pd.merge(
    df1_validated,
    df2_validated,
    on="record_id",
    suffixes=("_1", "_2"),
    how="inner",
)

if len(merged) == 0:
    logger.error("No common validated records found between the two datasets")
    sys.exit(1)

logger.info(f"Found {len(merged)} common validated records")

# Compute Cohen's kappa
labels1 = merged["validated_1"].astype(bool)
labels2 = merged["validated_2"].astype(bool)

kappa = cohen_kappa_score(labels1, labels2)
print(f"Cohen's kappa: {kappa:.4f}")

# Agreement statistics
agreement = (labels1 == labels2).mean()
print(f"Raw agreement: {agreement:.4f}")

# Confusion matrix breakdown
both_true = ((labels1 == True) & (labels2 == True)).sum()  # noqa: E712
both_false = ((labels1 == False) & (labels2 == False)).sum()  # noqa: E712
only_1_true = ((labels1 == True) & (labels2 == False)).sum()  # noqa: E712
only_2_true = ((labels1 == False) & (labels2 == True)).sum()  # noqa: E712

print("\nConfusion matrix:")
print(f"  Both validated=True:  {both_true}")
print(f"  Both validated=False: {both_false}")
print(f"  Only dir1 True:       {only_1_true}")
print(f"  Only dir2 True:       {only_2_true}")
