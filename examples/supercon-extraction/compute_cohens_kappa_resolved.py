"""Compute Cohen's kappa from the combined validation results CSV.

Usage:
    python compute_cohens_kappa_resolved.py --csv_path /path/to/combined_validation_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser(
    description="Compute Cohen's kappa from combined validation results CSV"
)
parser.add_argument(
    "--csv_path",
    type=Path,
    default=Path(__file__).parent
    / "data/new-supercon-papers/combined_validation_results RESOLVED.csv",
    help="Path to the combined validation results CSV",
)
parser.add_argument(
    "--col1",
    type=str,
    default="validated_joshua",
    help="First validation column name",
)
parser.add_argument(
    "--col2",
    type=str,
    default="validated_aaditya",
    help="Second validation column name",
)
args = parser.parse_args()

# Load the CSV
df = pd.read_csv(args.csv_path)

# Get the validation columns
labels1 = df[args.col1]
labels2 = df[args.col2]

# Filter out rows where either value is NaN
valid_mask = ~labels1.isna() & ~labels2.isna()
labels1 = labels1[valid_mask].astype(bool)
labels2 = labels2[valid_mask].astype(bool)

print(f"Total rows: {len(df)}")
print(f"Rows with both validations: {valid_mask.sum()}")
print()

# Compute Cohen's kappa
kappa = cohen_kappa_score(labels1, labels2)
print(f"Cohen's kappa: {kappa:.4f}")

# Raw agreement
agreement = (labels1 == labels2).mean()
print(f"Raw agreement: {agreement:.4f}")

# Confusion matrix breakdown
annotator1 = args.col1.replace("validated_", "")
annotator2 = args.col2.replace("validated_", "")

both_true = ((labels1 == True) & (labels2 == True)).sum()  # noqa: E712
both_false = ((labels1 == False) & (labels2 == False)).sum()  # noqa: E712
only_1_true = ((labels1 == True) & (labels2 == False)).sum()  # noqa: E712
only_2_true = ((labels1 == False) & (labels2 == True)).sum()  # noqa: E712

print()
print("Confusion matrix:")
print(f"  Both validated=True:       {both_true}")
print(f"  Both validated=False:      {both_false}")
print(f"  Only {annotator1} True:    {only_1_true}")
print(f"  Only {annotator2} True:    {only_2_true}")
