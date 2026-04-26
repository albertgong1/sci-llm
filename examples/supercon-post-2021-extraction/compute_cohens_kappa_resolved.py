"""Compute Cohen's kappa from the combined validation results CSV.

Usage:
    python compute_cohens_kappa_resolved.py --csv_path /path/to/combined_validation_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tabulate import tabulate

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

# Compute Cohen's kappa by property
print()
print("=" * 60)
print("Cohen's kappa by property:")
print("=" * 60)

# Filter to rows with both validations
df_valid = df[~df[args.col1].isna() & ~df[args.col2].isna()].copy()
df_valid[args.col1] = df_valid[args.col1].astype(bool)
df_valid[args.col2] = df_valid[args.col2].astype(bool)

# Group by property_name and compute kappa for each
property_stats: list[dict] = []
for prop_name, group in df_valid.groupby("property_name"):
    n = len(group)
    if n < 2:
        continue

    prop_labels1 = group[args.col1]
    prop_labels2 = group[args.col2]

    # Check if there's any variation (kappa undefined if all same)
    if prop_labels1.nunique() == 1 and prop_labels2.nunique() == 1:
        if prop_labels1.iloc[0] == prop_labels2.iloc[0]:
            kappa_prop = 1.0  # Perfect agreement (all same)
        else:
            kappa_prop = float("nan")
    else:
        try:
            kappa_prop = cohen_kappa_score(prop_labels1, prop_labels2)
        except ValueError:
            kappa_prop = float("nan")

    raw_agreement_prop = (prop_labels1 == prop_labels2).mean()
    both_true_prop = ((prop_labels1 == True) & (prop_labels2 == True)).sum()  # noqa: E712
    both_false_prop = ((prop_labels1 == False) & (prop_labels2 == False)).sum()  # noqa: E712
    disagree_prop = n - both_true_prop - both_false_prop

    # Compute chance agreement: P(both True by chance) + P(both False by chance)
    p1_true = prop_labels1.mean()
    p2_true = prop_labels2.mean()
    chance_agreement = p1_true * p2_true + (1 - p1_true) * (1 - p2_true)

    property_stats.append(
        {
            "property_name": prop_name,
            "n": n,
            "kappa": kappa_prop,
            "raw_agreement": raw_agreement_prop,
            "chance_agreement": chance_agreement,
            "both_true": both_true_prop,
            "both_false": both_false_prop,
            "disagree": disagree_prop,
        }
    )

# Sort by number of samples descending
property_stats = sorted(property_stats, key=lambda x: x["n"], reverse=True)

# Format for tabulate
table_data = []
for stat in property_stats:
    kappa_str = f"{stat['kappa']:.3f}" if not pd.isna(stat["kappa"]) else "N/A"
    table_data.append(
        [
            stat["property_name"],
            stat["n"],
            kappa_str,
            f"{stat['raw_agreement']:.3f}",
            f"{stat['chance_agreement']:.3f}",
            stat["both_true"],
            stat["both_false"],
            stat["disagree"],
        ]
    )

headers = ["Property", "N", "Kappa", "Raw", "Chance", "TT", "FF", "Dis"]
print(tabulate(table_data, headers=headers, tablefmt="github"))
