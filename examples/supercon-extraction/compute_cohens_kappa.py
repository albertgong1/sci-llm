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
from tabulate import tabulate

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

# Columns to include in the output
extra_cols = [
    "material_or_system",
    "property_name",
    "value_string",
    "condition1_name",
    "condition1_value",
    "condition2_name",
    "condition2_value",
    "condition3_name",
    "condition3_value",
    "location.page",
    "location.section",
    "location.source_type",
    "location.evidence",
    "location.figure_or_table",
    "flagged",
    "validator_note",
]

# Filter for rows where "validated" is not nan in both datasets
cols_to_keep = ["record_id", "validated"] + extra_cols
df1_validated = df1[~df1["validated"].isna()][cols_to_keep]
df2_validated = df2[~df2["validated"].isna()][
    ["record_id", "validated", "validator_note", "flagged"]
]

# Extract annotator names from directory names
annotator1 = str(args.output_dir1).split("-")[-1]
annotator2 = str(args.output_dir2).split("-")[-1]

# Merge on record_id to find common records
merged = pd.merge(
    df1_validated,
    df2_validated,
    on="record_id",
    suffixes=(f"_{annotator1}", f"_{annotator2}"),
    how="inner",
)

if len(merged) == 0:
    logger.error("No common validated records found between the two datasets")
    sys.exit(1)

logger.info(f"Found {len(merged)} common validated records")

# Compute Cohen's kappa
labels1 = merged[f"validated_{annotator1}"].astype(bool)
labels2 = merged[f"validated_{annotator2}"].astype(bool)

kappa = cohen_kappa_score(labels1, labels2)
print(f"Cohen's kappa: {kappa:.4f}")

# Agreement statistics
agreement = (labels1 == labels2).mean()
print(f"Raw agreement: {agreement:.4f}")

# Confusion matrix breakdown
both_true_mask = (labels1 == True) & (labels2 == True)  # noqa: E712
both_false_mask = (labels1 == False) & (labels2 == False)  # noqa: E712
only_1_true_mask = (labels1 == True) & (labels2 == False)  # noqa: E712
only_2_true_mask = (labels1 == False) & (labels2 == True)  # noqa: E712

print("\nConfusion matrix:")
print(f"  Both validated=True:      {both_true_mask.sum()}")
print(f"  Both validated=False:     {both_false_mask.sum()}")
print(f"  Only {annotator1} True:  {only_1_true_mask.sum()}")
print(f"  Only {annotator2} True:  {only_2_true_mask.sum()}")

# Property-wise breakdown of disagreements
disagreement_mask = only_1_true_mask | only_2_true_mask
if disagreement_mask.sum() > 0:
    disagreements = merged[disagreement_mask]
    property_counts = disagreements["property_name"].value_counts()
    print("\nDisagreements by property:")
    for prop, count in property_counts.items():
        print(f"  {prop}: {count}")

# Compute Cohen's kappa by property
print()
print("=" * 60)
print("Cohen's kappa by property:")
print("=" * 60)

property_stats: list[dict] = []
for prop_name, group in merged.groupby("property_name"):
    n = len(group)
    if n < 2:
        continue

    prop_labels1 = group[f"validated_{annotator1}"].astype(bool)
    prop_labels2 = group[f"validated_{annotator2}"].astype(bool)

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

# Add confusion matrix category column
merged["agreement_category"] = "unknown"
merged.loc[both_true_mask, "agreement_category"] = "both_true"
merged.loc[both_false_mask, "agreement_category"] = "both_false"
merged.loc[only_1_true_mask, "agreement_category"] = f"only_{annotator1}_true"
merged.loc[only_2_true_mask, "agreement_category"] = f"only_{annotator2}_true"

# Reorder columns so validation columns are grouped by annotator
validation_cols_1 = [
    f"validated_{annotator1}",
    f"validator_note_{annotator1}",
    f"flagged_{annotator1}",
]
validation_cols_2 = [
    f"validated_{annotator2}",
    f"validator_note_{annotator2}",
    f"flagged_{annotator2}",
]
other_cols = [
    c for c in merged.columns if c not in validation_cols_1 + validation_cols_2
]
merged = merged[other_cols + validation_cols_1 + validation_cols_2]

# Save rows validated by both annotators to CSV
output_csv = Path("both_annotators_validated.csv")
merged.to_csv(output_csv, index=False)
logger.info(f"Saved {len(merged)} rows to {output_csv}")
