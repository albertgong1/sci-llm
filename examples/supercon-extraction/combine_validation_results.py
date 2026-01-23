"""Combine validation results from two output directories.

Creates a combined CSV with separate columns for each annotator's validation results.
Papers validated by both annotators will have both columns filled; papers validated
by only one annotator will have only that annotator's columns filled.

Usage:
    python combine_validation_results.py --output_dir1 /path/to/output1 --output_dir2 /path/to/output2
"""

from argparse import ArgumentParser
import logging
import sys
from pathlib import Path

import pandas as pd

import pbench

logger = logging.getLogger(__name__)

parser = ArgumentParser(
    description="Combine validation results from two output directories"
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
    "--data_dir",
    "-d",
    type=Path,
    default=Path("data"),
    help="Path to data directory for output files",
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

# Extract annotator names from directory names
annotator1 = str(args.output_dir1).rstrip("/").split("-")[-1]
annotator2 = str(args.output_dir2).rstrip("/").split("-")[-1]
logger.info(f"Annotator 1: {annotator1}, Annotator 2: {annotator2}")

# Create record_id for matching (refno + id)
df1["record_id"] = df1["refno"].astype(str) + "||" + df1["id"].astype(str)
df2["record_id"] = df2["refno"].astype(str) + "||" + df2["id"].astype(str)

# Get unique papers (refnos) from each dataset where at least one record is validated
papers1 = set(df1[df1["validated"].notna()]["refno"].unique())
papers2 = set(df2[df2["validated"].notna()]["refno"].unique())

overlapping_papers = papers1 & papers2
only_in_1 = papers1 - papers2
only_in_2 = papers2 - papers1

logger.info(f"Papers validated by {annotator1}: {len(papers1)}")
logger.info(f"Papers validated by {annotator2}: {len(papers2)}")
logger.info(f"Overlapping papers: {len(overlapping_papers)}")
logger.info(f"Papers only validated by {annotator1}: {len(only_in_1)}")
logger.info(f"Papers only validated by {annotator2}: {len(only_in_2)}")
logger.info(f"Total unique papers validated: {len(papers1 | papers2)}")

# Columns for validation results (these will be duplicated per annotator)
validation_cols = ["validated", "validator_note", "flagged"]

# Columns that are shared (property data, not annotator-specific)
shared_cols = [
    "record_id",
    "refno",
    "id",
    "material_or_system",
    "sample_label",
    "property_name",
    "category",
    "value_string",
    "value_number",
    "units",
    "method",
    "notes",
    "location.page",
    "location.section",
    "location.source_type",
    "location.evidence",
    "location.figure_or_table",
    "condition1_name",
    "condition1_value",
    "condition2_name",
    "condition2_value",
    "condition3_name",
    "condition3_value",
    "condition4_name",
    "condition4_value",
    "condition5_name",
    "condition5_value",
    "paper_pdf_path",
    "agent",
    "model",
    "data_type",
    "is_target",
]

# Filter to only columns that exist in both dataframes
shared_cols = [c for c in shared_cols if c in df1.columns and c in df2.columns]

# Prepare df1 with annotator-specific validation columns
df1_cols = shared_cols + [f"{c}_{annotator1}" for c in validation_cols]
df1_renamed = df1[shared_cols + validation_cols].copy()
df1_renamed = df1_renamed.rename(
    columns={c: f"{c}_{annotator1}" for c in validation_cols}
)

# Prepare df2 with annotator-specific validation columns
df2_cols = [f"{c}_{annotator2}" for c in validation_cols]
df2_renamed = df2[["record_id"] + validation_cols].copy()
df2_renamed = df2_renamed.rename(
    columns={c: f"{c}_{annotator2}" for c in validation_cols}
)

# Outer merge to get all records
combined = pd.merge(
    df1_renamed,
    df2_renamed,
    on="record_id",
    how="outer",
)

# For records only in df2, we need to fill in the shared columns from df2
records_only_in_df2 = combined["refno"].isna()
if records_only_in_df2.any():
    # Get the record_ids that are only in df2
    missing_record_ids = combined.loc[records_only_in_df2, "record_id"].tolist()
    # Get the shared column data from df2 for these records
    df2_shared = df2[df2["record_id"].isin(missing_record_ids)][shared_cols].copy()
    df2_shared = df2_shared.set_index("record_id")

    # Fill in missing shared columns
    for col in shared_cols:
        if col == "record_id":
            continue
        combined.loc[records_only_in_df2, col] = combined.loc[
            records_only_in_df2, "record_id"
        ].map(df2_shared[col])

# Reorder columns: shared cols first, then annotator1 validation, then annotator2 validation
validation_cols_1 = [f"{c}_{annotator1}" for c in validation_cols]
validation_cols_2 = [f"{c}_{annotator2}" for c in validation_cols]
final_cols = shared_cols + validation_cols_1 + validation_cols_2
combined = combined[final_cols]

# Sort by refno and id
combined = combined.sort_values(["refno", "id"]).reset_index(drop=True)

# Print summary statistics
total_records = len(combined)
records_with_both = (
    combined[f"validated_{annotator1}"].notna()
    & combined[f"validated_{annotator2}"].notna()
)
records_only_1 = (
    combined[f"validated_{annotator1}"].notna()
    & combined[f"validated_{annotator2}"].isna()
)
records_only_2 = (
    combined[f"validated_{annotator1}"].isna()
    & combined[f"validated_{annotator2}"].notna()
)

# Add agreement_category column for records validated by both annotators
labels1 = combined[f"validated_{annotator1}"].astype("boolean")
labels2 = combined[f"validated_{annotator2}"].astype("boolean")

both_true_mask = (labels1 == True) & (labels2 == True)  # noqa: E712
both_false_mask = (labels1 == False) & (labels2 == False)  # noqa: E712
only_1_true_mask = (labels1 == True) & (labels2 == False)  # noqa: E712
only_2_true_mask = (labels1 == False) & (labels2 == True)  # noqa: E712

combined["agreement_category"] = ""
combined.loc[both_true_mask, "agreement_category"] = "both_true"
combined.loc[both_false_mask, "agreement_category"] = "both_false"
combined.loc[only_1_true_mask, "agreement_category"] = f"only_{annotator1}_true"
combined.loc[only_2_true_mask, "agreement_category"] = f"only_{annotator2}_true"
combined.loc[records_only_1, "agreement_category"] = f"only_{annotator1}_validated"
combined.loc[records_only_2, "agreement_category"] = f"only_{annotator2}_validated"

# Add resolved "validated" column
# - If only one annotator validated: use their result
# - If both agreed: use their agreed result
# - If both disagreed: set to "RESOLVE"
# - If neither validated: leave blank
combined["validated"] = ""
combined.loc[both_true_mask, "validated"] = "True"
combined.loc[both_false_mask, "validated"] = "False"
combined.loc[only_1_true_mask | only_2_true_mask, "validated"] = "RESOLVE"
combined.loc[records_only_1, "validated"] = combined.loc[
    records_only_1, f"validated_{annotator1}"
].astype(str)
combined.loc[records_only_2, "validated"] = combined.loc[
    records_only_2, f"validated_{annotator2}"
].astype(str)

print("\n=== Combined Validation Results ===")
print(f"Total records: {total_records}")
print(f"Records validated by both annotators: {records_with_both.sum()}")
print(f"Records validated only by {annotator1}: {records_only_1.sum()}")
print(f"Records validated only by {annotator2}: {records_only_2.sum()}")

# Confusion matrix for records validated by both
print("\n=== Confusion Matrix (records validated by both) ===")
print(f"  Both validated=True:      {both_true_mask.sum()}")
print(f"  Both validated=False:     {both_false_mask.sum()}")
print(f"  Only {annotator1} True:   {only_1_true_mask.sum()}")
print(f"  Only {annotator2} True:   {only_2_true_mask.sum()}")

# Paper-level statistics
print("\n=== Paper-Level Statistics ===")
print(f"Total unique papers: {len(papers1 | papers2)}")
print(f"Papers validated by both: {len(overlapping_papers)}")
print(f"Papers only by {annotator1}: {len(only_in_1)}")
print(f"Papers only by {annotator2}: {len(only_in_2)}")

# Save to CSV
args.data_dir.mkdir(parents=True, exist_ok=True)
output_csv = args.data_dir / "combined_validation_results.csv"
combined.to_csv(output_csv, index=False)
logger.info(f"Saved {len(combined)} rows to {output_csv}")
