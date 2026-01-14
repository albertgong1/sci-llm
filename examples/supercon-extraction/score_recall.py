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

import pandas as pd

# pbench imports
import pbench
from pbench_eval.metrics import compute_recall_per_material_property
from pbench_eval.utils import scorer_pymatgen, score_value
import logging

logger = logging.getLogger(__name__)

# Parse arguments
parser = ArgumentParser(
    description="Calculate recall scores for ground truth material properties"
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

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
    logger.info(f"Loading {csv_file.name}")
    df = pd.read_csv(csv_file)
    dfs.append(df)

df_matches = pd.concat(dfs, ignore_index=True)
logger.info(f"Loaded {len(df_matches)} total rows")

# Filter for rows where is_match is True
original_count = len(df_matches)
df_matches = df_matches[df_matches["is_match"] == True]  # noqa: E712
logger.info(
    f"Filtered to {len(df_matches)} rows where is_match=True (from {original_count} total)"
)

# Load rubric
rubric_path = Path("scoring") / "rubric.csv"
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

# Check for missing rubrics
missing_rubric = df["rubric"].isna().sum()
if missing_rubric > 0:
    logger.warning(f"{missing_rubric} rows have no matching rubric")

if False:
    # Group by ground truth material AND property name to calculate recall scores
    grouped = df.groupby(["material_or_system_gt", "property_name_gt"])
    logger.info(f"Processing {len(grouped)} unique (material, property) pairs...")

    results = []

    for (material_gt, property_gt), group in grouped:
        # if material_gt != "Hf1Rh0.7Pd0.3Si1":
        #     continue
        # import pdb; pdb.set_trace()
        # Check which rows have matching materials using scorer_pymatgen
        matching_rows = []

        for idx, row in group.iterrows():
            # Check if materials match using pymatgen
            if pd.notna(material_gt) and pd.notna(row["material_or_system_pred"]):
                if scorer_pymatgen(
                    str(material_gt), str(row["material_or_system_pred"])
                ):
                    matching_rows.append(row)

        num_matches = len(matching_rows)

        if num_matches == 0:
            # No matches, score is 0
            recall_score = 0.0
        else:
            # At least one match, calculate scores and take max
            scores = []

            for row in matching_rows:
                # Skip if values are missing
                if (
                    pd.isna(row["value_string_pred"])
                    or pd.isna(row["value_string_gt"])
                    or pd.isna(row["rubric"])
                ):
                    continue

                # Calculate score
                score = score_value(
                    pred_value=str(row["value_string_pred"]),
                    answer_value=str(row["value_string_gt"]),
                    rubric=str(row["rubric"]),
                    mapping=None,
                )
                scores.append(score)

            # Take maximum score
            recall_score = max(scores) if scores else 0.0

        results.append(
            {
                "material_or_system_gt": material_gt,
                "property_name_gt": property_gt,
                "value_string_gt": ", ".join(
                    list(
                        set(
                            [str(row["value_string_gt"]) for _, row in group.iterrows()]
                        )
                    )
                ),
                "num_property_matches": len(group),
                "num_property_material_matches": num_matches,
                "material_or_system_pred": ", ".join(
                    list(
                        set(
                            [
                                str(row["material_or_system_pred"])
                                for _, row in group.iterrows()
                            ]
                        )
                    )
                ),
                "recall_score": recall_score,
                "matches": ", ".join(
                    [
                        f"{row['property_name_pred']}: {row['value_string_pred']}"
                        for row in matching_rows
                    ]
                ),
            }
        )

    df_results = pd.DataFrame(results)
else:
    df_results = compute_recall_per_material_property(df)

# Print results
# logger.info("=" * 60)
# logger.info("RECALL SCORES")
# logger.info("=" * 60)

# for _, row in df_results.iterrows():
#     print(f"\nMaterial: {row['material_or_system_gt']}")
#     print(f"  Property: {row['property_name_gt']}")
#     print(f"  Total rows: {row['num_total_rows']}")
#     print(f"  Matches: {row['num_matches']}")
#     print(f"  Recall score: {row['recall_score']:.3f}")

# Summary statistics
logger.info("=" * 60)
logger.info("SUMMARY")
logger.info("=" * 60)
print(f"Total (material, property) pairs: {len(df_results)}")
print(f"Pairs with matches: {(df_results['num_property_material_matches'] > 0).sum()}")
print(f"Average recall score: {df_results['recall_score'].mean():.3f}")
print(f"Median recall score: {df_results['recall_score'].median():.3f}")

# save results to csv
score_dir = args.output_dir / "scores"
score_dir.mkdir(parents=True, exist_ok=True)
df_results.to_csv(score_dir / "score_recall.csv", index=False)
