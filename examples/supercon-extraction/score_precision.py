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
from tabulate import tabulate
import pandas as pd
import logging

# pbench imports
import pbench
from pbench_eval.metrics import compute_precision_per_material_property
from utils import RUBRIC_PATH, count_trials_per_agent_model, mean_sem_with_n

logger = logging.getLogger(__name__)

# Parse arguments
parser = ArgumentParser(
    description="Calculate precision scores for predicted material properties"
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)
# model used for property matching
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
    logger.debug(f"Loading {csv_file.name}")
    df = pd.read_csv(csv_file)
    dfs.append(df)

df_matches = pd.concat(dfs, ignore_index=True)
# NOTE: if judge is NaN, it means exact string match was used for matching
df_matches = df_matches[
    (df_matches["judge"] == model_name) | (df_matches["judge"].isna())
]
if False:
    df_matches = df_matches[
        (df_matches["agent"] == "gemini-cli")
        & (df_matches["model"] == "gemini/gemini-3-pro-preview")
    ]
if False:
    # only include rows where batch starts with 'bn1'
    df_matches = df_matches[df_matches["batch"].str.startswith("bn1")]
logger.info(
    f"Loaded {len(df_matches)} total rows using {model_name} for property matching"
)

# If jobs_dir was not provided, count unique refnos per agent/model from the data
if args.jobs_dir is None:
    trials_lookup = df_matches.groupby(["agent", "model"])["refno"].nunique().to_dict()
else:
    # Count number of trials (refnos) per agent/model
    # trials_lookup will be populated after df_matches is loaded if jobs_dir is not provided
    trials_lookup: dict[tuple[str, str], int] = {}
    trials_df = count_trials_per_agent_model(args.jobs_dir)
    trials_lookup = {
        (row["agent"], row["model"]): row["num_trials"]
        for _, row in trials_df.iterrows()
    }

# Load rubric
logger.info(f"Loading rubric from {RUBRIC_PATH}")
df_rubric = pd.read_csv(RUBRIC_PATH)
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
    logger.warning(f"{missing_rubric} out of {len(df)} rows have no matching rubric")

df_results = compute_precision_per_material_property(df, conversion_df=conversion_df)

for (agent, model, refno), group in df_results.groupby(
    ["agent", "model", "refno"], dropna=False
):
    # save results to csv
    scores_dir = args.output_dir / "scores" / agent / model
    scores_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = (
        args.output_dir / "scores" / agent / model / f"precision_results_{refno}.csv"
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(
        f"Saving precision results for {agent} {model} {refno} to {output_csv_path}"
    )
    # import pdb; pdb.set_trace()
    group.to_csv(output_csv_path, index=False)

# get the number of rows where we found at least one match
counta = lambda x: (x > 0).sum()  # noqa: E731
acc_by_refno = (
    df_results.groupby(["agent", "model", "refno"], dropna=False)
    .agg(
        precision_score=pd.NamedAgg(column="precision_score", aggfunc="mean"),
        property_matches=pd.NamedAgg(column="num_property_matches", aggfunc="count"),
        property_material_matches=pd.NamedAgg(
            column="num_property_material_matches", aggfunc=counta
        ),
        num_pred=pd.NamedAgg(column="id_pred", aggfunc="size"),
    )
    .reset_index()
)
# Merge trial counts into acc_by_refno for per-group normalization
acc_by_refno["num_trials"] = acc_by_refno.apply(
    lambda row: trials_lookup.get((row["agent"], row["model"]), 1), axis=1
)

acc = (
    acc_by_refno.groupby(["agent", "model"])
    .apply(
        lambda g: pd.Series(
            {
                "avg_precision": mean_sem_with_n(
                    g["precision_score"].tolist(), g["num_trials"].iloc[0]
                ),
                "avg_property_matches": mean_sem_with_n(
                    g["property_matches"].tolist(), g["num_trials"].iloc[0]
                ),
                "avg_property_material_matches": mean_sem_with_n(
                    g["property_material_matches"].tolist(), g["num_trials"].iloc[0]
                ),
                "successful_count": len(g),
                "avg_num_pred": mean_sem_with_n(
                    g["num_pred"].tolist(), g["num_trials"].iloc[0]
                ),
                "num_trials": g["num_trials"].iloc[0],
            }
        ),
        include_groups=False,
    )
    .reset_index()
)
# print acc as table using the tabulate library with 'github' format
print(tabulate(acc, headers="keys", tablefmt="github", showindex=False))
