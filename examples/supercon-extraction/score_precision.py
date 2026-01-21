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
from utils import RUBRIC_PATH, sem

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

NUM_REFNOS = 50

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
df_matches = df_matches[df_matches["judge"] == model_name]
if False:
    df_matches = df_matches[
        (df_matches["agent"] == "gemini-cli")
        & (df_matches["model"] == "gemini/gemini-3-pro-preview")
    ]
logger.info(
    f"Loaded {len(df_matches)} total rows using {model_name} for property matching"
)

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
    logger.warning(f"{missing_rubric} rows have no matching rubric")

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

counta = lambda x: (x > 0).sum()  # noqa: E731
acc_by_refno = (
    df_results.groupby(["agent", "model", "refno"], dropna=False)
    .agg(
        precision_score=pd.NamedAgg(column="precision_score", aggfunc="mean"),
        property_matches=pd.NamedAgg(column="num_property_matches", aggfunc="count"),
        property_material_matches=pd.NamedAgg(
            column="num_property_material_matches", aggfunc=counta
        ),
    )
    .reset_index()
)

mean_sem = lambda x: f"{x.sum() / NUM_REFNOS:.2f} ± {sem(x, NUM_REFNOS):.2f}"  # noqa: E731
acc = (
    acc_by_refno.groupby(["agent", "model"])
    .agg(
        avg_precision=pd.NamedAgg(column="precision_score", aggfunc=mean_sem),
        avg_property_matches=pd.NamedAgg(column="property_matches", aggfunc=mean_sem),
        avg_property_material_matches=pd.NamedAgg(
            column="property_material_matches", aggfunc=mean_sem
        ),
        count=pd.NamedAgg(column="model", aggfunc="count"),
    )
    .reset_index()
)
# print acc as table using the tabulate library with 'github' format
print(tabulate(acc, headers="keys", tablefmt="github", showindex=False))
