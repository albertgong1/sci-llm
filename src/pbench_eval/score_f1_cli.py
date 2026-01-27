"""Domain-agnostic CLI to calculate precision scores for predicted material properties.

This script reads predicted property matches, validates materials/conditions,
and scores property values against ground truth.

Example usage (supercon - material-based):
```bash
uv run pbench-score-precision \
    --output_dir ./out \
    --rubric_path scoring/rubric_4.csv \
    --conversion_factors_path scoring/si_conversion_factors.csv \
    --matching_mode material \
    --model_name gemini-3-pro-preview
```

Example usage (biosurfactants - condition-based):
```bash
uv run pbench-score-precision \
    --output_dir ./out \
    --rubric_path scoring/rubric.csv \
    --matching_mode conditions \
    --model_name gemini-3-pro-preview
```
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from tabulate import tabulate
import pandas as pd
import logging

import pbench
from pbench_eval.metrics import (
    compute_precision_per_material_property,
    compute_recall_per_material_property,
)
from pbench_eval.token_utils import (
    count_trials_per_group,
    count_zeroshot_trials_per_group,
)
from pbench_eval.stats import mean_sem_with_n
from pbench_eval.cli_utils import add_scoring_args

logger = logging.getLogger(__name__)


def load_rubric(rubric_path: Path) -> pd.DataFrame:
    """Load and filter rubric CSV.

    For biosurfactants-style rubrics with condition rows, filter to only property rows
    (where condition_name is empty/NaN).

    Args:
        rubric_path: Path to rubric CSV file

    Returns:
        DataFrame with rubric data (property rows only for biosurfactants-style)

    """
    df_rubric = pd.read_csv(rubric_path)

    # If this is a biosurfactants-style rubric with condition_name column,
    # filter to property rows only (where condition_name is empty)
    if "condition_name" in df_rubric.columns:
        df_property_rubric = df_rubric[
            df_rubric["condition_name"].isna() | (df_rubric["condition_name"] == "")
        ]
        logger.info(
            f"Filtered rubric from {len(df_rubric)} rows to {len(df_property_rubric)} property rows"
        )
        return df_property_rubric

    return df_rubric


def compute_precision_by_refno(args: ArgumentParser) -> pd.DataFrame:
    """Main function to compute precision scores.

    Args:
        args: Parsed command-line arguments

    Returns:
        DataFrame with precision scores per agent/model/reasoning_effort group

    """
    # Model used for property matching
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
        df = pd.read_csv(csv_file, dtype={"refno": str})
        dfs.append(df)

    df_matches = pd.concat(dfs, ignore_index=True)
    # NOTE: if judge is NaN, it means exact string match was used for matching
    df_matches = df_matches[
        (df_matches["judge"] == model_name) | (df_matches["judge"].isna())
    ]
    logger.info(
        f"Loaded {len(df_matches)} total rows using {model_name} for property matching"
    )

    group_cols = ["agent", "model"]

    # Load rubric
    logger.info(f"Loading rubric from {args.rubric_path}")
    df_rubric = load_rubric(args.rubric_path)
    logger.info(f"Loaded {len(df_rubric)} rows from rubric")

    # Join matches with rubric to get scoring method
    logger.info("Joining matches with rubric...")
    df = df_matches.merge(
        df_rubric[["property_name", "rubric"]],
        left_on="property_name_gt",
        right_on="property_name",
        how="left",
    )

    # Load conversion factors if provided
    conversion_df = None
    if args.conversion_factors_path:
        logger.info(f"Loading conversion factors from {args.conversion_factors_path}")
        conversion_df = pd.read_csv(args.conversion_factors_path, index_col=0)

    # Check for missing rubrics
    missing_rubric = df["rubric"].isna().sum()
    if missing_rubric > 0:
        logger.warning(
            f"{missing_rubric} out of {len(df)} rows have no matching rubric"
        )

    # Compute precision scores
    df_results = compute_precision_per_material_property(
        df,
        conversion_df=conversion_df,
        matching_mode=args.matching_mode,
        material_column=args.material_column,
        rubric_df=df_rubric if args.matching_mode == "conditions" else None,
    )

    # Aggregate results
    counta = lambda x: (x > 0).sum()  # noqa: E731
    refno_group_cols = group_cols + ["refno"]
    acc_by_refno = (
        df_results.groupby(refno_group_cols, dropna=False)
        .agg(
            precision_score=pd.NamedAgg(column="precision_score", aggfunc="mean"),
            property_matches=pd.NamedAgg(
                column="num_property_matches", aggfunc="count"
            ),
            property_material_matches=pd.NamedAgg(
                column="num_property_material_matches", aggfunc=counta
            ),
            num_pred=pd.NamedAgg(column="id_pred", aggfunc="size"),
        )
        .reset_index()
    )
    return acc_by_refno


def compute_recall_by_refno(args: ArgumentParser) -> pd.DataFrame:
    """Main function to compute recall scores.

    Args:
        args: Parsed command-line arguments

    Returns:
        DataFrame with recall scores per agent/model/reasoning_effort group

    """
    # Model used for property matching
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
        df = pd.read_csv(csv_file, dtype={"refno": str})
        dfs.append(df)

    df_matches = pd.concat(dfs, ignore_index=True)
    # NOTE: if judge is NaN, it means exact string match was used for matching
    df_matches = df_matches[
        (df_matches["judge"] == model_name) | (df_matches["judge"].isna())
    ]
    logger.info(
        f"Loaded {len(df_matches)} total rows using {model_name} for property matching"
    )

    group_cols = ["agent", "model"]

    # Load rubric
    logger.info(f"Loading rubric from {args.rubric_path}")
    df_rubric = load_rubric(args.rubric_path)
    logger.info(f"Loaded {len(df_rubric)} rows from rubric")

    # Join matches with rubric to get scoring method
    logger.info("Joining matches with rubric...")
    df = df_matches.merge(
        df_rubric[["property_name", "rubric"]],
        left_on="property_name_gt",
        right_on="property_name",
        how="left",
    )

    # Load conversion factors if provided
    conversion_df = None
    if args.conversion_factors_path:
        logger.info(f"Loading conversion factors from {args.conversion_factors_path}")
        conversion_df = pd.read_csv(args.conversion_factors_path, index_col=0)

    # Check for missing rubrics
    missing_rubric = df["rubric"].isna().sum()
    if missing_rubric > 0:
        logger.warning(
            f"{missing_rubric} out of {len(df)} rows have no matching rubric"
        )

    # Compute recall scores
    df_results = compute_recall_per_material_property(
        df,
        conversion_df=conversion_df,
        matching_mode=args.matching_mode,
        material_column=args.material_column,
        rubric_df=df_rubric if args.matching_mode == "conditions" else None,
    )

    # Save results per group
    for (agent, model, refno), group in df_results.groupby(
        ["agent", "model", "refno"], dropna=False
    ):
        scores_dir = args.output_dir / "scores" / agent / model
        scores_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = (
            args.output_dir / "scores" / agent / model / f"recall_results_{refno}.csv"
        )
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Saving recall results for {agent} {model} {refno} to {output_csv_path}"
        )
        group.to_csv(output_csv_path, index=False)

    # Aggregate results
    counta = lambda x: (x > 0).sum()  # noqa: E731
    refno_group_cols = group_cols + ["refno"]
    acc_by_refno = (
        df_results.groupby(refno_group_cols, dropna=False)
        .agg(
            recall_score=pd.NamedAgg(column="recall_score", aggfunc="mean"),
            property_matches=pd.NamedAgg(
                column="num_property_matches", aggfunc="count"
            ),
            property_material_matches=pd.NamedAgg(
                column="num_property_material_matches", aggfunc=counta
            ),
            num_gt=pd.NamedAgg(column="id_gt", aggfunc="size"),
        )
        .reset_index()
    )

    return acc_by_refno


def compute_f1_by_refno(args: ArgumentParser) -> pd.DataFrame:
    """Compute F1 scores by merging precision and recall results.

    Args:
        args: Parsed command-line arguments

    Returns:
        pd.DataFrame: DataFrame containing F1 scores by reference number

    """
    precision_by_refno = compute_precision_by_refno(args)
    recall_by_refno = compute_recall_by_refno(args)
    # Merge precision and recall results
    f1_by_refno = precision_by_refno.merge(
        recall_by_refno,
        on=["agent", "model", "refno"],
    )
    # compute F1 score
    f1_by_refno["f1_score"] = 2 * (
        (f1_by_refno["precision_score"] * f1_by_refno["recall_score"])
        / (f1_by_refno["precision_score"] + f1_by_refno["recall_score"] + 1e-8)
    )
    return f1_by_refno


def cli_main() -> None:
    """CLI main function to compute precision scores."""
    parser = ArgumentParser(
        description="Calculate precision scores for predicted material properties"
    )
    parser = pbench.add_base_args(parser)
    parser = add_scoring_args(parser)

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    # If jobs_dir was not provided, count trajectory JSONs in trajectories directory
    if args.jobs_dir is None:
        trials_lookup = count_zeroshot_trials_per_group(args.output_dir.resolve())
    else:
        trials_lookup = count_trials_per_group(args.jobs_dir)

    if False:
        precision_by_refno = compute_precision_by_refno(args)
        recall_by_refno = compute_recall_by_refno(args)
        # Merge precision and recall results
        f1_by_refno = precision_by_refno.merge(
            recall_by_refno,
            on=["agent", "model", "refno"],
        )
        # compute F1 score
        f1_by_refno["f1_score"] = 2 * (
            (f1_by_refno["precision_score"] * f1_by_refno["recall_score"])
            / (f1_by_refno["precision_score"] + f1_by_refno["recall_score"] + 1e-8)
        )
    else:
        f1_by_refno = compute_f1_by_refno(args)

    f1_by_refno["num_trials"] = f1_by_refno.apply(
        lambda row: trials_lookup.get((row["agent"], row["model"])), axis=1
    )

    f1 = (
        f1_by_refno.groupby(["agent", "model"])
        .apply(
            lambda g: pd.Series(
                {
                    "avg_f1_score": mean_sem_with_n(
                        g["f1_score"].tolist(), g["num_trials"].iloc[0]
                    ),
                    "successful_count": len(g),
                    "num_trials": g["num_trials"].iloc[0],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    print(tabulate(f1, headers="keys", tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    cli_main()
