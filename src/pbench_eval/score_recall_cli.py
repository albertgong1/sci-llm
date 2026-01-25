"""Domain-agnostic CLI to calculate recall scores for ground truth material properties.

This script reads ground truth property matches, validates materials/conditions,
and scores property values against predictions.

Example usage (supercon - material-based):
```bash
uv run pbench-score-recall \
    --output_dir ./out \
    --rubric_path scoring/rubric_4.csv \
    --conversion_factors_path scoring/si_conversion_factors.csv \
    --matching_mode material \
    --model_name gemini-3-pro-preview
```

Example usage (biosurfactants - condition-based):
```bash
uv run pbench-score-recall \
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
from pbench_eval.metrics import compute_recall_per_material_property
from pbench_eval.harbor_utils import count_trials_per_agent_model
from pbench_eval.stats import mean_sem_with_n

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


def cli_main() -> None:
    """CLI entry point."""
    parser = ArgumentParser(
        description="Calculate recall scores for ground truth material properties"
    )
    parser = pbench.add_base_args(parser)

    # Required arguments
    parser.add_argument(
        "--rubric_path",
        type=Path,
        required=True,
        help="Path to rubric CSV file",
    )

    # Optional arguments
    parser.add_argument(
        "--conversion_factors_path",
        type=Path,
        default=None,
        help="Path to SI conversion factors CSV file (optional)",
    )
    parser.add_argument(
        "--matching_mode",
        type=str,
        choices=["material", "conditions"],
        default="material",
        help="Matching mode: 'material' for supercon-style, 'conditions' for biosurfactants-style (default: material)",
    )
    parser.add_argument(
        "--material_column",
        type=str,
        default="material_or_system",
        help="Column name for material matching (default: material_or_system)",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

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
        df = pd.read_csv(csv_file, dtype=str)
        dfs.append(df)

    df_matches = pd.concat(dfs, ignore_index=True)
    # NOTE: if judge is NaN, it means exact string match was used for matching
    df_matches = df_matches[
        (df_matches["judge"] == model_name) | (df_matches["judge"].isna())
    ]
    logger.info(
        f"Loaded {len(df_matches)} total rows using {model_name} for property matching"
    )

    # If jobs_dir was not provided, count trajectory JSONs in trajectories directory
    if args.jobs_dir is None:
        trajectory_dir = args.output_dir / "trajectories"
        trials_lookup: dict[tuple[str, str], int] = {}
        if trajectory_dir.exists():
            # Count trajectory files per agent/model
            # Pattern: trajectory__agent={agent}__model={model}__refno={refno}.json
            import re

            trajectory_counts: dict[tuple[str, str], int] = {}
            for traj_file in trajectory_dir.glob("trajectory__*.json"):
                # Parse agent and model from filename
                match = re.match(
                    r"trajectory__agent=([^_]+)__model=([^_]+)__refno=.+\.json",
                    traj_file.name,
                )
                if match:
                    agent, model = match.groups()
                    # Convert model name back (-- to /)
                    model = model.replace("--", "/")
                    key = (agent, model)
                    trajectory_counts[key] = trajectory_counts.get(key, 0) + 1
            trials_lookup = trajectory_counts
            logger.info(f"Counted trials from {trajectory_dir}: {trials_lookup}")
        else:
            # Fallback to counting unique refnos from data
            logger.warning(
                f"Trajectories directory not found: {trajectory_dir}. "
                "Falling back to counting unique refnos from data."
            )
            trials_lookup = {
                k: v
                for k, v in df_matches.groupby(["agent", "model"])["refno"]
                .nunique()
                .to_dict()
                .items()
            }
    else:
        # Count number of trials (refnos) per agent/model
        trials_lookup: dict[tuple[str, str], int] = {}
        trials_df = count_trials_per_agent_model(args.jobs_dir)
        trials_lookup = {
            (row["agent"], row["model"]): row["num_trials"]
            for _, row in trials_df.iterrows()
        }

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
    acc_by_refno = (
        df_results.groupby(["agent", "model", "refno"], dropna=False)
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
    # Merge trial counts into acc_by_refno for per-group normalization
    acc_by_refno["num_trials"] = acc_by_refno.apply(
        lambda row: trials_lookup.get((row["agent"], row["model"]), 1), axis=1
    )

    acc = (
        acc_by_refno.groupby(["agent", "model"])
        .apply(
            lambda g: pd.Series(
                {
                    "avg_recall": mean_sem_with_n(
                        g["recall_score"].tolist(), g["num_trials"].iloc[0]
                    ),
                    "avg_property_matches": mean_sem_with_n(
                        g["property_matches"].tolist(), g["num_trials"].iloc[0]
                    ),
                    "avg_property_material_matches": mean_sem_with_n(
                        g["property_material_matches"].tolist(), g["num_trials"].iloc[0]
                    ),
                    "successful_count": len(g),
                    "avg_num_gt": mean_sem_with_n(
                        g["num_gt"].tolist(), g["num_trials"].iloc[0]
                    ),
                    "num_trials": g["num_trials"].iloc[0],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    # Print results as table
    print(tabulate(acc, headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    cli_main()
