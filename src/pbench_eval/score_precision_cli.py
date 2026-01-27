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

import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from tabulate import tabulate
import pandas as pd
import logging

import pbench
from pbench_eval.metrics import compute_precision_per_material_property
from pbench_eval.harbor_utils import count_trials_per_agent_model
from pbench_eval.token_utils import (
    count_trials_per_group,
    count_zeroshot_trials_per_group,
)
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
        description="Calculate precision scores for predicted material properties"
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

    # If jobs_dir was not provided, count trajectory JSONs in trajectories directory
    # and extract reasoning_effort if available
    reasoning_effort_lookup: dict[
        tuple[str, str, str], str
    ] = {}  # (agent, model, refno) -> reasoning_effort
    has_reasoning_effort = False
    if args.jobs_dir is None:
        if False:
            trajectory_dir = args.output_dir / "trajectories"
            trials_lookup: dict[tuple, int] = {}
            if trajectory_dir.exists():
                # First pass: check if any trajectory has reasoning_effort and extract values
                # Pattern: trajectory__agent={agent}__model={model}__refno={refno}.json
                for traj_file in trajectory_dir.glob("trajectory__*.json"):
                    match = re.match(
                        r"trajectory__agent=([^_]+)__model=([^_]+)__refno=(.+)\.json",
                        traj_file.name,
                    )
                    if match:
                        agent, model, refno = match.groups()
                        model = model.replace("--", "/")
                        try:
                            with open(traj_file) as f:
                                trajectory = json.load(f)
                            inf_gen_config = trajectory.get("inf_gen_config", {})
                            reasoning_effort = inf_gen_config.get(
                                "reasoning_effort", ""
                            )
                            if reasoning_effort is None:
                                reasoning_effort = ""
                            if reasoning_effort:
                                has_reasoning_effort = True
                            reasoning_effort_lookup[(agent, model, refno)] = (
                                reasoning_effort
                            )
                        except Exception:
                            reasoning_effort_lookup[(agent, model, refno)] = ""

                # Count trials per group
                trajectory_counts: dict[tuple, int] = {}
                for traj_file in trajectory_dir.glob("trajectory__*.json"):
                    match = re.match(
                        r"trajectory__agent=([^_]+)__model=([^_]+)__refno=(.+)\.json",
                        traj_file.name,
                    )
                    if match:
                        agent, model, refno = match.groups()
                        model = model.replace("--", "/")
                        if has_reasoning_effort:
                            reasoning_effort = reasoning_effort_lookup.get(
                                (agent, model, refno), ""
                            )
                            key = (agent, model, reasoning_effort)
                        else:
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
                group_cols_fallback = ["agent", "model"]
                trials_lookup = {
                    k: v
                    for k, v in df_matches.groupby(group_cols_fallback)["refno"]
                    .nunique()
                    .to_dict()
                    .items()
                }
        else:
            trials_lookup = count_zeroshot_trials_per_group(
                args.output_dir.resolve(),
                include_reasoning_effort=True,
            )
            has_reasoning_effort = True
    else:
        # Count number of trials (refnos) per agent/model
        if False:
            trials_lookup: dict[tuple, int] = {}
            trials_df = count_trials_per_agent_model(args.jobs_dir)
            trials_lookup = {
                (row["agent"], row["model"]): row["num_trials"]
                for _, row in trials_df.iterrows()
            }
        else:
            trials_lookup = count_trials_per_group(args.jobs_dir)

    # Determine grouping columns based on whether reasoning_effort exists
    if has_reasoning_effort:
        group_cols = ["agent", "model", "reasoning_effort"]
    else:
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

    # Add reasoning_effort column if available
    if has_reasoning_effort:
        df_results["reasoning_effort"] = df_results.apply(
            lambda row: reasoning_effort_lookup.get(
                (row["agent"], row["model"], row["refno"]), ""
            ),
            axis=1,
        )

    # Save results per group
    for (agent, model, refno), group in df_results.groupby(
        ["agent", "model", "refno"], dropna=False
    ):
        # import pdb; pdb.set_trace()
        scores_dir = args.output_dir / "scores" / agent / model
        scores_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = (
            args.output_dir
            / "scores"
            / agent
            / model
            / f"precision_results_{refno}.csv"
        )
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Saving precision results for {agent} {model} {refno} to {output_csv_path}"
        )
        group.to_csv(output_csv_path, index=False)

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

    # Merge trial counts into acc_by_refno for per-group normalization
    def get_trials_count(row: pd.Series) -> int:
        if has_reasoning_effort:
            key = (row["agent"], row["model"], row["reasoning_effort"])
        else:
            key = (row["agent"], row["model"])
        return trials_lookup.get(key, 1)

    acc_by_refno["num_trials"] = acc_by_refno.apply(get_trials_count, axis=1)

    acc = (
        acc_by_refno.groupby(group_cols)
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
    # Print results as table
    print(tabulate(acc, headers="keys", tablefmt="github", showindex=False))

    # Compute average precision per property_name
    # Rows: agent, model, reasoning_effort (if applicable)
    # Columns: property names
    property_group_cols = group_cols + ["refno", "property_name_pred"]
    precision_by_property = (
        df_results.groupby(property_group_cols, dropna=False)
        .agg(precision_score=pd.NamedAgg(column="precision_score", aggfunc="mean"))
        .reset_index()
    )

    # Add trial counts
    precision_by_property["num_trials"] = precision_by_property.apply(
        get_trials_count, axis=1
    )

    # Aggregate across refnos for each group and property
    property_agg_cols = group_cols + ["property_name_pred"]
    precision_by_property_agg = (
        precision_by_property.groupby(property_agg_cols, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "avg_precision": mean_sem_with_n(
                        g["precision_score"].tolist(), g["num_trials"].iloc[0]
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # Pivot to get property names as columns
    precision_pivot = precision_by_property_agg.pivot(
        index=group_cols, columns="property_name_pred", values="avg_precision"
    ).reset_index()

    # Sort property columns by descending occurrence count
    property_counts = df_results["property_name_pred"].value_counts()
    property_cols = [c for c in precision_pivot.columns if c not in group_cols]
    sorted_property_cols = sorted(
        property_cols, key=lambda x: property_counts.get(x, 0), reverse=True
    )
    precision_pivot = precision_pivot[group_cols + sorted_property_cols]

    # Save precision per property to CSV
    precision_per_property_path = args.output_dir / "precision_per_property.csv"
    precision_pivot.to_csv(precision_per_property_path, index=False)
    print(f"Saved precision per property to {precision_per_property_path}")


if __name__ == "__main__":
    cli_main()
