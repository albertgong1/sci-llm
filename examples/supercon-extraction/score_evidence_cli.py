"""CLI to calculate evidence F1 scores by comparing all pred-gt pairs per refno.

This script computes evidence metrics by comparing every predicted evidence string
against every ground truth evidence string within a refno, then taking the max.

Example usage:
```bash
uv run pbench-score-evidence \
    --output_dir ./out \
    --hf_repo kilian-group/supercon-extraction \
    --hf_split full
```
"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
from datasets import load_dataset
from slugify import slugify
from tabulate import tabulate

import pbench
from pbench_eval.harbor_utils import get_harbor_data
from pbench_eval.stats import mean_sem_with_n
from pbench_eval.token_utils import (
    count_trials_per_group,
    count_zeroshot_trials_per_group,
)
from pbench_eval.utils import compute_pairwise_evidence_scores

logger = logging.getLogger(__name__)


def compute_evidence_f1_for_refno(
    df_pred_refno: pd.DataFrame,
    df_gt_refno: pd.DataFrame,
    evidence_column: str = "location.evidence",
) -> dict[str, float]:
    """Compute evidence precision, recall, and F1 for a single refno.

    Args:
        df_pred_refno: Predictions for this refno
        df_gt_refno: Ground truth for this refno
        evidence_column: Column name containing evidence strings

    Returns:
        Dict with evidence_precision, evidence_recall, evidence_f1

    """
    # Extract evidence lists
    evidence_pred = df_pred_refno[evidence_column].fillna("").astype(str).tolist()
    evidence_gt = df_gt_refno[evidence_column].fillna("").astype(str).tolist()

    # Handle empty cases
    if not evidence_pred or not evidence_gt:
        return {
            "evidence_precision": 0.0,
            "evidence_recall": 0.0,
            "evidence_f1": 0.0,
        }

    # Compute pairwise scores: shape (n_pred, n_gt)
    scores_matrix = compute_pairwise_evidence_scores(evidence_pred, evidence_gt)

    # Precision: for each pred row, max over gt rows, then mean
    precision_scores = [max(row) for row in scores_matrix]
    evidence_precision = sum(precision_scores) / len(precision_scores)

    # Recall: for each gt row, max over pred rows, then mean
    # Transpose the matrix to get (n_gt, n_pred)
    n_pred, n_gt = len(scores_matrix), len(scores_matrix[0])
    recall_scores = []
    for j in range(n_gt):
        max_score = max(scores_matrix[i][j] for i in range(n_pred))
        recall_scores.append(max_score)
    evidence_recall = sum(recall_scores) / len(recall_scores)

    # F1
    evidence_f1 = (
        2
        * evidence_precision
        * evidence_recall
        / (evidence_precision + evidence_recall + 1e-8)
    )

    return {
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_f1": evidence_f1,
    }


def compute_evidence_recall_for_refno_by_page(
    df_pred_refno: pd.DataFrame,
    df_gt_refno: pd.DataFrame,
    evidence_column: str = "location.evidence",
    page_column: str = "location.page",
) -> dict[int, float]:
    """Compute evidence recall by page for a single refno.

    For each page in the GT, computes recall as:
    - For each GT evidence on that page, find max similarity to any prediction
    - Average these max scores to get page-level recall

    Args:
        df_pred_refno: Predictions for this refno
        df_gt_refno: Ground truth for this refno
        evidence_column: Column name containing evidence strings
        page_column: Column name containing page numbers

    Returns:
        Dict mapping page number -> evidence recall for that page

    """
    # Extract all prediction evidence (not filtered by page)
    evidence_pred = df_pred_refno[evidence_column].fillna("").astype(str).tolist()

    # Handle empty predictions
    if not evidence_pred:
        return {}

    # Get unique pages from GT (drop NaN and convert to int)
    gt_pages = df_gt_refno[page_column].dropna()
    if gt_pages.empty:
        return {}

    unique_pages = sorted(gt_pages.astype(int).unique())

    page_recalls: dict[int, float] = {}
    for page in unique_pages:
        # Filter GT to this page
        df_gt_page = df_gt_refno[df_gt_refno[page_column].astype(float) == float(page)]
        evidence_gt_page = df_gt_page[evidence_column].fillna("").astype(str).tolist()

        if not evidence_gt_page:
            continue

        # Compute pairwise scores: shape (n_pred, n_gt_page)
        scores_matrix = compute_pairwise_evidence_scores(
            evidence_pred, evidence_gt_page
        )

        # Recall: for each GT evidence on this page, max over all predictions
        n_pred = len(scores_matrix)
        n_gt_page = len(scores_matrix[0]) if scores_matrix else 0

        if n_gt_page == 0:
            continue

        recall_scores = []
        for j in range(n_gt_page):
            max_score = max(scores_matrix[i][j] for i in range(n_pred))
            recall_scores.append(max_score)

        page_recalls[page] = sum(recall_scores) / len(recall_scores)

    return page_recalls


def compute_evidence_recall_by_page(
    df_pred: pd.DataFrame,
    df_gt: pd.DataFrame,
    evidence_column: str = "location.evidence",
    page_column: str = "location.page",
) -> dict[tuple[str, str, int], list[float]]:
    """Compute evidence recall by page for all (agent, model) groups.

    Iterates over (agent, model) pairs from predictions crossed with refnos from
    ground truth to ensure missing predictions count as 0 recall.

    Args:
        df_pred: All predictions with columns [agent, model, refno, location.evidence]
        df_gt: Ground truth with columns [refno, location.evidence, location.page]
        evidence_column: Column name containing evidence strings
        page_column: Column name containing page numbers

    Returns:
        Dict mapping (agent, model, page) to list of recall values across refnos

    """
    from collections import defaultdict

    # Collect recall values: {(agent, model, page): [recall values across refnos]}
    recall_by_page: dict[tuple[str, str, int], list[float]] = defaultdict(list)

    # Get unique (agent, model) pairs from predictions
    agent_model_pairs = df_pred[["agent", "model"]].drop_duplicates().values.tolist()

    # Get unique refnos from ground truth
    gt_refnos = df_gt["refno"].unique()

    # Pre-compute slugified GT refnos for efficient matching
    df_gt = df_gt.copy()
    df_gt["_refno_slug"] = df_gt["refno"].str.lower().apply(slugify)

    for agent, model in agent_model_pairs:
        df_pred_am = df_pred[(df_pred["agent"] == agent) & (df_pred["model"] == model)]

        for gt_refno in gt_refnos:
            gt_refno_slug = slugify(str(gt_refno).lower())

            # Filter ground truth for this refno
            df_gt_refno = df_gt[df_gt["_refno_slug"] == gt_refno_slug]

            if df_gt_refno.empty:
                continue

            # Find matching predictions for this refno
            df_pred_refno = df_pred_am[
                df_pred_am["refno"].str.lower().apply(lambda x: slugify(x))
                == gt_refno_slug
            ]

            logger.debug(f"Processing agent={agent}, model={model}, refno={gt_refno}")
            # import pdb; pdb.set_trace()

            # Get unique pages from GT
            gt_pages = df_gt_refno[page_column].dropna().astype(int).unique()

            # If no predictions, recall is 0 for all GT pages
            if df_pred_refno.empty:
                logger.debug(f"No predictions for refno={gt_refno}, setting recall=0")
                for page in gt_pages:
                    recall_by_page[(agent, model, page)].append(0.0)
                continue

            # Compute per-page recall for this refno
            logger.debug(f"GT pages for refno={gt_refno}: {gt_pages}")
            page_recalls = compute_evidence_recall_for_refno_by_page(
                df_pred_refno,
                df_gt_refno,
                evidence_column=evidence_column,
                page_column=page_column,
            )

            for page in gt_pages:
                recall = page_recalls.get(page, 0.0)
                recall_by_page[(agent, model, page)].append(recall)

    return recall_by_page


def compute_evidence_f1_by_refno(
    df_pred: pd.DataFrame,
    df_gt: pd.DataFrame,
    evidence_column: str = "location.evidence",
) -> pd.DataFrame:
    """Compute evidence F1 scores for all (agent, model, refno) groups.

    Args:
        df_pred: All predictions with columns [agent, model, refno, location.evidence]
        df_gt: Ground truth with columns [refno, location.evidence]
        evidence_column: Column name containing evidence strings

    Returns:
        DataFrame with columns: agent, model, refno, evidence_precision,
        evidence_recall, evidence_f1

    """
    results = []

    for (agent, model, refno), group in df_pred.groupby(
        ["agent", "model", "refno"], dropna=False
    ):
        # Filter ground truth for this refno
        # NOTE: For Harbor evaluation, the refno for predictions is inferred from the trial dirname,
        # which is slugified. The refno in the GT is not slugified, so we need to slugify it for matching.
        df_gt_refno = df_gt[
            df_gt["refno"].str.lower().apply(lambda x: slugify(x))
            == slugify(refno.lower())
        ]

        if df_gt_refno.empty:
            logger.warning(f"No ground truth found for refno={refno}")
            continue

        # Compute evidence metrics
        metrics = compute_evidence_f1_for_refno(
            group, df_gt_refno, evidence_column=evidence_column
        )

        results.append(
            {
                "agent": agent,
                "model": model,
                "refno": refno,
                "num_pred": len(group),
                "num_gt": len(df_gt_refno),
                **metrics,
            }
        )

    return pd.DataFrame(results)


def cli_main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate evidence F1 scores by comparing all pred-gt pairs."
    )
    parser = pbench.add_base_args(parser)

    # Optional arguments
    parser.add_argument(
        "--evidence_column",
        type=str,
        default="location.evidence",
        help="Column name for evidence (default: location.evidence)",
    )
    parser.add_argument(
        "--jobs_dir_list",
        type=Path,
        nargs="*",
        default=[],
        help="List of jobs directories containing Harbor outputs. Each will be treated as a separate run.",
    )
    parser.add_argument(
        "--output_dir_list",
        type=Path,
        nargs="*",
        default=[],
        help="List of output directories containing predictions CSV files. Each will be treated as a separate run.",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    # Load predictions
    if len(args.jobs_dir_list) > 0:  # args.jobs_dir is not None:
        dfs = []
        for jobs_dir in args.jobs_dir_list:
            logger.info(f"Loading predictions from Harbor jobs: {jobs_dir}")
            df = get_harbor_data(jobs_dir)
            # Add run_id column and set it to output_dir name
            run_id = jobs_dir.name
            df["run_id"] = run_id
            dfs.append(df)
        df_pred = pd.concat(dfs, ignore_index=True)
    else:
        dfs = []
        for output_dir in args.output_dir_list:
            if False:
                pred_properties_dir = args.output_dir / args.preds_dirname
            else:
                pred_properties_dir = output_dir / args.preds_dirname
            if not pred_properties_dir.exists():
                logger.error(f"Directory not found: {pred_properties_dir}")
                sys.exit(1)

            csv_files = list(pred_properties_dir.glob("*.csv"))
            if not csv_files:
                logger.error(f"No CSV files found in {pred_properties_dir}")
                sys.exit(1)

            logger.info(f"Found {len(csv_files)} CSV file(s) in {pred_properties_dir}")
            for csv_file in csv_files:
                df = pd.read_csv(csv_file, dtype={"refno": str})
                # Add run_id column and set it to output_dir name
                run_id = output_dir.name
                df["run_id"] = run_id
                dfs.append(df)
        df_pred = pd.concat(dfs, ignore_index=True)

    logger.info(f"Loaded {len(df_pred)} prediction rows")

    # Load ground truth from HuggingFace
    hf_dataset_name = args.hf_repo
    hf_split_name = args.hf_split
    hf_revision = args.hf_revision or "main"

    if hf_dataset_name is None or hf_split_name is None:
        logger.error("--hf_repo and --hf_split are required")
        sys.exit(1)

    logger.info(
        f"Loading dataset from HuggingFace: {hf_dataset_name} "
        f"(revision={hf_revision}, split={hf_split_name})"
    )
    dataset = load_dataset(hf_dataset_name, split=hf_split_name, revision=hf_revision)
    df_gt: pd.DataFrame = dataset.to_pandas()

    # Explode properties to get one row per property
    df_gt = df_gt.explode(column="properties").reset_index(drop=True)
    df_gt = pd.concat(
        [df_gt[["refno"]], pd.json_normalize(df_gt["properties"])], axis=1
    )
    logger.info(f"Loaded {len(df_gt)} ground truth rows")

    # Check if evidence column exists
    evidence_column = args.evidence_column
    if evidence_column not in df_pred.columns:
        logger.error(
            f"Evidence column '{evidence_column}' not found in predictions. "
            f"Available columns: {list(df_pred.columns)}"
        )
        sys.exit(1)

    if evidence_column not in df_gt.columns:
        logger.error(
            f"Evidence column '{evidence_column}' not found in ground truth. "
            f"Available columns: {list(df_gt.columns)}"
        )
        sys.exit(1)

    # Count trials
    if len(args.jobs_dir_list) > 0:
        trials_lookup: dict[tuple, int] = {}
        for jobs_dir in args.jobs_dir_list:
            dir_trials = count_trials_per_group(jobs_dir)
            for key, count in dir_trials.items():
                trials_lookup[key] = trials_lookup.get(key, 0) + count
    else:
        # Aggregate trials across all output directories
        trials_lookup: dict[tuple, int] = {}
        for output_dir in args.output_dir_list:
            dir_trials = count_zeroshot_trials_per_group(output_dir.resolve())
            for key, count in dir_trials.items():
                trials_lookup[key] = trials_lookup.get(key, 0) + count

    # Compute evidence F1 by refno, grouped by run_id
    evidence_dfs = []
    for run_id, df_run in df_pred.groupby("run_id"):
        run_evidence = compute_evidence_f1_by_refno(
            df_run, df_gt, evidence_column=evidence_column
        )
        run_evidence["run_id"] = run_id
        evidence_dfs.append(run_evidence)
    evidence_by_refno = pd.concat(evidence_dfs, ignore_index=True)

    if evidence_by_refno.empty:
        logger.error(
            "No results computed. Check that refnos match between pred and gt."
        )
        sys.exit(1)

    # Add trial counts
    evidence_by_refno["num_trials"] = evidence_by_refno.apply(
        lambda row: trials_lookup.get((row["agent"], row["model"]), 1), axis=1
    )

    # Aggregate by agent/model
    group_cols = ["agent", "model"]
    aggregated = (
        evidence_by_refno.groupby(group_cols)
        .apply(
            lambda g: pd.Series(
                {
                    "avg_evidence_precision": mean_sem_with_n(
                        g["evidence_precision"].tolist(), g["num_trials"].iloc[0]
                    ),
                    "avg_evidence_recall": mean_sem_with_n(
                        g["evidence_recall"].tolist(), g["num_trials"].iloc[0]
                    ),
                    "avg_evidence_f1": mean_sem_with_n(
                        g["evidence_f1"].tolist(), g["num_trials"].iloc[0]
                    ),
                    "successful_count": len(g),
                    "num_trials": g["num_trials"].iloc[0],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # Print results
    print(tabulate(aggregated, headers="keys", tablefmt="github", showindex=False))

    # Save detailed results per refno
    output_path = args.output_dir / "evidence_f1_by_refno.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_by_refno.to_csv(output_path, index=False)
    logger.info(f"Saved detailed results to {output_path}")

    # Save summary table to tables/ directory for plotting
    # Create separate columns for mean and SEM (required by plotting scripts)
    from pbench_eval.stats import padded_mean, padded_sem

    summary_rows = []
    for _, row in aggregated.iterrows():
        group_data = evidence_by_refno[
            (evidence_by_refno["agent"] == row["agent"])
            & (evidence_by_refno["model"] == row["model"])
        ]
        n_trials = int(row["num_trials"])
        summary_rows.append(
            {
                "agent": row["agent"],
                "model": row["model"],
                "avg_evidence_precision": padded_mean(
                    group_data["evidence_precision"].tolist(), n_trials
                ),
                "avg_evidence_precision_sem": padded_sem(
                    group_data["evidence_precision"].tolist(), n_trials
                ),
                "avg_evidence_recall": padded_mean(
                    group_data["evidence_recall"].tolist(), n_trials
                ),
                "avg_evidence_recall_sem": padded_sem(
                    group_data["evidence_recall"].tolist(), n_trials
                ),
                "avg_evidence_f1": padded_mean(
                    group_data["evidence_f1"].tolist(), n_trials
                ),
                "avg_evidence_f1_sem": padded_sem(
                    group_data["evidence_f1"].tolist(), n_trials
                ),
                "successful_count": int(row["successful_count"]),
                "num_trials": n_trials,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / "evidence_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary table to {summary_path}")

    # Save location.page distribution for GT dataset
    page_column = "location.page"
    if page_column in df_gt.columns:
        page_values = df_gt[page_column].dropna()
        page_distribution = page_values.value_counts().sort_index().reset_index()
        page_distribution.columns = ["page", "count"]
        page_dist_path = tables_dir / "gt_page_distribution.csv"
        page_distribution.to_csv(page_dist_path, index=False)
        logger.info(f"Saved GT page distribution to {page_dist_path}")

        # Compute and save evidence recall by page, grouped by run_id
        if page_column in df_pred.columns:
            from collections import defaultdict

            from pbench_eval.stats import padded_mean, padded_sem

            # Aggregate recall values across runs
            combined_recall: dict[tuple[str, str, int], list[float]] = defaultdict(list)
            for run_id, df_run in df_pred.groupby("run_id"):
                run_recall = compute_evidence_recall_by_page(
                    df_run,
                    df_gt,
                    evidence_column=evidence_column,
                    page_column=page_column,
                )
                for key, values in run_recall.items():
                    combined_recall[key].extend(values)

            # Aggregate across refnos and runs
            results = []
            for (agent, model, page), recall_values in combined_recall.items():
                results.append(
                    {
                        "agent": agent,
                        "model": model,
                        "page": page,
                        "avg_evidence_recall": padded_mean(
                            recall_values, len(recall_values)
                        ),
                        "avg_evidence_recall_sem": padded_sem(
                            recall_values, len(recall_values)
                        ),
                        "count": len(recall_values),
                    }
                )
            recall_by_page = pd.DataFrame(results)
            if not recall_by_page.empty:
                # Sort by agent, model, page for readability
                recall_by_page = recall_by_page.sort_values(
                    ["agent", "model", "page"]
                ).reset_index(drop=True)
                recall_by_page_path = tables_dir / "evidence_recall_by_page.csv"
                recall_by_page.to_csv(recall_by_page_path, index=False)
                logger.info(f"Saved evidence recall by page to {recall_by_page_path}")
            else:
                logger.warning("No evidence recall by page computed")
        else:
            logger.warning(
                f"'{page_column}' column not found in predictions, "
                "skipping evidence recall by page"
            )
    else:
        logger.warning(f"'{page_column}' column not found in ground truth data")


if __name__ == "__main__":
    cli_main()
