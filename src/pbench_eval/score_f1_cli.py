"""Domain-agnostic CLI to calculate F1 scores for material property extraction.

This script computes F1 scores by combining precision and recall results.

Example usage (supercon - material-based):
```bash
uv run pbench-score-f1 \
    --output_dir ./out \
    --rubric_path scoring/rubric_4.csv \
    --conversion_factors_path scoring/si_conversion_factors.csv \
    --matching_mode material \
    --model_name gemini-3-pro-preview
```

Example usage (biosurfactants - condition-based):
```bash
uv run pbench-score-f1 \
    --output_dir ./out \
    --rubric_path scoring/rubric.csv \
    --matching_mode conditions \
    --model_name gemini-3-pro-preview
```
"""

from argparse import ArgumentParser, Namespace
from tabulate import tabulate
import pandas as pd
import logging

import pbench
from pbench_eval.token_utils import (
    count_trials_per_group,
    count_zeroshot_trials_per_group,
)
from pbench_eval.stats import mean_sem_with_n
from pbench_eval.cli_utils import add_scoring_args
from pbench_eval.score_precision_cli import compute_precision_by_refno
from pbench_eval.score_recall_cli import compute_recall_by_refno

logger = logging.getLogger(__name__)


def compute_f1_by_refno(args: Namespace) -> pd.DataFrame:
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
    # compute evidence F1 score
    # evidence_score_x is from precision, evidence_score_y is from recall
    f1_by_refno["evidence_f1_score"] = 2 * (
        (f1_by_refno["evidence_score_x"] * f1_by_refno["evidence_score_y"])
        / (f1_by_refno["evidence_score_x"] + f1_by_refno["evidence_score_y"] + 1e-8)
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
                    "avg_evidence_f1": mean_sem_with_n(
                        g["evidence_f1_score"].tolist(), g["num_trials"].iloc[0]
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
