"""Aggregate F1 scores with token usage or cost for property extraction tasks.

Todo:
- Overlay no agent harness results
- Add reasoning_effort support

Usage:
    uv run pbench-aggregate-accuracy-tokens --output-dir <OUTPUT_DIR> --x-axis <tokens|cost>

"""

from argparse import ArgumentParser, Namespace

import pandas as pd
from tabulate import tabulate

import pbench
from pbench_eval.cli_utils import add_scoring_args
from pbench_eval.score_f1_cli import compute_f1_by_refno
from pbench_eval.stats import padded_mean, padded_sem
from pbench_eval.token_utils import (
    collect_harbor_token_usage,
    collect_zeroshot_token_usage,
    count_trials_per_group,
    count_zeroshot_trials_per_group,
)


def aggregate_accuracy_tokens(args: Namespace) -> pd.DataFrame:
    """Aggregate F1 scores with token usage or cost metrics.

    Args:
        args: Parsed command-line arguments

    Returns:
        DataFrame with merged F1 and token/cost metrics

    """
    # Collect token usage with reasoning_effort
    if args.jobs_dir:
        records = collect_harbor_token_usage(
            args.jobs_dir.resolve(),
        )
        trials_lookup = count_trials_per_group(
            args.jobs_dir,
        )
    else:
        records = collect_zeroshot_token_usage(
            args.output_dir.resolve(),
        )
        trials_lookup = count_zeroshot_trials_per_group(
            args.output_dir.resolve(),
        )

    # Get F1 scores by refno
    f1_by_refno = compute_f1_by_refno(args)
    f1_by_refno["num_trials"] = f1_by_refno.apply(
        lambda row: trials_lookup.get((row["agent"], row["model"]), 0), axis=1
    )

    f1_mean = (
        f1_by_refno.groupby(["agent", "model"])
        .apply(
            lambda g: pd.Series(
                {
                    "avg_f1_score": padded_mean(
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
    f1_sem = (
        f1_by_refno.groupby(["agent", "model"])
        .apply(
            lambda g: pd.Series(
                {
                    "avg_f1_score": padded_sem(
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

    df_usage = pd.DataFrame(records)
    df_usage["total_tokens"] = (
        df_usage["total_prompt_tokens"] + df_usage["total_completion_tokens"]
    )

    # Determine which x-axis metric to use
    x_metric_col = "total_tokens" if args.x_axis == "tokens" else "total_cost_usd"
    x_metric_label = "avg_x_metric"

    x_metric_mean = (
        df_usage.groupby(["agent", "model_name"])
        .apply(
            lambda g: pd.Series(
                {
                    x_metric_label: padded_mean(
                        g[x_metric_col].tolist(),
                        trials_lookup.get((g.name[0], g.name[1]), len(g)),
                    ),
                    "successful_count": len(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    x_metric_sem = (
        df_usage.groupby(["agent", "model_name"])
        .apply(
            lambda g: pd.Series(
                {
                    x_metric_label: padded_sem(
                        g[x_metric_col].tolist(),
                        trials_lookup.get((g.name[0], g.name[1]), len(g)),
                    ),
                    "successful_count": len(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # Merge F1 and x-axis metric
    merged = f1_mean.merge(
        x_metric_mean, left_on=["agent", "model"], right_on=["agent", "model_name"]
    )
    merged = merged.merge(
        f1_sem[["agent", "model", "avg_f1_score"]],
        on=["agent", "model"],
        suffixes=("", "_sem"),
    )
    merged = merged.merge(
        x_metric_sem[["agent", "model_name", x_metric_label]],
        on=["agent", "model_name"],
        suffixes=("", "_sem"),
    )

    return merged


def cli_main() -> None:
    """CLI entry point for aggregating accuracy with token/cost metrics."""
    parser = ArgumentParser(
        description="Aggregate F1 scores with token usage or cost metrics."
    )
    parser = pbench.add_base_args(parser)
    parser = add_scoring_args(parser)
    parser.add_argument(
        "--x-axis",
        type=str,
        choices=["tokens", "cost"],
        default="tokens",
        help="X-axis metric: 'tokens' for total tokens or 'cost' for total cost in USD",
    )
    args = parser.parse_args()

    pbench.setup_logging(args.log_level)

    merged = aggregate_accuracy_tokens(args)

    # Save merged table for reference
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / f"f1_vs_{args.x_axis}_summary.csv"
    merged.to_csv(table_path, index=False)
    print(f"Saved summary table to {table_path}")
    print(tabulate(merged, headers="keys", tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    cli_main()
