# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: sci-llm
#     language: python
#     name: python3
# ---

# %%
"""Format token usage results from Harbor job directories or zeroshot output directories.

Usage:
  uv run python format_tokens.py -jd <job_dir>        # Harbor job directory
  uv run python format_tokens.py -od <output_dir>    # Zeroshot output directory

This script iterates over batches/trials in a job directory or trajectory JSON files in
an output directory, and prints combined token usage statistics.
"""

# %%
from argparse import ArgumentParser

from tabulate import tabulate

import pbench
from pbench_eval.token_utils import (
    collect_harbor_token_usage,
    collect_zeroshot_token_usage,
    count_trials_per_group,
    count_zeroshot_trials_per_group,
    format_token_statistics,
)

# %%
parser = ArgumentParser(
    description="Format token usage results from Harbor job directories or zeroshot output directories."
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()

# Validate arguments: need either jobs_dir or output_dir
if args.jobs_dir and args.output_dir:
    raise SystemExit("Cannot specify both --jobs_dir and --output_dir. Choose one.")
if not args.jobs_dir and not args.output_dir:
    raise SystemExit("Must specify either --jobs_dir or --output_dir.")

# %%
# Collect token usage with reasoning_effort
group_cols = ["agent", "model_name", "reasoning_effort"]

if args.jobs_dir:
    records = collect_harbor_token_usage(
        args.jobs_dir.resolve(),
        include_reasoning_effort=True,
    )
    # Use count_trials_per_group for proper trial counting from directory structure
    trials_lookup = count_trials_per_group(
        args.jobs_dir,
        include_reasoning_effort=True,
    )
else:
    records = collect_zeroshot_token_usage(
        args.output_dir.resolve(),
        include_reasoning_effort=True,
    )
    # Count trials from trajectory files, not from records
    trials_lookup = count_zeroshot_trials_per_group(
        args.output_dir.resolve(),
        include_reasoning_effort=True,
    )

# %%
# Format and print statistics
if not records:
    print("No results found.")
    raise SystemExit(0)

result = format_token_statistics(
    records,
    group_cols=group_cols,
    trials_lookup=trials_lookup,
    scale_factor=1e3,
    scale_suffix="K",
)

print("Mean ± SE Token Usage by Agent and Model (normalized by num_trials)")
print(
    tabulate(result.values.tolist(), headers=result.columns.tolist(), tablefmt="github")
)
