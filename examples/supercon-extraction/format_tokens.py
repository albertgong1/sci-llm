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
"""Format token usage results from Harbor job directories.

Usage:
  uv run python format_tokens.py -jd <job_dir>

This script iterates over batches in a job directory, reads the agent/trajectory.json
from each trial, and prints combined token usage statistics.
"""

# %%
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_utils import calculate_cost

# %%
parser = ArgumentParser(
    description="Format token usage results from Harbor job directories."
)
parser.add_argument(
    "--jobs-dir",
    "-jd",
    type=Path,
    help="Job directory containing batch subdirectories with trials.",
)
args = parser.parse_args()

jobs_dir = args.jobs_dir.resolve()
if not jobs_dir.exists():
    raise SystemExit(f"Jobs directory not found: {jobs_dir}")

# %%
# Collect results from all batches and trials
all_results: list[dict] = []

for batch_dir in tqdm(sorted(jobs_dir.iterdir()), desc="Batches"):
    if not batch_dir.is_dir():
        continue

    for trial_dir in sorted(batch_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        # Skip non-trial directories (e.g., config.json, result.json)
        if not (trial_dir / "agent").exists():
            continue

        trajectory_path = trial_dir / "agent/trajectory.json"
        if not trajectory_path.exists():
            continue

        try:
            with open(trajectory_path) as f:
                trajectory = json.load(f)
        except Exception as e:
            print(f"Skipping {trial_dir}: {e}")
            continue

        # Extract final metrics and model name
        final_metrics = trajectory.get("final_metrics", {})
        agent_info = trajectory.get("agent", {})
        agent_name = agent_info.get("name", "unknown")
        model_name = agent_info.get("model_name", "unknown")

        # Get step count: use "steps" field length for terminus-2, otherwise use final_metrics
        if agent_name == "terminus-2":
            total_steps = len(trajectory.get("steps", []))
        else:
            total_steps = final_metrics.get("total_steps", 0)

        # Get cost: use total_cost_usd if available, otherwise estimate from tokens
        total_cost = final_metrics.get("total_cost_usd")
        if total_cost is None:
            total_cost = calculate_cost(
                model=model_name,
                prompt_tokens=final_metrics.get("total_prompt_tokens", 0),
                completion_tokens=final_metrics.get("total_completion_tokens", 0),
                cached_tokens=final_metrics.get("total_cached_tokens", 0),
            )

        all_results.append(
            {
                "batch": batch_dir.name,
                "trial_id": trial_dir.name,
                "agent": agent_name,
                "model_name": model_name,
                "total_prompt_tokens": final_metrics.get("total_prompt_tokens", 0),
                "total_completion_tokens": final_metrics.get(
                    "total_completion_tokens", 0
                ),
                "total_cached_tokens": final_metrics.get("total_cached_tokens", 0),
                "total_steps": total_steps,
                "total_cost_usd": total_cost,
            }
        )

# %%
# Print formatted token usage results
if not all_results:
    print("No results found.")
    raise SystemExit(0)

df = pd.DataFrame(all_results)
df["total_tokens"] = df["total_prompt_tokens"] + df["total_completion_tokens"]

# Compute mean and standard error using groupby
metric_cols = [
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_cached_tokens",
    "total_tokens",
    "total_steps",
    "total_cost_usd",
]
grouped = df.groupby(["agent", "model_name"])[metric_cols]
means = grouped.mean()
stds = grouped.std(ddof=1)
counts = grouped.size()
se = stds.div(np.sqrt(counts), axis=0)

# Scale token columns to millions (except steps and cost)
scale_cols = [
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_cached_tokens",
    "total_tokens",
]
means[scale_cols] = means[scale_cols] / 1e6
se[scale_cols] = se[scale_cols] / 1e6

# Format as "mean ± se" strings
display_cols = [
    "Prompt (M)",
    "Completion (M)",
    "Cached (M)",
    "Total (M)",
    "Steps",
    "Cost ($)",
]
result = pd.DataFrame(index=means.index)
for old_col, new_col in zip(metric_cols, display_cols):
    result[new_col] = means[old_col].combine(
        se[old_col], lambda m, s: f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f}"
    )
result["n"] = counts

print("Mean ± SE Token Usage by Agent and Model")
print(result.to_markdown())
