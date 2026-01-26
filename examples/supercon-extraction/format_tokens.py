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

This script iterates over batches/trials in a job directory or usage JSON files in
an output directory, and prints combined token usage statistics.
"""

# %%
import json
import re
from argparse import ArgumentParser
from tabulate import tabulate

import pandas as pd
from tqdm import tqdm

import pbench
from llm_utils import calculate_cost
from utils import count_trials_per_agent_model, mean_sem_with_n

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
# Collect results from all batches and trials
all_results: list[dict] = []

if args.jobs_dir:
    # Harbor job directory format
    jobs_dir = args.jobs_dir.resolve()
    if not jobs_dir.exists():
        raise SystemExit(f"Jobs directory not found: {jobs_dir}")

    for batch_dir in tqdm(sorted(jobs_dir.iterdir()), desc="Batches"):
        if not batch_dir.is_dir():
            continue

        # Get agent and model name from batch config.json (consistent with count_trials_per_agent_model)
        agent_name, model_name = "unknown", "unknown"
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    agent_name = config["agents"][0].get("name", "unknown")
                    model_name = config["agents"][0].get("model_name", "unknown")
            except Exception:
                pass

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

            # Extract final metrics
            final_metrics = trajectory.get("final_metrics", {})
            agent_info = trajectory.get("agent", {})

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

else:
    # Zeroshot output directory format
    output_dir = args.output_dir.resolve()
    usage_dir = output_dir / "usage"
    if not usage_dir.exists():
        raise SystemExit(f"Usage directory not found: {usage_dir}")

    # Parse usage JSON files: usage__model=<model>__refno=<refno>.json
    # or usage__agent=<agent>__model=<model>__refno=<refno>.json
    usage_pattern = re.compile(r"usage__(?:agent=.+__)?model=(.+?)__refno=(.+)\.json")

    for usage_file in tqdm(sorted(usage_dir.glob("*.json")), desc="Usage files"):
        match = usage_pattern.match(usage_file.name)
        if not match:
            print(f"Skipping unrecognized file: {usage_file.name}")
            continue

        model_name = match.group(1)
        refno = match.group(2)

        try:
            with open(usage_file) as f:
                usage = json.load(f)
        except Exception as e:
            print(f"Skipping {usage_file}: {e}")
            continue

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        thinking_tokens = usage.get("thinking_tokens", 0)

        # Calculate cost from tokens
        total_cost = calculate_cost(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

        all_results.append(
            {
                "batch": "zeroshot",
                "trial_id": refno,
                "agent": "zeroshot",
                "model_name": model_name,
                "total_prompt_tokens": prompt_tokens,
                "total_completion_tokens": completion_tokens,
                "total_cached_tokens": cached_tokens,
                "total_thinking_tokens": thinking_tokens,
                "total_steps": 1,
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

# Fill missing thinking tokens with 0 for Harbor results
if "total_thinking_tokens" not in df.columns:
    df["total_thinking_tokens"] = 0
df["total_thinking_tokens"] = df["total_thinking_tokens"].fillna(0)

# Get trial counts for normalization
if args.jobs_dir:
    trials_df = count_trials_per_agent_model(args.jobs_dir)
    trials_lookup: dict[tuple[str, str], int] = {
        (row["agent"], row["model"]): row["num_trials"]
        for _, row in trials_df.iterrows()
    }
else:
    # For zeroshot, count unique refnos per agent/model
    trials_lookup = df.groupby(["agent", "model_name"])["trial_id"].nunique().to_dict()

# Compute mean and SEM normalized by number of trials
metric_cols = [
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_cached_tokens",
    "total_thinking_tokens",
    "total_tokens",
    "total_steps",
    "total_cost_usd",
]

# Scale token columns to millions (except steps and cost)
scale_cols = [
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_cached_tokens",
    "total_thinking_tokens",
    "total_tokens",
]
df[scale_cols] = df[scale_cols] / 1e6

# Group and compute mean ± SEM using trial counts
display_cols = [
    "Prompt (M)",
    "Completion (M)",
    "Cached (M)",
    "Thinking (M)",
    "Total (M)",
    "Steps",
    "Cost ($)",
]


def compute_stats(g: pd.DataFrame) -> pd.Series:
    """Compute mean ± SEM for each metric column, normalized by number of trials.

    Args:
        g: DataFrame group for a specific (agent, model_name).

    Returns:
        pd.Series with mean ± SEM for each metric.

    """
    key = g.name  # (agent, model_name) tuple from groupby
    n_trials = trials_lookup.get(key, len(g))
    stats = {
        new_col: mean_sem_with_n(g[old_col].tolist(), n_trials)
        for old_col, new_col in zip(metric_cols, display_cols)
    }
    stats["successful_count"] = len(g)
    stats["num_trials"] = n_trials
    return pd.Series(stats)


result = (
    df.groupby(["agent", "model_name"])
    .apply(compute_stats, include_groups=False)
    .reset_index()
)

print("Mean ± SE Token Usage by Agent and Model (normalized by num_trials)")
print(
    tabulate(result.values.tolist(), headers=result.columns.tolist(), tablefmt="github")
)
