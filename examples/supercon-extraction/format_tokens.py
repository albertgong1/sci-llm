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
from tabulate import tabulate
from collections import defaultdict

from tqdm import tqdm

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
        model_name = agent_info.get("model_name", "unknown")

        all_results.append(
            {
                "batch": batch_dir.name,
                "trial_id": trial_dir.name,
                "model_name": model_name,
                "total_prompt_tokens": final_metrics.get("total_prompt_tokens", 0),
                "total_completion_tokens": final_metrics.get(
                    "total_completion_tokens", 0
                ),
                "total_cached_tokens": final_metrics.get("total_cached_tokens", 0),
                "total_steps": final_metrics.get("total_steps", 0),
            }
        )

# %%
# Print formatted token usage results
if not all_results:
    print("No results found.")
    raise SystemExit(0)

results_by_model = defaultdict(list)
for r in all_results:
    results_by_model[r["model_name"]].append(r)

# Create table data with models as rows and metrics as columns
table_data = []
headers = [
    "Model",
    "Prompt (M)",
    "Completion (M)",
    "Cached (M)",
    "Total (M)",
    "Steps",
    "n",
]

for model_name in sorted(results_by_model.keys()):
    model_results = results_by_model[model_name]

    # Extract token arrays for this model
    prompt_tokens = np.array([r["total_prompt_tokens"] for r in model_results])
    completion_tokens = np.array([r["total_completion_tokens"] for r in model_results])
    cached_tokens = np.array([r["total_cached_tokens"] for r in model_results])
    steps = np.array([r["total_steps"] for r in model_results])
    total_tokens = prompt_tokens + completion_tokens

    n_trials = len(model_results)

    # Calculate mean and standard error
    if n_trials > 1:
        mean_prompt = np.mean(prompt_tokens) / 1e6
        se_prompt = np.std(prompt_tokens, ddof=1) / np.sqrt(n_trials) / 1e6

        mean_completion = np.mean(completion_tokens) / 1e6
        se_completion = np.std(completion_tokens, ddof=1) / np.sqrt(n_trials) / 1e6

        mean_cached = np.mean(cached_tokens) / 1e6
        se_cached = np.std(cached_tokens, ddof=1) / np.sqrt(n_trials) / 1e6

        mean_total = np.mean(total_tokens) / 1e6
        se_total = np.std(total_tokens, ddof=1) / np.sqrt(n_trials) / 1e6

        mean_steps = np.mean(steps)
        se_steps = np.std(steps, ddof=1) / np.sqrt(n_trials)

        table_data.append(
            [
                model_name,
                f"{mean_prompt:.4f} ± {se_prompt:.4f}",
                f"{mean_completion:.4f} ± {se_completion:.4f}",
                f"{mean_cached:.4f} ± {se_cached:.4f}",
                f"{mean_total:.4f} ± {se_total:.4f}",
                f"{mean_steps:.1f} ± {se_steps:.2f}",
                n_trials,
            ]
        )
    else:
        # Only one trial, no SE
        mean_prompt = prompt_tokens[0] / 1e6
        mean_completion = completion_tokens[0] / 1e6
        mean_cached = cached_tokens[0] / 1e6
        mean_total = total_tokens[0] / 1e6
        mean_steps = steps[0]

        table_data.append(
            [
                model_name,
                f"{mean_prompt:.4f}",
                f"{mean_completion:.4f}",
                f"{mean_cached:.4f}",
                f"{mean_total:.4f}",
                f"{mean_steps:.1f}",
                n_trials,
            ]
        )

print("Mean ± 1 standard error Token Usage by Model")
print(tabulate(table_data, headers=headers, tablefmt="github"))
