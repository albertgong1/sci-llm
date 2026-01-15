#! /usr/bin/env -S uv run --env-file=.env -- python
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
"""Format accuracy results from Harbor job directories.

Usage:
  uv run python format_accuracy.py -jd <job_dir>

This script iterates over batches in a job directory, reads the verifier/details.json
from each trial, and prints combined accuracy statistics.
"""

# %%
import json
from argparse import ArgumentParser
from pathlib import Path

# %%
parser = ArgumentParser(
    description="Format accuracy results from Harbor job directories."
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

for batch_dir in sorted(jobs_dir.iterdir()):
    if not batch_dir.is_dir():
        continue

    for trial_dir in sorted(batch_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        # Skip non-trial directories (e.g., config.json, result.json)
        if not (trial_dir / "verifier").exists():
            continue

        details_path = trial_dir / "verifier/details.json"
        if not details_path.exists():
            continue

        try:
            with open(details_path) as f:
                details = json.load(f)
        except Exception as e:
            print(f"Skipping {trial_dir}: {e}")
            continue

        all_results.append(
            {
                "batch": batch_dir.name,
                "trial_id": trial_dir.name,
                "refno": details.get("refno", "unknown"),
                "task": details.get("task", "unknown"),
                "reward": details.get("reward", 0.0),
                "correct": details.get("correct", 0),
                "total": details.get("total", 0),
                "n_predictions": details.get("n_predictions", 0),
            }
        )

# %%
# Print formatted accuracy results
if not all_results:
    print("No results found.")
    raise SystemExit(0)

print("=" * 80)
print("ACCURACY RESULTS")
print("=" * 80)

total_correct = 0
total_total = 0

for r in all_results:
    reward_pct = r["reward"] * 100
    print(
        f"  {r['trial_id']:<40} "
        f"reward={reward_pct:5.1f}%  "
        f"({r['correct']:3d}/{r['total']:3d})"
    )
    total_correct += r["correct"]
    total_total += r["total"]

# Overall summary
print("=" * 80)
overall_acc = total_correct / total_total * 100 if total_total > 0 else 0
print(
    f"OVERALL: {len(all_results)} trials, {total_correct}/{total_total} correct ({overall_acc:.1f}%)"
)
print("=" * 80)
