"""Script to plot failure rate as a horizontal bar plot for SuperCon tasks.

Failure rate is defined as 1 - successful_count_x / num_trials.

Usage:
    python scripts/plot_failure_rate.py

"""

import pbench

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from pbench_eval.plotting_utils import (
    TICK_FONT_SIZE,
    LABEL_FONT_SIZE,
    OUTWARD,
    get_display_label,
)

parser = ArgumentParser(
    description="Plot failure rate as a horizontal bar plot for SuperCon tasks."
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()

pbench.setup_logging(args.log_level)

# Output directories for SuperCon data
output_dirs = [
    Path("examples/supercon-extraction/out-supercon"),
    Path("examples/supercon-extraction/out-supercon-no-agent"),
]

# Collect data from all output directories
rows = []
for output_dir in output_dirs:
    csv_path = output_dir / "tables" / "f1_vs_cost_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            n = row["num_trials"]
            p = 1 - row["successful_count_x"] / n  # failure rate
            # Standard error for a proportion: sqrt(p * (1-p) / n)
            se = np.sqrt(p * (1 - p) / n)
            rows.append(
                {
                    "agent": row["agent"],
                    "model": row["model"],
                    "failure_rate": p,
                    "failure_rate_se": se,
                }
            )

# Create dataframe and aggregate by agent/model (in case of duplicates)
plot_df = pd.DataFrame(rows)

# Skip small models
SKIP_MODELS = {"gemini-3-flash-preview", "gpt-5-mini-2025-08-07"}
plot_df = plot_df[~plot_df["model"].apply(lambda m: m.split("/")[-1]).isin(SKIP_MODELS)]

plot_df = plot_df.groupby(["agent", "model"], as_index=False).agg(
    {"failure_rate": "mean", "failure_rate_se": "mean"}
)

# Add display label after aggregation
plot_df["label"] = plot_df.apply(
    lambda row: get_display_label(row["agent"], row["model"], multiline=False), axis=1
)

# Sort by failure rate (descending so highest failure is at top)
plot_df = plot_df.sort_values("failure_rate", ascending=True)

# Create horizontal bar plot
fig, ax = plt.subplots(figsize=(3.25, 2))

ax.barh(
    plot_df["label"],
    plot_df["failure_rate"],
    xerr=plot_df["failure_rate_se"],
    color="#E57373",
    edgecolor="none",
    capsize=3,
    error_kw={"elinewidth": 1},
)

ax.set_xlabel("Failure Rate", fontsize=LABEL_FONT_SIZE)
max_failure = plot_df["failure_rate"].max() + plot_df["failure_rate_se"].max()
ax.set_xlim(0, max_failure * 1.1)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE, direction="out")
ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, direction="out")
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_position(("outward", OUTWARD))
ax.spines["bottom"].set_position(("outward", OUTWARD))

plt.tight_layout()

# Save figure
figures_dir = args.output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
fig_path = figures_dir / "failure_rate.pdf"
plt.savefig(fig_path, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
plt.close()
