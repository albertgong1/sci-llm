"""Script to format F1 vs. cost (USD) or total tokens for property extraction tasks.

Todo:
- Overlay no agent harness results
- Add reasoning_effort support

Usage:
    python scripts/format_accuracy_tokens.py --jobs-dir <HARBOR_JOBS_DIR> --output-dir <OUTPUT_DIR> --x-axis <tokens|cost>

"""

import pbench

import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path

from pbench_eval.plotting_utils import (
    TICK_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
)

# Model -> color mapping for consistent visualization
MODEL_COLORS: dict[str, str] = {
    "gpt-5-mini-2025-08-07": "#1f77b4",  # blue
    "gpt-5.1-2025-11-13": "#2ca02c",  # green
    "gpt-5.2-2025-12-11": "#9467bd",  # purple
    "gemini-3-flash-preview": "#ff7f0e",  # orange
    "gemini-3-pro-preview": "#d62728",  # red
}

parser = ArgumentParser(
    description="Format token usage results from Harbor job directories or zeroshot output directories."
)
parser = pbench.add_base_args(parser)
# parser = add_scoring_args(parser)
parser.add_argument(
    "--x-axis",
    type=str,
    choices=["tokens", "cost"],
    default="tokens",
    help="X-axis metric: 'tokens' for total tokens or 'cost' for total cost in USD",
)
args = parser.parse_args()

pbench.setup_logging(args.log_level)

# Collect token usage with reasoning_effort
group_cols = ["agent", "model_name"]

# Read merged table from examples subdirectories
output_dirs = [
    ("supercon", Path("examples/supercon-extraction/out-post-2021")),
    ("supercon", Path("examples/supercon-extraction/out-post-2021-no-agent")),
    ("biosurfactants", Path("examples/biosurfactants-extraction/out-biosurfactants")),
    ("biosurfactants", Path("examples/biosurfactants-extraction/out-no-agent")),
    # ("cdw", Path("examples/cdw-extraction/out-cdw")),
]

# Set x-axis label based on metric
x_axis_label = "Average Tokens" if args.x_axis == "tokens" else "Average Cost (USD)"
x_metric_label = "avg_x_metric"

# Plot with error bars and labels
legend_handles = []
domains = ["supercon", "biosurfactants", "cdw"]
fig, axs = plt.subplots(1, len(domains), figsize=(6.75, 2.25), sharey=True)
for domain, output_dir in output_dirs:
    ax = axs[domains.index(domain)]
    # Read F1 scores
    merged = pd.read_csv(output_dir / "tables" / f"f1_vs_{args.x_axis}_summary.csv")

    # Plot each point with color based on model
    for idx, row in merged.iterrows():
        model_name = row["model"].split("/")[-1]
        color = MODEL_COLORS.get(model_name, "#333333")
        ax.errorbar(
            row[x_metric_label],
            row["avg_f1_score"],
            xerr=row[f"{x_metric_label}_sem"],
            yerr=row["avg_f1_score_sem"],
            fmt="o",
            color=color,
            capsize=5,
            capthick=2,
        )
        # Annotate only with agent harness
        ax.annotate(
            row["agent"],
            (row[x_metric_label], row["avg_f1_score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )
        if model_name not in [h.get_label() for h in legend_handles]:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=model_name,
                )
            )
    ax.set_title(domain.capitalize(), fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel(x_axis_label, fontsize=LABEL_FONT_SIZE)
    if domain == domains[0]:
        ax.set_ylabel("Average F1 Score", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(axis="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Add legend for models at the bottom (only include models with results)
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(legend_handles),
    fontsize=LEGEND_FONT_SIZE,
    frameon=False,
    bbox_to_anchor=(0.5, -0.15),
)

plt.tight_layout()
figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)
fig_name = f"f1_vs_{args.x_axis}.pdf"
fig_path = figures_dir / fig_name
plt.savefig(fig_path, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
plt.close()
