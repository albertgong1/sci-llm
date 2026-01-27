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

# Agent -> color mapping for consistent visualization
# zeroshot and terminus-2 use same color but different markers for gpt vs gemini
AGENT_COLORS: dict[str, str] = {
    "zeroshot-gpt": "#555555",  # gray
    "zeroshot-gemini": "#555555",  # gray
    "codex": "#2ca02c",  # green
    "gemini-cli": "#ff7f0e",  # orange
    "terminus-2-gpt": "#d62728",  # red
    "terminus-2-gemini": "#d62728",  # red
}

# Agent -> marker mapping (gpt = circle, gemini = square)
AGENT_MARKERS: dict[str, str] = {
    "zeroshot-gpt": "o",  # circle
    "zeroshot-gemini": "s",  # square
    "codex": "o",  # circle
    "gemini-cli": "o",  # circle
    "terminus-2-gpt": "o",  # circle
    "terminus-2-gemini": "s",  # square
}

# Domain -> display name mapping for plot titles
DOMAIN_ALIASES: dict[str, str] = {
    "supercon": "SuperCon",
    "biosurfactants": "Biosurfactants",
    "cdw": "CDW",
}

# Model -> alpha mapping (large models = high alpha, small models = low alpha)
MODEL_ALPHAS: dict[str, float] = {
    "gpt-5-mini-2025-08-07": 0.4,  # small
    "gemini-3-flash-preview": 0.4,  # small
    "gpt-5.1-2025-11-13": 1.0,  # large
    "gpt-5.2-2025-12-11": 1.0,  # large
    "gemini-3-pro-preview": 1.0,  # large
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
    ("supercon", Path("examples/supercon-extraction/out-post-2021")),  # harbor
    ("supercon", Path("examples/supercon-extraction/out-post-2021-no-agent")),
    (
        "biosurfactants",
        Path("examples/biosurfactants-extraction/out-biosurfactants"),
    ),  # harbor
    ("biosurfactants", Path("examples/biosurfactants-extraction/out-no-agent")),
    ("cdw", Path("examples/cdw-extraction/out-cdw")),  # harbor
]

# Set x-axis label based on metric
x_axis_label = "Tokens" if args.x_axis == "tokens" else "Cost (USD)"
x_metric_label = "avg_x_metric"

# Plot with error bars and labels
legend_handles = []
domains = ["supercon", "biosurfactants", "cdw"]
fig, axs = plt.subplots(1, len(domains), figsize=(6.75, 2.25), sharey=True)
for domain, output_dir in output_dirs:
    ax = axs[domains.index(domain)]
    # Read F1 scores
    merged = pd.read_csv(output_dir / "tables" / f"f1_vs_{args.x_axis}_summary.csv")

    # Plot each point with color based on agent, alpha based on model
    for idx, row in merged.iterrows():
        agent_name = row["agent"]
        model_name = row["model"].split("/")[-1]
        # zeroshot and terminus-2 use different colors based on model provider
        if agent_name in ("zeroshot", "terminus-2"):
            suffix = "-gemini" if "gemini" in model_name else "-gpt"
            color_key = agent_name + suffix
        else:
            color_key = agent_name
        color = AGENT_COLORS.get(color_key, "#333333")
        marker = AGENT_MARKERS.get(color_key, "o")
        alpha = MODEL_ALPHAS.get(model_name, 0.7)
        ax.errorbar(
            row[x_metric_label],
            row["avg_f1_score"],
            xerr=row[f"{x_metric_label}_sem"],
            yerr=row["avg_f1_score_sem"],
            fmt=marker,
            color=color,
            alpha=alpha,
            capsize=0,
        )
        # Use model family as label for zeroshot, otherwise use color_key
        if agent_name == "zeroshot":
            legend_label = "Gemini" if "gemini" in model_name else "GPT"
        else:
            legend_label = color_key
        if legend_label not in [h.get_label() for h in legend_handles]:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=legend_label,
                )
            )
    ax.set_title(DOMAIN_ALIASES.get(domain, domain), fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel(x_axis_label, fontsize=LABEL_FONT_SIZE)
    if domain == domains[0]:
        ax.set_ylabel("F1 Score", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(axis="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Add legend for agents at the bottom (only include agents with results)
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
