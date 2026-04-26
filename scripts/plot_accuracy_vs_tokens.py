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
import matplotlib.gridspec as gridspec

from pbench_eval.plotting_utils import (
    TICK_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    OUTWARD,
    get_display_label,
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
    "qwen": "#9467bd",  # purple
}

# Agent -> marker mapping (gpt = circle, gemini = square)
AGENT_MARKERS: dict[str, str] = {
    "zeroshot-gpt": "o",  # circle
    "zeroshot-gemini": "s",  # square
    "codex": "o",  # circle
    "gemini-cli": "s",  # square
    "terminus-2-gpt": "o",  # circle
    "terminus-2-gemini": "s",  # square
    "qwen": "D",  # diamond
}

# Domain -> display name mapping for plot titles
DOMAIN_ALIASES: dict[str, str] = {
    "supercon-post-2021": "SuperCon",
    "biosurfactants": "Biosurfactants",
    "cdw": "CDW",
    "tc": "Tc",
    "flux": "Flux",
}

# Agent color_key -> legend display name mapping
DISPLAY_NAMES: dict[str, str] = {
    "zeroshot-gpt": "GPT",
    "zeroshot-gemini": "Gemini",
    "codex": "codex",
    "gemini-cli": "gemini-cli",
    "terminus-2-gpt": "terminus-2 (GPT)",
    "terminus-2-gemini": "terminus-2 (Gemini)",
    "qwen": "Qwen",
}

# Mapping from CSV agent names (in normalized_score_vs_cost_tokens.csv) to color keys
CSV_AGENT_TO_COLOR_KEY: dict[str, str] = {
    "Gemini-CLI": "gemini-cli",
    "Codex": "codex",
    "Qwen": "qwen",
    "Terminus (GPT)": "terminus-2-gpt",
    "Terminus (Gemini)": "terminus-2-gemini",
    "Gemini-3-Pro": "zeroshot-gemini",
    "GPT-5.1": "zeroshot-gpt",
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
    (
        "supercon-post-2021",
        Path("examples/supercon-extraction/out-post-2021"),
    ),  # harbor
    ("supercon-post-2021", Path("examples/supercon-extraction/out-post-2021-no-agent")),
    (
        "biosurfactants",
        Path("examples/biosurfactants-extraction/out-biosurfactants"),
    ),  # harbor
    ("biosurfactants", Path("examples/biosurfactants-extraction/out-no-agent")),
    # ("cdw", Path("examples/cdw-extraction/out-cdw")),  # harbor
]

# Set x-axis label based on metric
x_axis_label = "Tokens (M)" if args.x_axis == "tokens" else "Cost (USD)"
token_scale = 1e6 if args.x_axis == "tokens" else 1
x_metric_label = "avg_x_metric"

# Plot with error bars and labels
legend_handles = []
domains = ["supercon-post-2021", "biosurfactants", "flux", "tc"]

# 1x4 single row layout with two groups, ICML two-column width
fig = plt.figure(figsize=(6.75, 2.25))
outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
inner_left = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[0], wspace=0.15
)
inner_right = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[1], wspace=0.15
)
ax0 = fig.add_subplot(inner_left[0, 0])
axs = [
    ax0,
    fig.add_subplot(inner_left[0, 1], sharey=ax0),
    None,  # placeholder, set below
    None,
]
ax2 = fig.add_subplot(inner_right[0, 0])
axs[2] = ax2
axs[3] = fig.add_subplot(inner_right[0, 1], sharey=ax2)

# --- Plot SuperCon and Biosurfactants from per-domain output directories ---
for domain, output_dir in output_dirs:
    ax = axs[domains.index(domain)]
    merged = pd.read_csv(output_dir / "tables" / f"f1_vs_{args.x_axis}_summary.csv")

    # Skip small models
    SKIP_MODELS = {"gemini-3-flash-preview", "gpt-5-mini-2025-08-07"}
    merged = merged[
        ~merged["model"].apply(lambda m: m.split("/")[-1]).isin(SKIP_MODELS)
    ]

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
            row[x_metric_label] / token_scale,
            row["avg_score"],
            xerr=row[f"{x_metric_label}_sem"] / token_scale,
            yerr=row["avg_score_sem"],
            fmt=marker,
            color=color,
            alpha=alpha,
            capsize=0,
            clip_on=False,
        )
        legend_label = get_display_label(row["agent"], row["model"], multiline=False)
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
        ax.set_ylabel("Recovery Rate", fontsize=LABEL_FONT_SIZE)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, direction="out")
    ax.grid(axis="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", OUTWARD))
    ax.spines["bottom"].set_position(("outward", OUTWARD))

# --- Plot Tc and Flux from normalized CSV ---
csv_data = pd.read_csv(Path("data/normalized_score_vs_cost_tokens.csv"))
x_col = "avg_tokens" if args.x_axis == "tokens" else "avg_cost"
x_sem_col = f"{x_col}_sem"

for task_name in ["tc", "flux"]:
    ax = axs[domains.index(task_name)]
    task_df = csv_data[(csv_data["task"] == task_name) & (csv_data["agent"] != "Qwen")]

    for _, row in task_df.iterrows():
        agent_str: str = row["agent"]
        color_key = CSV_AGENT_TO_COLOR_KEY.get(agent_str, agent_str)
        color = AGENT_COLORS.get(color_key, "#333333")
        marker = AGENT_MARKERS.get(color_key, "o")
        # Baselines get full alpha, agents get slightly lower
        alpha = 1.0 if row["type"] == "baseline" else 0.7
        ax.errorbar(
            row[x_col] / token_scale,
            row["avg_score"],
            xerr=row[x_sem_col] / token_scale,
            yerr=row["avg_score_sem"],
            fmt=marker,
            color=color,
            alpha=alpha,
            capsize=0,
            clip_on=False,
        )
        legend_label = DISPLAY_NAMES.get(color_key, color_key)
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
    ax.set_title(DOMAIN_ALIASES.get(task_name, task_name), fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel(x_axis_label, fontsize=LABEL_FONT_SIZE)
    if task_name == "flux":
        ax.set_ylabel("Accuracy Score", fontsize=LABEL_FONT_SIZE)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, direction="out")
    ax.grid(axis="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", OUTWARD))
    ax.spines["bottom"].set_position(("outward", OUTWARD))

# Add legend for agents at the bottom (only include agents with results)
# Sort handles: GPT/codex variants first (top row), Gemini variants second (bottom row)
gpt_labels = ["GPT", "codex", "terminus-2 (GPT)"]
gemini_labels = ["Gemini", "gemini-cli", "terminus-2 (Gemini)", "Qwen"]
sorted_handles = sorted(
    legend_handles,
    key=lambda h: (
        0 if h.get_label() in gpt_labels else 1,
        gpt_labels.index(h.get_label())
        if h.get_label() in gpt_labels
        else gemini_labels.index(h.get_label())
        if h.get_label() in gemini_labels
        else 0,
    ),
)
fig.legend(
    handles=sorted_handles,
    loc="lower center",
    ncol=len(sorted_handles),
    fontsize=LEGEND_FONT_SIZE,
    frameon=False,
    bbox_to_anchor=(0.5, -0.15),
)
figures_dir = args.output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
fig_name = f"f1_vs_{args.x_axis}.pdf"
fig_path = figures_dir / fig_name
plt.savefig(fig_path, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
plt.close()
