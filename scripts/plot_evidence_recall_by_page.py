"""Script to plot evidence recall by page number for each agent/model combination.

Usage:
    python scripts/plot_evidence_recall_by_page.py

Generates:
    - figures/evidence_recall_by_page.pdf

"""

import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path

import pbench
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

# Agent -> line style mapping (gpt = solid, gemini = dashed)
AGENT_LINESTYLES: dict[str, str] = {
    "zeroshot-gpt": "-",
    "zeroshot-gemini": "--",
    "codex": "-",
    "gemini-cli": "-",
    "terminus-2-gpt": "-",
    "terminus-2-gemini": "--",
}

# Domain -> display name mapping for plot titles
DOMAIN_ALIASES: dict[str, str] = {
    "supercon": "SuperCon",
    "biosurfactants": "Biosurfactants",
    "cdw": "CDW",
}

# Agent color_key -> legend display name mapping
DISPLAY_NAMES: dict[str, str] = {
    "zeroshot-gpt": "GPT",
    "zeroshot-gemini": "Gemini",
    "codex": "codex",
    "gemini-cli": "gemini-cli",
    "terminus-2-gpt": "terminus-2 (GPT)",
    "terminus-2-gemini": "terminus-2 (Gemini)",
}

parser = ArgumentParser(
    description="Plot evidence recall by page number for each agent/model."
)
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--max_page",
    type=int,
    default=60,
    help="Maximum page number to display on x-axis (default: 60)",
)
args = parser.parse_args()

pbench.setup_logging(args.log_level)

# Read evidence recall by page from examples subdirectories
output_dirs = [
    ("supercon", Path("examples/supercon-extraction/out-post-2021")),  # harbor
    ("supercon", Path("examples/supercon-extraction/out-post-2021-no-agent")),
    (
        "biosurfactants",
        Path("examples/biosurfactants-extraction/out-biosurfactants"),
    ),  # harbor
    ("biosurfactants", Path("examples/biosurfactants-extraction/out-no-agent")),
]

# Collect data from all output directories
all_data: dict[str, list[pd.DataFrame]] = {"supercon": [], "biosurfactants": []}
for domain, output_dir in output_dirs:
    recall_path = output_dir / "tables" / "evidence_recall_by_page.csv"
    if not recall_path.exists():
        print(f"Warning: {recall_path} not found, skipping")
        continue
    df = pd.read_csv(recall_path)
    all_data[domain].append(df)

# Check if we have any data
has_data = any(dfs for dfs in all_data.values())
if not has_data:
    print(
        "No data found. Run pbench-score-evidence first to generate "
        "evidence_recall_by_page.csv tables."
    )
    exit(1)

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# Filter to domains that have data
domains = [d for d in ["supercon", "biosurfactants"] if all_data.get(d)]

fig, axs = plt.subplots(1, len(domains), figsize=(6.75, 2.5), sharey=True)

# Handle single subplot case
if len(domains) == 1:
    axs = [axs]

legend_handles = []
legend_labels = []

for i, domain in enumerate(domains):
    ax = axs[i]
    dfs = all_data[domain]

    if not dfs:
        ax.set_title(DOMAIN_ALIASES.get(domain, domain), fontsize=LABEL_FONT_SIZE)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue

    # Combine all DataFrames for this domain
    df_combined = pd.concat(dfs, ignore_index=True)

    # Plot each (agent, model) combination as a line
    for (agent, model), group in df_combined.groupby(["agent", "model"]):
        # Sort by page number and filter to max_page
        group = group.sort_values("page")
        group = group[group["page"] <= args.max_page]

        if group.empty:
            continue

        # Determine color_key based on agent and model provider
        model_name = model.split("/")[-1]
        if agent in ("zeroshot", "terminus-2"):
            suffix = "-gemini" if "gemini" in model_name else "-gpt"
            color_key = agent + suffix
        else:
            color_key = agent

        color = AGENT_COLORS.get(color_key, "#333333")
        linestyle = AGENT_LINESTYLES.get(color_key, "-")
        display_name = DISPLAY_NAMES.get(color_key, color_key)

        # Plot line
        line = ax.plot(
            group["page"],
            group["avg_evidence_recall"],
            color=color,
            linestyle=linestyle,
            alpha=0.8,
            linewidth=1.5,
        )[0]

        # Add error band
        ax.fill_between(
            group["page"],
            group["avg_evidence_recall"] - group["avg_evidence_recall_sem"],
            group["avg_evidence_recall"] + group["avg_evidence_recall_sem"],
            color=color,
            alpha=0.15,
        )

        # Add vertical line at max page for this agent/model
        max_page_for_agent = int(group["page"].max())
        ax.axvline(
            x=max_page_for_agent,
            color=color,
            linestyle=":",
            alpha=0.5,
            linewidth=1.0,
        )

        # Track legend entries (deduplicate by color_key)
        if display_name not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(display_name)

    # Determine max page for this domain from the data
    domain_max_page = (
        int(df_combined["page"].max()) if not df_combined.empty else args.max_page
    )
    domain_max_page = min(domain_max_page, args.max_page)  # Cap at args.max_page

    ax.set_xlabel("Page Number", fontsize=LABEL_FONT_SIZE)
    if i == 0:
        ax.set_ylabel("Evidence Recall", fontsize=LABEL_FONT_SIZE)
    ax.set_title(DOMAIN_ALIASES.get(domain, domain), fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(axis="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1, domain_max_page)
    ax.set_ylim(0, 1)

# Add legend at bottom (only include agents with results)
# Sort handles: GPT/codex variants first (top row), Gemini variants second (bottom row)
if legend_handles:
    gpt_labels = ["GPT", "codex", "terminus-2 (GPT)"]
    # Create (handle, label) pairs and sort
    handle_label_pairs = list(zip(legend_handles, legend_labels))
    sorted_pairs = sorted(
        handle_label_pairs,
        key=lambda x: (
            0 if x[1] in gpt_labels else 1,
            gpt_labels.index(x[1]) if x[1] in gpt_labels else 0,
        ),
    )
    sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig.legend(
        sorted_handles,
        sorted_labels,
        loc="lower center",
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, -0.15),
    )

plt.tight_layout()
fig_path = figures_dir / "evidence_recall_by_page.pdf"
plt.savefig(fig_path, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
plt.close()
