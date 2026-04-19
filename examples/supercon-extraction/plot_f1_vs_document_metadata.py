"""Plot F1 score vs. document metadata (number of GT properties, number of pages).

Usage:
    uv run python plot_f1_vs_document_metadata.py --output_dir out-supercon --data_dir data-arxiv
"""

from argparse import ArgumentParser
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pbench
from pbench_eval.plotting_utils import (
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    MARKER_SIZE,
    TICK_FONT_SIZE,
)

# Agent -> color mapping
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
    "zeroshot-gpt": "o",
    "zeroshot-gemini": "s",
    "codex": "o",
    "gemini-cli": "s",
    "terminus-2-gpt": "o",
    "terminus-2-gemini": "s",
}

# Display names for legend
DISPLAY_NAMES: dict[str, str] = {
    "zeroshot-gpt": "GPT",
    "zeroshot-gemini": "Gemini",
    "codex": "codex",
    "gemini-cli": "gemini-cli",
    "terminus-2-gpt": "terminus-2 (GPT)",
    "terminus-2-gemini": "terminus-2 (Gemini)",
}


def get_color_key(agent: str, model: str) -> str:
    """Get the color key for an agent+model combination."""
    model_short = model.split("/")[-1]
    if agent in ("zeroshot", "terminus-2"):
        suffix = "-gemini" if "gemini" in model_short else "-gpt"
        return agent + suffix
    return agent


def get_page_count(refno: str, paper_dir: Path) -> int | None:
    """Get the number of pages in a PDF file."""
    paper_path = paper_dir / f"{refno}.pdf"
    if not paper_path.exists():
        return None
    doc = fitz.open(paper_path)
    num_pages = len(doc)
    doc.close()
    return num_pages


parser = ArgumentParser(
    description="Plot F1 score vs. document metadata (num GT properties, num pages)"
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

# Load F1 scores by refno
f1_path = args.output_dir / "scores" / "f1_by_refno.csv"
if not f1_path.exists():
    raise FileNotFoundError(
        f"F1 scores file not found: {f1_path}. "
        "Run pbench-score-f1 first to generate it."
    )
df = pd.read_csv(f1_path, dtype={"refno": str})

# Compute number of pages per document
paper_dir = args.data_dir / "Paper_DB"
df["num_pages"] = df["refno"].apply(lambda r: get_page_count(r, paper_dir))

# Create figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.5), sharey=True)

# Collect legend handles
legend_handles: list[plt.Line2D] = []

# Plot each (agent, model) combination
for (agent, model), group in df.groupby(["agent", "model"]):
    color_key = get_color_key(agent, model)
    color = AGENT_COLORS.get(color_key, "#333333")
    marker = AGENT_MARKERS.get(color_key, "o")

    # Plot F1 vs. num_gt (left subplot)
    axs[0].scatter(
        group["num_gt"],
        group["f1_score"],
        c=color,
        marker=marker,
        s=MARKER_SIZE**2,
        alpha=0.7,
        label=DISPLAY_NAMES.get(color_key, color_key),
    )

    # Plot F1 vs. num_pages (right subplot)
    axs[1].scatter(
        group["num_pages"],
        group["f1_score"],
        c=color,
        marker=marker,
        s=MARKER_SIZE**2,
        alpha=0.7,
    )

    # Add line of best fit for this agent/model (left subplot - F1 vs. num_gt)
    x_gt = group["num_gt"].dropna()
    y_gt = group.loc[x_gt.index, "f1_score"]
    if len(x_gt) > 1:
        z_gt = np.polyfit(x_gt, y_gt, 1)
        p_gt = np.poly1d(z_gt)
        x_fit_gt = np.linspace(x_gt.min(), x_gt.max(), 100)
        axs[0].plot(
            x_fit_gt,
            p_gt(x_fit_gt),
            color=color,
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )

    # Add line of best fit for this agent/model (right subplot - F1 vs. num_pages)
    group_pages = group.dropna(subset=["num_pages"])
    x_pages = group_pages["num_pages"]
    y_pages = group_pages["f1_score"]
    if len(x_pages) > 1:
        z_pages = np.polyfit(x_pages, y_pages, 1)
        p_pages = np.poly1d(z_pages)
        x_fit_pages = np.linspace(x_pages.min(), x_pages.max(), 100)
        axs[1].plot(
            x_fit_pages,
            p_pages(x_fit_pages),
            color=color,
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )

    # Add to legend if not already present
    legend_label = DISPLAY_NAMES.get(color_key, color_key)
    if legend_label not in [h.get_label() for h in legend_handles]:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markersize=6,
                label=legend_label,
            )
        )

# Configure left subplot (F1 vs. num_gt)
axs[0].set_xlabel("Number of GT Properties", fontsize=LABEL_FONT_SIZE)
axs[0].set_ylabel("F1 Score", fontsize=LABEL_FONT_SIZE)
axs[0].tick_params(axis="both", labelsize=TICK_FONT_SIZE)
axs[0].grid(axis="both", alpha=0.3)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)

# Configure right subplot (F1 vs. num_pages)
axs[1].set_xlabel("Number of Pages", fontsize=LABEL_FONT_SIZE)
axs[1].tick_params(axis="both", labelsize=TICK_FONT_SIZE)
axs[1].grid(axis="both", alpha=0.3)
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)

# Add legend at bottom
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(legend_handles),
    fontsize=LEGEND_FONT_SIZE,
    frameon=False,
    bbox_to_anchor=(0.5, -0.1),
)

plt.tight_layout()

# Save figure
figures_dir = args.output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
fig_path = figures_dir / "f1_vs_document_metadata.pdf"
plt.savefig(fig_path, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
plt.close()
