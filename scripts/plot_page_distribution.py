"""Script to plot the distribution of location.page for ground truth datasets.

Usage:
    python scripts/plot_page_distribution.py

Generates:
    - figures/gt_page_distribution.pdf

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

# Domain -> color mapping
DOMAIN_COLORS: dict[str, str] = {
    "supercon": "#1f77b4",  # blue
    "biosurfactants": "#ff7f0e",  # orange
    "cdw": "#2ca02c",  # green
}

# Domain -> display name mapping for plot titles
DOMAIN_ALIASES: dict[str, str] = {
    "supercon": "SuperCon",
    "biosurfactants": "Biosurfactants",
    "cdw": "CDW",
}

parser = ArgumentParser(
    description="Plot the distribution of location.page for GT datasets."
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()

pbench.setup_logging(args.log_level)

# Read page distribution from examples subdirectories
output_dirs = [
    ("supercon", Path("examples/supercon-extraction/out-post-2021")),
    (
        "biosurfactants",
        Path("examples/biosurfactants-extraction/out-biosurfactants"),
    ),
]

# Collect data from all output directories
all_data: dict[str, pd.DataFrame] = {}
for domain, output_dir in output_dirs:
    page_dist_path = output_dir / "tables" / "gt_page_distribution.csv"
    if not page_dist_path.exists():
        print(f"Warning: {page_dist_path} not found, skipping")
        continue
    df = pd.read_csv(page_dist_path)
    all_data[domain] = df

if not all_data:
    print(
        "No data found. Run pbench-score-evidence first to generate page distributions."
    )
    exit(1)

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# Create single figure with overlaid distributions
domains = list(all_data.keys())
fig, ax = plt.subplots(figsize=(3.375, 2.25))

bar_width = 0.35
for i, domain in enumerate(domains):
    df = all_data[domain]
    color = DOMAIN_COLORS.get(domain, "#333333")
    label = DOMAIN_ALIASES.get(domain, domain)

    # Normalize to density
    total = df["count"].sum()
    density = df["count"] / total

    # Offset bars for each domain
    offset = (i - (len(domains) - 1) / 2) * bar_width
    ax.bar(
        df["page"] + offset,
        density,
        color=color,
        alpha=0.8,
        width=bar_width,
        label=label,
    )

ax.set_xlabel("Page Number", fontsize=LABEL_FONT_SIZE)
ax.set_ylabel("Density", fontsize=LABEL_FONT_SIZE)
ax.set_xlim(0, 60)
ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_position(("outward", 5))
ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=False)

plt.tight_layout()
fig_path = figures_dir / "gt_page_distribution.pdf"
plt.savefig(fig_path, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
plt.close()
