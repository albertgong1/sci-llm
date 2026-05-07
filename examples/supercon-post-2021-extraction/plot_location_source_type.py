"""Plot the distribution of location.source_type per property in the GT dataset.

Usage:
    uv run python examples/supercon-extraction/plot_location_source_type.py \
        --hf_repo <repo> --hf_split <split>
"""

from argparse import ArgumentParser
import logging
from pathlib import Path

from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

import pbench
from pbench_eval.plotting_utils import LABEL_FONT_SIZE, LEGEND_FONT_SIZE, TICK_FONT_SIZE

logger = logging.getLogger(__name__)

parser = ArgumentParser(
    description="Plot location source type distribution per property."
)
parser = pbench.add_base_args(parser)

args = parser.parse_args()
pbench.setup_logging(args.log_level)

# Load ground truth from HuggingFace
hf_dataset_name = args.hf_repo
hf_split_name = args.hf_split
hf_revision = args.hf_revision or "main"

logger.info(
    f"Loading dataset from HuggingFace: {hf_dataset_name} "
    f"(revision={hf_revision}, split={hf_split_name})"
)
dataset = load_dataset(hf_dataset_name, split=hf_split_name, revision=hf_revision)
df_gt: pd.DataFrame = dataset.to_pandas()

# Explode properties to get one row per property
df_gt = df_gt.explode(column="properties").reset_index(drop=True)
df_gt = pd.concat([df_gt[["refno"]], pd.json_normalize(df_gt["properties"])], axis=1)
logger.info(f"Loaded {len(df_gt)} ground truth rows")

# Group by property_name and calculate percentage breakdown of source types
source_type_col = "location.source_type"
counts = df_gt.groupby(["property_name", source_type_col]).size().unstack(fill_value=0)
pct: pd.DataFrame = counts.div(counts.sum(axis=1), axis=0) * 100

# Sort properties by total count (descending)
property_order = counts.sum(axis=1).sort_values(ascending=True).index
pct = pct.loc[property_order]

# Define consistent colors and column order for source types
source_type_order = ["figure", "table", "caption", "text"]
source_type_colors: dict[str, str] = {
    "figure": "#C44E52",
    "table": "#DD8452",
    "caption": "#55A868",
    "text": "#4C72B0",
}
# Reorder columns to match desired order (keep only columns that exist)
ordered_cols = [col for col in source_type_order if col in pct.columns]
pct = pct[ordered_cols]
colors: dict[str, str] = {
    col: source_type_colors.get(col, "#999999") for col in pct.columns
}

# Create stacked horizontal bar plot
fig, ax = plt.subplots(figsize=(6.75, max(2.5, len(pct) * 0.35)))
pct.plot.barh(stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)

ax.set_xlabel("Percentage (%)", fontsize=LABEL_FONT_SIZE)
ax.set_ylabel("")
ax.set_xlim(0, 100)
ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
ax.legend(
    title="Source type",
    fontsize=LEGEND_FONT_SIZE,
    title_fontsize=LEGEND_FONT_SIZE,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=len(pct.columns),
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()

# Save figure
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / "location_source_type_distribution.pdf"
plt.savefig(fig_path, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
plt.close()
