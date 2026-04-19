"""Compare property frequencies between supercon-extraction and supercon-post-2021-extraction datasets.

Usage:
    python compare_dataset_properties.py
    python compare_dataset_properties.py --output_dir results/
"""

from datasets import load_dataset
from argparse import ArgumentParser
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import pbench
from utils import (
    TICK_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
)

# Dataset configurations
DATASETS = {
    "supercon": {
        "name": "kilian-group/supercon-extraction",
        "split": "full",
        "revision": "main",
    },
    "supercon-post-2021": {
        "name": "kilian-group/supercon-post-2021-extraction",
        "split": "full",
        "revision": "main",
    },
}

parser = ArgumentParser(
    description="Compare property frequencies between supercon datasets"
)
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--top_n",
    type=int,
    default=30,
    help="Number of top properties to display in the comparison",
)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

logger = logging.getLogger(__name__)

output_dir = args.output_dir or Path(".")
output_dir.mkdir(parents=True, exist_ok=True)


def load_and_process_dataset(config: dict[str, str]) -> pd.DataFrame:
    """Load a HuggingFace dataset and extract property information.

    Args:
        config: Dict with keys 'name', 'split', 'revision'

    Returns:
        DataFrame with exploded properties and property_name column

    """
    logger.info(
        f"Loading dataset: {config['name']} "
        f"(revision={config['revision']}, split={config['split']})"
    )
    dataset = load_dataset(
        config["name"], split=config["split"], revision=config["revision"]
    )
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df)} documents")

    # Explode properties to get one row per property
    df = df.explode(column="properties").reset_index(drop=True)
    # Normalize the properties JSON into columns
    df = pd.concat(
        [df.drop(columns=["properties"]), pd.json_normalize(df["properties"])], axis=1
    )
    logger.info(f"Total properties: {len(df)}")
    return df


# Load both datasets
dfs: dict[str, pd.DataFrame] = {}
for dataset_key, config in DATASETS.items():
    dfs[dataset_key] = load_and_process_dataset(config)

# Compute property frequencies for each dataset
property_counts: dict[str, pd.Series] = {}
for dataset_key, df in dfs.items():
    counts = df["property_name"].value_counts()
    property_counts[dataset_key] = counts
    logger.info(f"{dataset_key}: {len(counts)} unique property names")

# Create a combined dataframe with counts from both datasets
all_properties = set()
for counts in property_counts.values():
    all_properties.update(counts.index)

comparison_data = []
for prop in all_properties:
    row = {"property_name": prop}
    for dataset_key in DATASETS.keys():
        row[f"{dataset_key}_count"] = property_counts[dataset_key].get(prop, 0)
        row[f"{dataset_key}_pct"] = (
            property_counts[dataset_key].get(prop, 0) / len(dfs[dataset_key]) * 100
        )
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

# Sort by supercon percentage
df_comparison = df_comparison.sort_values("supercon_pct", ascending=False)

# Save full comparison to CSV
csv_path = output_dir / "property_frequency_comparison.csv"
df_comparison.to_csv(csv_path, index=False)
logger.info(f"Saved full comparison to {csv_path}")

# Print summary statistics
print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
for dataset_key, df in dfs.items():
    n_docs = df["refno"].nunique()
    n_props = len(df)
    n_unique_props = df["property_name"].nunique()
    print(f"\n{dataset_key}:")
    print(f"  Documents: {n_docs}")
    print(f"  Total properties: {n_props}")
    print(f"  Unique property names: {n_unique_props}")
    print(f"  Avg properties per doc: {n_props / n_docs:.1f}")

# Print top N properties comparison (normalized)
print("\n" + "=" * 80)
print(f"TOP {args.top_n} PROPERTIES BY SUPERCON PERCENTAGE")
print("=" * 80)
print(f"\n{'Property Name':<50} {'supercon %':>12} {'post-2021 %':>12}")
print("-" * 74)
for _, row in df_comparison.head(args.top_n).iterrows():
    print(
        f"{row['property_name']:<50} "
        f"{row['supercon_pct']:>11.2f}% "
        f"{row['supercon-post-2021_pct']:>11.2f}%"
    )

# Identify properties unique to each dataset
supercon_only = df_comparison[
    (df_comparison["supercon_count"] > 0)
    & (df_comparison["supercon-post-2021_count"] == 0)
]
post2021_only = df_comparison[
    (df_comparison["supercon_count"] == 0)
    & (df_comparison["supercon-post-2021_count"] > 0)
]
shared = df_comparison[
    (df_comparison["supercon_count"] > 0)
    & (df_comparison["supercon-post-2021_count"] > 0)
]

print("\n" + "=" * 80)
print("PROPERTY OVERLAP ANALYSIS")
print("=" * 80)
print(f"\nProperties only in supercon: {len(supercon_only)}")
print(f"Properties only in supercon-post-2021: {len(post2021_only)}")
print(f"Properties in both datasets: {len(shared)}")

if len(supercon_only) > 0:
    print("\nTop properties unique to supercon:")
    for _, row in supercon_only.head(10).iterrows():
        print(f"  - {row['property_name']} ({int(row['supercon_count'])} occurrences)")

if len(post2021_only) > 0:
    print("\nTop properties unique to supercon-post-2021:")
    for _, row in post2021_only.head(10).iterrows():
        print(
            f"  - {row['property_name']} ({int(row['supercon-post-2021_count'])} occurrences)"
        )

# Create visualization
fig_dir = output_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Bar chart comparing top properties (normalized)
# fig, ax = plt.subplots(figsize=(3.25, 2))
fig, ax = plt.subplots(figsize=(6.75, 2))
top_props = df_comparison.head(args.top_n)
x = range(len(top_props))
width = 0.35

bars1 = ax.bar(
    [i - width / 2 for i in x],
    top_props["supercon_pct"],
    width,
    label="supercon",
    color="steelblue",
)
bars2 = ax.bar(
    [i + width / 2 for i in x],
    top_props["supercon-post-2021_pct"],
    width,
    label="supercon-post-2021",
    color="coral",
)

ax.set_xlabel("Property Name", fontsize=LABEL_FONT_SIZE)
ax.set_ylabel("Percentage (%)", fontsize=LABEL_FONT_SIZE)
ax.set_title(
    f"Top {args.top_n} Property Frequencies (Normalized)",
    fontsize=LABEL_FONT_SIZE,
)
ax.set_xticks(x)
ax.set_xticklabels(top_props["property_name"], fontsize=TICK_FONT_SIZE)
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")
    label.set_rotation_mode("anchor")
ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
ax.legend(fontsize=LEGEND_FONT_SIZE)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.spines["left"].set_position(("outward", OUTWARD))
# ax.spines["bottom"].set_position(("outward", OUTWARD))

plt.tight_layout()
fig_path = fig_dir / "property_frequency_comparison.pdf"
plt.savefig(fig_path, bbox_inches="tight")
logger.info(f"Saved figure to {fig_path}")
plt.close()

print(f"\nResults saved to: {output_dir}")
print(f"  - {csv_path}")
print(f"  - {fig_path}")
