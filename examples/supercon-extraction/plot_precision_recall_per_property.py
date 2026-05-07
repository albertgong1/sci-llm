"""Plot precision and recall per property as bar charts.

Usage:
```bash
uv run python plot_precision_recall_per_property.py \
    --output_dir ./out-0123 \
    --rubric_path scoring/rubric_4.csv
```
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pbench

parser = ArgumentParser()
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--rubric_path",
    type=Path,
    required=True,
    help="Path to rubric CSV file (used to determine property data types)",
)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

# Load rubric to get property data types and categories
df_rubric = pd.read_csv(args.rubric_path)
non_string_properties = set(
    df_rubric[df_rubric["data_type"].isin(["Float", "Double"])][
        "property_name"
    ].tolist()
)
# Map property_name -> category and property_name -> rubric for grouping
property_to_category: dict[str, str] = dict(
    zip(df_rubric["property_name"], df_rubric["category"])
)
property_to_rubric: dict[str, str] = dict(
    zip(df_rubric["property_name"], df_rubric["rubric"])
)

# Load precision and recall per property CSVs
precision_path = args.output_dir / "precision_per_property.csv"
recall_path = args.output_dir / "recall_per_property.csv"

df_precision = pd.read_csv(precision_path)
df_recall = pd.read_csv(recall_path)

# Determine group columns (agent, model, and optionally reasoning_effort)
group_cols = ["agent", "model"]
if "reasoning_effort" in df_precision.columns:
    group_cols.append("reasoning_effort")

# Get property columns (all columns except group columns)
precision_property_cols = [c for c in df_precision.columns if c not in group_cols]
recall_property_cols = [c for c in df_recall.columns if c not in group_cols]

# Create figures directory
figures_dir = args.output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)


def parse_mean_sem(value: str) -> tuple[float, float]:
    """Parse a value like '0.63 +/- 0.03' into (mean, sem)."""
    if pd.isna(value) or value == "":
        return np.nan, np.nan
    if "+/-" in str(value):
        parts = str(value).split("+/-")
        return float(parts[0].strip()), float(parts[1].strip())
    return float(value), 0.0


def group_properties(
    property_cols: list[str],
    prop_to_group: dict[str, str],
) -> list[str]:
    """Reorder property columns so they are grouped by the given mapping."""
    grouped: dict[str, list[str]] = {}
    for prop in property_cols:
        group = prop_to_group.get(prop, "Other")
        grouped.setdefault(group, []).append(prop)
    # Flatten in group order (preserving original order within each group)
    ordered: list[str] = []
    for group in grouped:
        ordered.extend(grouped[group])
    return ordered


def plot_metric_per_property(
    df: pd.DataFrame,
    property_cols: list[str],
    metric_name: str,
    output_path: Path,
    non_string_props: set[str],
    prop_to_group: dict[str, str],
) -> None:
    """Plot horizontal bar chart of metric per property, visually grouped."""
    # Reorder properties by group
    property_cols = group_properties(property_cols, prop_to_group)

    n_groups = len(df)
    n_props = len(property_cols)
    fig_height = max(6, n_props * 0.5)
    fig, ax = plt.subplots(figsize=(6.75, fig_height))

    y = np.arange(n_props)
    height = 0.8 / n_groups

    for i, (_, row) in enumerate(df.iterrows()):
        # Create label from group columns
        label_parts = [str(row[col]) for col in group_cols]
        label = " / ".join(label_parts)

        means = []
        sems = []
        for prop in property_cols:
            mean, sem = parse_mean_sem(row[prop])
            means.append(mean)
            sems.append(sem)

        offset = (i - n_groups / 2 + 0.5) * height
        ax.barh(
            y + offset,
            means,
            height,
            label=label,
            xerr=sems,
            capsize=2,
            error_kw={"alpha": 0.5},
        )

    ax.set_ylabel("")
    ax.set_xlabel(metric_name.capitalize())
    ax.set_yticks(y)
    ax.set_yticklabels(property_cols)

    # Bold labels for non-string properties
    for label in ax.get_yticklabels():
        if label.get_text() in non_string_props:
            label.set_fontweight("bold")

    # Draw horizontal separators and group labels between groups
    categories = [prop_to_group.get(p, "Other") for p in property_cols]
    group_start = 0
    for j in range(1, len(categories) + 1):
        if j == len(categories) or categories[j] != categories[group_start]:
            # Draw separator line at group boundary
            if j < len(categories):
                ax.axhline(
                    y=j - 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5
                )
            # Place category label at the center of the group
            mid = (group_start + j - 1) / 2.0
            ax.text(
                1.02,
                mid / (n_props - 1) if n_props > 1 else 0.5,
                categories[group_start],
                ha="left",
                va="center",
                fontsize=7,
                fontstyle="italic",
                color="gray",
                transform=ax.get_yaxis_transform(),
            )
            group_start = j

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {metric_name} plot to {output_path}")


# Plot precision per property
plot_metric_per_property(
    df_precision,
    precision_property_cols,
    "precision",
    figures_dir / "precision_per_property.pdf",
    non_string_properties,
    property_to_category,
)

# Plot recall per property
plot_metric_per_property(
    df_recall,
    recall_property_cols,
    "recall",
    figures_dir / "recall_per_property.pdf",
    non_string_properties,
    property_to_category,
)

# Plot precision per property grouped by rubric
plot_metric_per_property(
    df_precision,
    precision_property_cols,
    "precision",
    figures_dir / "precision_per_property_by_rubric.pdf",
    non_string_properties,
    property_to_rubric,
)

# Plot recall per property grouped by rubric
plot_metric_per_property(
    df_recall,
    recall_property_cols,
    "recall",
    figures_dir / "recall_per_property_by_rubric.pdf",
    non_string_properties,
    property_to_rubric,
)
