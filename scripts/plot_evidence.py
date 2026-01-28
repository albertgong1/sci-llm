"""Script to plot evidence metrics (precision, recall, F1) as separate bar plot figures.

Usage:
    python scripts/plot_evidence.py

Generates three figures:
    - evidence_precision_bars.pdf
    - evidence_recall_bars.pdf
    - evidence_f1_bars.pdf

"""

import pbench

import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path

from pbench_eval.plotting_utils import (
    TICK_FONT_SIZE,
    LABEL_FONT_SIZE,
)

# Agent -> color mapping for consistent visualization
AGENT_COLORS: dict[str, str] = {
    "zeroshot": "#555555",  # gray
    "codex": "#2ca02c",  # green
    "gemini-cli": "#ff7f0e",  # orange
    "terminus-2": "#d62728",  # red
}

# Domain -> display name mapping for plot titles
DOMAIN_ALIASES: dict[str, str] = {
    "supercon": "SuperCon",
    "biosurfactants": "Biosurfactants",
    "cdw": "CDW",
}

# Agent -> legend display name mapping
DISPLAY_NAMES: dict[str, str] = {
    "zeroshot": "Zeroshot",
    "codex": "Codex",
    "gemini-cli": "Gemini CLI",
    "terminus-2": "Terminus-2",
}

# Metrics to plot
METRICS = [
    {
        "name": "precision",
        "column": "avg_evidence_precision",
        "sem_column": "avg_evidence_precision_sem",
        "label": "Evidence Precision",
    },
    {
        "name": "recall",
        "column": "avg_evidence_recall",
        "sem_column": "avg_evidence_recall_sem",
        "label": "Evidence Recall",
    },
    {
        "name": "f1",
        "column": "avg_evidence_f1",
        "sem_column": "avg_evidence_f1_sem",
        "label": "Evidence F1",
    },
]

parser = ArgumentParser(
    description="Plot evidence metrics as bar plots for each domain."
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()

pbench.setup_logging(args.log_level)

# Read evidence summary from examples subdirectories
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
all_data: dict[str, list[dict]] = {"supercon": [], "biosurfactants": []}
for domain, output_dir in output_dirs:
    evidence_path = output_dir / "tables" / "evidence_summary.csv"
    if not evidence_path.exists():
        print(f"Warning: {evidence_path} not found, skipping")
        continue
    df = pd.read_csv(evidence_path)
    for _, row in df.iterrows():
        all_data[domain].append(
            {
                "agent": row["agent"],
                "model": row["model"],
                "avg_evidence_f1": row["avg_evidence_f1"],
                "avg_evidence_f1_sem": row.get("avg_evidence_f1_sem", 0),
                "avg_evidence_precision": row["avg_evidence_precision"],
                "avg_evidence_precision_sem": row.get("avg_evidence_precision_sem", 0),
                "avg_evidence_recall": row["avg_evidence_recall"],
                "avg_evidence_recall_sem": row.get("avg_evidence_recall_sem", 0),
            }
        )

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

domains = ["supercon", "biosurfactants"]

# Generate a separate figure for each metric
for metric in METRICS:
    metric_name = metric["name"]
    metric_column = metric["column"]
    metric_sem_column = metric["sem_column"]
    metric_label = metric["label"]

    fig, axs = plt.subplots(1, len(domains), figsize=(6, 3), sharey=False)

    for i, domain in enumerate(domains):
        ax = axs[i]
        data = all_data[domain]
        if not data:
            ax.set_title(DOMAIN_ALIASES.get(domain, domain), fontsize=LABEL_FONT_SIZE)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        df = pd.DataFrame(data)

        # Create labels combining agent and model
        df["label"] = df.apply(
            lambda row: f"{row['agent']}\n({row['model'].split('/')[-1][:15]})", axis=1
        )

        # Sort by the current metric descending
        df = df.sort_values(metric_column, ascending=True)

        # Create horizontal bar plot
        y_pos = range(len(df))
        colors = [AGENT_COLORS.get(row["agent"], "#333333") for _, row in df.iterrows()]

        ax.barh(
            y_pos,
            df[metric_column],
            xerr=df[metric_sem_column],
            color=colors,
            alpha=0.8,
            capsize=3,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["label"], fontsize=TICK_FONT_SIZE - 1)
        ax.set_xlabel(metric_label, fontsize=LABEL_FONT_SIZE)
        ax.set_title(DOMAIN_ALIASES.get(domain, domain), fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_name = f"evidence_{metric_name}_bars.pdf"
    fig_path = figures_dir / fig_name
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")
    plt.close()
