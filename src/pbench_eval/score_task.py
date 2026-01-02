#! /usr/bin/env -S uv run --env-file=.env -- python
"""Score extraction predictions based on property-specific rubrics.

This script loads prediction JSON files from <output_dir>/<domain>/preds/*.json,
scores them according to property-specific rubrics, and generates analysis outputs:
- <output_dir>/<domain>/scores/scores__task={task}__model={model}.csv
- <output_dir>/<domain>/analysis/analysis_by_material.csv
- <output_dir>/<domain>/analysis/analysis_by_property.csv
- <output_dir>/<domain>/figures/scores_by_material.pdf
- <output_dir>/<domain>/figures/scores_by_property.pdf

Usage:
```bash
./src/pbench_eval/score_task.py \
    --domain supercon \
    --task tc \
    --model_name gemini-2.5-flash \
    -od out/
```
"""

import argparse
import json
import re
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import pbench
from pbench_eval.utils import scorer_categorical, scorer_pymatgen, scorer_si, score_value

logger = logging.getLogger(__name__)

# Load clusters globally for the categorical scorer (clusters of "method of X" properties)
# This script utilizes `property_clusters.json` (if present) to map values in
# "method of X" properties to canonical categories.
CLUSTER_FILE = Path(__file__).parent / "assets" / "property_clusters.json"
CLUSTERS = {}
if CLUSTER_FILE.exists():
    try:
        with open(CLUSTER_FILE, "r") as f:
            CLUSTERS = json.load(f)
        logger.info(f"Loaded property clusters from {CLUSTER_FILE}")
    except Exception as e:
        logger.warning(f"Failed to load property clusters: {e}")
else:
    logger.warning(f"Property clusters file not found at {CLUSTER_FILE}")


def score_row(
    pred_value: str, answer_value: str, rubric: str, property_name: str | None = None
) -> float | None:
    """Score a single prediction based on the rubric.

    Returns:
        1.0 if prediction matches answer according to rubric (0.0 otherwise).
        None if scoring could not be performed (e.g., missing values).
    """
    if pd.isna(pred_value) or pd.isna(answer_value):
        return None

    # Get specific mapping for this property if available
    mapping = CLUSTERS.get(property_name, None) if property_name else None
    
    return score_value(
        str(pred_value), str(answer_value), rubric=rubric, mapping=mapping
    )


def load_json_predictions(json_path: Path) -> pd.DataFrame:
    """Load predictions from a single JSON file.

    Args:
        json_path: Path to the JSON prediction file.

    Returns:
        DataFrame with predictions.

    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert to list of dictionaries (flattening the structure)
    records = []
    for record in data:
        # Flatten metadata into the record
        flat_record = {
            "refno": record["refno"],
            "material": record["material"],
            "property_name": record["property_name"],
            "property_value": record["true"]["value"],
            "property_unit": record["true"]["unit"],
            "pred_value": record["pred"]["value"],
            "pred_unit": record["pred"]["unit"],
            "response": record["response"]["pred"],
            "rubric": record.get("rubric"),  # Extract rubric if present
        }
        # Add metadata fields
        if "metadata" in record:
            for key, value in record["metadata"].items():
                flat_record[f"metadata_{key}"] = value
        records.append(flat_record)

    return pd.DataFrame(records)


def load_all_predictions(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Load predictions from JSON files in <output_dir>/<domain>/preds/*.json.
    Performs filtering by task, split, and model name if specified.

    Returns:
        DataFrame with all predictions combined.

    """
    preds_dir = args.output_dir / args.domain / "preds"

    if not preds_dir.exists():
        raise ValueError(f"Predictions directory not found: {preds_dir}")

    # Find all matching JSON files
    pattern_parts = []
    if args.task:
        pattern_parts.append(f"task={args.task}")
    if args.split:
        pattern_parts.append(f"split={args.split}")
    if args.model_name:
        pattern_parts.append(f"model={args.model_name.replace('/', '--')}")

    if pattern_parts:
        # Filter by task and/or model
        all_jsons = list(preds_dir.glob("*.json"))
        filtered_jsons = []
        for json_file in all_jsons:
            filename = json_file.stem
            if all(part in filename for part in pattern_parts):
                filtered_jsons.append(json_file)
        json_files = sorted(filtered_jsons)
    else:
        # Load all JSON files
        json_files = sorted(preds_dir.glob("*.json"))

    if not json_files:
        raise ValueError(
            f"No JSON files found in {preds_dir} matching task={args.task}, split={args.split}, model={args.model_name}"
        )

    logger.info(f"Loading {len(json_files)} JSON file(s) from {preds_dir}...")
    dfs = []
    for json_file in json_files:
        logger.info(f"  Loading {json_file.name}")
        dfs.append(load_json_predictions(json_file))

    return pd.concat(dfs, ignore_index=True)


def load_rubric_mapping(rubric_path: Path) -> dict[str, str]:
    """Load the rubric mapping from a CSV file.

    Args:
        rubric_path: Path to the rubric CSV file.

    Returns:
        Dictionary mapping property name to rubric name.

    """
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found at {rubric_path}")

    df = pd.read_csv(rubric_path)
    if "property_name" not in df.columns or "rubric" not in df.columns:
        raise ValueError(
            f"Rubric CSV at {rubric_path} must contain 'property_name' and 'rubric' columns."
        )

    return dict(zip(df["property_name"], df["rubric"]))

def score_predictions(
    preds_df: pd.DataFrame, rubric_path: Path, output_path: Path
) -> pd.DataFrame:
    """Score all predictions and write results to output CSV.

    Args:
        preds_df: DataFrame with predictions.
        rubric_path: Path to the rubric CSV file.
        output_path: Path to save the scored predictions.

    Returns:
        DataFrame with scores added.

    """
    rubric_mapping = load_rubric_mapping(rubric_path)

    # Score each row
    scores = []
    for _, row in preds_df.iterrows():
        property_name = str(row["property_name"])
        rubric = rubric_mapping.get(property_name)

        if rubric is None and "rubric" in row:
             rubric = row["rubric"]

        if rubric is None or pd.isna(rubric):
            scores.append(None)
            continue

        pred_value = str(row["pred_value"])
        answer_value = str(row["property_value"])

        score = score_row(pred_value, answer_value, rubric, property_name=property_name)
        scores.append(score)

    # Add scores column and save
    preds_df["score"] = scores
    preds_df.to_csv(output_path, index=False)
    logger.info(f"Scored predictions saved to: {output_path}")

    return preds_df


def compute_mean_se(scores: pd.Series) -> tuple[float, float, int]:
    """Compute mean, standard error, and count for a group of scores.

    Returns:
        Tuple of (mean, standard_error, count).

    """
    # Filter out None values
    valid_scores = scores.dropna().astype(float)
    n = len(valid_scores)

    if n == 0:
        return float("nan"), float("nan"), 0

    mean_val = float(valid_scores.mean())
    se_val = float(valid_scores.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    return mean_val, se_val, n


def analyze_scores(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze scores by material and property, print tables and save to CSV.

    Args:
        df: DataFrame with 'material', 'property_name', and 'score' columns.
        output_dir: Directory to save the analysis CSV files.

    """
    # Analysis by material
    material_stats = []
    for material, group in df.groupby("material"):
        mean, se, n = compute_mean_se(group["score"])
        material_stats.append(
            {
                "material": material,
                "mean": mean,
                "se": se,
                "n": n,
                "mean ± se": f"{mean:.3f} ± {se:.3f}" if not np.isnan(mean) else "N/A",
            }
        )

    material_df = pd.DataFrame(material_stats)
    material_df = material_df.sort_values("mean", ascending=False, na_position="last")

    # Analysis by property
    property_stats = []
    for prop, group in df.groupby("property_name"):
        mean, se, n = compute_mean_se(group["score"])
        property_stats.append(
            {
                "property_name": prop,
                "mean": mean,
                "se": se,
                "n": n,
                "mean ± se": f"{mean:.3f} ± {se:.3f}" if not np.isnan(mean) else "N/A",
            }
        )

    property_df = pd.DataFrame(property_stats)
    property_df = property_df.sort_values("mean", ascending=False, na_position="last")

    # Print markdown tables
    print("\n" + "=" * 60)
    print("ANALYSIS: Mean Score by Material")
    print("=" * 60)
    print(
        tabulate(
            material_df[["material", "mean ± se", "n"]],
            headers=["Material", "Mean ± SE", "N"],
            tablefmt="github",
            showindex=False,
        )
    )

    print("\n" + "=" * 60)
    print("ANALYSIS: Mean Score by Property")
    print("=" * 60)
    print(
        tabulate(
            property_df[["property_name", "mean ± se", "n"]],
            headers=["Property", "Mean ± SE", "N"],
            tablefmt="github",
            showindex=False,
        )
    )

    # Save to CSV
    material_csv = output_dir / "analysis_by_material.csv"
    property_csv = output_dir / "analysis_by_property.csv"

    material_df.to_csv(material_csv, index=False)
    property_df.to_csv(property_csv, index=False)

    logger.info("Analysis saved to:")
    logger.info(f"  - {material_csv}")
    logger.info(f"  - {property_csv}")


def plot_scores(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean scores by material and property with SE error bars."""
    # Filter out rows with None scores
    df_valid = df.dropna(subset=["score"])

    # Plot mean scores by material (sorted by score descending)
    material_stats = (
        df_valid.groupby("material")["score"]
        .agg(["mean", "sem"])
        .sort_values("mean", ascending=False)
    )
    plt.figure(figsize=(10, 6))
    plt.bar(
        material_stats.index,
        material_stats["mean"],
        yerr=material_stats["sem"],
        capsize=3,
    )
    plt.xlabel("Material")
    plt.ylabel("Mean Score")
    plt.title("Mean Scores by Material")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    material_plot_path = output_dir / "scores_by_material.pdf"
    plt.savefig(material_plot_path)
    plt.close()

    # Plot mean scores by property (sorted by score descending)
    property_stats = (
        df_valid.groupby("property_name")["score"]
        .agg(["mean", "sem"])
        .sort_values("mean", ascending=False)
    )
    plt.figure(figsize=(12, 6))
    plt.bar(
        property_stats.index,
        property_stats["mean"],
        yerr=property_stats["sem"],
        capsize=3,
    )
    plt.xlabel("Property")
    plt.ylabel("Mean Score")
    plt.title("Mean Scores by Property")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    property_plot_path = output_dir / "scores_by_property.pdf"
    plt.savefig(property_plot_path)

    logger.info("Plots saved to:")
    logger.info(f"  - {material_plot_path}")
    logger.info(f"  - {property_plot_path}")
    plt.close()


def main() -> None:
    """Main function to score predictions and generate analysis outputs."""
    parser = argparse.ArgumentParser(
        description="Score extraction predictions based on property-specific rubrics."
    )
    parser = pbench.add_base_args(parser)
    parser.add_argument(
        "--rubric_csv_filename",
        type=str,
        default="rubric.csv",
        help="Filename of the rubric CSV file (default: rubric.csv)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print and save analysis tables (mean ± SE by material and property)",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    # Load predictions
    preds_df = load_all_predictions(args)
    logger.info(f"Loaded {len(preds_df)} predictions")

    # Determine output filename based on task, split, and model (extracted from metadata if not specified)
    if args.task is None and "metadata_task" in preds_df.columns:
        task_from_data = (
            preds_df["metadata_task"].iloc[0] if len(preds_df) > 0 else "unknown"
        )
    else:
        task_from_data = args.task

    if args.split is None and "metadata_split" in preds_df.columns:
        split_from_data = (
            preds_df["metadata_split"].iloc[0] if len(preds_df) > 0 else "unknown"
        )
    else:
        split_from_data = args.split

    if args.model_name is None and "metadata_model" in preds_df.columns:
        model_from_data = (
            preds_df["metadata_model"].iloc[0] if len(preds_df) > 0 else "unknown"
        )
    else:
        model_from_data = args.model_name

    task_part = f"task={task_from_data}" if task_from_data else "all-tasks"
    split_part = f"split={split_from_data}" if split_from_data else "all-splits"
    model_part = (
        f"model={model_from_data.replace('/', '--')}"
        if model_from_data
        else "all-models"
    )
    output_stem = f"scores__{task_part}__{split_part}__{model_part}"

    scores_dir = args.output_dir / args.domain / "scores"
    output_path = scores_dir / f"{output_stem}.csv"

    # Create output directory if it doesn't exist
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Score predictions
    rubric_path = pbench.ASSETS_DIR / args.domain / args.rubric_csv_filename
    scored_df = score_predictions(preds_df, rubric_path, output_path)

    # Analyze scores by material and property
    analysis_dir = args.output_dir / args.domain / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analyze_scores(scored_df, analysis_dir)

    # Plot scores by material and property
    figures_dir = args.output_dir / args.domain / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_scores(scored_df, figures_dir)


if __name__ == "__main__":
    # Configure logging only when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main()
