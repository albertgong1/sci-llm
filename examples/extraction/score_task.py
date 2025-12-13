"""Score extraction predictions based on property-specific rubrics.

This script loads prediction JSON files from OUTPUT_DIR/preds, scores them
according to property-specific rubrics, and generates analysis outputs:
- OUTPUT_DIR/scores/scores__task={task}__model={model}.csv
- OUTPUT_DIR/analysis/analysis_by_material.csv
- OUTPUT_DIR/analysis/analysis_by_property.csv
- OUTPUT_DIR/figures/scores_by_material.pdf
- OUTPUT_DIR/figures/scores_by_property.pdf

Example usage:
    python score_task.py -od out --task tc --model gemini-2.5-flash
    python score_task.py -od out --task tc --model gemini-2.5-flash --rubric rubric.csv
"""

import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tabulate import tabulate

from pbench_eval.utils import scorer_categorical, scorer_pymatgen, scorer_si


def parse_numeric_value(value: str) -> float | None:
    """Parse a numeric value from a string, handling scientific notation and parenthetical uncertainties.

    Examples:
        >>> parse_numeric_value("7.441(5)")
        7.441
        >>> parse_numeric_value("3.00E+22")
        3e+22
        >>> parse_numeric_value("2.8")
        2.8

    """
    if value is None or pd.isna(value):
        return None

    value_str = str(value).strip()

    # Handle NOT_FOUND or empty strings
    if value_str.upper() == "NOT_FOUND" or value_str == "":
        return None

    # Remove parenthetical uncertainty, e.g., "7.441(5)" -> "7.441"
    value_str = re.sub(r"\(\d+\)", "", value_str)

    try:
        return float(value_str)
    except ValueError:
        return None


def load_rubric_mapping(rubric_path: Path) -> dict[str, str]:
    """Load the property_name -> rubric mapping from the rubric CSV."""
    rubric_df = pd.read_csv(rubric_path)
    return dict(zip(rubric_df["property_name"], rubric_df["rubric"]))


def score_row(pred_value: str, answer_value: str, rubric: str) -> bool | None:
    """Score a single prediction based on the rubric.

    Returns:
        True if prediction matches answer according to rubric.
        False if prediction does not match.
        None if scoring could not be performed (e.g., parse error, unknown rubric).

    """
    if rubric == "0.1% SI":
        pred_num = parse_numeric_value(pred_value)
        answer_num = parse_numeric_value(answer_value)
        if pred_num is None or answer_num is None:
            return None
        return scorer_si(pred_num, answer_num)

    elif rubric == "pymatgen":
        if pd.isna(pred_value) or pd.isna(answer_value):
            return None
        return scorer_pymatgen(str(pred_value), str(answer_value))

    elif rubric == "categorical":
        if pd.isna(pred_value) or pd.isna(answer_value):
            return None
        return scorer_categorical(str(pred_value), str(answer_value))

    else:
        # Unknown rubric
        return None


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
    for idx, record in data.items():
        # Flatten metadata into the record
        flat_record = {
            "idx": idx,
            "refno": record["refno"],
            "material": record["material"],
            "property_name": record["property_name"],
            "property_value": record["property_value"],
            "property_unit": record["property_unit"],
            "pred_value": record["pred_value"],
            "pred_unit": record["pred_unit"],
            "response": record["response"],
        }
        # Add metadata fields
        if "metadata" in record:
            for key, value in record["metadata"].items():
                flat_record[f"metadata_{key}"] = value
        records.append(flat_record)

    return pd.DataFrame(records)


def load_all_predictions(
    output_dir: Path, task: str | None = None, model: str | None = None
) -> pd.DataFrame:
    """Load predictions from JSON files in output_dir/preds.

    Args:
        output_dir: Output directory containing the preds subdirectory.
        task: Optional task name to filter files (e.g., "tc").
        model: Optional model name to filter files (e.g., "gemini-2.5-flash").

    Returns:
        DataFrame with all predictions combined.

    """
    # Load from output_dir/preds
    preds_dir = output_dir / "preds"

    if not preds_dir.exists():
        raise ValueError(f"Predictions directory not found: {preds_dir}")

    # Find all matching JSON files
    pattern_parts = []
    if task:
        pattern_parts.append(f"task={task}")
    if model:
        pattern_parts.append(f"model={model.replace('/', '--')}")

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
            f"No JSON files found in {preds_dir} matching task={task}, model={model}"
        )

    print(f"Loading {len(json_files)} JSON file(s) from {preds_dir}...")
    dfs = []
    for json_file in json_files:
        print(f"  Loading {json_file.name}")
        dfs.append(load_json_predictions(json_file))

    return pd.concat(dfs, ignore_index=True)


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

        if rubric is None:
            scores.append(None)
            continue

        pred_value = str(row["pred_value"])
        answer_value = str(row["property_value"])

        score = score_row(pred_value, answer_value, rubric)
        scores.append(score)

    # Add scores column and save
    preds_df["score"] = scores
    preds_df.to_csv(output_path, index=False)
    print(f"Scored predictions saved to: {output_path}")

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

    print("\nAnalysis saved to:")
    print(f"  - {material_csv}")
    print(f"  - {property_csv}")


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

    print("\nPlots saved to:")
    print(f"  - {material_plot_path}")
    print(f"  - {property_plot_path}")
    plt.close()


def main() -> None:
    """Main function to score predictions and generate analysis outputs."""
    parser = argparse.ArgumentParser(
        description="Score extraction predictions based on property-specific rubrics."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter by task name (e.g., 'tc'). Only used if preds_path is a directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model name (e.g., 'gemini-2.5-flash'). Only used if preds_path is a directory.",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path(__file__).parent / "rubric.csv",
        help="Path to the rubric CSV file (default: rubric.csv)",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=Path,
        default=Path("out"),
        help="Output directory (default: out)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print and save analysis tables (mean ± SE by material and property)",
    )

    args = parser.parse_args()

    # Load predictions
    preds_df = load_all_predictions(args.output_dir, args.task, args.model)
    print(f"Loaded {len(preds_df)} predictions")

    # Determine output filename based on task and model (extracted from metadata if not specified)
    if args.task is None and "metadata_task" in preds_df.columns:
        task_from_data = (
            preds_df["metadata_task"].iloc[0] if len(preds_df) > 0 else "unknown"
        )
    else:
        task_from_data = args.task

    if args.model is None and "metadata_model" in preds_df.columns:
        model_from_data = (
            preds_df["metadata_model"].iloc[0] if len(preds_df) > 0 else "unknown"
        )
    else:
        model_from_data = args.model

    task_part = f"task={task_from_data}" if task_from_data else "all-tasks"
    model_part = (
        f"model={model_from_data.replace('/', '--')}"
        if model_from_data
        else "all-models"
    )
    output_stem = f"scores__{task_part}__{model_part}"

    scores_dir = args.output_dir / "scores"
    output_path = scores_dir / f"{output_stem}.csv"

    # Create output directory if it doesn't exist
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Score predictions
    scored_df = score_predictions(preds_df, args.rubric, output_path)

    # Analyze scores by material and property
    analysis_dir = args.output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analyze_scores(scored_df, analysis_dir)

    # Plot scores by material and property
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_scores(scored_df, figures_dir)


if __name__ == "__main__":
    main()
