"""Score extraction predictions based on property-specific rubrics.

Example usage:
    python score.py preds__gemini-2.5-flash.csv
    python score.py preds__gemini-2.5-flash.csv --output my_scores.csv
    python score.py preds__gemini-2.5-flash.csv --rubric data/rubric.csv
    python score.py preds__gemini-2.5-flash.csv --analyze
"""

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tabulate import tabulate

from utils import scorer_categorical, scorer_pymatgen, scorer_si


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


def score_predictions(preds_path: Path, rubric_path: Path, output_path: Path) -> pd.DataFrame:
    """Score all predictions and write results to output CSV.
    
    Returns:
        DataFrame with scores added.
    """
    # Load data
    preds_df = pd.read_csv(preds_path)
    rubric_mapping = load_rubric_mapping(rubric_path)
    
    # Score each row
    scores = []
    for _, row in preds_df.iterrows():
        property_name = str(row["property_name"])
        rubric = rubric_mapping.get(property_name)
        
        if rubric is None:
            scores.append(None)
            continue
        
        pred_value = str(row["gemini-2.5-flash-pred-value"])
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
        material_stats.append({
            "material": material,
            "mean": mean,
            "se": se,
            "n": n,
            "mean ± se": f"{mean:.3f} ± {se:.3f}" if not np.isnan(mean) else "N/A",
        })
    
    material_df = pd.DataFrame(material_stats)
    material_df = material_df.sort_values("mean", ascending=False, na_position="last")
    
    # Analysis by property
    property_stats = []
    for prop, group in df.groupby("property_name"):
        mean, se, n = compute_mean_se(group["score"])
        property_stats.append({
            "property_name": prop,
            "mean": mean,
            "se": se,
            "n": n,
            "mean ± se": f"{mean:.3f} ± {se:.3f}" if not np.isnan(mean) else "N/A",
        })
    
    property_df = pd.DataFrame(property_stats)
    property_df = property_df.sort_values("mean", ascending=False, na_position="last")
    
    # Print markdown tables
    print("\n" + "=" * 60)
    print("ANALYSIS: Mean Score by Material")
    print("=" * 60)
    print(tabulate(
        material_df[["material", "mean ± se", "n"]],
        headers=["Material", "Mean ± SE", "N"],
        tablefmt="github",
        showindex=False,
    ))
    
    print("\n" + "=" * 60)
    print("ANALYSIS: Mean Score by Property")
    print("=" * 60)
    print(tabulate(
        property_df[["property_name", "mean ± se", "n"]],
        headers=["Property", "Mean ± SE", "N"],
        tablefmt="github",
        showindex=False,
    ))
    
    # Save to CSV
    material_csv = output_dir / "analysis_by_material.csv"
    property_csv = output_dir / "analysis_by_property.csv"
    
    material_df.to_csv(material_csv, index=False)
    property_df.to_csv(property_csv, index=False)
    
    print(f"\nAnalysis saved to:")
    print(f"  - {material_csv}")
    print(f"  - {property_csv}")


def plot_scores(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean scores by material and property with SE error bars."""
    # Filter out rows with None scores
    df_valid = df.dropna(subset=["score"])
    
    # Plot mean scores by material
    material_stats = df_valid.groupby("material")["score"].agg(["mean", "sem"])
    plt.figure(figsize=(10, 6))
    plt.bar(material_stats.index, material_stats["mean"], yerr=material_stats["sem"], capsize=3)
    plt.xlabel("Material")
    plt.ylabel("Mean Score")
    plt.title("Mean Scores by Material")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    material_plot_path = output_dir / "scores_by_material.pdf"
    plt.savefig(material_plot_path)
    plt.close()

    # Plot mean scores by property
    property_stats = df_valid.groupby("property_name")["score"].agg(["mean", "sem"])
    plt.figure(figsize=(12, 6))
    plt.bar(property_stats.index, property_stats["mean"], yerr=property_stats["sem"], capsize=3)
    plt.xlabel("Property")
    plt.ylabel("Mean Score")
    plt.title("Mean Scores by Property")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    property_plot_path = output_dir / "scores_by_property.pdf"
    plt.savefig(property_plot_path)
    
    print(f"\nPlots saved to:")
    print(f"  - {material_plot_path}")
    print(f"  - {property_plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Score extraction predictions based on property-specific rubrics.")
    parser.add_argument(
        "preds_csv",
        type=Path,
        help="Path to the predictions CSV file",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path(__file__).parent / "data" / "rubric.csv",
        help="Path to the rubric CSV file (default: data/rubric.csv)",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=Path,
        default=Path("out"),
        help="Path for output CSV (default: out)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print and save analysis tables (mean ± SE by material and property)",
    )
    
    args = parser.parse_args()
    
    # Determine output path
    input_stem = args.preds_csv.stem
    # Replace 'preds__' prefix with 'scores__' if present
    if input_stem.startswith("preds__"):
        output_stem = input_stem.replace("preds__", "scores__", 1)
    else:
        output_stem = f"scores__{input_stem}"
    scores_dir = args.output_dir / "scores"
    output_path = scores_dir / f"{output_stem}.csv"
    
    # Create output directory if it doesn't exist
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    # Score predictions
    scored_df = score_predictions(args.preds_csv, args.rubric, output_path)
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

