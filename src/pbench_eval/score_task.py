#! /usr/bin/env -S uv run --env-file=.env -- python
"""Score extraction predictions or analyze existing Harbor results.

This script can:
1. Load prediction JSON files and score them (Legacy/Development mode).
2. Load an existing CSV of scored results (Harbor mode) and generate analysis tables.

Analysis outputs:
- <output_dir>/<domain>/analysis/analysis_by_material.csv
- <output_dir>/<domain>/analysis/analysis_by_property.csv

Usage:
```bash
python src/pbench_eval/score_task.py \
    --domain precedent-search \
    --input_csv out/harbor/precedent-search/results.csv \
    --analyze
```
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

# Hack: Ensure project root is in sys.path if pbench is not installed
try:
    import pbench
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root / "src"))
    import pbench

from pbench_eval.utils import score_value

logger = logging.getLogger(__name__)

# Updated path for property clusters
CLUSTER_FILE = Path("examples/supercon-extraction/scoring/property_clusters_gemini-3-pro-preview.json")

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
    """Score a single prediction based on the rubric."""
    if pd.isna(pred_value) or pd.isna(answer_value):
        return None

    # Get specific mapping for this property if available
    mapping = CLUSTERS.get(property_name, None) if property_name else None
    
    return score_value(
        str(pred_value), str(answer_value), rubric=rubric, mapping=mapping
    )


def compute_mean_se(scores: pd.Series) -> tuple[float, float, int]:
    """Compute mean, standard error, and count for a group of scores."""
    valid_scores = scores.dropna().astype(float)
    n = len(valid_scores)

    if n == 0:
        return float("nan"), float("nan"), 0

    mean_val = float(valid_scores.mean())
    se_val = float(valid_scores.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    return mean_val, se_val, n


def analyze_scores(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze scores by material and property, print tables and save to CSV."""
    if "score" not in df.columns:
        logger.error("DataFrame must contain 'score' column for analysis.")
        return

    # Analysis by material
    material_stats = []
    if "material" in df.columns:
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
        material_csv = output_dir / "analysis_by_material.csv"
        material_df.to_csv(material_csv, index=False)

    # Analysis by property
    property_stats = []
    prop_col = "property_name" if "property_name" in df.columns else "property"
    if prop_col in df.columns:
        for prop, group in df.groupby(prop_col):
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
        property_csv = output_dir / "analysis_by_property.csv"
        property_df.to_csv(property_csv, index=False)


def load_json_predictions(json_path: Path) -> pd.DataFrame:
    """Load predictions from a single JSON file."""
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
    """Load predictions from JSON files in <output_dir>/<domain>/preds/*.json."""
    preds_dir = args.output_dir / args.domain / "preds"

    combined_csv = args.output_dir / args.domain / "results.csv"
    if combined_csv.exists():
        logger.info(f"Loading pre-combined results from {combined_csv}")
        return pd.read_csv(combined_csv)

    if not preds_dir.exists():
         # If input_csv was provided, main() handles it. Here we just fail if directory missing.
        raise ValueError(f"Predictions directory not found: {preds_dir}")

    # Load all JSON files (Harbor generates one per trial)
    json_files = sorted(preds_dir.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {preds_dir}")

    logger.info(f"Loading {len(json_files)} JSON file(s) from {preds_dir}...")
    dfs = []
    for json_file in json_files:
        dfs.append(load_json_predictions(json_file))

    return pd.concat(dfs, ignore_index=True)


def load_rubric_mapping(rubric_path: Path) -> dict[str, str]:
    """Load the rubric mapping from a CSV file."""
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
    """Score all predictions and write results to output CSV."""
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


def main() -> None:
    """Main function to score and analyze."""
    parser = argparse.ArgumentParser(
        description="Score and analyze extraction results."
    )
    parser = pbench.add_base_args(parser)
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Path to an existing CSV file containing predictions and scores.",
    )
    parser.add_argument(
        "--rubric_csv_filename",
        type=str,
        default=None,
        help="Filename of the rubric CSV file.",
    )
    parser.add_argument(
        "--cluster_file",
        type=str,
        default=None,
        help="Path to property clusters JSON file.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print and save analysis tables.",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    # Load property clusters
    global CLUSTERS
    cluster_path = Path(args.cluster_file) if args.cluster_file else CLUSTER_FILE
    if cluster_path.exists():
        try:
            with open(cluster_path, "r") as f:
                CLUSTERS = json.load(f)
            logger.info(f"Loaded property clusters from {cluster_path}")
        except Exception as e:
            logger.warning(f"Failed to load property clusters: {e}")
    else:
        logger.warning(f"Property clusters file not found at {cluster_path}")

    analysis_dir = args.output_dir / args.domain / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Try explicit CSV
    if args.input_csv:
        try:
           df = pd.read_csv(args.input_csv)
           logger.info(f"Loaded {len(df)} rows from {args.input_csv}")
        except Exception as e:
           logger.error(f"Failed to load CSV: {e}")
           sys.exit(1)
           
    else:
        # 2. Try loading JSONs from standard directory structure
        try:
            df = load_all_predictions(args)
            logger.info(f"Loaded {len(df)} rows from JSONs")
            
            # Score them if they don't have scores (Harbor JSONs coming from collect_harbor_results.py MIGHT miss scores? 
            # No, details.json HAS scores, but we didn't explicitly put 'score' in the pbench formatted JSON.
            # Let's check collect_harbor_results.py logic again. 
            # It saves 'true' and 'pred', but NOT 'score' in the root. 
            # So we MUST re-score here.
            
            # Setup Rubric
            if args.rubric_csv_filename:
                rubric_path = Path(args.rubric_csv_filename)
            elif args.domain == "precedent-search":
                rubric_path = pbench.ASSETS_DIR / "hard" / "rubric.csv"
            else:
                rubric_path = pbench.ASSETS_DIR / args.domain / "rubric.csv"
            
            scores_dir = args.output_dir / args.domain / "scores"
            scores_dir.mkdir(parents=True, exist_ok=True)
            output_csv = scores_dir / "scored_results.csv"
            
            logger.info("Scoring predictions...")
            df = score_predictions(df, rubric_path, output_csv)
            
        except ValueError as e:
             logger.error(e)
             sys.exit(1)

    if args.analyze:
        analyze_scores(df, analysis_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main()
