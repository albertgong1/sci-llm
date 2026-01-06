"""Precision/Recall scoring for material property extraction.

This script compares extracted properties against a Ground Truth (GT) database
to calculate precision, recall, and F1 scores. It handles the alignment of
predicted values to GT values using property-specific rubrics.

Usage:
    python score_pr.py --preds preds.csv --gt gt.csv --rubric rubric.csv --output_dir out
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
try:
    from dotenv import load_dotenv
    load_dotenv()
    import os
    if "HF_TOKEN" in os.environ:
        logging.info("HF_TOKEN found in environment variables.")
    else:
        logging.warning("HF_TOKEN NOT found in environment. Ensure it is set in .env if accessing gated datasets.")
except ImportError:
    logging.warning("python-dotenv not found, .env file will not be loaded.")

# Try to import scorers from pbench_eval.utils
# Adjust the import path if necessary based on where this script is run from
try:
    from pbench_eval.utils import (
        scorer_categorical,
        scorer_si,
        scorer_space_group,
    )
    # Pymatgen might not be installed
    try:
        from pbench_eval.utils import scorer_pymatgen
    except ImportError:
        logging.warning("pymatgen scorer not available. 'pymatgen' rubric will default to string matching.")
        def scorer_pymatgen(pred, gt):
            return str(pred).strip() == str(gt).strip()

except ImportError:
    # If running from examples/extraction, we might need to add src to path
    import sys
    sys.path.append(str(Path(__file__).parents[3] / "src"))
    try:
        from pbench_eval.utils import (
            scorer_categorical,
            scorer_si,
            scorer_space_group,
        )
        try:
            from pbench_eval.utils import scorer_pymatgen
        except ImportError:
            logging.warning("pymatgen scorer not available. 'pymatgen' rubric will default to string matching.")
            def scorer_pymatgen(pred, gt):
                return str(pred).strip() == str(gt).strip()
    except ImportError as e:
        logging.error(f"Could not import pbench_eval.utils: {e}")
        # dummy fallbacks
        def scorer_categorical(p, g, mapping=None): return str(p) == str(g)
        def scorer_si(p, g): return str(p) == str(g)
        def scorer_space_group(p, g): return str(p) == str(g)
        def scorer_pymatgen(p, g): return str(p) == str(g)


def parse_numeric_value(value: Any) -> Optional[float]:
    """Parse a numeric value from a string, handling scientific notation and parenthetical uncertainties.
    
    Copied from score.py to avoid circular dependencies if score.py imports this.
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


def load_rubric(rubric_path: Path) -> Dict[str, str]:
    """Load rubric mapping from CSV.
    
    Expects columns: 'label', 'rubric' (or 'property_name', 'rubric')
    Returns: dict {property_label: rubric_type}
    """
    df = pd.read_csv(rubric_path)
    
    # Handle column naming variations
    if "label" in df.columns:
        key_col = "label"
    elif "property_name" in df.columns:
        key_col = "property_name"
    else:
        raise ValueError(f"Rubric CSV at {rubric_path} must have 'label' or 'property_name' column.")
        
    if "rubric" not in df.columns:
        raise ValueError(f"Rubric CSV at {rubric_path} must have 'rubric' column.")

    # Drop rows where rubric is NaN
    df = df.dropna(subset=[key_col, "rubric"])
    
    return dict(zip(df[key_col], df["rubric"]))


def load_clusters(cluster_path: Path) -> Dict[str, Dict[str, str]]:
    """Load property clusters from JSON."""
    if not cluster_path.exists():
        logging.warning(f"Clusters file not found at {cluster_path}. Categorical scoring will be exact match.")
        return {}
    
    with open(cluster_path, "r") as f:
        return json.load(f)


def score_pair(pred: Any, gt: Any, rubric: str, property_name: Optional[str] = None, clusters: Optional[Dict] = None) -> bool:
    """Score a single prediction-GT pair based on rubric."""
    
    if rubric == "0.1% SI":
        p_val = parse_numeric_value(pred)
        g_val = parse_numeric_value(gt)
        if p_val is None or g_val is None:
            return False
        return scorer_si(p_val, g_val)
        
    elif rubric == "pymatgen":
        if pd.isna(pred) or pd.isna(gt):
            return False
        return scorer_pymatgen(str(pred), str(gt))
        
    elif rubric == "categorical":
        if pd.isna(pred) or pd.isna(gt):
            return False
        
        mapping = clusters.get(property_name, None) if (clusters and property_name) else None
        return scorer_categorical(str(pred), str(gt), mapping=mapping)
        
    elif rubric == "space group" or rubric == "space_group":
        if pd.isna(pred) or pd.isna(gt):
            return False
        return scorer_space_group(str(pred), str(gt))
        
    else:
        # Check for numeric rubrics (e.g. "0.1% SI" might appear as just "SI" in some versions?)
        # For now, default to exact string match if unknown
        # logging.warning(f"Unknown rubric '{rubric}', defaulting to string match.")
        return str(pred).strip() == str(gt).strip()


def match_properties(
    preds_list: List[Any], 
    gt_list: List[Any], 
    rubric: str,
    property_name: Optional[str] = None,
    clusters: Optional[Dict] = None
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Match predictions to ground truth values using greedy bipartite matching.
    
    Returns:
        (tp, fp, fn, match_details)
    """
    # Create simple bipartite graph edges (pred_idx, gt_idx) where match is True
    matches = []
    for i, p in enumerate(preds_list):
        for j, g in enumerate(gt_list):
            if score_pair(p, g, rubric, property_name, clusters):
                matches.append((i, j))
    
    # Greedy matching: pick edges
    used_preds = set()
    used_gts = set()
    
    match_details = []
    
    def fmt(v): return str(v) if v is not None else ""
    
    # TP
    for p_idx, g_idx in matches:
        if p_idx not in used_preds and g_idx not in used_gts:
            used_preds.add(p_idx)
            used_gts.add(g_idx)
            match_details.append({
                "pred_value": fmt(preds_list[p_idx]),
                "gt_value": fmt(gt_list[g_idx]),
                "status": "TP"
            })
            
    # FP (Unmatched Preds)
    for i, p in enumerate(preds_list):
        if i not in used_preds:
             match_details.append({
                "pred_value": fmt(p),
                "gt_value": "",
                "status": "FP"
            })
            
    # FN (Unmatched GTs)
    for j, g in enumerate(gt_list):
        if j not in used_gts:
             match_details.append({
                "pred_value": "",
                "gt_value": fmt(g),
                "status": "FN"
            })
            
    tp = len(used_preds)
    fp = len(preds_list) - len(used_preds)
    fn = len(gt_list) - len(used_gts)
    
    return tp, fp, fn, match_details


def main():
    parser = argparse.ArgumentParser(description="Calculate Precision/Recall for extracted properties.")
    parser.add_argument("--preds", type=Path, required=True, help="Path to predictions CSV")
    parser.add_argument("--gt", type=str, required=True, help="Path to Ground Truth CSV or Hugging Face dataset name")
    parser.add_argument("--rubric", type=Path, required=True, help="Path to Rubric CSV")
    
    # Default clusters path
    default_clusters = Path(__file__).parent.parent / "assets" / "property_clusters_gemini-3-pro-preview.json"
    parser.add_argument("--clusters", type=Path, default=default_clusters, help="Path to Property Clusters JSON")
    
    parser.add_argument("--space_groups", type=Path, help="Path to space_groups_normalized.json")
    
    parser.add_argument("--output_dir", "-od", type=Path, default=Path("out"), help="Output directory")
    
    args = parser.parse_args()
    
    # Inject space groups if provided
    if args.space_groups and args.space_groups.exists():
        logging.info(f"Loading space groups from {args.space_groups}")
        try:
            with open(args.space_groups, "r") as f:
                sg_data = json.load(f)
            
            # Try to inject into pbench_eval.utils
            import sys
            if 'pbench_eval.utils' in sys.modules:
                sys.modules['pbench_eval.utils'].SPACE_GROUPS = sg_data
                logging.info("Injected space groups into pbench_eval.utils")
            else:
                logging.warning("pbench_eval.utils not found in sys.modules, could not inject space groups.")
        except Exception as e:
            logging.error(f"Failed to load/inject space groups: {e}")
    
    # Ensure output dir exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info(f"Loading predictions from {args.preds}")
    if args.preds.is_dir():
        # Glob recursively or non-recursively? User said "in the folder", assume flat or simple.
        # Let's just find all CSVs
        files = list(args.preds.glob("*.csv"))
        if not files:
            logging.error(f"No CSV files found in {args.preds}")
            return
        
        logging.info(f"Found {len(files)} CSV files in directory.")
        dfs = []
        for p in files:
            try:
                df = pd.read_csv(p)
                dfs.append(df)
            except Exception as e:
                logging.warning(f"Failed to read {p}: {e}")
        
        if not dfs:
            logging.error("No valid CSV files loaded.")
            return
            
        preds_df = pd.concat(dfs, ignore_index=True)
    else:
        preds_df = pd.read_csv(args.preds)
    
    # GT Loading moved below to handle HF datasets
    
    logging.info(f"Loading rubric from {args.rubric}")
    rubric_map = load_rubric(args.rubric)
    
    logging.info(f"Loading clusters from {args.clusters}")
    clusters = load_clusters(args.clusters)
    
    # Ensure columns exist
    # Preds: refno, property_name, value_string
    # GT: refno, label, property_value
    
    # 1. Prediction Value Column
    pred_val_col = "value_string"
    if pred_val_col not in preds_df.columns:
        # Fallback for other formats
        possible = [c for c in preds_df.columns if "value" in c.lower() and "num" not in c.lower()]
        if possible:
            pred_val_col = possible[0]
            logging.warning(f"Prediction value column 'value_string' not found. Using '{pred_val_col}' instead.")
        else:
            raise ValueError("Could not identify value column in predictions. Expected 'value_string'.")

    # 2. Load GT (Handle HF Dataset vs CSV)
    # Check if gt arg looks like a file path
    gt_path = str(args.gt)
    is_hf_dataset = not (Path(gt_path).exists() or gt_path.endswith('.csv'))
    
    gt_df = pd.DataFrame() # Initialize
    
    if is_hf_dataset:
        logging.info(f"Arguments suggest GT is a Hugging Face dataset: {gt_path}")
        try:
            from datasets import load_dataset
            # Load specific revision v2.0.1
            ds = load_dataset(gt_path, split="test", revision="v2.0.1")
            
            # Since we specified split="test", 'ds' is the Dataset itself, not a Dict
            logging.info(f"Loaded split 'test' from {gt_path}")
            data = ds
            
            # Flatten 'properties' column
            # Expected structure: refno, properties=[{property_name: ..., value_string: ...}, ...]
            gt_records = []
            for item in data:
                refno = item['refno']
                for prop in item['properties']:
                    # We expect prop to be a dict
                    record = {
                        'refno': refno,
                        'label': prop.get('property_name'), 
                        'property_value': prop.get('value_string'),
                        # Add other fields if useful for debugging
                        'category': prop.get('category')
                    }
                    gt_records.append(record)
            
            gt_df = pd.DataFrame(gt_records)
            logging.info(f"Flattened HF dataset into {len(gt_df)} rows.")
            
        except ImportError:
            logging.error("The 'datasets' library is required to load Hugging Face datasets. Please install it.")
            return
        except Exception as e:
            logging.error(f"Failed to load HF dataset: {e}")
            return
            
    else:
        # Load as CSV
        gt_df = pd.read_csv(args.gt)

    # 3. GT Value Column (If CSV)
    # If loaded from HF, we already standardized to 'property_value'
    gt_val_col = "property_value"
    if gt_val_col not in gt_df.columns:
        # Fallback
        possible = [c for c in gt_df.columns if "value" in c.lower()]
        if possible:
            gt_val_col = possible[0]
            logging.warning(f"GT value column 'property_value' not found. Using '{gt_val_col}' instead.")
        else:
            raise ValueError("Could not identify value column in GT. Expected 'property_value'.")
            
    # 4. GT Label Column (used for matching with Pred property_name)
    # If loaded from HF, we already standardized to 'label'
    gt_label_col = "label"
    if gt_label_col not in gt_df.columns:
         # Fallback to property_name if label not found, but warn
        if "property_name" in gt_df.columns:
            gt_label_col = "property_name"
            logging.warning("GT 'label' column not found. Using 'property_name' as label.")
        else:
            raise ValueError("Could not identify label column in GT. Expected 'label'.")

    # Process per Refno
    # We want to match Pred(property_name) -> GT(gt_label_col)
    
    results = []
    all_match_details = []
    
    all_refnos = set(preds_df["refno"].unique()) | set(gt_df["refno"].unique())
    
    # Load Aliases if available
    aliases = {}
    alias_path = args.output_dir / "property_aliases.json"
    if alias_path.exists():
        try:
            with open(alias_path, "r") as f:
                aliases = json.load(f)
            logging.info(f"Loaded {len(aliases)} property aliases from {alias_path}")
        except Exception as e:
            logging.warning(f"Failed to load property aliases: {e}")

    # Load Rubric
    rubric_map = load_rubric(args.rubric) # Changed from load_rubric_mapping to load_rubric to match existing function name
    
    # Load Clusters if provided
    clusters = {}
    if args.clusters:
        clusters = load_clusters(Path(args.clusters))

    # --- SCORING LOGIC ---
    results = []
    
    # Group by refno (paper)
    # We need to intersect the refnos present in both
    pred_refnos = set(preds_df['refno'].unique())
    gt_refnos = set(gt_df['refno'].unique())
    
    common_refnos = pred_refnos.intersection(gt_refnos)
    logging.info(f"Found {len(common_refnos)} papers in common between Predictions and Ground Truth.")
    
    for refno in common_refnos:
        p_df = preds_df[preds_df['refno'] == refno]
        g_df = gt_df[gt_df['refno'] == refno]
        
        # Get all properties for this paper
        # Apply Aliases to Predictions -> determine 'effective_name'
        # But wait, we iterate by *property name*.
        # Strategy:
        # 1. Map all predicted property names to their canonical (GT) names using aliases.
        # 2. Add a column 'scoring_name' to p_df.
        
        p_df = p_df.copy()
        p_df['scoring_name'] = p_df['property_name'].apply(lambda x: aliases.get(x, x))
        
        # Properties to score: Union of (Mapped Preds) and (GT Labels)
        pred_props = set(p_df['scoring_name'].dropna().unique())
        gt_props = set(g_df[gt_label_col].dropna().unique())
        
        all_props = pred_props.union(gt_props)
        
        for prop in all_props:
            # Get values for this property
            # Preds: match on 'scoring_name'
            p_vals = p_df[p_df['scoring_name'] == prop][pred_val_col].dropna().tolist()
            # GT: match on gt_label_col
            g_vals = g_df[g_df[gt_label_col] == prop][gt_val_col].dropna().tolist()
            
            # If both empty (shouldn't happen due to union), skip
            if not p_vals and not g_vals:
                continue
                
            # Get rubric for this property
            # If prop not in rubric, default to exact string match or skip? 
            # score_row handles unknown rubric by returning None, but we need a rubric string.
            # default to "exact" if not found? Or warn?
            rubric = rubric_map.get(prop, "exact") 
            
            # Bipartite Matching
            # ... (rest of matching logic)e TP, FP, FN
            tp, fp, fn, details = match_properties(p_vals, g_vals, rubric, property_name=prop, clusters=clusters)
            
            # Enrich details with context
            for d in details:
                d["refno"] = refno
                d["property_name"] = prop
                d["rubric"] = rubric
                all_match_details.append(d)
                
            if tp > 0 or fp > 0 or fn > 0:
                results.append({
                    "refno": refno,
                    "property_name": prop,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "rubric": rubric,
                    "gt_count": len(g_vals),
                    "pred_count": len(p_vals),
                    "details": details
                })
            
    # Aggregate results
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        logging.warning("No matching refnos or properties found.")
        return

    # Calculate global metrics per property
    prop_stats = results_df.groupby("property_name")[["tp", "fp", "fn"]].sum().reset_index()
    
    # Calculate P/R/F1
    prop_stats["Precision"] = prop_stats["tp"] / (prop_stats["tp"] + prop_stats["fp"])
    prop_stats["Recall"] = prop_stats["tp"] / (prop_stats["tp"] + prop_stats["fn"])
    prop_stats["F1"] = 2 * (prop_stats["Precision"] * prop_stats["Recall"]) / (prop_stats["Precision"] + prop_stats["Recall"])
    
    # Handle division by zero (fillna with 0)
    prop_stats = prop_stats.fillna(0.0)
    
    # Save
    summary_path = args.output_dir / "score_pr_summary.csv"
    prop_stats.to_csv(summary_path, index=False)
    
    detailed_path = args.output_dir / "score_pr_detailed.csv"
    results_df.to_csv(detailed_path, index=False)
    
    # Save Detailed Matches
    if all_match_details:
        matches_df = pd.DataFrame(all_match_details)
        matches_path = args.output_dir / "score_pr_matches.csv"
        # Reorder columns for readability
        cols = ["refno", "property_name", "status", "pred_value", "gt_value", "rubric"]
        # Filter cols that exist
        cols = [c for c in cols if c in matches_df.columns]
        matches_df = matches_df[cols]
        matches_df.to_csv(matches_path, index=False)
        logging.info(f"Detailed matches saved to {matches_path}")
    
    # Save Flattened GT
    gt_dump_path = args.output_dir / "gt_dump.csv"
    gt_df.to_csv(gt_dump_path, index=False)
    logging.info(f"Flattened GT saved to {gt_dump_path}")

    # Generate Report
    report_path = args.output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("Property Extraction Scoring Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Predictions: {args.preds}\n")
        f.write(f"Ground Truth: {args.gt}\n\n")
        
        f.write(f"Total Papers Processed: {len(all_refnos)}\n")
        f.write(f"Total Properties Scored: {len(prop_stats)}\n\n")

        f.write("-" * 20 + "\n")
        f.write("Metric Definitions\n")
        f.write("-" * 20 + "\n")
        f.write("TP (True Positive):  Extracted value matches Ground Truth (within tolerance).\n")
        f.write("FP (False Positive): Extracted value does NOT match Ground Truth, or formatted incorrectly.\n")
        f.write("FN (False Negative): Ground Truth value exists but was not Extracted.\n")
        f.write("Macro Average:       Unweighted mean of metrics across all property types (treats rare properties equally to common ones).\n\n")
        
        # Overall Stats
        total_tp = prop_stats["tp"].sum()
        total_fp = prop_stats["fp"].sum()
        total_fn = prop_stats["fn"].sum()
        macro_prec = prop_stats["Precision"].mean()
        macro_rec = prop_stats["Recall"].mean()
        macro_f1 = prop_stats["F1"].mean()
        
        f.write("-" * 20 + "\n")
        f.write("Aggregate Metrics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total TP: {int(total_tp)}\n")
        f.write(f"Total FP: {int(total_fp)}\n")
        f.write(f"Total FN: {int(total_fn)}\n")
        f.write(f"Macro Precision: {macro_prec:.3f}\n")
        f.write(f"Macro Recall:    {macro_rec:.3f}\n")
        f.write(f"Macro F1:        {macro_f1:.3f}\n\n")

        f.write("-" * 20 + "\n")
        f.write("Per-Property Breakdown\n")
        f.write("-" * 20 + "\n")
        from tabulate import tabulate
        prop_stats_sorted = prop_stats.sort_values("F1", ascending=False)
        table = tabulate(prop_stats_sorted, headers="keys", tablefmt="github", showindex=False, floatfmt=".3f")
        f.write(table)
        f.write("\n")

    logging.info(f"Report saved to {report_path}")

    # Generate Global Property Alignment Report
    alignment_path = args.output_dir / "property_alignment.txt"
    
    # Global sets of property names
    # Preds: property_name
    all_pred_props = set(preds_df["property_name"].dropna().unique())
    # GT: use identified gt_label_col
    all_gt_props = set(gt_df[gt_label_col].dropna().unique())
    
    intersection = sorted(list(all_pred_props & all_gt_props))
    only_in_preds = sorted(list(all_pred_props - all_gt_props))
    only_in_gt = sorted(list(all_gt_props - all_pred_props))
    
    with open(alignment_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("GLOBAL Property Name Alignment Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Unique Properties in Predictions: {len(all_pred_props)}\n")
        f.write(f"Total Unique Properties in Ground Truth: {len(all_gt_props)}\n")
        f.write(f"Overlapping Properties (Exact Match): {len(intersection)}\n\n")
        
        f.write("-" * 20 + "\n")
        f.write("MATCHING PROPERTIES (In Both)\n")
        f.write("-" * 20 + "\n")
        for p in intersection:
            f.write(f"  [MATCH] {p}\n")
        f.write("\n")
            
        f.write("-" * 20 + "\n")
        f.write("UNMATCHED PREDICTIONS (Only in Extracted)\n")
        f.write("-" * 20 + "\n")
        for p in only_in_preds:
            f.write(f"  [PRED_ONLY] {p}\n")
        f.write("\n")
            
        f.write("-" * 20 + "\n")
        f.write("UNMATCHED GROUND TRUTH (Only in GT)\n")
        f.write("-" * 20 + "\n")
        for p in only_in_gt:
            f.write(f"  [GT_ONLY]   {p}\n")
        f.write("\n")

    logging.info(f"Property alignment report saved to {alignment_path}")
    
    # ---------------------------------------------------------
    # Generate Per-Paper Property Alignment Reports
    # ---------------------------------------------------------
    alignment_dir = args.output_dir / "property_alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by refno
    pred_groups = preds_df.groupby("refno")
    gt_groups = gt_df.groupby("refno")
    
    all_refnos = sorted(list(set(pred_groups.groups.keys()) | set(gt_groups.groups.keys())))
    
    count_saved = 0
    for refno in all_refnos:
        # Get props for this paper
        p_props = set()
        if refno in pred_groups.groups:
            p_props = set(pred_groups.get_group(refno)["property_name"].dropna().unique())
            
        g_props = set()
        if refno in gt_groups.groups:
            g_props = set(gt_groups.get_group(refno)[gt_label_col].dropna().unique())
            
        if not p_props and not g_props:
            continue
            
        # Comparison
        paper_intersect = sorted(list(p_props & g_props))
        paper_only_preds = sorted(list(p_props - g_props))
        paper_only_gt = sorted(list(g_props - p_props))
        
        file_path = alignment_dir / f"property_alignment_refno={refno}.txt"
        with open(file_path, "w") as f:
            f.write(f"Property Alignment Report for: {refno}\n")
            f.write("="*60 + "\n\n")
            
            f.write("MATCHING PROPERTIES:\n")
            for p in paper_intersect:
                f.write(f"  [MATCH] {p}\n")
            f.write("\n")
            
            f.write("UNMATCHED PREDICTIONS (Pred Only):\n")
            for p in paper_only_preds:
                f.write(f"  [PRED_ONLY] {p}\n")
            f.write("\n")
            
            f.write("UNMATCHED GROUND TRUTH (GT Only):\n")
            for p in paper_only_gt:
                f.write(f"  [GT_ONLY]   {p}\n")
        
        count_saved += 1
        
    logging.info(f"Saved {count_saved} per-paper alignment reports to {alignment_dir}")

    # ---------------------------------------------------------
    # Generate Per-Paper Calculation Reports (TP/FP/FN Counts)
    # ---------------------------------------------------------
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # We already have 'results' list with dictionary for each paper
    # [ { 'refno': X, 'tp': Y, 'fp': Z, 'fn': W, 'details': [...] }, ... ]
    
    for res in results:
        refno = res['refno']
        tp = res['tp']
        fp = res['fp']
        fn = res['fn']
        details = res['details']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        r_path = reports_dir / f"report_refno={refno}.txt"
        with open(r_path, "w") as f:
            f.write(f"Extraction Report for Paper: {refno}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall:    {recall:.2f}\n")
            f.write(f"F1 Score:  {f1:.2f}\n")
            f.write(f"Counts:    TP={tp}, FP={fp}, FN={fn}\n\n")
            
            f.write("-" * 20 + "\n")
            f.write("DETAILED MATCHES\n")
            f.write("-" * 20 + "\n")
            
            # Write TPs first
            tps = [d for d in details if d['status'] == 'TP']
            if tps:
                f.write("\nTrue Positives (Correctly Extracted):\n")
                f.write(f"{'Property':<30} | {'Predicted':<20} | {'GT Value'}\n")
                f.write("-" * 80 + "\n")
                for d in tps:
                    f.write(f"{d['property_name']:<30} | {d['pred_value']:<20} | {d['gt_value']}\n")
            
            # Write FPs
            fps = [d for d in details if d['status'] == 'FP']
            if fps:
                f.write("\nFalse Positives (Extracted but not in GT or Incorrect Value):\n")
                f.write(f"{'Property':<30} | {'Value'}\n")
                f.write("-" * 50 + "\n")
                for d in fps:
                    f.write(f"{d['property_name']:<30} | {d['pred_value']}\n")

            # Write FNs
            fns = [d for d in details if d['status'] == 'FN']
            if fns:
                f.write("\nFalse Negatives (Missed from GT):\n")
                f.write(f"{'Property':<30} | {'GT Value'}\n")
                f.write("-" * 50 + "\n")
                for d in fns:
                    f.write(f"{d['property_name']:<30} | {d['gt_value']}\n")

    logging.info(f"Saved {len(results)} per-paper detailed reports to {reports_dir}")

    logging.info(f"Summary saved to {summary_path}")
    # Print nice table
    
    # Auto-run Mismatch Analysis to generate alias candidates
    try:
        from analyze_mismatches import analyze_mismatches_from_df
        logging.info("Running automatic mismatch analysis...")
        analyze_mismatches_from_df(matches_df, args.output_dir, top_n=200)
    except ImportError:
        # If in examples/extraction, simple import works. If run from root, might need path adjustment
        # Fallback for path issues
        logging.warning("Could not import analyze_mismatches for auto-generation. Ensure it is in the python path.")
    except Exception as e:
        logging.error(f"Error running automatic mismatch analysis: {e}")


if __name__ == "__main__":
    main()
