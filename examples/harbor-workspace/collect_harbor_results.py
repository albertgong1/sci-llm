"""Collect Harbor verifier outputs into PBench-style prediction JSON files.

Usage:
  uv run python examples/harbor-workspace/collect_harbor_results.py
  uv run python examples/harbor-workspace/collect_harbor_results.py \
    --workspace /path/to/workspace
  uv run python examples/harbor-workspace/collect_harbor_results.py \
    --trials-dir /path/to/trials \
    --output-dir /path/to/out/harbor/precedent-search/preds

This script reads `verifier/details.json` from each trial and writes one JSON file
per trial with PBench-style records (refno/material/property_name/true/pred/response).
"""

import argparse
import json
from pathlib import Path


def default_workspace_root() -> Path:
    """Return the workspace root (directory containing this script)."""
    return Path(__file__).resolve().parent


def main() -> int:
    """Collect trial results and write one JSON file per trial."""
    parser = argparse.ArgumentParser(
        description="Collect Harbor results into PBench format."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root (default: this script's directory).",
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=None,
        help="Trials directory (default: <workspace>/trials).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <workspace>/out/harbor/precedent-search/preds).",
    )
    args = parser.parse_args()

    workspace = (args.workspace or default_workspace_root()).resolve()
    if args.trials_dir is None:
        args.trials_dir = workspace / "trials"
    if args.output_dir is None:
        args.output_dir = workspace / "out" / "harbor" / "precedent-search" / "preds"

    if not args.trials_dir.exists():
        raise SystemExit(f"Trials directory not found: {args.trials_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for trial_dir in args.trials_dir.iterdir():
        if not trial_dir.is_dir():
            continue

        details_path = trial_dir / "verifier/details.json"
        if not details_path.exists():
            continue

        try:
            with open(details_path) as f:
                details = json.load(f)
        except Exception as e:
            print(f"Skipping {trial_dir}: {e}")
            continue

        # Extract info
        rows = details.get("rows", [])
        if not rows:
            continue

        # Convert to pbench format list-of-dicts
        # Need: refno, material, property_name, true{value, unit}, pred{value, unit}, response{pred}
        # In details.json, we have this flat in 'rows'

        pbench_records = []
        for row in rows:
            # pred_raw can be None if no prediction was matched
            pred_raw = row.get("pred_raw") or {}

            # Fallback to 'value' if 'value_string' is missing (compatibility with extraction task)
            pred_val_str = pred_raw.get("value_string") or pred_raw.get("value") or ""

            record = {
                "refno": details.get("refno", "unknown"),
                "material": row["material"],
                "property_name": row["property_name"],
                "true": {"value": row["answer_value"], "unit": row["answer_unit"]},
                "pred": {"value": row["pred_value"], "unit": row["pred_unit"]},
                "rubric": row.get("rubric"),
                "response": {
                    "pred": pred_val_str,
                    "source_doi": pred_raw.get("source_doi"),
                    "conditions": pred_raw.get("conditions"),
                    "related_materials": pred_raw.get("related_materials"),
                },
                # Add metadata to enable filtering in score_task.py
                "metadata": {
                    "task": details.get("task", "precedent-search"),
                    "trial_id": trial_dir.name,
                },
            }
            pbench_records.append(record)

        # Save one JSON per trial
        out_name = f"{trial_dir.name}.json"
        out_path = args.output_dir / out_name
        with open(out_path, "w") as f:
            json.dump(pbench_records, f, indent=2)

        count += 1

    print(f"Collected results from {count} trials to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
